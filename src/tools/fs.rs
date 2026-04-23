use anyhow::{Context, Result, anyhow};
use std::ffi::OsString;
use std::fs;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use super::ToolResult;

#[derive(Debug, Clone)]
pub struct FsTool {
    root: PathBuf,
}

impl FsTool {
    // TODO: consider exposing a `stat_path` helper on the public tools/fs.rs surface
    // so callers can query metadata (size, is_dir, modified) without reading file contents.

    pub fn new(root: PathBuf) -> Result<Self> {
        let root = root
            .canonicalize()
            .with_context(|| format!("failed to resolve startup directory '{}'", root.display()))?;
        Ok(Self { root })
    }

    pub fn read_file(&self, input: &str) -> Result<ToolResult> {
        let path = self.resolve(input)?;
        // Read raw bytes first so we can return a clearer error when the file isn't valid UTF-8.
        let bytes = fs::read(&path).with_context(|| {
            format!(
                "failed to read file '{input}' inside startup directory '{}'",
                self.root.display()
            )
        })?;

        let body = String::from_utf8(bytes).map_err(|_| {
            anyhow!("file '{input}' contains invalid UTF-8; this tool only supports reading text files")
        })?;

        let preview = body.lines().next().unwrap_or_default().to_string();
        Ok(ToolResult::text(preview, body))
    }

    pub fn list_dir(&self, input: &str) -> Result<ToolResult> {
        let path = self.resolve(input)?;
        let mut lines = Vec::new();

        for entry in fs::read_dir(&path).with_context(|| {
            format!(
                "failed to list directory '{input}' inside startup directory '{}'",
                self.root.display()
            )
        })? {
            let entry = entry.with_context(|| {
                format!(
                    "failed to read an entry from directory '{input}' inside startup directory '{}'",
                    self.root.display()
                )
            })?;
            lines.push(entry.file_name().to_string_lossy().to_string());
        }

        lines.sort();
        Ok(ToolResult::text(
            format!("{} entries", lines.len()),
            lines.join("\n"),
        ))
    }

    // Note: This two-pass approach (resolving an existing prefix and then canonicalizing the final path)
    // reduces symlink-escape attacks but cannot eliminate a TOCTOU race — the filesystem may change
    // between checks. For a read-only tool that's scoped to a startup directory, this risk is acceptable
    // because we never write and we additionally verify the canonicalized final path stays inside root.
    fn resolve(&self, input: &str) -> Result<PathBuf> {
        let relative = self.normalize_relative_path(input)?;
        let candidate = self.root.join(&relative);
        let resolved = self.resolve_existing_prefix(&candidate)?;

        if !resolved.starts_with(&self.root) {
            return Err(anyhow!(
                "path '{input}' escapes the startup directory '{}'",
                self.root.display()
            ));
        }

        match fs::symlink_metadata(&candidate) {
            Ok(_) => {
                let final_path = candidate.canonicalize().with_context(|| {
                    format!(
                        "failed to resolve '{input}' inside startup directory '{}'",
                        self.root.display()
                    )
                })?;
                if !final_path.starts_with(&self.root) {
                    return Err(anyhow!(
                        "path '{input}' escapes the startup directory '{}'",
                        self.root.display()
                    ));
                }
                Ok(final_path)
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(candidate),
            Err(error) => Err(error).with_context(|| {
                format!(
                    "failed to inspect '{input}' inside startup directory '{}'",
                    self.root.display()
                )
            }),
        }
    }

    fn normalize_relative_path(&self, input: &str) -> Result<PathBuf> {
        let path = Path::new(input);
        if path.is_absolute() {
            return Err(anyhow!(
                "path '{input}' must stay inside the startup directory '{}'",
                self.root.display()
            ));
        }

        let mut normalized = PathBuf::new();
        for component in path.components() {
            match component {
                Component::Normal(part) => normalized.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(anyhow!(
                        "path '{input}' must stay inside the startup directory '{}'",
                        self.root.display()
                    ));
                }
            }
        }

        Ok(normalized)
    }

    fn resolve_existing_prefix(&self, path: &Path) -> Result<PathBuf> {
        let mut current = path;
        let mut suffix = Vec::<OsString>::new();

        loop {
            match fs::symlink_metadata(current) {
                Ok(_) => {
                    let mut resolved = current.canonicalize().with_context(|| {
                        format!(
                            "failed to resolve '{current}' inside startup directory '{root}'",
                            root = self.root.display(),
                            current = current.display()
                        )
                    })?;
                    for component in suffix.iter().rev() {
                        resolved.push(component);
                    }
                    return Ok(resolved);
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    let name = current.file_name().ok_or_else(|| {
                        anyhow!(
                            "path '{}' must stay inside the startup directory '{}'",
                            path.display(),
                            self.root.display()
                        )
                    })?;
                    suffix.push(name.to_os_string());
                    current = current.parent().ok_or_else(|| {
                        anyhow!(
                            "path '{}' must stay inside the startup directory '{}'",
                            path.display(),
                            self.root.display()
                        )
                    })?;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "failed to inspect '{}' inside startup directory '{}'",
                            current.display(),
                            self.root.display()
                        )
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolResult;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// RAII guard to ensure test directories are removed even if a test panics.
    struct TestDirGuard(PathBuf);
    impl Drop for TestDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn rejects_escape_outside_root() {
        let tool = FsTool::new(std::env::current_dir().unwrap()).unwrap();
        let error = tool.read_file("../../etc/passwd").unwrap_err().to_string();
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn rejects_absolute_paths() {
        let tool = FsTool::new(std::env::current_dir().unwrap()).unwrap();
        let error = tool.read_file("/etc/passwd").unwrap_err().to_string();
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn reads_file_contents() {
        let root = test_root("read");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let _guard = TestDirGuard(root.clone());
        fs::write(root.join("note.txt"), "hello\nworld").unwrap();

        let tool = FsTool::new(root.clone()).unwrap();
        let result = tool.read_file("note.txt").unwrap();

        match result {
            ToolResult::Text { preview, body } => {
                assert_eq!(preview, "hello");
                assert_eq!(body, "hello\nworld");
            }
        }
    }

    #[test]
    fn lists_directory_entries() {
        let root = test_root("list");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("nested")).unwrap();
        let _guard = TestDirGuard(root.clone());
        fs::write(root.join("nested").join("file.txt"), "hello").unwrap();

        let tool = FsTool::new(root.clone()).unwrap();
        let result = tool.list_dir("nested").unwrap();
        let body = text_body(&result);
        assert!(body.contains("file.txt"));
    }

    #[test]
    fn rejects_symlink_target_outside_root() {
        let root = test_root("symlink");
        let outside = test_root("outside");
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&outside);
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        let _root_guard = TestDirGuard(root.clone());
        let _outside_guard = TestDirGuard(outside.clone());
        fs::write(outside.join("secret.txt"), "secret").unwrap();
        std::os::unix::fs::symlink(outside.join("secret.txt"), root.join("escape.txt")).unwrap();

        let tool = FsTool::new(root.clone()).unwrap();
        let error = tool.read_file("escape.txt").unwrap_err().to_string();
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn missing_file_error_has_context() {
        let root = test_root("missing-file");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let _guard = TestDirGuard(root.clone());

        let tool = FsTool::new(root.clone()).unwrap();
        let error = tool.read_file("missing.txt").unwrap_err().to_string();
        assert!(error.contains("failed to read file 'missing.txt'"));
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn missing_directory_error_has_context() {
        let root = test_root("missing-dir");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let _guard = TestDirGuard(root.clone());

        let tool = FsTool::new(root.clone()).unwrap();
        let error = tool.list_dir("missing").unwrap_err().to_string();
        assert!(error.contains("failed to list directory 'missing'"));
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn listing_file_error_has_context() {
        let root = test_root("list-file");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let _guard = TestDirGuard(root.clone());
        fs::write(root.join("note.txt"), "hello").unwrap();

        let tool = FsTool::new(root.clone()).unwrap();
        let error = tool.list_dir("note.txt").unwrap_err().to_string();
        assert!(error.contains("failed to list directory 'note.txt'"));
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn rejects_missing_startup_root() {
        let root = test_root("missing-root");
        let error = FsTool::new(root.clone()).unwrap_err().to_string();
        assert!(error.contains("failed to resolve startup directory"));
        assert!(error.contains(&root.display().to_string()));
    }

    #[test]
    fn reads_in_root_symlink_target() {
        let root = test_root("symlink-in-root");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("nested")).unwrap();
        let _guard = TestDirGuard(root.clone());
        fs::write(root.join("nested").join("file.txt"), "hello").unwrap();
        std::os::unix::fs::symlink(
            root.join("nested").join("file.txt"),
            root.join("link.txt"),
        )
        .unwrap();

        let tool = FsTool::new(root.clone()).unwrap();
        let result = tool.read_file("link.txt").unwrap();

        match result {
            ToolResult::Text { preview, body } => {
                assert_eq!(preview, "hello");
                assert_eq!(body, "hello");
            }
        }
    }

    #[test]
    fn reading_directory_error_has_context() {
        let root = test_root("read-dir");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("nested")).unwrap();
        let _guard = TestDirGuard(root.clone());

        let tool = FsTool::new(root.clone()).unwrap();
        let error = tool.read_file("nested").unwrap_err().to_string();
        assert!(error.contains("failed to read file 'nested'"));
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn invalid_utf8_returns_distinct_error() {
        let root = test_root("invalid-utf8");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let _guard = TestDirGuard(root.clone());
        fs::write(root.join("binary.dat"), vec![0xff, 0xfe, 0xfd]).unwrap();

        let tool = FsTool::new(root.clone()).unwrap();
        let error = tool.read_file("binary.dat").unwrap_err().to_string();
        assert!(error.contains("contains invalid UTF-8"));
    }

    fn text_body(result: &ToolResult) -> &str {
        match result {
            ToolResult::Text { body, .. } => body,
        }
    }

    fn test_root(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::current_dir()
            .unwrap()
            .join("target")
            .join("test-artifacts")
            .join(format!("glass-fs-{name}-{unique}"))
    }
}
