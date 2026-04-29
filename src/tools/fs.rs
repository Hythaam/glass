use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::{Value, json};
use std::ffi::OsString;
use std::fmt::Display;
use std::fs;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use super::{ToolExecutor, ToolResult};
use crate::llm::{ToolCall, ToolDefinition, ToolFunction};

#[derive(Debug, Clone)]
pub struct FsTool {
    root: PathBuf,
}

impl FsTool {
    pub fn new(root: PathBuf) -> Result<Self> {
        let root = root
            .canonicalize()
            .with_context(|| format!("failed to resolve startup directory '{}'", root.display()))?;
        Ok(Self { root })
    }

    pub fn read_file(&self, input: &str) -> Result<ToolResult> {
        let path = self.resolve(input)?;
        // Read raw bytes first so we can return a clearer error when the file isn't valid UTF-8.
        let bytes = fs::read(&path)
            .with_context(|| self.inside_startup_dir(format!("failed to read file '{input}'")))?;

        let body = String::from_utf8(bytes).map_err(|_| {
            anyhow!(
                "file '{input}' contains invalid UTF-8; this tool only supports reading text files"
            )
        })?;

        let preview = body.lines().next().unwrap_or_default().to_string();
        Ok(ToolResult::text(preview, body))
    }

    pub fn list_dir(&self, input: &str) -> Result<ToolResult> {
        let path = self.resolve(input)?;
        let mut lines = Vec::new();

        for entry in fs::read_dir(&path).with_context(|| {
            self.inside_startup_dir(format!("failed to list directory '{input}'"))
        })? {
            let entry = entry.with_context(|| {
                self.inside_startup_dir(format!("failed to read an entry from directory '{input}'"))
            })?;
            lines.push(entry.file_name().to_string_lossy().to_string());
        }

        lines.sort();
        Ok(ToolResult::text(
            format!("{} entries", lines.len()),
            lines.join("\n"),
        ))
    }

    fn write_file(&self, input: &str, contents: &str) -> Result<ToolResult> {
        let path = self.resolve(input)?;
        let parent = path
            .parent()
            .ok_or_else(|| anyhow!("{}", self.must_stay_inside_root(input)))?;
        fs::create_dir_all(parent).with_context(|| {
            self.inside_startup_dir(format!("failed to create parent directories for '{input}'"))
        })?;
        fs::write(&path, contents)
            .with_context(|| self.inside_startup_dir(format!("failed to write file '{input}'")))?;
        Ok(ToolResult::text(
            format!("wrote {input}"),
            format!("wrote {input}"),
        ))
    }

    fn edit_file(&self, input: &str, edits: &[FsTextEdit]) -> Result<ToolResult> {
        let path = self.resolve(input)?;
        let bytes = fs::read(&path)
            .with_context(|| self.inside_startup_dir(format!("failed to read file '{input}'")))?;
        let mut body = String::from_utf8(bytes).map_err(|_| {
            anyhow!(
                "file '{input}' contains invalid UTF-8; this tool only supports reading text files"
            )
        })?;

        for (index, edit) in edits.iter().enumerate() {
            body = self.apply_text_edit(input, index, &body, edit)?;
        }

        fs::write(&path, &body)
            .with_context(|| self.inside_startup_dir(format!("failed to write file '{input}'")))?;
        Ok(ToolResult::text(
            format!("edited {input}"),
            format!("edited {input}"),
        ))
    }

    // Note: This two-pass approach (resolving an existing prefix and then canonicalizing the final path)
    // reduces symlink-escape attacks but cannot eliminate a TOCTOU race — the filesystem may change
    // between checks. For a tool that's scoped to a startup directory, this risk is acceptable because
    // we additionally verify the canonicalized final path stays inside root before filesystem access.
    // Important: the second canonicalize() call after confirming the candidate exists is intentional and
    // must not be removed. It ensures any symlinks on the final path are resolved and produces a canonical
    // absolute target that we re-check against self.root. Without this final canonicalize, a symlink could be
    // swapped between the earlier prefix resolution and the final existence check, allowing the tool to return
    // a path that actually points outside the startup directory. While this does not eliminate TOCTOU races,
    // keeping the final canonicalize materially reduces the attack surface by verifying the resolved target
    // immediately before returning.
    fn resolve(&self, input: &str) -> Result<PathBuf> {
        let relative = self.normalize_relative_path(input)?;
        let candidate = self.root.join(&relative);
        let resolved = self.resolve_existing_prefix(&candidate)?;

        self.ensure_within_root(&resolved, input)?;

        match fs::symlink_metadata(&candidate) {
            Ok(_) => {
                let final_path = candidate.canonicalize().with_context(|| {
                    self.inside_startup_dir(format!("failed to resolve '{input}'"))
                })?;
                self.ensure_within_root(&final_path, input)?;
                Ok(final_path)
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(resolved),
            Err(error) => Err(error)
                .with_context(|| self.inside_startup_dir(format!("failed to inspect '{input}'"))),
        }
    }

    fn normalize_relative_path(&self, input: &str) -> Result<PathBuf> {
        if input.is_empty() {
            return Err(anyhow!(
                "path is empty; provide a file or directory path inside {}",
                self.startup_dir_label()
            ));
        }

        let path = Path::new(input);
        if path.is_absolute() {
            return Err(anyhow!("{}", self.must_stay_inside_root(input)));
        }

        let mut normalized = PathBuf::new();
        for component in path.components() {
            match component {
                Component::Normal(part) => normalized.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(anyhow!("{}", self.must_stay_inside_root(input)));
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
                        self.inside_startup_dir(format!(
                            "failed to resolve '{}'",
                            current.display()
                        ))
                    })?;
                    for component in suffix.iter().rev() {
                        resolved.push(component);
                    }
                    return Ok(resolved);
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    let name = current.file_name().ok_or_else(|| {
                        anyhow!(
                            "{}",
                            self.must_stay_inside_root(&path.display().to_string())
                        )
                    })?;
                    suffix.push(name.to_os_string());
                    current = current.parent().ok_or_else(|| {
                        anyhow!(
                            "{}",
                            self.must_stay_inside_root(&path.display().to_string())
                        )
                    })?;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        self.inside_startup_dir(format!(
                            "failed to inspect '{}'",
                            current.display()
                        ))
                    });
                }
            }
        }
    }

    fn ensure_within_root(&self, path: &Path, input: &str) -> Result<()> {
        if path.starts_with(&self.root) {
            Ok(())
        } else {
            Err(anyhow!("{}", self.escapes_root(input)))
        }
    }

    fn inside_startup_dir(&self, detail: String) -> String {
        format!("{detail} inside {}", self.startup_dir_label())
    }

    fn startup_dir_label(&self) -> String {
        format!("startup directory '{}'", self.root.display())
    }

    fn escapes_root(&self, input: &str) -> String {
        format!("path '{input}' escapes {}", self.startup_dir_label())
    }

    fn must_stay_inside_root(&self, input: impl Display) -> String {
        format!(
            "path '{input}' must stay inside {}",
            self.startup_dir_label()
        )
    }

    fn apply_text_edit(
        &self,
        input: &str,
        index: usize,
        current: &str,
        edit: &FsTextEdit,
    ) -> Result<String> {
        if edit.old_text.is_empty() {
            return Err(anyhow!(
                "edit {} for file '{input}' has empty old_text",
                index + 1
            ));
        }

        let matches = current.match_indices(&edit.old_text).collect::<Vec<_>>();
        match matches.len() {
            0 => Err(anyhow!(
                "edit {} for file '{input}' did not match any text",
                index + 1
            )),
            1 => {
                let start = matches[0].0;
                let end = start + edit.old_text.len();
                let mut updated = String::with_capacity(
                    current.len() - edit.old_text.len() + edit.new_text.len(),
                );
                updated.push_str(&current[..start]);
                updated.push_str(&edit.new_text);
                updated.push_str(&current[end..]);
                Ok(updated)
            }
            _ => Err(anyhow!(
                "edit {} for file '{input}' matched multiple locations",
                index + 1
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum FsToolArguments {
    ReadFile {
        path: String,
    },
    ListDir {
        path: String,
    },
    WriteFile {
        path: String,
        contents: String,
    },
    EditFile {
        path: String,
        edits: Vec<FsTextEdit>,
    },
}

#[derive(Debug, Deserialize)]
struct FsTextEdit {
    old_text: String,
    new_text: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FsToolArgumentShape {
    Tagged(FsToolArguments),
    Nested { args: FsPathOnlyArguments },
    PathOnly(FsPathOnlyArguments),
}

#[derive(Debug, Deserialize)]
struct FsPathOnlyArguments {
    path: String,
}

impl ToolExecutor for FsTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            r#type: "function".into(),
            function: ToolFunction {
                name: "fs".into(),
                description: "Read and write UTF-8 text files inside the startup directory".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["read_file", "list_dir", "write_file", "edit_file"],
                            "description": "Operation to run. Use read_file for UTF-8 text files, list_dir for directories, write_file for overwrite/create, and edit_file for targeted edits."
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative path inside the startup directory"
                        },
                        "contents": {
                            "type": "string",
                            "description": "UTF-8 text to write when op is write_file"
                        },
                        "edits": {
                            "type": "array",
                            "description": "Ordered text replacements to apply when op is edit_file",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_text": { "type": "string" },
                                    "new_text": { "type": "string" }
                                },
                                "required": ["old_text", "new_text"]
                            }
                        }
                    },
                    "required": ["op", "path"]
                }),
            },
        }]
    }

    fn execute(&mut self, call: &ToolCall) -> Result<ToolResult> {
        if call.name != "fs" {
            return Err(anyhow!("unsupported tool '{}'", call.name));
        }

        let arguments = self
            .parse_arguments(&call.arguments_json)
            .with_context(|| format!("invalid arguments for tool '{}'", call.name))?;

        match arguments {
            FsToolArguments::ReadFile { path } => self.read_file(&path),
            FsToolArguments::ListDir { path } => self.list_dir(&path),
            FsToolArguments::WriteFile { path, contents } => self.write_file(&path, &contents),
            FsToolArguments::EditFile { path, edits } => self.edit_file(&path, &edits),
        }
    }
}

impl FsTool {
    fn parse_arguments(&self, raw: &str) -> Result<FsToolArguments> {
        let value: Value = serde_json::from_str(raw)?;
        if value.get("op").is_some() {
            return serde_json::from_value(value).map_err(Into::into);
        }

        let shape: FsToolArgumentShape = serde_json::from_value(value)?;
        match shape {
            FsToolArgumentShape::Tagged(arguments) => Ok(arguments),
            FsToolArgumentShape::Nested { args } | FsToolArgumentShape::PathOnly(args) => {
                self.infer_operation_from_path(args.path)
            }
        }
    }

    fn infer_operation_from_path(&self, path: String) -> Result<FsToolArguments> {
        let resolved = self.resolve(&path)?;
        let metadata = fs::metadata(&resolved).with_context(|| {
            format!(
                "failed to inspect '{path}' inside startup directory '{}'",
                self.root.display()
            )
        })?;

        if metadata.is_dir() {
            return Ok(FsToolArguments::ListDir { path });
        }

        if metadata.is_file() {
            return Ok(FsToolArguments::ReadFile { path });
        }

        Err(anyhow!(
            "path '{path}' is neither a regular file nor a directory inside startup directory '{}'",
            self.root.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::fs::TestDir;
    use crate::tools::ToolResult;

    fn startup_tool() -> FsTool {
        FsTool::new(std::env::current_dir().unwrap()).unwrap()
    }

    fn tool_for(root: &TestDir) -> FsTool {
        FsTool::new(root.path().to_path_buf()).unwrap()
    }

    fn assert_text_result(result: ToolResult, expected_preview: &str, expected_body: &str) {
        assert_eq!(result.preview(), expected_preview);
        assert_eq!(result.body(), expected_body);
    }

    #[test]
    fn rejects_paths_outside_root() {
        let tool = startup_tool();

        for path in ["../../etc/passwd", "/etc/passwd"] {
            let error = tool.read_file(path).unwrap_err().to_string();
            assert!(
                error.contains("startup directory"),
                "path={path} error={error}"
            );
        }
    }

    #[test]
    fn reads_file_contents() {
        let root = TestDir::new("fs-read");
        root.write_text("note.txt", "hello\nworld");

        let tool = tool_for(&root);
        let result = tool.read_file("note.txt").unwrap();

        assert_text_result(result, "hello", "hello\nworld");
    }

    #[test]
    fn lists_directory_entries() {
        let root = TestDir::new("fs-list");
        root.write_text("nested/file.txt", "hello");

        let tool = tool_for(&root);
        let result = tool.list_dir("nested").unwrap();
        let body = text_body(&result);
        assert!(body.contains("file.txt"));
    }

    #[test]
    fn rejects_symlink_target_outside_root() {
        let root = TestDir::new("fs-symlink");
        let outside = TestDir::new("fs-outside");
        outside.write_text("secret.txt", "secret");
        std::os::unix::fs::symlink(outside.child("secret.txt"), root.child("escape.txt")).unwrap();

        let tool = tool_for(&root);
        let error = tool.read_file("escape.txt").unwrap_err().to_string();
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn missing_file_error_has_context() {
        let root = TestDir::new("fs-missing-file");

        let tool = tool_for(&root);
        let error = tool.read_file("missing.txt").unwrap_err().to_string();
        assert!(error.contains("failed to read file 'missing.txt'"));
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn missing_directory_error_has_context() {
        let root = TestDir::new("fs-missing-dir");

        let tool = tool_for(&root);
        let error = tool.list_dir("missing").unwrap_err().to_string();
        assert!(error.contains("failed to list directory 'missing'"));
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn listing_file_error_has_context() {
        let root = TestDir::new("fs-list-file");
        root.write_text("note.txt", "hello");

        let tool = tool_for(&root);
        let error = tool.list_dir("note.txt").unwrap_err().to_string();
        assert!(error.contains("failed to list directory 'note.txt'"));
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn rejects_missing_startup_root() {
        let root = TestDir::new("fs-missing-root");
        let path = root.path().to_path_buf();
        drop(root);
        let error = FsTool::new(path.clone()).unwrap_err().to_string();
        assert!(error.contains("failed to resolve startup directory"));
        assert!(error.contains(&path.display().to_string()));
    }

    #[test]
    fn reads_in_root_symlink_target() {
        let root = TestDir::new("fs-symlink-in-root");
        root.write_text("nested/file.txt", "hello");
        std::os::unix::fs::symlink(root.child("nested/file.txt"), root.child("link.txt")).unwrap();

        let tool = tool_for(&root);
        let result = tool.read_file("link.txt").unwrap();

        assert_text_result(result, "hello", "hello");
    }

    #[test]
    fn reading_directory_error_has_context() {
        let root = TestDir::new("fs-read-dir");
        root.create_dir_all("nested");

        let tool = tool_for(&root);
        let error = tool.read_file("nested").unwrap_err().to_string();
        assert!(error.contains("failed to read file 'nested'"));
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn invalid_utf8_returns_distinct_error() {
        let root = TestDir::new("fs-invalid-utf8");
        root.write_bytes("binary.dat", &[0xff, 0xfe, 0xfd]);

        let tool = tool_for(&root);
        let error = tool.read_file("binary.dat").unwrap_err().to_string();
        assert!(error.contains("contains invalid UTF-8"));
    }

    #[test]
    fn execute_infers_list_dir_when_op_is_missing() {
        let root = TestDir::new("fs-infer-list-dir");
        root.write_text("nested/file.txt", "hello");

        let mut tool = tool_for(&root);
        let result = tool
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"path":"nested"}"#.into(),
            })
            .unwrap();

        assert!(text_body(&result).contains("file.txt"));
    }

    #[test]
    fn execute_infers_read_file_when_op_is_missing() {
        let root = TestDir::new("fs-infer-read-file");
        root.write_text("note.txt", "hello\nworld");

        let mut tool = tool_for(&root);
        let result = tool
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"path":"note.txt"}"#.into(),
            })
            .unwrap();

        assert_eq!(text_body(&result), "hello\nworld");
    }

    #[test]
    fn execute_unwraps_nested_args_payload() {
        let root = TestDir::new("fs-nested-args");
        root.write_text("nested/file.txt", "hello");

        let mut tool = FsTool::new(root.path().to_path_buf()).unwrap();
        let result = tool
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"args":{"path":"nested"}}"#.into(),
            })
            .unwrap();

        assert!(text_body(&result).contains("file.txt"));
    }

    #[test]
    fn execute_rejects_unknown_op_even_when_path_can_be_inferred() {
        let root = TestDir::new("fs-unknown-op");
        root.write_text("nested/file.txt", "hello");

        let mut tool = tool_for(&root);
        let error = tool
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"bogus","path":"nested"}"#.into(),
            })
            .unwrap_err()
            .to_string();

        assert!(error.contains("invalid arguments for tool 'fs'"));
    }

    #[test]
    fn execute_write_file_creates_missing_parent_dirs() {
        let root = TestDir::new("fs-write-create");

        let mut tool = tool_for(&root);
        let result = tool
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"write_file","path":"nested/new.txt","contents":"hello"}"#
                    .into(),
            })
            .unwrap();

        assert!(text_body(&result).contains("nested/new.txt"));
        assert_eq!(
            fs::read_to_string(root.child("nested/new.txt")).unwrap(),
            "hello"
        );
    }

    #[test]
    fn execute_edit_file_replaces_exact_match() {
        let root = TestDir::new("fs-edit");
        root.write_text("note.txt", "alpha\nbeta\ngamma\n");

        let mut tool = tool_for(&root);
        let result = tool
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"edit_file","path":"note.txt","edits":[{"old_text":"beta","new_text":"BETA"}]}"#.into(),
            })
            .unwrap();

        assert!(text_body(&result).contains("note.txt"));
        assert_eq!(
            fs::read_to_string(root.child("note.txt")).unwrap(),
            "alpha\nBETA\ngamma\n"
        );
    }

    #[test]
    fn definitions_advertise_write_and_edit_file_ops() {
        let root = TestDir::new("fs-definitions");
        let tool = tool_for(&root);
        let definitions = tool.definitions();
        let parameters = &definitions[0].function.parameters;
        let ops = parameters["properties"]["op"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|value| value.as_str())
            .collect::<Vec<_>>();

        assert!(ops.contains(&"write_file"));
        assert!(ops.contains(&"edit_file"));
    }

    #[test]
    fn edit_file_rejects_zero_match() {
        let root = TestDir::new("fs-edit-no-match");
        root.write_text("note.txt", "alpha\nbeta\ngamma\n");

        let mut tool = tool_for(&root);
        let error = tool
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"edit_file","path":"note.txt","edits":[{"old_text":"delta","new_text":"DELTA"}]}"#.into(),
            })
            .unwrap_err()
            .to_string();

        assert!(error.contains("did not match any text"));
    }

    #[test]
    fn edit_file_rejects_multiple_matches() {
        let root = TestDir::new("fs-edit-multi-match");
        root.write_text("note.txt", "beta\nbeta\n");

        let mut tool = tool_for(&root);
        let error = tool
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"edit_file","path":"note.txt","edits":[{"old_text":"beta","new_text":"BETA"}]}"#.into(),
            })
            .unwrap_err()
            .to_string();

        assert!(error.contains("matched multiple locations"));
    }

    #[test]
    fn edit_file_rejects_invalid_utf8() {
        let root = TestDir::new("fs-edit-invalid-utf8");
        root.write_bytes("binary.dat", &[0xff, 0xfe, 0xfd]);

        let mut tool = tool_for(&root);
        let error = tool
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"edit_file","path":"binary.dat","edits":[{"old_text":"a","new_text":"b"}]}"#.into(),
            })
            .unwrap_err()
            .to_string();

        assert!(error.contains("contains invalid UTF-8"));
    }

    #[test]
    fn write_file_rejects_paths_outside_root() {
        let root = TestDir::new("fs-write-outside");

        let mut tool = tool_for(&root);
        let error = tool
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json:
                    r#"{"op":"write_file","path":"../../escape.txt","contents":"hello"}"#.into(),
            })
            .unwrap_err()
            .to_string();

        assert!(error.contains("startup directory"));
    }

    fn text_body(result: &ToolResult) -> &str {
        result.body()
    }
}
