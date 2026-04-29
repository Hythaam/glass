use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) struct TestDir {
    path: PathBuf,
}

impl TestDir {
    pub(crate) fn new(prefix: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::current_dir()
            .unwrap()
            .join("target")
            .join("test-artifacts")
            .join(format!("glass-{prefix}-{unique}"));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn child(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.path.join(relative)
    }

    pub(crate) fn create_dir_all(&self, relative: impl AsRef<Path>) -> PathBuf {
        let path = self.child(relative);
        fs::create_dir_all(&path).unwrap();
        path
    }

    pub(crate) fn write_text(&self, relative: impl AsRef<Path>, contents: &str) -> PathBuf {
        let path = self.child(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, contents).unwrap();
        path
    }

    pub(crate) fn write_bytes(&self, relative: impl AsRef<Path>, contents: &[u8]) -> PathBuf {
        let path = self.child(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, contents).unwrap();
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::TestDir;

    #[test]
    fn removes_directory_when_dropped() {
        let path = {
            let dir = TestDir::new("cleanup");
            let path = dir.path().to_path_buf();
            assert!(path.exists());
            path
        };

        assert!(!path.exists());
    }
}
