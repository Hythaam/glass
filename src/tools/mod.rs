pub mod fs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolResult {
    Text { preview: String, body: String },
}

impl ToolResult {
    pub fn text(preview: impl Into<String>, body: impl Into<String>) -> Self {
        Self::Text {
            preview: preview.into(),
            body: body.into(),
        }
    }
}
