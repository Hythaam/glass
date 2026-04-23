pub mod fs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolResult {
    Text { preview: String, body: String },
}
