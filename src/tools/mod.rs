pub mod fs;

use anyhow::Result;

use crate::llm::{ToolCall, ToolDefinition};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    preview: String,
    body: String,
}

impl ToolResult {
    pub fn text(preview: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            preview: preview.into(),
            body: body.into(),
        }
    }

    pub fn preview(&self) -> &str {
        &self.preview
    }

    pub fn body(&self) -> &str {
        &self.body
    }
}

pub trait ToolExecutor {
    fn definitions(&self) -> Vec<ToolDefinition>;
    fn execute(&mut self, call: &ToolCall) -> Result<ToolResult>;
}
