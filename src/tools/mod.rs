pub mod fs;

use anyhow::Result;

use crate::llm::{ToolCall, ToolDefinition};

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

    pub fn preview(&self) -> &str {
        match self {
            Self::Text { preview, .. } => preview,
        }
    }

    pub fn body(&self) -> &str {
        match self {
            Self::Text { body, .. } => body,
        }
    }
}

pub trait ToolExecutor {
    fn definitions(&self) -> Vec<ToolDefinition>;
    fn execute(&mut self, call: &ToolCall) -> Result<ToolResult>;
}
