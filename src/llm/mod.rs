pub mod ollama;

use futures::{future::BoxFuture, stream::BoxStream};
use reqwest::Url;
use serde::Serialize;
use serde_json::Value;
use std::error::Error;
use std::fmt;

pub const DEFAULT_OLLAMA_MODEL: &str = "llama3.1:8b";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderStreamItem {
    AssistantDelta(String),
    ToolCall(ToolCall),
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    Validation(String),
    Transport(String),
    Protocol(String),
}

impl ProviderError {
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation(message.into())
    }

    pub fn transport(message: impl Into<String>) -> Self {
        Self::Transport(message.into())
    }

    pub fn protocol(message: impl Into<String>) -> Self {
        Self::Protocol(message.into())
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(message) | Self::Transport(message) | Self::Protocol(message) => {
                f.write_str(message)
            }
        }
    }
}

impl Error for ProviderError {}

pub type ProviderResult<T> = Result<T, ProviderError>;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatToolCall {
    pub function: ToolFunctionCall,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolFunctionCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolCall {
    pub fn arguments_value(&self) -> ProviderResult<Value> {
        serde_json::from_str(&self.arguments_json).map_err(|error| {
            ProviderError::protocol(format!(
                "invalid tool-call arguments for '{}': {error}",
                self.name
            ))
        })
    }

    pub fn as_chat_tool_call(&self) -> ProviderResult<ChatToolCall> {
        Ok(ChatToolCall {
            function: ToolFunctionCall {
                name: self.name.clone(),
                arguments: self.arguments_value()?,
            },
        })
    }
}

pub trait Provider {
    #[allow(dead_code)]
    fn base_url(&self) -> &Url;
    #[allow(dead_code)]
    fn model(&self) -> &str;
    fn validate<'a>(&'a self) -> BoxFuture<'a, ProviderResult<()>>;
    fn stream_chat<'a>(&'a self, request: ChatRequest) -> BoxStream<'a, ProviderResult<ProviderStreamItem>>;
}
