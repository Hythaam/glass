pub mod ollama;

use futures::future::BoxFuture;
use reqwest::Url;
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
}

pub trait Provider {
    fn base_url(&self) -> &Url;
    fn model(&self) -> &str;
    fn validate<'a>(&'a self) -> BoxFuture<'a, ProviderResult<()>>;
}
