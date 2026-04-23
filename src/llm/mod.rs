pub mod ollama;

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
