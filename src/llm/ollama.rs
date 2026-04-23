use anyhow::{Result, anyhow};
use futures::future::BoxFuture;
use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::Value;

use super::{
    DEFAULT_OLLAMA_MODEL, Provider, ProviderError, ProviderResult, ProviderStreamItem, ToolCall,
};

#[derive(Debug, Clone)]
pub struct OllamaProvider {
    base_url: Url,
    model: String,
    client: Client,
}

impl OllamaProvider {
    pub fn new(base_url: Url) -> Self {
        Self::with_model(base_url, DEFAULT_OLLAMA_MODEL)
    }

    pub fn with_model(base_url: Url, model: impl Into<String>) -> Self {
        Self {
            base_url,
            model: model.into(),
            client: Client::new(),
        }
    }

    pub async fn validate_model_available(&self) -> ProviderResult<()> {
        let url = self.base_url.join("api/tags").map_err(|error| {
            ProviderError::validation(format!("invalid Ollama base URL: {error}"))
        })?;

        let response = self.client.get(url.clone()).send().await.map_err(|error| {
            ProviderError::transport(format!("failed to contact Ollama at {url}: {error}"))
        })?;

        if !response.status().is_success() {
            return Err(ProviderError::validation(format!(
                "Ollama model validation failed at {url}: server returned {}",
                response.status()
            )));
        }

        let body = response.text().await.map_err(|error| {
            ProviderError::transport(format!(
                "failed to read Ollama model validation response from {url}: {error}"
            ))
        })?;

        validate_model_listing(&body, &self.model)
    }
}

impl Provider for OllamaProvider {
    fn base_url(&self) -> &Url {
        &self.base_url
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn validate<'a>(&'a self) -> BoxFuture<'a, ProviderResult<()>> {
        Box::pin(async move { self.validate_model_available().await })
    }
}

#[derive(Debug, Deserialize)]
struct ChatChunk {
    #[serde(default)]
    message: Option<ChunkMessage>,
    done: bool,
}

#[derive(Debug, Deserialize)]
struct ChunkMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCall {
    function: ChunkFunction,
}

#[derive(Debug, Deserialize)]
struct ChunkFunction {
    name: String,
    arguments: Value,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagModel>,
}

#[derive(Debug, Deserialize)]
struct TagModel {
    name: String,
}

pub fn parse_chat_chunk(input: &str) -> Result<ProviderStreamItem> {
    let chunk: ChatChunk = serde_json::from_str(input)?;
    if chunk.done {
        return Ok(ProviderStreamItem::Done);
    }
    if let Some(message) = chunk.message {
        if let Some(content) = message.content {
            return Ok(ProviderStreamItem::AssistantDelta(content));
        }
        if let Some(mut calls) = message.tool_calls {
            let call = calls.pop().ok_or_else(|| anyhow!("tool_calls was empty"))?;
            return Ok(ProviderStreamItem::ToolCall(ToolCall {
                name: call.function.name,
                arguments_json: serde_json::to_string(&call.function.arguments)?,
            }));
        }
    }
    Err(anyhow!("unsupported Ollama chat chunk"))
}

fn default_model_available(input: &str) -> Result<()> {
    validate_model_listing(input, DEFAULT_OLLAMA_MODEL).map_err(|error| anyhow!(error.to_string()))
}

fn validate_model_listing(input: &str, expected_model: &str) -> ProviderResult<()> {
    let tags: TagsResponse = serde_json::from_str(input).map_err(|error| {
        ProviderError::protocol(format!("invalid Ollama model list response: {error}"))
    })?;

    if tags.models.iter().any(|model| model.name == expected_model) {
        return Ok(());
    }

    Err(ProviderError::validation(format!(
        "required Ollama model {expected_model} is unavailable from the configured server"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ProviderStreamItem;

    #[test]
    fn parses_assistant_delta_chunk() {
        let item = parse_chat_chunk(r#"{"message":{"content":"hello"},"done":false}"#).unwrap();
        assert_eq!(item, ProviderStreamItem::AssistantDelta("hello".into()));
    }

    #[test]
    fn parses_tool_call_chunk() {
        let item = parse_chat_chunk(
            r#"{"message":{"tool_calls":[{"function":{"name":"fs","arguments":{"op":"list_dir","path":"src"}}}]},"done":false}"#,
        )
        .unwrap();
        match item {
            ProviderStreamItem::ToolCall(call) => {
                assert_eq!(call.name, "fs");
                assert!(call.arguments_json.contains("\"list_dir\""));
            }
            other => panic!("unexpected item: {other:?}"),
        }
    }

    #[test]
    fn parses_done_chunk() {
        let item = parse_chat_chunk(r#"{"done":true}"#).unwrap();
        assert_eq!(item, ProviderStreamItem::Done);
    }

    #[test]
    fn accepts_validation_response_with_default_model() {
        let body = r#"{"models":[{"name":"llama3.1:8b"},{"name":"other"}]}"#;
        assert!(default_model_available(body).is_ok());
    }

    #[test]
    fn rejects_validation_response_without_default_model() {
        let body = r#"{"models":[{"name":"other"}]}"#;
        let error = default_model_available(body).unwrap_err().to_string();
        assert!(error.contains(DEFAULT_OLLAMA_MODEL));
    }
}
