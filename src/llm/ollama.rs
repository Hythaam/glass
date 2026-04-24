use futures::{StreamExt, future::{BoxFuture, ready}, stream::{self, BoxStream}};
use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::{
    ChatRequest, DEFAULT_OLLAMA_MODEL, Provider, ProviderError, ProviderResult,
    ProviderStreamItem, ToolCall,
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

    async fn validate_model_available(&self) -> ProviderResult<()> {
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

    fn stream_chat<'a>(&'a self, request: ChatRequest) -> BoxStream<'a, ProviderResult<ProviderStreamItem>> {
        let url = match self.base_url.join("api/chat") {
            Ok(url) => url,
            Err(error) => {
                return Box::pin(stream::once(ready(Err(ProviderError::validation(
                    format!("invalid Ollama base URL: {error}"),
                )))));
            }
        };

        let client = self.client.clone();
        let model = self.model.clone();
        let (tx, rx) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            let payload = OllamaChatRequest {
                model,
                messages: request.messages,
                tools: request.tools,
                stream: true,
            };

            let response = match client.post(url.clone()).json(&payload).send().await {
                Ok(response) => response,
                Err(error) => {
                    let _ = tx.send(Err(ProviderError::transport(format!(
                        "failed to contact Ollama at {url}: {error}"
                    ))));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let _ = tx.send(Err(ProviderError::transport(format!(
                    "Ollama chat request failed at {url}: server returned {status}{}",
                    if body.is_empty() {
                        String::new()
                    } else {
                        format!(" with body: {body}")
                    }
                ))));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = Vec::new();

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(chunk) => {
                        buffer.extend_from_slice(&chunk);
                        while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
                            let line = buffer.drain(..=position).collect::<Vec<_>>();
                            if !forward_chat_line(&tx, &line) {
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        let _ = tx.send(Err(ProviderError::transport(format!(
                            "failed to read Ollama stream from {url}: {error}"
                        ))));
                        return;
                    }
                }
            }

            if !buffer.is_empty() {
                let _ = forward_chat_line(&tx, &buffer);
            }
        });

        Box::pin(UnboundedReceiverStream::new(rx))
    }
}

#[derive(Debug, serde::Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<super::ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tools: Vec<super::ToolDefinition>,
    stream: bool,
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

pub fn parse_chat_chunk(input: &str) -> ProviderResult<Option<ProviderStreamItem>> {
    let chunk: ChatChunk = serde_json::from_str(input)
        .map_err(|error| ProviderError::protocol(format!("invalid Ollama chat chunk: {error}")))?;
    if chunk.done {
        return Ok(Some(ProviderStreamItem::Done));
    }
    if let Some(message) = chunk.message {
        if let Some(calls) = message.tool_calls {
            let mut calls = calls.into_iter();
            let call = calls
                .next()
                .ok_or_else(|| ProviderError::protocol("tool_calls was empty"))?;
            if calls.next().is_some() {
                return Err(ProviderError::protocol(
                    "multiple tool calls in a single Ollama chunk are not supported yet",
                ));
            }
            return Ok(Some(ProviderStreamItem::ToolCall(ToolCall {
                name: call.function.name,
                arguments_json: serde_json::to_string(&call.function.arguments).map_err(
                    |error| {
                        ProviderError::protocol(format!(
                            "failed to serialize Ollama tool-call arguments: {error}"
                        ))
                    },
                )?,
            })));
        }
        if let Some(content) = message.content.filter(|content| !content.is_empty()) {
            return Ok(Some(ProviderStreamItem::AssistantDelta(content)));
        }
        return Ok(None);
    }
    Ok(None)
}

fn validate_model_listing(input: &str, expected_model: &str) -> ProviderResult<()> {
    let tags: TagsResponse = serde_json::from_str(input).map_err(|error| {
        ProviderError::protocol(format!("invalid Ollama model list response: {error}"))
    })?;

    if tags.models.iter().any(|model| model.name == expected_model) {
        return Ok(());
    }

    let available = tags
        .models
        .iter()
        .map(|model| model.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    Err(ProviderError::validation(format!(
        "required Ollama model {expected_model} is unavailable from the configured server; available models: [{available}]"
    )))
}

fn forward_chat_line(
    tx: &mpsc::UnboundedSender<ProviderResult<ProviderStreamItem>>,
    line: &[u8],
) -> bool {
    let line = match std::str::from_utf8(line) {
        Ok(line) => line.trim(),
        Err(error) => {
            let _ = tx.send(Err(ProviderError::protocol(format!(
                "invalid UTF-8 in Ollama chat stream: {error}"
            ))));
            return false;
        }
    };

    if line.is_empty() {
        return true;
    }

    match parse_chat_chunk(line) {
        Ok(Some(item)) => tx.send(Ok(item)).is_ok(),
        Ok(None) => true,
        Err(error) => {
            let _ = tx.send(Err(error));
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatMessage, ChatRequest, ChatRole, ProviderStreamItem, ToolDefinition, ToolFunction};
    use futures::StreamExt;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn parses_assistant_delta_chunk() {
        let item = parse_chat_chunk(r#"{"message":{"content":"hello"},"done":false}"#).unwrap();
        assert_eq!(
            item,
            Some(ProviderStreamItem::AssistantDelta("hello".into()))
        );
    }

    #[test]
    fn parses_tool_call_chunk() {
        let item = parse_chat_chunk(
            r#"{"message":{"tool_calls":[{"function":{"name":"fs","arguments":{"op":"list_dir","path":"src"}}}]},"done":false}"#,
        )
        .unwrap();
        match item {
            Some(ProviderStreamItem::ToolCall(call)) => {
                assert_eq!(call.name, "fs");
                assert!(call.arguments_json.contains("\"list_dir\""));
            }
            other => panic!("unexpected item: {other:?}"),
        }
    }

    #[test]
    fn parses_tool_call_chunk_with_empty_content() {
        let item = parse_chat_chunk(
            r#"{"message":{"content":"","tool_calls":[{"function":{"name":"fs","arguments":{"op":"list_dir","path":"src"}}}]},"done":false}"#,
        )
        .unwrap();
        match item {
            Some(ProviderStreamItem::ToolCall(call)) => {
                assert_eq!(call.name, "fs");
                assert!(call.arguments_json.contains("\"list_dir\""));
            }
            other => panic!("unexpected item: {other:?}"),
        }
    }

    #[test]
    fn parses_done_chunk() {
        let item = parse_chat_chunk(r#"{"done":true}"#).unwrap();
        assert_eq!(item, Some(ProviderStreamItem::Done));
    }

    #[test]
    fn accepts_validation_response_with_default_model() {
        let body = r#"{"models":[{"name":"llama3.1:8b"},{"name":"other"}]}"#;
        assert!(validate_model_listing(body, DEFAULT_OLLAMA_MODEL).is_ok());
    }

    #[test]
    fn rejects_validation_response_without_default_model() {
        let body = r#"{"models":[{"name":"other"}]}"#;
        let error = validate_model_listing(body, DEFAULT_OLLAMA_MODEL)
            .unwrap_err()
            .to_string();
        assert!(error.contains(DEFAULT_OLLAMA_MODEL));
        assert!(error.contains("other"));
    }

    #[test]
    fn rejects_multi_tool_call_chunk() {
        let error = parse_chat_chunk(
            r#"{"message":{"tool_calls":[{"function":{"name":"fs","arguments":{"op":"list_dir","path":"src"}}},{"function":{"name":"fs","arguments":{"op":"read_file","path":"src/main.rs"}}}]},"done":false}"#,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("multiple tool calls"));
    }

    #[test]
    fn ignores_empty_message_chunk() {
        let item = parse_chat_chunk(r#"{"message":{},"done":false}"#).unwrap();
        assert_eq!(item, None);
    }

    #[test]
    fn ignores_chunk_without_message() {
        let item = parse_chat_chunk(r#"{"done":false}"#).unwrap();
        assert_eq!(item, None);
    }

    #[tokio::test]
    async fn stream_chat_posts_tools_and_yields_stream_items() {
        let received_body = Arc::new(Mutex::new(String::new()));
        let server = TestServer::spawn(chunked_response(&[
            r#"{"message":{"content":"hello"},"done":false}"#,
            r#"{"message":{"tool_calls":[{"function":{"name":"fs","arguments":{"op":"list_dir","path":"src"}}}]},"done":false}"#,
            r#"{"done":true}"#,
        ]), received_body.clone())
        .await;
        let provider = OllamaProvider::new(server.url());
        let request = ChatRequest {
            messages: vec![
                ChatMessage {
                    role: ChatRole::User,
                    content: "show src".into(),
                    tool_name: None,
                    tool_calls: None,
                },
                ChatMessage {
                    role: ChatRole::Tool,
                    content: "agent.rs\nmain.rs".into(),
                    tool_name: Some("fs".into()),
                    tool_calls: None,
                },
            ],
            tools: vec![ToolDefinition {
                r#type: "function".into(),
                function: ToolFunction {
                    name: "fs".into(),
                    description: "Read files from the startup directory".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "op": { "type": "string" },
                            "path": { "type": "string" }
                        },
                        "required": ["op", "path"]
                    }),
                },
            }],
        };

        let items = provider
            .stream_chat(request)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<ProviderResult<Vec<_>>>()
            .unwrap();

        assert_eq!(
            items,
            vec![
                ProviderStreamItem::AssistantDelta("hello".into()),
                ProviderStreamItem::ToolCall(ToolCall {
                    name: "fs".into(),
                    arguments_json: r#"{"op":"list_dir","path":"src"}"#.into(),
                }),
                ProviderStreamItem::Done,
            ]
        );

        let body = received_body.lock().unwrap().clone();
        assert!(body.contains("\"tools\""));
        assert!(body.contains("\"tool_name\":\"fs\""));
    }

    struct TestServer {
        url: Url,
    }

    impl TestServer {
        async fn spawn(response: String, received_body: Arc<Mutex<String>>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = read_http_request(&mut socket).await;
                *received_body.lock().unwrap() = request;
                socket.write_all(response.as_bytes()).await.unwrap();
            });
            Self {
                url: Url::parse(&format!("http://{address}/")).unwrap(),
            }
        }

        fn url(&self) -> Url {
            self.url.clone()
        }
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 1024];

        loop {
            let count = socket.read(&mut chunk).await.unwrap();
            buffer.extend_from_slice(&chunk[..count]);
            if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }

        let header_end = buffer
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap()
            + 4;
        let headers = String::from_utf8(buffer[..header_end].to_vec()).unwrap();
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.strip_prefix("content-length: ")
                    .or_else(|| line.strip_prefix("Content-Length: "))
            })
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let mut body = buffer[header_end..].to_vec();
        while body.len() < content_length {
            let count = socket.read(&mut chunk).await.unwrap();
            body.extend_from_slice(&chunk[..count]);
        }

        String::from_utf8(body).unwrap()
    }

    fn chunked_response(lines: &[&str]) -> String {
        let mut response =
            "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\ntransfer-encoding: chunked\r\n\r\n"
                .to_string();
        for line in lines {
            let payload = format!("{line}\n");
            response.push_str(&format!("{:x}\r\n{}\r\n", payload.len(), payload));
        }
        response.push_str("0\r\n\r\n");
        response
    }
}
