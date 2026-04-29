use futures::{
    StreamExt,
    future::{BoxFuture, ready},
    stream::{self, BoxStream},
};
use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::{
    ChatRequest, Provider, ProviderError, ProviderResult, ProviderStreamItem, RequestTokenUsage,
    ToolCall,
};

#[derive(Debug, Clone)]
pub struct LlamaCppProvider {
    base_url: Url,
    model: String,
    client: Client,
}

impl LlamaCppProvider {
    pub fn new(base_url: Url) -> Self {
        Self {
            base_url,
            model: String::new(),
            client: Client::new(),
        }
    }

    async fn validate_model_available(&mut self) -> ProviderResult<()> {
        let url = self.base_url.join("api/tags").map_err(|error| {
            ProviderError::validation(format!("invalid llama.cpp base URL: {error}"))
        })?;

        let response = self.client.get(url.clone()).send().await.map_err(|error| {
            ProviderError::transport(format!(
                "failed to contact llama.cpp server at {url}: {error}"
            ))
        })?;

        if !response.status().is_success() {
            return Err(ProviderError::validation(format!(
                "llama.cpp model validation failed at {url}: server returned {}",
                response.status()
            )));
        }

        let body = response.text().await.map_err(|error| {
            ProviderError::transport(format!(
                "failed to read llama.cpp model validation response from {url}: {error}"
            ))
        })?;

        self.model = select_model_from_listing(&body)?;
        Ok(())
    }

    fn chat_url(&self) -> ProviderResult<Url> {
        self.base_url.join("api/chat").map_err(|error| {
            ProviderError::validation(format!("invalid llama.cpp base URL: {error}"))
        })
    }
}

impl Provider for LlamaCppProvider {
    fn base_url(&self) -> &Url {
        &self.base_url
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn validate<'a>(&'a mut self) -> BoxFuture<'a, ProviderResult<()>> {
        Box::pin(async move { self.validate_model_available().await })
    }

    fn stream_chat<'a>(
        &'a self,
        request: ChatRequest,
    ) -> BoxStream<'a, ProviderResult<ProviderStreamItem>> {
        if self.model.is_empty() {
            return Box::pin(stream::once(ready(Err(ProviderError::validation(
                "llama.cpp provider must be validated before streaming chat".to_string(),
            )))));
        }

        let url = match self.chat_url() {
            Ok(url) => url,
            Err(error) => return Box::pin(stream::once(ready(Err(error)))),
        };

        let client = self.client.clone();
        let model = self.model.clone();
        let (tx, rx) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            let payload = LlamaCppChatRequest {
                model,
                messages: request.messages,
                tools: request.tools,
                stream: true,
            };

            let response = match send_chat_request(&client, url.clone(), payload).await {
                Ok(response) => response,
                Err(error) => {
                    let _ = tx.send(Err(error));
                    return;
                }
            };

            if let Err(error) = forward_chat_response(&tx, response, &url).await {
                let _ = tx.send(Err(error));
            }
        });

        Box::pin(UnboundedReceiverStream::new(rx))
    }
}

#[derive(Debug, serde::Serialize)]
struct LlamaCppChatRequest {
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
struct OpenAiChatChunk {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
    #[serde(default)]
    timings: Option<OpenAiTimings>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    #[serde(default)]
    delta: OpenAiDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAiTimings {
    #[serde(default)]
    prompt_n: Option<usize>,
    #[serde(default)]
    predicted_n: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAiDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCallDelta {
    index: usize,
    #[serde(default)]
    function: OpenAiFunctionDelta,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAiFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
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

#[derive(Debug)]
enum ParsedStreamLine {
    Done,
    Ollama(ChatChunk),
    OpenAi(OpenAiChatChunk),
}

#[cfg(test)]
fn parse_chat_chunk(input: &str) -> ProviderResult<Option<ProviderStreamItem>> {
    let Some(line) = parse_stream_line(input)? else {
        return Ok(None);
    };

    match line {
        ParsedStreamLine::Done => Ok(Some(ProviderStreamItem::Done { usage: None })),
        ParsedStreamLine::Ollama(chunk) => parse_ollama_chat_chunk(chunk),
        ParsedStreamLine::OpenAi(chunk) => parse_stateless_openai_chat_chunk(chunk),
    }
}

fn select_model_from_listing(input: &str) -> ProviderResult<String> {
    let tags: TagsResponse = serde_json::from_str(input).map_err(|error| {
        ProviderError::protocol(format!("invalid llama.cpp model list response: {error}"))
    })?;

    if let Some(model) = tags.models.first() {
        return Ok(model.name.clone());
    }

    Err(ProviderError::validation(
        "llama.cpp server advertised no models".to_string(),
    ))
}

fn forward_chat_line(
    tx: &mpsc::UnboundedSender<ProviderResult<ProviderStreamItem>>,
    line: &[u8],
    state: &mut StreamParserState,
) -> ProviderResult<ForwardLineStatus> {
    let line = match std::str::from_utf8(line) {
        Ok(line) => line.trim(),
        Err(error) => {
            return Err(ProviderError::protocol(format!(
                "invalid UTF-8 in llama.cpp chat stream: {error}"
            )));
        }
    };

    if line.is_empty() {
        return Ok(ForwardLineStatus::Continue);
    }

    match parse_chat_line(line, state) {
        Ok(Some(item)) => Ok(send_stream_item(tx, item)),
        Ok(None) => Ok(ForwardLineStatus::Continue),
        Err(error) => Err(error),
    }
}

fn normalized_stream_line(input: &str) -> Option<&str> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }
    Some(
        input
            .strip_prefix("data:")
            .map(str::trim_start)
            .unwrap_or(input),
    )
}

#[derive(Debug, Default)]
struct StreamParserState {
    pending_tool_call: Option<PendingToolCall>,
    pending_usage: Option<RequestTokenUsage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForwardLineStatus {
    Continue,
    Stop,
}

#[derive(Debug, Default)]
struct PendingToolCall {
    name: Option<String>,
    arguments_json: String,
}

fn parse_chat_line(
    input: &str,
    state: &mut StreamParserState,
) -> ProviderResult<Option<ProviderStreamItem>> {
    let Some(line) = parse_stream_line(input)? else {
        return Ok(None);
    };

    match line {
        ParsedStreamLine::Done => Ok(Some(ProviderStreamItem::Done {
            usage: state.pending_usage.take(),
        })),
        ParsedStreamLine::Ollama(chunk) => parse_ollama_chat_chunk(chunk),
        ParsedStreamLine::OpenAi(chunk) => parse_openai_chat_chunk(chunk, state),
    }
}

fn parse_openai_chat_chunk(
    chunk: OpenAiChatChunk,
    state: &mut StreamParserState,
) -> ProviderResult<Option<ProviderStreamItem>> {
    if let Some(usage) = chunk_request_usage(&chunk) {
        state.pending_usage = Some(usage);
    }

    for choice in chunk.choices {
        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                ensure_single_tool_call_index(tool_call.index)?;

                let pending = state
                    .pending_tool_call
                    .get_or_insert_with(PendingToolCall::default);
                if let Some(name) = tool_call.function.name {
                    pending.name = Some(name);
                }
                if let Some(arguments) = tool_call.function.arguments {
                    pending.arguments_json.push_str(&arguments);
                }
            }
        }

        if matches!(choice.finish_reason.as_deref(), Some("tool_calls")) {
            return finalize_pending_tool_call(state);
        }

        if let Some(content) = choice.delta.content.filter(|content| !content.is_empty()) {
            return Ok(Some(ProviderStreamItem::AssistantDelta(content)));
        }
    }

    Ok(None)
}

fn parse_ollama_chat_chunk(chunk: ChatChunk) -> ProviderResult<Option<ProviderStreamItem>> {
    if chunk.done {
        return Ok(Some(ProviderStreamItem::Done { usage: None }));
    }

    let Some(message) = chunk.message else {
        return Ok(None);
    };

    if let Some(calls) = message.tool_calls {
        let mut calls = calls.into_iter();
        let call = calls
            .next()
            .ok_or_else(|| ProviderError::protocol("tool_calls was empty"))?;
        if calls.next().is_some() {
            return Err(ProviderError::protocol(
                "multiple tool calls in a single llama.cpp chunk are not supported yet",
            ));
        }
        return Ok(Some(ProviderStreamItem::ToolCall(
            ToolCall::from_json_value(
                call.function.name,
                call.function.arguments,
                "failed to serialize llama.cpp tool-call arguments",
            )?,
        )));
    }

    Ok(message
        .content
        .filter(|content| !content.is_empty())
        .map(ProviderStreamItem::AssistantDelta))
}

#[cfg(test)]
fn parse_stateless_openai_chat_chunk(
    chunk: OpenAiChatChunk,
) -> ProviderResult<Option<ProviderStreamItem>> {
    for choice in chunk.choices {
        if let Some(content) = choice.delta.content.filter(|content| !content.is_empty()) {
            return Ok(Some(ProviderStreamItem::AssistantDelta(content)));
        }
    }
    Ok(None)
}

fn parse_stream_line(input: &str) -> ProviderResult<Option<ParsedStreamLine>> {
    let Some(input) = normalized_stream_line(input) else {
        return Ok(None);
    };

    if input == "[DONE]" {
        return Ok(Some(ParsedStreamLine::Done));
    }

    let value: Value = serde_json::from_str(input)
        .map_err(|_| ProviderError::protocol("invalid llama.cpp chat chunk"))?;

    if value.get("choices").is_some() {
        return serde_json::from_value(value)
            .map(ParsedStreamLine::OpenAi)
            .map(Some)
            .map_err(|_| ProviderError::protocol("invalid llama.cpp chat chunk"));
    }

    if value.get("message").is_some() || value.get("done").is_some() {
        return serde_json::from_value(value)
            .map(ParsedStreamLine::Ollama)
            .map(Some)
            .map_err(|_| ProviderError::protocol("invalid llama.cpp chat chunk"));
    }

    Err(ProviderError::protocol("invalid llama.cpp chat chunk"))
}

fn ensure_single_tool_call_index(index: usize) -> ProviderResult<()> {
    if index == 0 {
        Ok(())
    } else {
        Err(ProviderError::protocol(
            "multiple tool calls in a single llama.cpp stream are not supported yet",
        ))
    }
}

fn chunk_request_usage(chunk: &OpenAiChatChunk) -> Option<RequestTokenUsage> {
    if let Some(usage) = &chunk.usage {
        return Some(RequestTokenUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        });
    }

    let timings = chunk.timings.as_ref()?;
    let prompt_tokens = timings.prompt_n?;
    let completion_tokens = timings.predicted_n?;
    Some(RequestTokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    })
}

fn finalize_pending_tool_call(
    state: &mut StreamParserState,
) -> ProviderResult<Option<ProviderStreamItem>> {
    let pending = match state.pending_tool_call.take() {
        Some(pending) => pending,
        None => return Ok(None),
    };

    let name = pending
        .name
        .ok_or_else(|| ProviderError::protocol("llama.cpp tool call is missing a function name"))?;
    let arguments: Value = serde_json::from_str(&pending.arguments_json).map_err(|error| {
        ProviderError::protocol(format!(
            "invalid fragmented llama.cpp tool-call arguments: {error}"
        ))
    })?;

    Ok(Some(ProviderStreamItem::ToolCall(
        ToolCall::from_json_value(
            name,
            arguments,
            "failed to serialize fragmented llama.cpp tool-call arguments",
        )?,
    )))
}

async fn send_chat_request(
    client: &Client,
    url: Url,
    payload: LlamaCppChatRequest,
) -> ProviderResult<reqwest::Response> {
    let response = client
        .post(url.clone())
        .json(&payload)
        .send()
        .await
        .map_err(|error| {
            ProviderError::transport(format!(
                "failed to contact llama.cpp server at {url}: {error}"
            ))
        })?;

    if response.status().is_success() {
        Ok(response)
    } else {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        Err(ProviderError::transport(format!(
            "llama.cpp chat request failed at {url}: server returned {status}{}",
            if body.is_empty() {
                String::new()
            } else {
                format!(" with body: {body}")
            }
        )))
    }
}

async fn forward_chat_response(
    tx: &mpsc::UnboundedSender<ProviderResult<ProviderStreamItem>>,
    response: reqwest::Response,
    url: &Url,
) -> ProviderResult<()> {
    let mut stream = response.bytes_stream();
    let mut buffer = StreamLineBuffer::default();
    let mut parser_state = StreamParserState::default();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(chunk) => {
                for line in buffer.push_chunk(&chunk) {
                    match forward_chat_line(tx, &line, &mut parser_state) {
                        Ok(ForwardLineStatus::Continue) => {}
                        Ok(ForwardLineStatus::Stop) => return Ok(()),
                        Err(error) => {
                            let _ = tx.send(Err(error));
                            return Ok(());
                        }
                    }
                }
            }
            Err(error) => {
                return Err(ProviderError::transport(format!(
                    "failed to read llama.cpp stream from {url}: {error}"
                )));
            }
        }
    }

    if let Some(line) = buffer.finish() {
        if let Err(error) = forward_chat_line(tx, &line, &mut parser_state) {
            let _ = tx.send(Err(error));
        }
    }

    Ok(())
}

fn send_stream_item(
    tx: &mpsc::UnboundedSender<ProviderResult<ProviderStreamItem>>,
    item: ProviderStreamItem,
) -> ForwardLineStatus {
    if tx.send(Ok(item)).is_ok() {
        ForwardLineStatus::Continue
    } else {
        ForwardLineStatus::Stop
    }
}

#[derive(Debug, Default)]
struct StreamLineBuffer {
    buffer: Vec<u8>,
}

impl StreamLineBuffer {
    fn push_chunk(&mut self, chunk: &[u8]) -> Vec<Vec<u8>> {
        self.buffer.extend_from_slice(chunk);
        let mut lines = Vec::new();
        while let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
            lines.push(self.buffer.drain(..=position).collect());
        }
        lines
    }

    fn finish(self) -> Option<Vec<u8>> {
        (!self.buffer.is_empty()).then_some(self.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{
        ChatMessage, ChatRequest, ChatRole, ProviderStreamItem, ToolDefinition, ToolFunction,
    };
    use futures::StreamExt;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn parses_assistant_delta_chunk() {
        let item = parse_chat_chunk(&ollama_content_chunk("hello")).unwrap();
        assert_eq!(
            item,
            Some(ProviderStreamItem::AssistantDelta("hello".into()))
        );
    }

    #[test]
    fn parses_tool_call_chunk() {
        let item = parse_chat_chunk(&ollama_tool_call_chunk(
            None,
            json!({"op":"list_dir","path":"src"}),
        ))
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
        let item = parse_chat_chunk(&ollama_tool_call_chunk(
            Some(""),
            json!({"op":"list_dir","path":"src"}),
        ))
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
        let item = parse_chat_chunk(&ollama_done_chunk()).unwrap();
        assert_eq!(item, Some(ProviderStreamItem::Done { usage: None }));
    }

    #[tokio::test]
    async fn validate_selects_first_advertised_model() {
        let received_body = Arc::new(Mutex::new(String::new()));
        let server = TestServer::spawn(
            json_response(r#"{"models":[{"name":"first-model"},{"name":"other"}]}"#),
            received_body,
        )
        .await;
        let mut provider = LlamaCppProvider::new(server.url());

        provider.validate().await.unwrap();

        assert_eq!(provider.model(), "first-model");
    }

    #[tokio::test]
    async fn validate_rejects_empty_model_list() {
        let received_body = Arc::new(Mutex::new(String::new()));
        let server = TestServer::spawn(json_response(r#"{"models":[]}"#), received_body).await;
        let mut provider = LlamaCppProvider::new(server.url());

        let error = provider.validate().await.unwrap_err().to_string();

        assert!(error.contains("no models"));
        assert!(error.contains("llama.cpp"));
    }

    #[test]
    fn parse_errors_refer_to_llama_cpp() {
        let error = parse_chat_chunk("not json").unwrap_err().to_string();

        assert!(error.contains("llama.cpp"));
    }

    #[test]
    fn rejects_multi_tool_call_chunk() {
        let error = parse_chat_chunk(&ollama_multi_tool_call_chunk())
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

    #[test]
    fn parses_openai_sse_assistant_delta_chunk() {
        let item = parse_chat_chunk(&openai_content_chunk("Hi")).unwrap();

        assert_eq!(item, Some(ProviderStreamItem::AssistantDelta("Hi".into())));
    }

    #[test]
    fn parses_openai_sse_done_chunk() {
        let item = parse_chat_chunk("data: [DONE]").unwrap();
        assert_eq!(item, Some(ProviderStreamItem::Done { usage: None }));
    }

    #[test]
    fn parses_openai_sse_done_chunk_with_usage_from_timings() {
        let mut state = StreamParserState::default();

        let item = parse_chat_line(&openai_stop_chunk_with_timings(19, 2), &mut state).unwrap();
        assert_eq!(item, None);

        let done = parse_chat_line("data: [DONE]", &mut state).unwrap();
        let done_debug = format!("{done:?}");

        assert!(done_debug.contains("prompt_tokens: 19"));
        assert!(done_debug.contains("completion_tokens: 2"));
        assert!(done_debug.contains("total_tokens: 21"));
    }

    #[test]
    fn prioritizes_fragmented_tool_call_over_content_in_final_openai_chunk() {
        let mut state = StreamParserState::default();

        let first = parse_chat_line(
            &openai_tool_call_fragment_chunk("fs", "{", None, None),
            &mut state,
        )
        .unwrap();
        assert_eq!(first, None);

        let second = parse_chat_line(
            &openai_tool_call_fragment_chunk(
                "fs",
                r#""op":"list_dir","path":"src"}"#,
                Some("working"),
                Some("tool_calls"),
            ),
            &mut state,
        )
        .unwrap();

        assert_eq!(
            second,
            Some(ProviderStreamItem::ToolCall(ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"list_dir","path":"src"}"#.into(),
            }))
        );
    }

    #[tokio::test]
    async fn stream_chat_posts_tools_and_yields_stream_items() {
        let received_requests = Arc::new(Mutex::new(Vec::new()));
        let server = TestServer::spawn_sequence(
            vec![
                json_response(r#"{"models":[{"name":"first-model"}]}"#),
                chunked_response(&[
                    ollama_content_chunk("hello"),
                    ollama_tool_call_chunk(None, json!({"op":"list_dir","path":"src"})),
                    ollama_done_chunk(),
                ]),
            ],
            received_requests.clone(),
        )
        .await;
        let mut provider = LlamaCppProvider::new(server.url());
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

        provider.validate().await.unwrap();
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
                ProviderStreamItem::Done { usage: None },
            ]
        );

        let body = received_requests.lock().unwrap().last().unwrap().clone();
        assert!(body.contains("\"tools\""));
        assert!(body.contains("\"tool_name\":\"fs\""));
    }

    #[tokio::test]
    async fn stream_chat_yields_openai_sse_fragmented_tool_call() {
        let received_requests = Arc::new(Mutex::new(Vec::new()));
        let server = TestServer::spawn_sequence(
            vec![
                json_response(r#"{"models":[{"name":"qwen"}]}"#),
                chunked_response(&openai_fragmented_tool_call_chunks(
                    "fs",
                    &[
                        "{",
                        "\"op\":\"",
                        "list",
                        "_dir",
                        "\"",
                        ",\"path\":\"",
                        "src",
                        "\"",
                        "}",
                    ],
                )),
            ],
            received_requests,
        )
        .await;
        let mut provider = LlamaCppProvider::new(server.url());
        provider.validate().await.unwrap();

        let items = provider
            .stream_chat(ChatRequest::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<ProviderResult<Vec<_>>>()
            .unwrap();

        assert_eq!(
            items,
            vec![
                ProviderStreamItem::ToolCall(ToolCall {
                    name: "fs".into(),
                    arguments_json: r#"{"op":"list_dir","path":"src"}"#.into(),
                }),
                ProviderStreamItem::Done { usage: None },
            ]
        );
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

        async fn spawn_sequence(
            responses: Vec<String>,
            received_requests: Arc<Mutex<Vec<String>>>,
        ) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            tokio::spawn(async move {
                for response in responses {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let request = read_http_request(&mut socket).await;
                    received_requests.lock().unwrap().push(request);
                    socket.write_all(response.as_bytes()).await.unwrap();
                }
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

    fn chunked_response(lines: &[String]) -> String {
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

    fn json_response(body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        )
    }

    fn ollama_content_chunk(content: &str) -> String {
        json!({"message":{"content":content},"done":false}).to_string()
    }

    fn ollama_tool_call_chunk(content: Option<&str>, arguments: serde_json::Value) -> String {
        let mut message = json!({
            "tool_calls": [{
                "function": {
                    "name": "fs",
                    "arguments": arguments,
                }
            }]
        });
        if let Some(content) = content {
            message["content"] = json!(content);
        }
        json!({"message": message, "done": false}).to_string()
    }

    fn ollama_multi_tool_call_chunk() -> String {
        json!({
            "message": {
                "tool_calls": [
                    {"function": {"name": "fs", "arguments": {"op": "list_dir", "path": "src"}}},
                    {"function": {"name": "fs", "arguments": {"op": "read_file", "path": "src/main.rs"}}}
                ]
            },
            "done": false
        })
        .to_string()
    }

    fn ollama_done_chunk() -> String {
        json!({"done": true}).to_string()
    }

    fn openai_content_chunk(content: &str) -> String {
        format!(
            "data: {}",
            json!({
                "choices": [{
                    "finish_reason": serde_json::Value::Null,
                    "index": 0,
                    "delta": {"content": content}
                }],
                "object": "chat.completion.chunk"
            })
        )
    }

    fn openai_stop_chunk_with_timings(prompt_n: u32, predicted_n: u32) -> String {
        format!(
            "data: {}",
            json!({
                "choices": [{
                    "finish_reason": "stop",
                    "index": 0,
                    "delta": {}
                }],
                "object": "chat.completion.chunk",
                "timings": {
                    "prompt_n": prompt_n,
                    "predicted_n": predicted_n
                }
            })
        )
    }

    fn openai_tool_call_fragment_chunk(
        name: &str,
        arguments: &str,
        content: Option<&str>,
        finish_reason: Option<&str>,
    ) -> String {
        let mut delta = json!({
            "tool_calls": [{
                "index": 0,
                "function": {
                    "arguments": arguments,
                }
            }]
        });
        if !name.is_empty() {
            delta["tool_calls"][0]["function"]["name"] = json!(name);
        }
        if let Some(content) = content {
            delta["content"] = json!(content);
        }

        format!(
            "data: {}",
            json!({
                "choices": [{
                    "finish_reason": finish_reason,
                    "index": 0,
                    "delta": delta
                }],
                "object": "chat.completion.chunk"
            })
        )
    }

    fn openai_fragmented_tool_call_chunks(name: &str, fragments: &[&str]) -> Vec<String> {
        let mut chunks = Vec::with_capacity(fragments.len() + 2);
        for (index, fragment) in fragments.iter().enumerate() {
            let delta = if index == 0 {
                json!({
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": fragment,
                        }
                    }]
                })
            } else {
                json!({
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": fragment,
                        }
                    }]
                })
            };
            chunks.push(format!(
                "data: {}",
                json!({
                    "choices": [{
                        "finish_reason": serde_json::Value::Null,
                        "index": 0,
                        "delta": delta
                    }],
                    "object": "chat.completion.chunk"
                })
            ));
        }
        chunks.push(format!(
            "data: {}",
            json!({
                "choices": [{
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "delta": {}
                }],
                "object": "chat.completion.chunk"
            })
        ));
        chunks.push("data: [DONE]".to_string());
        chunks
    }
}
