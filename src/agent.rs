use anyhow::Result;
use futures::StreamExt;

use crate::context::SessionContext;
use crate::llm::{Provider, ProviderStreamItem};
use crate::tools::ToolExecutor;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    AssistantDelta {
        turn_id: u64,
        text: String,
    },
    AssistantDone {
        turn_id: u64,
    },
    ToolStarted {
        turn_id: u64,
        tool_name: String,
    },
    ToolOutputDelta {
        turn_id: u64,
        tool_name: String,
        text: String,
    },
    ToolFinished {
        turn_id: u64,
        tool_name: String,
        preview: String,
    },
    TurnError {
        turn_id: u64,
        message: String,
    },
}

#[allow(dead_code)]
pub struct Agent<P, T> {
    provider: P,
    tools: T,
    context: SessionContext,
    next_turn_id: u64,
}

#[allow(dead_code)]
impl<P, T> Agent<P, T> {
    pub fn new(provider: P, tools: T, context: SessionContext) -> Self {
        Self {
            provider,
            tools,
            context,
            next_turn_id: 1,
        }
    }

    #[cfg(test)]
    fn provider(&self) -> &P {
        &self.provider
    }

    #[cfg(test)]
    fn context(&self) -> &SessionContext {
        &self.context
    }
}

impl<P, T> Agent<P, T>
where
    P: Provider,
    T: ToolExecutor,
{
    #[allow(dead_code)]
    pub async fn run_turn<F>(&mut self, input: &str, mut emit: F) -> Result<()>
    where
        F: FnMut(AgentEvent),
    {
        let turn_id = self.next_turn_id;
        self.next_turn_id += 1;

        self.context.push_user(input);

        loop {
            let mut request = self.context.chat_request()?;
            request.tools = self.tools.definitions();
            let mut stream = self.provider.stream_chat(request);
            let mut assistant_text = String::new();
            let mut requested_tool = false;

            while let Some(item) = stream.next().await {
                let item = match item {
                    Ok(item) => item,
                    Err(error) => {
                        emit(AgentEvent::TurnError {
                            turn_id,
                            message: error.to_string(),
                        });
                        self.context.prune_if_needed();
                        return Ok(());
                    }
                };

                match item {
                    ProviderStreamItem::AssistantDelta(text) => {
                        assistant_text.push_str(&text);
                        emit(AgentEvent::AssistantDelta { turn_id, text });
                    }
                    ProviderStreamItem::ToolCall(call) => {
                        requested_tool = true;
                        if !assistant_text.is_empty() {
                            self.context.push_assistant(&assistant_text);
                            assistant_text.clear();
                        }
                        self.context.push_assistant_tool_call(&call)?;

                        let tool_name = call.name.clone();
                        emit(AgentEvent::ToolStarted {
                            turn_id,
                            tool_name: tool_name.clone(),
                        });

                        match self.tools.execute(&call) {
                            Ok(result) => {
                                let body = result.body().to_string();
                                self.context.push_tool_output(&tool_name, &body);
                                for chunk in tool_output_chunks(&body) {
                                    emit(AgentEvent::ToolOutputDelta {
                                        turn_id,
                                        tool_name: tool_name.clone(),
                                        text: chunk,
                                    });
                                }
                                emit(AgentEvent::ToolFinished {
                                    turn_id,
                                    tool_name,
                                    preview: result.preview().to_string(),
                                });
                            }
                            Err(error) => {
                                self.context
                                    .push_tool_output(&tool_name, &format!("error: {error}"));
                                emit(AgentEvent::TurnError {
                                    turn_id,
                                    message: error.to_string(),
                                });
                                self.context.prune_if_needed();
                                return Ok(());
                            }
                        }

                        break;
                    }
                    ProviderStreamItem::Done => {
                        if !assistant_text.is_empty() {
                            self.context.push_assistant(&assistant_text);
                        }
                        emit(AgentEvent::AssistantDone { turn_id });
                        self.context.prune_if_needed();
                        return Ok(());
                    }
                }
            }

            if !requested_tool {
                emit(AgentEvent::TurnError {
                    turn_id,
                    message: "provider stream ended before signaling completion".into(),
                });
                self.context.prune_if_needed();
                return Ok(());
            }
        }
    }
}

fn tool_output_chunks(body: &str) -> Vec<String> {
    const CHUNK_BYTES: usize = 256;

    if body.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for piece in body.split_inclusive('\n') {
        if !current.is_empty() && current.len() + piece.len() > CHUNK_BYTES {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(piece);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::SessionContext;
    use crate::llm::{
        ChatRequest, ChatRole, Provider, ProviderError, ProviderResult, ProviderStreamItem,
        ToolCall,
    };
    use crate::tools::{ToolExecutor, ToolResult};
    use futures::{future::{BoxFuture, ready}, stream::{self, BoxStream}};
    use reqwest::Url;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    #[tokio::test]
    async fn turn_with_tool_call_emits_tool_and_assistant_events() {
        let provider = FakeProvider::new(vec![
            ProviderStreamItem::ToolCall(ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"list_dir","path":"src"}"#.into(),
            }),
            ProviderStreamItem::AssistantDelta("done".into()),
            ProviderStreamItem::Done,
        ]);
        let tools = FakeTools::with_text("src files", "agent.rs\nmain.rs");
        let mut agent = Agent::new(provider, tools, SessionContext::new(512));
        let mut events = Vec::new();

        agent
            .run_turn("show src", |event| events.push(event))
            .await
            .unwrap();

        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolStarted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolOutputDelta { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolFinished { .. }))
        );
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::AssistantDelta { text, .. } if text == "done")
        ));
        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::AssistantDone { .. })));
    }

    #[tokio::test]
    async fn tool_output_is_added_to_context_before_follow_up_provider_call() {
        let provider = FakeProvider::new(vec![
            ProviderStreamItem::ToolCall(ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"read_file","path":"src/agent.rs"}"#.into(),
            }),
            ProviderStreamItem::Done,
        ]);
        let tools = FakeTools::with_text("agent preview", "agent body");
        let mut agent = Agent::new(provider, tools, SessionContext::new(512));

        agent.run_turn("read agent", |_| {}).await.unwrap();

        let requests = agent.provider().requests();
        assert_eq!(requests.len(), 2);
        assert!(
            requests[0]
                .messages
                .iter()
                .any(|message| message.content.contains("read agent"))
        );
        assert!(requests[1].messages.iter().any(|message| {
            message.role == ChatRole::Tool
                && message.tool_name.as_deref() == Some("fs")
                && message.content == "agent body"
        }));
    }

    #[tokio::test]
    async fn assistant_text_before_tool_call_is_preserved_for_follow_up_requests() {
        let provider = FakeProvider::new(vec![
            ProviderStreamItem::AssistantDelta("checking".into()),
            ProviderStreamItem::ToolCall(ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"list_dir","path":"src"}"#.into(),
            }),
            ProviderStreamItem::AssistantDelta("done".into()),
            ProviderStreamItem::Done,
        ]);
        let tools = FakeTools::with_text("src files", "agent.rs\nmain.rs");
        let mut agent = Agent::new(provider, tools, SessionContext::new(512));

        agent.run_turn("show src", |_| {}).await.unwrap();

        let requests = agent.provider().requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].messages.iter().any(|message| {
            message.role == ChatRole::Assistant && message.content == "checking"
        }));
        assert!(requests[1].messages.iter().any(|message| message.role == ChatRole::Tool));
    }

    #[tokio::test]
    async fn tool_errors_are_recorded_in_context_and_emit_turn_error() {
        let provider = FakeProvider::new(vec![ProviderStreamItem::ToolCall(ToolCall {
            name: "fs".into(),
            arguments_json: r#"{"op":"read_file","path":"missing.txt"}"#.into(),
        })]);
        let tools = FakeTools::failing("read failed");
        let mut agent = Agent::new(provider, tools, SessionContext::new(512));
        let mut events = Vec::new();

        agent
            .run_turn("read missing", |event| events.push(event))
            .await
            .unwrap();

        let request = agent.context().chat_request().unwrap();
        assert!(request.messages.iter().any(|message| {
            message.role == ChatRole::Tool && message.content.contains("error: read failed")
        }));
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::TurnError { message, .. } if message.contains("read failed"))
        ));
    }

    #[tokio::test]
    async fn provider_errors_emit_turn_error_event() {
        let provider = FakeProvider::failing("provider down");
        let tools = FakeTools::with_text("unused", "unused");
        let mut agent = Agent::new(provider, tools, SessionContext::new(512));
        let mut events = Vec::new();

        agent.run_turn("hello", |event| events.push(event)).await.unwrap();

        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::TurnError { message, .. } if message.contains("provider down"))
        ));
    }

    #[derive(Debug)]
    struct FakeProvider {
        #[allow(dead_code)]
        base_url: Url,
        queued_batches: RefCell<VecDeque<ProviderResult<Vec<ProviderStreamItem>>>>,
        requests: RefCell<Vec<ChatRequest>>,
    }

    impl FakeProvider {
        fn new(items: Vec<ProviderStreamItem>) -> Self {
            Self {
                base_url: Url::parse("http://localhost:11434").unwrap(),
                queued_batches: RefCell::new(queue_items_into_batches(items)),
                requests: RefCell::new(Vec::new()),
            }
        }

        fn failing(message: &str) -> Self {
            Self {
                base_url: Url::parse("http://localhost:11434").unwrap(),
                queued_batches: RefCell::new(VecDeque::from([Err(ProviderError::transport(
                    message,
                ))])),
                requests: RefCell::new(Vec::new()),
            }
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.requests.borrow().clone()
        }
    }

    impl Provider for FakeProvider {
        fn base_url(&self) -> &Url {
            &self.base_url
        }

        fn model(&self) -> &str {
            "fake"
        }

        fn validate<'a>(&'a self) -> BoxFuture<'a, ProviderResult<()>> {
            Box::pin(ready(Ok(())))
        }

        fn stream_chat<'a>(
            &'a self,
            request: ChatRequest,
        ) -> BoxStream<'a, ProviderResult<ProviderStreamItem>> {
            self.requests.borrow_mut().push(request);
            let batch = self
                .queued_batches
                .borrow_mut()
                .pop_front()
                .unwrap_or_else(|| Ok(Vec::new()));
            match batch {
                Ok(items) => Box::pin(stream::iter(items.into_iter().map(Ok))),
                Err(error) => Box::pin(stream::once(ready(Err(error)))),
            }
        }
    }

    #[derive(Debug)]
    struct FakeTools {
        result: std::result::Result<ToolResult, String>,
    }

    impl FakeTools {
        fn with_text(preview: &str, body: &str) -> Self {
            Self {
                result: Ok(ToolResult::text(preview, body)),
            }
        }

        fn failing(message: &str) -> Self {
            Self {
                result: Err(message.into()),
            }
        }
    }

    impl ToolExecutor for FakeTools {
        fn definitions(&self) -> Vec<crate::llm::ToolDefinition> {
            Vec::new()
        }

        fn execute(&mut self, call: &ToolCall) -> Result<ToolResult> {
            if call.name != "fs" {
                return Err(ProviderError::protocol("unsupported fake tool").into());
            }

            match &self.result {
                Ok(result) => Ok(result.clone()),
                Err(message) => Err(ProviderError::protocol(message).into()),
            }
        }
    }

    fn queue_items_into_batches(
        items: Vec<ProviderStreamItem>,
    ) -> VecDeque<ProviderResult<Vec<ProviderStreamItem>>> {
        let mut queued_items: VecDeque<_> = items.into();
        let mut batches = VecDeque::new();

        while !queued_items.is_empty() {
            let mut batch = Vec::new();
            while let Some(item) = queued_items.pop_front() {
                let stop = matches!(
                    item,
                    ProviderStreamItem::ToolCall(_) | ProviderStreamItem::Done
                );
                batch.push(item);
                if stop {
                    break;
                }
            }
            batches.push_back(Ok(batch));
        }

        batches
    }
}
