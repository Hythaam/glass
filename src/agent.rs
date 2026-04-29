use anyhow::Result;
use futures::StreamExt;

use crate::context::SessionContext;
use crate::llm::{Provider, ProviderStreamItem, RequestTokenUsage};
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AgentStatus {
    pub model_name: String,
    pub context_tokens: usize,
    pub context_limit: usize,
    pub last_request_usage: Option<RequestTokenUsage>,
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
    pub fn status_snapshot(&self) -> AgentStatus {
        AgentStatus {
            model_name: self.provider.model().to_string(),
            context_tokens: self.context.estimated_tokens_total(),
            context_limit: self.context.limit(),
            last_request_usage: self.context.latest_request_usage(),
        }
    }

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
                        drop(stream);
                        return self.finish_turn_error(turn_id, error.to_string(), &mut emit);
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
                                drop(stream);
                                return self.finish_turn_error(
                                    turn_id,
                                    error.to_string(),
                                    &mut emit,
                                );
                            }
                        }

                        break;
                    }
                    ProviderStreamItem::Done { usage } => {
                        if assistant_text.is_empty() {
                            drop(stream);
                            return self.finish_turn_error(
                                turn_id,
                                "provider completed with no assistant content or tool calls; the configured llama.cpp model may not support tool calling",
                                &mut emit,
                            );
                        }

                        self.context
                            .push_assistant_with_usage(&assistant_text, usage);
                        emit(AgentEvent::AssistantDone { turn_id });
                        self.context.prune_if_needed();
                        return Ok(());
                    }
                }
            }

            if !requested_tool {
                drop(stream);
                return self.finish_turn_error(
                    turn_id,
                    "provider stream ended before signaling completion",
                    &mut emit,
                );
            }
        }
    }

    fn finish_turn_error<F>(
        &mut self,
        turn_id: u64,
        message: impl Into<String>,
        emit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(AgentEvent),
    {
        emit(AgentEvent::TurnError {
            turn_id,
            message: message.into(),
        });
        self.context.prune_if_needed();
        Ok(())
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
    use crate::llm::{ChatRole, ProviderStreamItem, RequestTokenUsage, ToolCall};
    use crate::test_support::fakes::{FakeProvider, FakeTools};

    fn test_agent(provider: FakeProvider, tools: FakeTools) -> Agent<FakeProvider, FakeTools> {
        Agent::new(provider, tools, SessionContext::new(512, None))
    }

    fn fs_call(op: &str, path: &str) -> ProviderStreamItem {
        ProviderStreamItem::ToolCall(ToolCall {
            name: "fs".into(),
            arguments_json: format!(r#"{{"op":"{op}","path":"{path}"}}"#),
        })
    }

    fn assert_has_event(events: &[AgentEvent], predicate: impl Fn(&AgentEvent) -> bool) {
        assert!(
            events.iter().any(predicate),
            "missing expected event: {events:?}"
        );
    }

    #[tokio::test]
    async fn turn_with_tool_call_emits_tool_and_assistant_events() {
        let provider = FakeProvider::new(vec![
            fs_call("list_dir", "src"),
            ProviderStreamItem::AssistantDelta("done".into()),
            ProviderStreamItem::Done { usage: None },
        ]);
        let tools = FakeTools::with_text("src files", "agent.rs\nmain.rs");
        let mut agent = test_agent(provider, tools);
        let mut events = Vec::new();

        agent
            .run_turn("show src", |event| events.push(event))
            .await
            .unwrap();

        assert_has_event(&events, |event| {
            matches!(event, AgentEvent::ToolStarted { .. })
        });
        assert_has_event(&events, |event| {
            matches!(event, AgentEvent::ToolOutputDelta { .. })
        });
        assert_has_event(&events, |event| {
            matches!(event, AgentEvent::ToolFinished { .. })
        });
        assert_has_event(
            &events,
            |event| matches!(event, AgentEvent::AssistantDelta { text, .. } if text == "done"),
        );
        assert_has_event(&events, |event| {
            matches!(event, AgentEvent::AssistantDone { .. })
        });
    }

    #[tokio::test]
    async fn tool_output_is_added_to_context_before_follow_up_provider_call() {
        let provider = FakeProvider::new(vec![
            fs_call("read_file", "src/agent.rs"),
            ProviderStreamItem::Done { usage: None },
        ]);
        let tools = FakeTools::with_text("agent preview", "agent body");
        let mut agent = test_agent(provider, tools);

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
            fs_call("list_dir", "src"),
            ProviderStreamItem::AssistantDelta("done".into()),
            ProviderStreamItem::Done { usage: None },
        ]);
        let tools = FakeTools::with_text("src files", "agent.rs\nmain.rs");
        let mut agent = test_agent(provider, tools);

        agent.run_turn("show src", |_| {}).await.unwrap();

        let requests = agent.provider().requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].messages.iter().any(|message| {
            message.role == ChatRole::Assistant && message.content == "checking"
        }));
        assert!(
            requests[1]
                .messages
                .iter()
                .any(|message| message.role == ChatRole::Tool)
        );
    }

    #[tokio::test]
    async fn tool_errors_are_recorded_in_context_and_emit_turn_error() {
        let provider = FakeProvider::new(vec![fs_call("read_file", "missing.txt")]);
        let tools = FakeTools::failing("read failed");
        let mut agent = test_agent(provider, tools);
        let mut events = Vec::new();

        agent
            .run_turn("read missing", |event| events.push(event))
            .await
            .unwrap();

        let request = agent.context().chat_request().unwrap();
        assert!(request.messages.iter().any(|message| {
            message.role == ChatRole::Tool && message.content.contains("error: read failed")
        }));
        assert_has_event(
            &events,
            |event| matches!(event, AgentEvent::TurnError { message, .. } if message.contains("read failed")),
        );
    }

    #[tokio::test]
    async fn provider_errors_emit_turn_error_event() {
        let provider = FakeProvider::failing("provider down");
        let tools = FakeTools::with_text("unused", "unused");
        let mut agent = test_agent(provider, tools);
        let mut events = Vec::new();

        agent
            .run_turn("hello", |event| events.push(event))
            .await
            .unwrap();

        assert_has_event(
            &events,
            |event| matches!(event, AgentEvent::TurnError { message, .. } if message.contains("provider down")),
        );
    }

    #[tokio::test]
    async fn empty_completed_response_emits_turn_error() {
        let provider = FakeProvider::new(vec![ProviderStreamItem::Done { usage: None }]);
        let tools = FakeTools::with_text("unused", "unused");
        let mut agent = test_agent(provider, tools);
        let mut events = Vec::new();

        agent
            .run_turn("use the tool", |event| events.push(event))
            .await
            .unwrap();

        assert_has_event(
            &events,
            |event| matches!(event, AgentEvent::TurnError { message, .. } if message.contains("no assistant content or tool calls")),
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::AssistantDone { .. }))
        );
    }

    #[tokio::test]
    async fn assistant_turn_records_provider_usage_in_context() {
        let provider = FakeProvider::new(vec![
            ProviderStreamItem::AssistantDelta("done".into()),
            ProviderStreamItem::Done {
                usage: Some(RequestTokenUsage {
                    prompt_tokens: 19,
                    completion_tokens: 2,
                    total_tokens: 21,
                }),
            },
        ]);
        let tools = FakeTools::with_text("unused", "unused");
        let mut agent = test_agent(provider, tools);

        agent.run_turn("hello", |_| {}).await.unwrap();

        let assistant_debug = format!("{:?}", agent.context().recent_turns().last().unwrap());
        assert!(assistant_debug.contains("prompt_tokens: 19"));
        assert!(assistant_debug.contains("completion_tokens: 2"));
        assert!(assistant_debug.contains("total_tokens: 21"));
    }
}
