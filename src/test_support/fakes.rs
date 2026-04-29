use anyhow::Result;
use futures::{
    future::{BoxFuture, ready},
    stream::{self, BoxStream},
};
use reqwest::Url;
use std::cell::RefCell;
use std::collections::VecDeque;

use crate::llm::{
    ChatRequest, Provider, ProviderError, ProviderResult, ProviderStreamItem, ToolCall,
};
use crate::tools::{ToolExecutor, ToolResult};

#[derive(Debug)]
pub(crate) struct FakeProvider {
    base_url: Url,
    queued_batches: RefCell<VecDeque<ProviderResult<Vec<ProviderStreamItem>>>>,
    requests: RefCell<Vec<ChatRequest>>,
}

impl FakeProvider {
    pub(crate) fn new(items: Vec<ProviderStreamItem>) -> Self {
        Self {
            base_url: Url::parse("http://localhost:11434").unwrap(),
            queued_batches: RefCell::new(queue_items_into_batches(items)),
            requests: RefCell::new(Vec::new()),
        }
    }

    pub(crate) fn failing(message: &str) -> Self {
        Self {
            base_url: Url::parse("http://localhost:11434").unwrap(),
            queued_batches: RefCell::new(VecDeque::from([Err(ProviderError::transport(message))])),
            requests: RefCell::new(Vec::new()),
        }
    }

    pub(crate) fn idle() -> Self {
        Self::new(Vec::new())
    }

    pub(crate) fn requests(&self) -> Vec<ChatRequest> {
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

    fn validate<'a>(&'a mut self) -> BoxFuture<'a, ProviderResult<()>> {
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
pub(crate) struct FakeTools {
    result: std::result::Result<ToolResult, String>,
}

impl FakeTools {
    pub(crate) fn with_text(preview: &str, body: &str) -> Self {
        Self {
            result: Ok(ToolResult::text(preview, body)),
        }
    }

    pub(crate) fn failing(message: &str) -> Self {
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
                ProviderStreamItem::ToolCall(_) | ProviderStreamItem::Done { .. }
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

#[cfg(test)]
mod tests {
    use super::{FakeProvider, FakeTools};
    use crate::llm::{ChatMessage, ChatRequest, ChatRole, Provider, ToolCall};
    use crate::tools::ToolExecutor;
    use futures::StreamExt;

    #[tokio::test]
    async fn provider_records_requests() {
        let provider = FakeProvider::idle();
        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".into(),
                tool_name: None,
                tool_calls: None,
            }],
            tools: Vec::new(),
        };

        let items: Vec<_> = provider.stream_chat(request.clone()).collect().await;

        assert!(items.is_empty());
        assert_eq!(provider.requests().len(), 1);
        assert_eq!(
            provider.requests()[0].messages[0].content,
            request.messages[0].content
        );
    }

    #[test]
    fn tools_return_configured_text() {
        let mut tools = FakeTools::with_text("preview", "body");
        let result = tools
            .execute(&ToolCall {
                name: "fs".into(),
                arguments_json: "{}".into(),
            })
            .unwrap();

        assert_eq!(result.preview(), "preview");
        assert_eq!(result.body(), "body");
    }
}
