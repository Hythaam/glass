use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Hardcoded server address for the OpenAI-compatible API.
const DEFAULT_BASE_URL: &str = "http://192.168.1.167:11434/v1";

/// A single chat message with role and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// A streaming chunk from the LLM response.
#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

/// Async HTTP client for the OpenAI-compatible API.
#[allow(dead_code)]
pub struct ApiClient {
    base_url: String,
    model: String,
    http_client: Client,
}

impl ApiClient {
    /// Create a new API client with the default hardcoded base URL.
    pub fn new(model: &str) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.to_string(),
            http_client: Client::new(),
        }
    }

    /// Send a non-streaming chat request and return the full assistant response.
    #[allow(dead_code)]
    pub async fn send_message(
        &self,
        messages: &[ChatMessage],
    ) -> Result<String> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
        });

        let resp = self
            .http_client
            .post(format!("{}/chat/completions", self.base_url))
            .json(&body)
            .send()
            .await
            .context("Failed to send chat request")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "<could not read error body>".to_string());
            return Err(anyhow::anyhow!(
                "API error ({}): {}",
                status,
                text
            ));
        }

        let chat_resp: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse chat response")?;

        chat_resp
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("No content in response"))
    }

    /// Send a streaming chat request and return a channel receiver for tokens.
    /// The caller receives tokens as `Ok(String)` values.
    /// When streaming is complete, an `Ok(String)` with an empty string is sent
    /// to signal the end of the stream.
    pub async fn send_stream_channel(
        &self,
        messages: &[ChatMessage],
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String>>> {
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        let base_url = self.base_url.clone();
        let model = self.model.clone();
        // Serialize messages here so we can move them into the spawned task.
        let messages_json = serde_json::to_value(messages)
            .context("Failed to serialize messages")?;

        tokio::spawn(async move {
            let client = Client::new();
            let body = serde_json::json!({
                "model": model,
                "messages": messages_json,
                "stream": true,
            });

            let resp = match client
                .post(format!("{}/chat/completions", base_url))
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(e.into())).await;
                    return;
                }
            };

            let status = resp.status();
            if !status.is_success() {
                let text = resp
                    .text()
                    .await
                    .unwrap_or_else(|_| "<could not read error body>".to_string());
                let _ = tx
                    .send(Err(anyhow::anyhow!(
                        "API error ({}): {}",
                        status,
                        text
                    )))
                    .await;
                return;
            }

            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(Err(e.into())).await;
                        break;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE lines from the buffer
                let mut lines = Vec::new();
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();
                    lines.push(line);
                }

                for line in lines {
                    if !line.starts_with("data: ") {
                        continue;
                    }
                    let json_str = &line["data: ".len()..];
                    if json_str == "[DONE]" {
                        break;
                    }
                    match serde_json::from_str::<StreamChunk>(json_str) {
                        Ok(stream_chunk) => {
                            for choice in stream_chunk.choices {
                                if let Some(content) = choice.delta.content {
                                    let _ = tx.send(Ok(content)).await;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(anyhow::anyhow!(
                                "Failed to parse stream: {}: {}",
                                e,
                                json_str
                            )))
                            .await;
                        }
                    }
                }
            }

            // Signal completion with an empty string
            let _ = tx.send(Ok(String::new())).await;
        });

        Ok(rx)
    }
}
