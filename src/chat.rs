use std::env;

use anyhow::{Context, Result, anyhow, bail};
use async_openai::{Client, config::OpenAIConfig};
use futures_util::StreamExt;
use serde_json::{Value, json};

pub struct ChatClient {
    client: Client<OpenAIConfig>,
    model: String,
}

impl ChatClient {
    pub fn from_env() -> Result<Self> {
        let host = env::var("GLASS_OPENAI_HOST").context(
            "GLASS_OPENAI_HOST is required and should point at an OpenAI-compatible base URL",
        )?;
        let api_key = env::var("OPENAI_API_KEY")
            .or_else(|_| env::var("GLASS_OPENAI_API_KEY"))
            .context("OPENAI_API_KEY or GLASS_OPENAI_API_KEY is required")?;
        let model = env::var("GLASS_OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_owned());

        let config = OpenAIConfig::new()
            .with_api_base(host)
            .with_api_key(api_key);

        Ok(Self {
            client: Client::with_config(config),
            model,
        })
    }

    pub async fn stream_chat_completion<F>(
        &self,
        messages: Vec<Value>,
        mut on_delta: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Result<()>,
    {
        if messages.is_empty() {
            bail!("cannot create a chat request with no context messages");
        }

        let request = json!({
            "model": self.model,
            "messages": messages,
            "stream": true
        });

        let mut stream = self
            .client
            .chat()
            .create_stream_byot::<_, Value>(request)
            .await
            .context("failed to start streaming chat completion")?;

        let mut output = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("failed while reading streamed chat completion")?;
            let delta = chunk["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap_or_default();
            if !delta.is_empty() {
                output.push_str(delta);
                on_delta(&output)?;
            }
        }

        if output.is_empty() {
            return Err(anyhow!("model response was empty"));
        }

        Ok(output)
    }
}
