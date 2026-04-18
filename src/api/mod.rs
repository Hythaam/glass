use reqwest::blocking::Client;
use serde_json::Value;

pub struct ApiClient {
    client: Client,
    url: String,
}

impl ApiClient {
    pub fn new(url: String) -> Self {
        Self { client: Client::new(), url }
    }

    /// Send a user message to the OpenAPI-compatible server and return the assistant reply as a string.
    /// This uses a very small JSON shape and falls back to returning the raw body when the expected
    /// field isn't present. Edge cases are ignored as requested.
    pub fn send_message(&self, message: &str) -> Result<String, Box<dyn std::error::Error>> {
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": message}]
        });

        let resp = self.client.post(&self.url).json(&body).send()?;
        let text = resp.text()?;

        if let Ok(json) = serde_json::from_str::<Value>(&text) {
            if let Some(content) = json.get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c0| c0.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|v| v.as_str()) {
                return Ok(content.to_string());
            }
        }

        // Fallback: return raw response body
        Ok(text)
    }
}
