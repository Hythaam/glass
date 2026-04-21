use crate::api::ChatMessage;

/// Manages the conversation history for a single chat session.
pub struct Session {
    messages: Vec<ChatMessage>,
}

impl Session {
    /// Create a new session with an optional system prompt.
    pub fn new(system_prompt: Option<&str>) -> Self {
        let mut messages = Vec::new();
        if let Some(prompt) = system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: prompt.to_string(),
            });
        }
        Self { messages }
    }

    /// Add a user message to the session.
    /// Returns a clone of the message for display purposes.
    pub fn add_user_message(&mut self, content: &str) -> ChatMessage {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: content.to_string(),
        };
        self.messages.push(msg.clone());
        msg
    }

    /// Add an assistant message to the session.
    /// Returns a clone of the message for display purposes.
    pub fn add_assistant_message(&mut self, content: &str) -> ChatMessage {
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: content.to_string(),
        };
        self.messages.push(msg.clone());
        msg
    }

    /// Get a reference to all messages in the session.
    /// This is what gets sent to the API.
    pub fn get_messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Clear all messages from the session.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}
