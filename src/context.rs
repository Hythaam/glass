use serde_json::{Value, json};

use crate::session::{MessageRole, SessionEvent};

pub fn build_chat_messages(events: &[SessionEvent]) -> Vec<Value> {
    events
        .iter()
        .filter(|event| event.include_in_context)
        .filter_map(|event| {
            let role = event.role?;
            Some(json!({
                "role": role.as_str(),
                "content": event.message,
            }))
        })
        .collect()
}

pub fn role_for_event_name(name: &str) -> Option<MessageRole> {
    match name {
        "user" => Some(MessageRole::User),
        "assistant" => Some(MessageRole::Assistant),
        "system" => Some(MessageRole::System),
        "developer" => Some(MessageRole::Developer),
        "tool" => Some(MessageRole::Tool),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};

    use super::*;
    use crate::session::{EventKind, SessionEvent};

    #[test]
    fn build_chat_messages_uses_include_in_context() {
        let timestamp = DateTime::parse_from_rfc3339("2026-05-12T22:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let visible = SessionEvent {
            sequence: 1,
            event: EventKind::UserMessage,
            role: Some(MessageRole::User),
            timestamp,
            include_in_context: true,
            message: "visible".to_owned(),
        };
        let hidden = SessionEvent {
            sequence: 2,
            event: EventKind::SystemEvent,
            role: Some(MessageRole::System),
            timestamp,
            include_in_context: false,
            message: "hidden".to_owned(),
        };

        let messages = build_chat_messages(&[visible, hidden]);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "visible");
    }
}
