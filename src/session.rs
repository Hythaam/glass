use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use saphyr::{LoadableYamlNode, Mapping, Scalar, Yaml, YamlEmitter};

const DEFAULT_SESSION_PREFIX: &str = "glass-session-";
const DEFAULT_SESSION_SUFFIX: &str = ".yaml";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    SystemEvent,
}

impl EventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UserMessage => "user_message",
            Self::AssistantMessage => "assistant_message",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
            Self::SystemEvent => "system_event",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "user_message" => Some(Self::UserMessage),
            "assistant_message" => Some(Self::AssistantMessage),
            "tool_call" => Some(Self::ToolCall),
            "tool_result" => Some(Self::ToolResult),
            "system_event" => Some(Self::SystemEvent),
            _ => None,
        }
    }
}

impl fmt::Display for EventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Developer,
    Tool,
}

impl MessageRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Developer => "developer",
            Self::Tool => "tool",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "user" => Some(Self::User),
            "assistant" => Some(Self::Assistant),
            "system" => Some(Self::System),
            "developer" => Some(Self::Developer),
            "tool" => Some(Self::Tool),
            _ => None,
        }
    }
}

impl fmt::Display for MessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionEvent {
    pub sequence: usize,
    pub event: EventKind,
    pub role: Option<MessageRole>,
    pub timestamp: DateTime<Utc>,
    pub include_in_context: bool,
    pub message: String,
}

impl SessionEvent {
    pub fn user_message(sequence: usize, message: impl Into<String>) -> Self {
        Self::new(
            sequence,
            EventKind::UserMessage,
            Some(MessageRole::User),
            message,
        )
    }

    pub fn assistant_message(sequence: usize, message: impl Into<String>) -> Self {
        Self::new(
            sequence,
            EventKind::AssistantMessage,
            Some(MessageRole::Assistant),
            message,
        )
    }

    pub fn new(
        sequence: usize,
        event: EventKind,
        role: Option<MessageRole>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            sequence,
            event,
            role,
            timestamp: Utc::now(),
            include_in_context: default_include_in_context(event),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Session {
    path: PathBuf,
}

impl Session {
    pub fn resolve(session_name: Option<&str>, cwd: &Path) -> Result<Self> {
        let path = match session_name {
            Some(name) => cwd.join(name),
            None => cwd.join(next_default_session_name(cwd)?),
        };

        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load_events(&self) -> Result<Vec<SessionEvent>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let raw = fs::read_to_string(&self.path)
            .with_context(|| format!("failed to read session file {}", self.path.display()))?;

        if raw.trim().is_empty() {
            return Ok(Vec::new());
        }

        let docs = Yaml::load_from_str(&raw)
            .map_err(|error| anyhow!("failed to parse {}: {error}", self.path.display()))?;

        docs.iter()
            .enumerate()
            .map(|(index, doc)| parse_event(doc).with_context(|| format!("document {}", index + 1)))
            .collect()
    }

    pub fn append_event(&self, event: &SessionEvent) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create session directory {}", parent.display())
            })?;
        }

        let yaml = serialize_event(event)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("failed to open session file {}", self.path.display()))?;

        if file.metadata()?.len() > 0 {
            writeln!(file).context("failed to separate YAML documents")?;
        }

        file.write_all(yaml.as_bytes())
            .with_context(|| format!("failed to append session event to {}", self.path.display()))
    }

    pub fn next_sequence(&self) -> Result<usize> {
        Ok(self.load_events()?.len() + 1)
    }
}

pub fn default_include_in_context(event: EventKind) -> bool {
    !matches!(event, EventKind::SystemEvent)
}

fn next_default_session_name(cwd: &Path) -> Result<String> {
    let mut next_index = 1usize;

    for entry in fs::read_dir(cwd)
        .with_context(|| format!("failed to read workspace directory {}", cwd.display()))?
    {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
            continue;
        };

        if !name.starts_with(DEFAULT_SESSION_PREFIX) || !name.ends_with(DEFAULT_SESSION_SUFFIX) {
            continue;
        }

        let index = &name[DEFAULT_SESSION_PREFIX.len()..name.len() - DEFAULT_SESSION_SUFFIX.len()];
        let Ok(parsed) = index.parse::<usize>() else {
            continue;
        };

        next_index = next_index.max(parsed + 1);
    }

    Ok(format!(
        "{DEFAULT_SESSION_PREFIX}{next_index}{DEFAULT_SESSION_SUFFIX}"
    ))
}

fn parse_event(doc: &Yaml<'_>) -> Result<SessionEvent> {
    if !doc.is_mapping() {
        bail!("expected YAML mapping document");
    }

    let event = required_str(doc, "event")?;
    let sequence = required_i64(doc, "sequence")?;
    if sequence < 1 {
        bail!("sequence must be >= 1");
    }

    let timestamp = required_str(doc, "timestamp")?;
    let timestamp = DateTime::parse_from_rfc3339(timestamp)
        .with_context(|| format!("invalid timestamp {timestamp:?}"))?
        .with_timezone(&Utc);

    let include_in_context = required_bool(doc, "include_in_context")?;
    let message = required_str(doc, "message")?.to_owned();
    let role = optional_str(doc, "role").and_then(MessageRole::from_str);

    Ok(SessionEvent {
        sequence: sequence as usize,
        event: EventKind::from_str(event).ok_or_else(|| anyhow!("unsupported event {event:?}"))?,
        role,
        timestamp,
        include_in_context,
        message,
    })
}

fn serialize_event(event: &SessionEvent) -> Result<String> {
    let mut mapping = Mapping::new();
    mapping.insert(
        Yaml::value_from_str("sequence"),
        Yaml::Value(Scalar::Integer(event.sequence as i64)),
    );
    mapping.insert(
        Yaml::value_from_str("event"),
        Yaml::scalar_from_string(event.event.as_str().to_owned()),
    );
    if let Some(role) = event.role {
        mapping.insert(
            Yaml::value_from_str("role"),
            Yaml::scalar_from_string(role.as_str().to_owned()),
        );
    }
    mapping.insert(
        Yaml::value_from_str("timestamp"),
        Yaml::scalar_from_string(event.timestamp.to_rfc3339()),
    );
    mapping.insert(
        Yaml::value_from_str("include_in_context"),
        Yaml::Value(Scalar::Boolean(event.include_in_context)),
    );
    mapping.insert(
        Yaml::value_from_str("message"),
        Yaml::scalar_from_string(event.message.clone()),
    );

    let doc = Yaml::Mapping(mapping);
    let mut out = String::new();
    let mut emitter = YamlEmitter::new(&mut out);
    emitter.multiline_strings(true);
    emitter.dump(&doc)?;
    Ok(out)
}

fn required_str<'a>(doc: &'a Yaml<'a>, key: &str) -> Result<&'a str> {
    optional_str(doc, key).ok_or_else(|| anyhow!("missing string field {key:?}"))
}

fn optional_str<'a>(doc: &'a Yaml<'a>, key: &str) -> Option<&'a str> {
    doc.as_mapping_get(key).and_then(Yaml::as_str)
}

fn required_bool(doc: &Yaml<'_>, key: &str) -> Result<bool> {
    doc.as_mapping_get(key)
        .and_then(Yaml::as_bool)
        .ok_or_else(|| anyhow!("missing boolean field {key:?}"))
}

fn required_i64(doc: &Yaml<'_>, key: &str) -> Result<i64> {
    doc.as_mapping_get(key)
        .and_then(Yaml::as_integer)
        .ok_or_else(|| anyhow!("missing integer field {key:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn next_default_session_name_skips_existing_indices() {
        let temp = tempdir().unwrap();
        fs::write(temp.path().join("glass-session-1.yaml"), "---\n").unwrap();
        fs::write(temp.path().join("glass-session-4.yaml"), "---\n").unwrap();

        let session = Session::resolve(None, temp.path()).unwrap();

        assert_eq!(session.path(), &temp.path().join("glass-session-5.yaml"));
    }

    #[test]
    fn load_events_filters_unknown_fields_without_failing() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("session.yaml");
        fs::write(
            &path,
            r#"---
sequence: 1
event: user_message
role: user
timestamp: 2026-05-12T22:30:00Z
include_in_context: true
message: hello
extra_field: preserve-me
"#,
        )
        .unwrap();

        let session = Session { path };
        let events = session.load_events().unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].message, "hello");
        assert!(events[0].include_in_context);
    }

    #[test]
    fn append_event_round_trips_through_yaml() {
        let temp = tempdir().unwrap();
        let session = Session::resolve(Some("session.yaml"), temp.path()).unwrap();
        let mut event = SessionEvent::user_message(1, "hello\nworld");
        event.timestamp = DateTime::parse_from_rfc3339("2026-05-12T22:30:00Z")
            .unwrap()
            .with_timezone(&Utc);

        session.append_event(&event).unwrap();
        let loaded = session.load_events().unwrap();

        assert_eq!(loaded, vec![event]);
    }
}
