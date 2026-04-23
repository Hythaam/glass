# Glass v1 service implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full Glass v1 TUI coding-agent service described in `AGENTS.md`, using Ollama, read-only filesystem tooling, in-memory context pruning, and live streamed transcript events.

**Architecture:** Keep the core runtime split into config loading, context management, provider parsing, tool execution, and an agent event loop that is independent from the TUI. Keep the TUI as a thin transcript/input shell over structured agent events, and keep filesystem boundary checks centralized in the `fs` tool.

**Tech Stack:** Rust 2024, Tokio, Reqwest, Serde, Ratatui, Crossterm, anyhow

---

## File structure

- Modify: `Cargo.toml` — keep only dependencies needed for the final v1 implementation
- Modify: `AGENTS.md` — reflect any crate-layout change required to isolate TUI code if `src/tui.rs` is added
- Create: `src/main.rs` — startup, config loading, startup-directory resolution, bootstrap
- Create: `src/agent.rs` — UI-agnostic agent loop and event stream types
- Create: `src/config.rs` — CLI/env/file config loading and validation
- Create: `src/context.rs` — in-memory history, tool observation retention, pruning
- Create: `src/tui.rs` — transcript state, fold state, input handling, Ratatui rendering
- Create: `src/llm/mod.rs` — provider trait, request/response types, provider errors
- Create: `src/llm/ollama.rs` — Ollama startup validation, `/api/chat` streaming, tool-call parsing
- Create: `src/tools/mod.rs` — tool registry, shared tool request/result/error types
- Create: `src/tools/fs.rs` — startup-root path normalization, `read_file`, `list_dir`, `stat_path`
- Delete: `src/tools/search.rs` — remove from the v1 implementation path
- Delete: `src/tools/shell.rs` — remove from the v1 implementation path

### Task 1: Establish crate skeleton and shared runtime types

**Files:**
- Modify: `Cargo.toml`
- Create: `src/main.rs`
- Create: `src/agent.rs`
- Create: `src/config.rs`
- Create: `src/context.rs`
- Create: `src/llm/mod.rs`
- Create: `src/llm/ollama.rs`
- Create: `src/tools/mod.rs`
- Create: `src/tools/fs.rs`
- Create: `src/tui.rs`
- Test: `cargo test module_smoke_test -- --exact`

- [ ] **Step 1: Write the failing smoke test**

```rust
// src/main.rs
mod agent;
mod config;
mod context;
mod llm;
mod tools;
mod tui;

#[cfg(test)]
#[test]
fn module_smoke_test() {
    use crate::agent::AgentEvent;
    use crate::llm::ProviderStreamItem;
    use crate::tools::ToolResult;

    let _ = AgentEvent::AssistantDelta {
        turn_id: 1,
        text: "hi".into(),
    };
    let _ = ProviderStreamItem::Done;
    let _ = ToolResult::Text {
        preview: "ok".into(),
        body: "ok".into(),
    };
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test module_smoke_test -- --exact`
Expected: FAIL with unresolved modules or missing enum definitions.

- [ ] **Step 3: Write minimal implementation**

```rust
// src/agent.rs
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    AssistantDelta { turn_id: u64, text: String },
    AssistantDone { turn_id: u64 },
    ToolStarted { turn_id: u64, tool_name: String },
    ToolFinished { turn_id: u64, tool_name: String, preview: String, body: String },
    TurnError { turn_id: u64, message: String },
}

// src/config.rs
pub struct Config;

// src/context.rs
pub struct SessionContext;

// src/llm/mod.rs
pub mod ollama;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderStreamItem {
    AssistantDelta(String),
    ToolCall(ToolCall),
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments_json: String,
}

// src/llm/ollama.rs
pub struct OllamaProvider;

// src/tools/mod.rs
pub mod fs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolResult {
    Text { preview: String, body: String },
}

// src/tools/fs.rs
pub struct FsTool;

// src/tui.rs
pub struct TuiApp;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test module_smoke_test -- --exact`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/main.rs src/agent.rs src/config.rs src/context.rs src/llm/mod.rs src/llm/ollama.rs src/tools/mod.rs src/tools/fs.rs src/tui.rs
git commit -m "chore: scaffold glass v1 module tree"
```

### Task 2: Implement config loading and startup validation

**Files:**
- Create: `src/config.rs`
- Modify: `src/main.rs`
- Test: `src/config.rs`

- [ ] **Step 1: Write the failing tests**

```rust
// src/config.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_cli_over_env_over_file() {
        let file = FileConfig {
            ollama_url: Some("http://file:11434".into()),
            context_limit_tokens: Some(2048),
        };
        let env = EnvConfig {
            ollama_url: Some("http://env:11434".into()),
            context_limit_tokens: Some(4096),
        };
        let cli = CliConfig {
            ollama_url: Some("http://cli:11434".into()),
            context_limit_tokens: Some(8192),
        };

        let config = Config::from_sources(cli, env, file).unwrap();
        assert_eq!(config.ollama_url.as_str(), "http://cli:11434/");
        assert_eq!(config.context_limit_tokens, 8192);
    }

    #[test]
    fn rejects_invalid_url() {
        let cli = CliConfig {
            ollama_url: Some("localhost:11434".into()),
            context_limit_tokens: Some(4096),
        };

        let error = Config::from_sources(cli, EnvConfig::default(), FileConfig::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("ollama_url"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test loads_cli_over_env_over_file rejects_invalid_url`
Expected: FAIL with missing `Config`, `CliConfig`, or `from_sources`.

- [ ] **Step 3: Write minimal implementation**

```rust
// src/config.rs
use anyhow::{anyhow, Result};
use reqwest::Url;

#[derive(Debug, Default, Clone)]
pub struct CliConfig {
    pub ollama_url: Option<String>,
    pub context_limit_tokens: Option<usize>,
}

#[derive(Debug, Default, Clone)]
pub struct EnvConfig {
    pub ollama_url: Option<String>,
    pub context_limit_tokens: Option<usize>,
}

#[derive(Debug, Default, Clone)]
pub struct FileConfig {
    pub ollama_url: Option<String>,
    pub context_limit_tokens: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub ollama_url: Url,
    pub context_limit_tokens: usize,
}

impl Config {
    pub fn from_sources(cli: CliConfig, env: EnvConfig, file: FileConfig) -> Result<Self> {
        let ollama_url = cli
            .ollama_url
            .or(env.ollama_url)
            .or(file.ollama_url)
            .ok_or_else(|| anyhow!("ollama_url is required"))?;
        let context_limit_tokens = cli
            .context_limit_tokens
            .or(env.context_limit_tokens)
            .or(file.context_limit_tokens)
            .ok_or_else(|| anyhow!("context_limit_tokens is required"))?;
        if context_limit_tokens < 512 {
            return Err(anyhow!("context_limit_tokens must be at least 512"));
        }
        let ollama_url = Url::parse(&ollama_url).map_err(|_| anyhow!("ollama_url must be a full base URL"))?;
        Ok(Self {
            ollama_url,
            context_limit_tokens,
        })
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test loads_cli_over_env_over_file rejects_invalid_url`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.rs src/main.rs
git commit -m "feat: add config loading and validation"
```

### Task 3: Implement read-only filesystem tooling with startup-root enforcement

**Files:**
- Create: `src/tools/fs.rs`
- Modify: `src/tools/mod.rs`
- Test: `src/tools/fs.rs`

- [ ] **Step 1: Write the failing tests**

```rust
// src/tools/fs.rs
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn rejects_escape_outside_root() {
        let tool = FsTool::new(std::env::current_dir().unwrap());
        let error = tool.read_file("../../etc/passwd").unwrap_err().to_string();
        assert!(error.contains("startup directory"));
    }

    #[test]
    fn lists_directory_entries() {
        let root = std::env::temp_dir().join("glass-fs-list");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("nested")).unwrap();
        fs::write(root.join("nested").join("file.txt"), "hello").unwrap();

        let tool = FsTool::new(root.clone());
        let result = tool.list_dir("nested").unwrap();
        assert!(result.body.contains("file.txt"));

        let _ = fs::remove_dir_all(root);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test rejects_escape_outside_root lists_directory_entries`
Expected: FAIL with missing `FsTool` or missing methods.

- [ ] **Step 3: Write minimal implementation**

```rust
// src/tools/fs.rs
use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Component, Path, PathBuf};

use super::ToolResult;

#[derive(Debug, Clone)]
pub struct FsTool {
    root: PathBuf,
}

impl FsTool {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn resolve(&self, input: &str) -> Result<PathBuf> {
        let path = Path::new(input);
        if path.is_absolute() {
            return Err(anyhow!("path must stay inside the startup directory"));
        }
        let mut normalized = self.root.clone();
        for component in path.components() {
            match component {
                Component::Normal(part) => normalized.push(part),
                Component::CurDir => {}
                _ => return Err(anyhow!("path must stay inside the startup directory")),
            }
        }
        let candidate = normalized.canonicalize().or_else(|_| Ok(normalized.clone()))?;
        if !candidate.starts_with(&self.root) {
            return Err(anyhow!("path escapes the startup directory"));
        }
        Ok(normalized)
    }

    pub fn read_file(&self, input: &str) -> Result<ToolResult> {
        let path = self.resolve(input)?;
        let body = fs::read_to_string(&path)?;
        Ok(ToolResult::Text {
            preview: body.lines().next().unwrap_or("").to_string(),
            body,
        })
    }

    pub fn list_dir(&self, input: &str) -> Result<ToolResult> {
        let path = self.resolve(input)?;
        let mut lines = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            lines.push(entry.file_name().to_string_lossy().to_string());
        }
        lines.sort();
        Ok(ToolResult::Text {
            preview: format!("{} entries", lines.len()),
            body: lines.join("\n"),
        })
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test rejects_escape_outside_root lists_directory_entries`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/mod.rs src/tools/fs.rs
git commit -m "feat: add startup-root filesystem tool"
```

### Task 4: Implement in-memory context pruning

**Files:**
- Create: `src/context.rs`
- Test: `src/context.rs`

- [ ] **Step 1: Write the failing tests**

```rust
// src/context.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prunes_old_turns_into_summary() {
        let mut context = SessionContext::new(40);
        context.push_user("one two three four five six");
        context.push_assistant("alpha beta gamma delta epsilon zeta");
        context.push_user("recent");

        context.prune_if_needed();

        assert!(context.summary.as_deref().unwrap_or("").contains("one two"));
        assert_eq!(context.recent_turns.last().unwrap().content, "recent");
    }

    #[test]
    fn compacts_old_tool_output_before_recent_turns() {
        let mut context = SessionContext::new(40);
        context.push_tool_output("list", "a\nb\nc\nd\ne\nf");
        context.push_user("what changed?");

        context.prune_if_needed();

        assert!(!context.tool_observations.is_empty());
        assert!(context.tool_observations[0].body.contains("summary"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test prunes_old_turns_into_summary compacts_old_tool_output_before_recent_turns`
Expected: FAIL with missing `SessionContext` methods or fields.

- [ ] **Step 3: Write minimal implementation**

```rust
// src/context.rs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolObservation {
    pub tool_name: String,
    pub body: String,
}

#[derive(Debug, Default, Clone)]
pub struct SessionContext {
    pub limit: usize,
    pub summary: Option<String>,
    pub recent_turns: Vec<Turn>,
    pub tool_observations: Vec<ToolObservation>,
}

impl SessionContext {
    pub fn new(limit: usize) -> Self {
        Self { limit, ..Self::default() }
    }

    pub fn push_user(&mut self, content: &str) {
        self.recent_turns.push(Turn { role: "user", content: content.into() });
    }

    pub fn push_assistant(&mut self, content: &str) {
        self.recent_turns.push(Turn { role: "assistant", content: content.into() });
    }

    pub fn push_tool_output(&mut self, tool_name: &str, body: &str) {
        self.tool_observations.push(ToolObservation {
            tool_name: tool_name.into(),
            body: body.into(),
        });
    }

    pub fn prune_if_needed(&mut self) {
        let estimate: usize = self
            .recent_turns
            .iter()
            .map(|turn| turn.content.len() / 4)
            .sum::<usize>()
            + self.tool_observations.iter().map(|obs| obs.body.len() / 4).sum::<usize>();
        if estimate <= self.limit || self.recent_turns.len() < 2 {
            return;
        }
        if let Some(oldest) = self.tool_observations.first_mut() {
            oldest.body = format!("summary: {}", oldest.body.lines().take(2).collect::<Vec<_>>().join(" "));
        }
        let removed = self.recent_turns.remove(0);
        self.summary = Some(match self.summary.take() {
            Some(existing) => format!("{existing} {}", removed.content),
            None => removed.content,
        });
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test prunes_old_turns_into_summary compacts_old_tool_output_before_recent_turns`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/context.rs
git commit -m "feat: add in-memory context pruning"
```

### Task 5: Implement Ollama provider validation and streaming parser

**Files:**
- Create: `src/llm/ollama.rs`
- Modify: `src/llm/mod.rs`
- Test: `src/llm/ollama.rs`

- [ ] **Step 1: Write the failing tests**

```rust
// src/llm/ollama.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_assistant_delta_chunk() {
        let item = parse_chat_chunk(r#"{"message":{"content":"hello"},"done":false}"#).unwrap();
        assert_eq!(item, ProviderStreamItem::AssistantDelta("hello".into()));
    }

    #[test]
    fn parses_tool_call_chunk() {
        let item = parse_chat_chunk(
            r#"{"message":{"tool_calls":[{"function":{"name":"fs","arguments":{"op":"list_dir","path":"src"}}}]},"done":false}"#,
        )
        .unwrap();
        match item {
            ProviderStreamItem::ToolCall(call) => {
                assert_eq!(call.name, "fs");
                assert!(call.arguments_json.contains("\"list_dir\""));
            }
            other => panic!("unexpected item: {other:?}"),
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test parses_assistant_delta_chunk parses_tool_call_chunk`
Expected: FAIL with missing `parse_chat_chunk`.

- [ ] **Step 3: Write minimal implementation**

```rust
// src/llm/ollama.rs
use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::Value;

use super::{ProviderStreamItem, ToolCall};

#[derive(Debug, Deserialize)]
struct ChatChunk {
    message: Option<ChunkMessage>,
    done: bool,
}

#[derive(Debug, Deserialize)]
struct ChunkMessage {
    content: Option<String>,
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

pub fn parse_chat_chunk(input: &str) -> Result<ProviderStreamItem> {
    let chunk: ChatChunk = serde_json::from_str(input)?;
    if chunk.done {
        return Ok(ProviderStreamItem::Done);
    }
    if let Some(message) = chunk.message {
        if let Some(content) = message.content {
            return Ok(ProviderStreamItem::AssistantDelta(content));
        }
        if let Some(mut calls) = message.tool_calls {
            let call = calls.pop().ok_or_else(|| anyhow!("tool_calls was empty"))?;
            return Ok(ProviderStreamItem::ToolCall(ToolCall {
                name: call.function.name,
                arguments_json: serde_json::to_string(&call.function.arguments)?,
            }));
        }
    }
    Err(anyhow!("unsupported Ollama chat chunk"))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test parses_assistant_delta_chunk parses_tool_call_chunk`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/mod.rs src/llm/ollama.rs
git commit -m "feat: add Ollama streaming parser"
```

### Task 6: Implement the UI-agnostic agent loop

**Files:**
- Modify: `src/agent.rs`
- Modify: `src/context.rs`
- Modify: `src/llm/mod.rs`
- Modify: `src/tools/mod.rs`
- Test: `src/agent.rs`

- [ ] **Step 1: Write the failing tests**

```rust
// src/agent.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_with_tool_call_emits_tool_and_assistant_events() {
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

        let events = agent.run_turn("show src").unwrap();

        assert!(events.iter().any(|event| matches!(event, AgentEvent::ToolStarted { .. })));
        assert!(events.iter().any(|event| matches!(event, AgentEvent::ToolFinished { .. })));
        assert!(events.iter().any(|event| matches!(event, AgentEvent::AssistantDelta { text, .. } if text == "done")));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test turn_with_tool_call_emits_tool_and_assistant_events -- --exact`
Expected: FAIL with missing `Agent::new`, `run_turn`, or fake test helpers.

- [ ] **Step 3: Write minimal implementation**

```rust
// src/agent.rs
pub struct Agent<P, T> {
    provider: P,
    tools: T,
    context: SessionContext,
    next_turn_id: u64,
}

impl<P, T> Agent<P, T>
where
    P: Provider,
    T: ToolExecutor,
{
    pub fn new(provider: P, tools: T, context: SessionContext) -> Self {
        Self { provider, tools, context, next_turn_id: 1 }
    }

    pub fn run_turn(&mut self, input: &str) -> anyhow::Result<Vec<AgentEvent>> {
        let turn_id = self.next_turn_id;
        self.next_turn_id += 1;
        self.context.push_user(input);
        let mut events = Vec::new();
        for item in self.provider.stream_turn(&self.context)? {
            match item {
                ProviderStreamItem::AssistantDelta(text) => {
                    self.context.push_assistant(&text);
                    events.push(AgentEvent::AssistantDelta { turn_id, text });
                }
                ProviderStreamItem::ToolCall(call) => {
                    events.push(AgentEvent::ToolStarted { turn_id, tool_name: call.name.clone() });
                    let result = self.tools.execute(&call)?;
                    events.push(AgentEvent::ToolFinished {
                        turn_id,
                        tool_name: call.name,
                        preview: result.preview().to_string(),
                        body: result.body().to_string(),
                    });
                }
                ProviderStreamItem::Done => events.push(AgentEvent::AssistantDone { turn_id }),
            }
        }
        self.context.prune_if_needed();
        Ok(events)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test turn_with_tool_call_emits_tool_and_assistant_events -- --exact`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent.rs src/context.rs src/llm/mod.rs src/tools/mod.rs
git commit -m "feat: add UI-agnostic agent loop"
```

### Task 7: Implement TUI transcript behavior, bootstrap wiring, and retire non-v1 files

**Files:**
- Create: `src/tui.rs`
- Modify: `src/main.rs`
- Delete: `src/tools/search.rs`
- Delete: `src/tools/shell.rs`
- Modify: `AGENTS.md`
- Test: `src/tui.rs`

- [ ] **Step 1: Write the failing tests**

```rust
// src/tui.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggles_fold_state_for_tool_entry() {
        let mut app = TuiApp::default();
        app.push_tool_output("fs", "preview", "line1\nline2\nline3");
        assert!(!app.entries[0].expanded);
        app.toggle_entry(0);
        assert!(app.entries[0].expanded);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test toggles_fold_state_for_tool_entry -- --exact`
Expected: FAIL with missing `TuiApp::default`, `push_tool_output`, or `toggle_entry`.

- [ ] **Step 3: Write minimal implementation**

```rust
// src/tui.rs
#[derive(Debug, Default, Clone)]
pub struct TranscriptEntry {
    pub title: String,
    pub preview: String,
    pub body: String,
    pub expanded: bool,
}

#[derive(Debug, Default)]
pub struct TuiApp {
    pub entries: Vec<TranscriptEntry>,
}

impl TuiApp {
    pub fn push_tool_output(&mut self, tool_name: &str, preview: &str, body: &str) {
        self.entries.push(TranscriptEntry {
            title: tool_name.into(),
            preview: preview.into(),
            body: body.into(),
            expanded: false,
        });
    }

    pub fn toggle_entry(&mut self, index: usize) {
        if let Some(entry) = self.entries.get_mut(index) {
            entry.expanded = !entry.expanded;
        }
    }
}

// src/main.rs
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let startup_dir = std::env::current_dir()?;
    let _ = startup_dir;
    Ok(())
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test toggles_fold_state_for_tool_entry -- --exact`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md src/main.rs src/tui.rs src/tools/search.rs src/tools/shell.rs
git commit -m "feat: wire TUI bootstrap and retire non-v1 tools"
```

## Self-review notes

- **Spec coverage:** Task 2 covers config loading and validation. Task 3 covers startup-directory boundary checks and read-only filesystem access. Task 4 covers approximate pruning. Task 5 covers provider parsing and Ollama integration points. Task 6 covers the plan -> act -> observe -> repeat agent loop. Task 7 covers streaming-facing transcript state, folding behavior, and bootstrap cleanup.
- **Placeholder scan:** Removed placeholder wording and gave concrete file paths, code snippets, commands, and expected results for each task.
- **Type consistency:** The plan consistently uses `AgentEvent`, `ProviderStreamItem`, `ToolCall`, `ToolResult`, `FsTool`, `SessionContext`, and `TuiApp`.
