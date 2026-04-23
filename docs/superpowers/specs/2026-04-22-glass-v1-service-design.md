# Glass v1 service design

## Problem

The crate needs a full v1 implementation that matches `AGENTS.md`: a Rust TUI chat client for the current directory, backed only by Ollama, with automatic read-only filesystem tooling, in-memory context management, and live streaming of both assistant and tool output.

The current source tree is effectively empty, so the design needs to define the full runtime structure while preserving the architectural boundaries required for future headless support.

## Goals

- Implement the full v1 runtime described in `AGENTS.md`
- Keep the core agent loop independent from Ratatui and Crossterm
- Support only read-only filesystem access inside the startup directory
- Stream assistant text, tool activity, and tool output live in the TUI
- Keep session history in memory only, with approximate-budget pruning
- Validate Ollama configuration and fail fast at startup when it is invalid or the default model is unavailable
- Cover pure logic with unit tests only

## Non-goals

- Headless mode
- Persistent session history
- Multiple model providers
- Dedicated search or shell tools
- Filesystem writes of any kind
- Integration tests or end-to-end UI harnesses

## Recommended approach

Build the service in thin, testable slices around a small set of stable boundaries:

1. `config.rs` handles loading and validation
2. `tools/fs.rs` owns all startup-directory path checks and read-only filesystem operations
3. `llm/ollama.rs` owns Ollama HTTP requests, startup model validation, streaming response parsing, and tool-call decoding
4. `context.rs` owns in-memory history and approximate pruning
5. `agent.rs` owns the plan -> act -> observe -> repeat loop and emits structured transcript events
6. `main.rs` and a small TUI layer wire input/output around the agent without moving protocol logic into the UI

This keeps the risky logic isolated, matches the requested crate layout, and leaves room for a later headless frontend without rewriting the agent loop.

## Alternatives considered

### 1. Vertical slice first

Implement one end-to-end chat flow quickly, then refactor toward the target architecture.

**Trade-off:** Faster to demo, but likely to leak TUI concerns into the agent loop and create cleanup work.

### 2. TUI-first shell

Build the full transcript and input experience first with mocked agent events, then wire in the backend.

**Trade-off:** Good for interaction details, but delays the hard correctness work around provider parsing, boundary checks, and tool execution.

### 3. Core-first modular build (recommended)

Define the core runtime contracts first, then connect the TUI as a thin shell.

**Trade-off:** Slightly more up-front design work, but the implementation stays closer to `AGENTS.md` and is easier to test.

## Architecture

### Runtime flow

`main.rs` resolves the startup directory from `cwd`, loads validated config, constructs the provider, tool registry, context store, agent, and TUI app, then hands control to the TUI loop.

For each submitted user message:

1. The TUI sends the message to the agent and subscribes to streamed agent events.
2. The agent builds a provider request from the current context plus the advertised tool schema.
3. The Ollama provider streams response chunks back as assistant text and may emit tool-call requests.
4. The agent forwards assistant chunks to the TUI immediately.
5. When a tool call appears, the agent executes it automatically, emits tool-start and tool-result events, stores the observation in context, and continues the loop with the tool result included.
6. The cycle ends when Ollama produces a normal assistant completion without another tool request.

The TUI never interprets Ollama protocol payloads or touches the filesystem directly.

### Core boundaries

#### `agent.rs`

The agent owns conversation orchestration and nothing UI-specific. It should expose an async API that accepts a user message and yields a stream of structured events such as:

- assistant text delta
- assistant message complete
- tool invocation started
- tool output delta or chunk
- tool result complete
- fatal turn error

The agent keeps internal reasoning private. Only assistant-facing text and tool-facing transcript events are surfaced.

#### `llm/mod.rs`

The provider boundary should expose:

- startup validation
- a streamed chat-turn interface
- tool schema advertisement
- typed provider errors

The trait remains generic enough for future providers, but v1 only implements Ollama.

#### `llm/ollama.rs`

The Ollama implementation uses:

- configured base URL only
- built-in default model `llama3.1:8b`
- `/api/tags` or equivalent startup validation to confirm the model exists
- `/api/chat` with streaming enabled

The parser should translate Ollama JSON chunks into typed stream items:

- assistant text fragments
- tool call requests with name and structured arguments
- completion markers

Malformed provider output should fail the active turn with actionable error text rather than being ignored.

#### `tools/mod.rs`

The tool registry should stay simple because v1 has a single tool surface. It should define:

- a `Tool` trait or equivalent dispatcher contract
- a registry for looking up tools by name
- structured tool arguments and results
- shared error types with clear user-facing context

The registry still matters even with one tool because the provider communicates in terms of named tool calls.

#### `tools/fs.rs`

The filesystem tool owns all path normalization and boundary enforcement. Supported operations are:

- `read_file`
- `list_dir`
- `stat_path`

Each operation accepts a path relative to the startup directory. Before use, the path is normalized, joined to the startup root, and rejected if its resolved target escapes the root by `..`, absolute-path bypass, or symlink traversal.

All filesystem access is read-only. Unsupported operations return explicit errors.

#### `context.rs`

The context store keeps:

- recent raw user and assistant turns
- recent raw tool observations needed for follow-up questions
- a rolling internal summary for older conversation

Pruning is triggered by an approximate token estimate, not an exact tokenizer. The estimator can use a simple heuristic such as character count divided by a constant ratio, as long as the limit is treated consistently and documented as approximate.

When pruning begins:

1. Older raw tool output is compacted first into short summaries.
2. Older conversation turns are merged into an internal summary.
3. Recent raw turns remain intact.

The summary is internal context only and is not shown as assistant output.

### TUI design

The v1 TUI consists of a single transcript pane plus a bottom composer.

Transcript entries are typed renderables:

- user message
- assistant message
- tool activity header
- tool output preview
- expanded tool output body
- inline error

Behavior rules:

- assistant text streams into the active assistant entry
- tool activity appears inline in transcript order
- long tool output is folded behind a short preview
- folded tool sections expand and collapse in place on mouse click
- `Enter` sends
- `Shift+Enter` inserts newline
- `PageUp` and `PageDown` scroll transcript
- `Ctrl+C` quits

Fold state, scrolling state, hit testing, and input editing stay in the TUI layer. The agent only emits events.

### Configuration

`config.rs` loads the v1 config values from:

1. CLI flags
2. environment variables
3. `~/.config/glass/config.toml`

Supported keys:

- `--ollama-url` / `GLASS_OLLAMA_URL` / `ollama_url`
- `--context-limit-tokens` / `GLASS_CONTEXT_LIMIT_TOKENS` / `context_limit_tokens`

Validation rules:

- `ollama_url` must be a full base URL with scheme and host, and port when required
- `context_limit_tokens` must be a positive integer above a small minimum practical threshold
- precedence is CLI > environment > config file

Startup should fail with clear errors when config is invalid or when the default model is unavailable from the configured Ollama server.

### Error handling

Errors should remain local and explicit:

- config errors stop startup immediately
- provider validation errors stop startup immediately
- turn-scoped provider errors become inline transcript errors and end the active turn cleanly
- tool errors are surfaced inline with actionable context and are also fed back into the model as tool observations when appropriate
- boundary violations are rejected before filesystem access occurs

No broad fallback behavior should mask invalid input or malformed provider output.

### Testing strategy

Unit tests should cover the pure logic called out in `AGENTS.md`:

- config precedence and validation
- path normalization and startup-root boundary enforcement
- symlink escape rejection where practical on supported platforms
- context pruning behavior around the approximate token limit
- Ollama request shaping
- Ollama streamed response and tool-call parsing
- agent state transitions for normal reply, tool call, tool error, and provider error paths
- TUI-local folding and hit-testing helpers if kept as pure functions

## Implementation notes

- Remove `src/tools/search.rs` from the active v1 module tree
- Do not add an OpenAI provider path
- Keep any TUI-specific types out of `agent.rs`
- Keep tool output structured enough that the TUI can preview and fold it without re-parsing raw strings
- Prefer small enums and plain structs over layered abstractions

## Scope check

This design is still small enough for a single implementation plan because each module has one clear responsibility and the runtime has a single provider, a single tool family, and a single frontend.
