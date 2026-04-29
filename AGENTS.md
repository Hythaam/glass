# Glass

Glass is a Rust TUI coding-agent harness focused on simplicity.

v1 exists to let a user chat about the current directory.

## Operating rule

When behavior is not specified in this file, ask a clarifying question rather than guessing. Keep unresolved decisions explicit as `TBD` instead of silently inventing behavior in code.

## v1 scope

- Primary user story: chat about the current directory
- Frontend: TUI only
- Future extension: headless mode later, so core logic must stay UI-agnostic
- Supported platforms: Linux and macOS
- LLM provider: llama.cpp only
- Tool approval: none; tool calls auto-run
- Tool boundary: startup directory only
- Session history: in-memory only for the active session
- Streaming: assistant output and tool output must stream live in the TUI
- Testing: unit tests only

## v1 non-goals

- Headless mode
- Persistent session history
- Multiple LLM providers
- Dedicated search tooling
- Additional config beyond the server URL, context pruning limit, and optional system prompt file
- Test requirements beyond unit tests

## Crate layout (v1)

```text
src/
├── main.rs              # Startup, config loading, startup-directory resolution, TUI bootstrap
├── agent.rs             # Core loop: plan -> act -> observe -> repeat
├── tools/
│   ├── mod.rs           # Tool trait + dispatch registry
│   └── fs.rs            # File operations inside the startup directory
├── llm/
│   ├── mod.rs           # Provider interface
│   └── llama_cpp.rs     # llama.cpp implementation for v1
├── context.rs           # In-memory session history and pruning
├── config.rs            # CLI/env/config-file loading for server settings
└── tui/
    ├── mod.rs           # TUI module wiring
    ├── app.rs           # Terminal event loop, task spawning, and draw orchestration
    ├── state.rs         # Input/composer/status state transitions
    ├── transcript.rs    # Transcript entries, folding, selection, and scrolling
    ├── render.rs        # Shared layout/render helpers
    └── tests.rs         # TUI-focused unit tests
```

Remove the dedicated search tool from v1. Directory inspection should happen through `fs`.

The active provider implementation lives in `llm/llama_cpp.rs`. The config surface still uses `ollama_url` naming because v1 talks to llama.cpp through an Ollama-compatible HTTP API shape.

## Runtime model

1. Start from the current working directory. The startup directory is the tool root even when it is a subdirectory of a Git repo or not a Git repo at all.
2. Load the v1 config values: the llama.cpp server base URL, the context pruning limit, and the optional system prompt file.
3. Start one in-memory chat session.
4. Accept user input in the TUI.
5. Send session context to the llama.cpp backend through its Ollama-compatible chat API.
6. Stream assistant output to the TUI.
7. When the assistant requests a tool, execute it automatically.
8. Stream tool status and tool output to the TUI.
9. Feed tool observations back into the loop.
10. End the session when the user quits. Do not persist history between runs.

The TUI must also support local slash commands:
- `/exit` closes the harness immediately
- `/new` clears the visible transcript and starts a fresh in-memory agent context

## Architecture rules

- Keep the agent loop independent from Ratatui/Crossterm types.
- Keep tool interfaces explicit and structured.
- Keep startup-directory boundary checks centralized and testable.
- Prefer simple, readable modules over layered abstractions.
- Do not add new dependencies without a clear reason.
- Do not silently code around a `TBD`.

## Module responsibilities

### `main.rs`

- Parse startup inputs.
- Determine the startup directory from the current working directory and use it as the tool root without walking to a Git repo root.
- Load config.
- Construct the agent, provider, tools, and TUI.
- Own startup and shutdown only.

### `agent.rs`

- Own the plan -> act -> observe -> repeat loop.
- Turn user messages plus current context into provider requests.
- Detect and execute tool requests automatically.
- Stream assistant and tool events to the TUI.
- Stop cleanly on user exit or fatal error.
- Keep internal planning text internal. Only assistant output, tool events, and provider-exposed thinking may stream to the TUI.

### `tui/`

- Own the Ratatui/Crossterm frontend only.
- `app.rs` owns the terminal loop, task spawning, and draw orchestration.
- `state.rs` owns composer/status/input transitions.
- `transcript.rs` owns transcript rendering state, folding, selection, and scrolling.
- `render.rs` owns shared layout and rendering helpers.
- `tests.rs` owns TUI-focused unit coverage.

### `tools/mod.rs`

- Define the common tool interface and registry.
- v1 tools are only `fs`.
- Tool results must be structured enough for both the agent and the TUI.
- Tool errors must include actionable context.

### `tools/fs.rs`

- Perform file operations inside the startup directory only.
- Normalize and validate all paths before touching the filesystem.
- v1 filesystem access supports structured UTF-8 text reads and writes.
- Allowed operations are `read_file`, `list_dir`, `write_file`, and `edit_file`. Filesystem metadata may be used internally to validate paths or infer which of the read operations to run when `op` is omitted, but there is no separate metadata-reading tool operation in v1. `write_file` may create missing parent directories inside the startup directory. Deletes and renames remain out of scope for v1.

### `llm/mod.rs`

- Define the provider interface.
- v1 only needs llama.cpp, but the boundary should stay clean enough for later extension.
- The interface must support:
  - normal turns
  - streaming output
  - the mechanism used for tool requests
  - structured provider errors

### `llm/llama_cpp.rs`

- Implement the v1 provider.
- Use only the configured llama.cpp server address.
- The implementation speaks to llama.cpp using an Ollama-compatible HTTP surface. That compatibility is why config names still use `ollama_url` even though v1 supports llama.cpp only.
- Discover the first available model from the configured llama.cpp server at startup and fail fast with a clear startup error if the server advertises no models.
- Use the server's Ollama-compatible `/api/tags` discovery endpoint at startup and `/api/chat` with streaming enabled and its tool-calling shape for tool requests.

### `context.rs`

- Hold in-memory conversation history for the active session only.
- Store tool observations needed for the current run.
- Never persist session data to disk in v1.
- Use a configurable approximate token limit to decide when pruning begins.
- When estimated context approaches that limit, summarize the oldest conversation into a compact internal summary while keeping recent raw turns.
- Retain raw tool output only for recent tool calls needed by the active conversation; summarize older tool observations before pruning them.

### `config.rs`

- Load config from:
  - CLI flags
  - environment variables
  - `~/.config/glass/config.toml`
- The v1 config values are the llama.cpp server base URL, the context pruning limit, and an optional system prompt file.
- Validate config at startup and fail fast with a clear error.
- Use the CLI flag name `--ollama-url`.
- Use the environment variable name `GLASS_OLLAMA_URL`.
- Use the TOML key `ollama_url`.
- These names are retained for compatibility with the Ollama-style HTTP API exposed by the llama.cpp server used in v1.
- Use the CLI flag name `--context-limit-tokens`.
- Use the environment variable name `GLASS_CONTEXT_LIMIT_TOKENS`.
- Use the TOML key `context_limit_tokens`.
- Use the CLI flag name `--system-prompt-file`.
- Use the environment variable name `GLASS_SYSTEM_PROMPT_FILE`.
- Use the TOML key `system_prompt_file`.
- Apply precedence in the order CLI flag > environment variable > config file.
- Require the value to be a full base URL, including scheme and port when needed.
- Treat the context limit as an approximate token budget for pruning rather than an exact tokenizer guarantee.
- Treat the system prompt file as a UTF-8 text file to load at startup and inject as the first system message in provider requests.

## Startup directory boundary rules

- All tool execution is scoped to the startup directory.
- No tool may intentionally read or write outside the startup directory.
- Boundary validation must happen before executing a tool.
- Reject any path or symlink whose resolved target escapes the startup directory.

## TUI rules

- The TUI is the only frontend in v1.
- Assistant output must stream live.
- Tool activity and tool output must stream live.
- Provider-exposed thinking may stream live when the backend emits it, but Glass internal planning must remain hidden.
- Keep TUI code isolated so headless mode can be added later without rewriting the agent loop.
- Use a single scrolling transcript pane with a bottom input composer; show tool activity, tool output, and provider-exposed thinking inline in the transcript.
- Use `Enter` to send, `Ctrl+J` to insert a newline, `Shift+Enter` as an additional newline shortcut when the terminal reports it, `PageUp`/`PageDown` to scroll the transcript, mouse-wheel scrolling over the transcript for small scroll steps, and `Ctrl+C` to quit.
- Treat slash-prefixed local commands as TUI-accessible harness controls: `/exit` quits immediately and `/new` starts a fresh in-memory session while preserving configured system settings.
- Fold long tool output and completed provider-thinking blocks inline behind a short preview and allow them to be expanded and collapsed in place on mouse click; do not add a separate pager in v1.

## Testing

- Unit tests only in v1.
- Prioritize tests for pure logic:
  - config parsing and validation
  - configured context-limit pruning behavior
  - startup-directory boundary checks
  - path normalization
  - provider request and response parsing
  - agent-loop state transitions that can be tested without the TUI

## Change discipline

- Update this file whenever scope, architecture, or module ownership changes.
- When something important is unresolved, ask instead of guessing.
