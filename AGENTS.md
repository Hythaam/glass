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
- LLM provider: Ollama only
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
- Additional config beyond the Ollama server address
- Test requirements beyond unit tests

## Crate layout (v1)

```text
coding_agent/
├── main.rs              # Startup, config loading, startup-directory resolution, TUI bootstrap
├── agent.rs             # Core loop: plan -> act -> observe -> repeat
├── tools/
│   ├── mod.rs           # Tool trait + dispatch registry
│   ├── shell.rs         # Run shell commands inside the startup directory
│   └── fs.rs            # File operations inside the startup directory
├── llm/
│   ├── mod.rs           # Provider interface
│   └── ollama.rs        # Ollama implementation for v1
├── context.rs           # In-memory session history and pruning
└── config.rs            # CLI/env/config-file loading for Ollama server address
```

Remove the dedicated search tool from v1. Directory inspection should happen through `shell` and `fs`.

If the current crate still contains `tools/search.rs`, remove it from the v1 path.  
If the current crate still contains `llm/openai.rs`, rename or replace it so the crate reflects the actual Ollama-only v1 scope.

## Runtime model

1. Start from the current working directory. The startup directory is the tool root even when it is a subdirectory of a Git repo or not a Git repo at all.
2. Load the single v1 config value: the Ollama server address.
3. Start one in-memory chat session.
4. Accept user input in the TUI.
5. Send session context to the Ollama backend.
6. Stream assistant output to the TUI.
7. When the assistant requests a tool, execute it automatically.
8. Stream tool status and tool output to the TUI.
9. Feed tool observations back into the loop.
10. End the session when the user quits. Do not persist history between runs.

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
- Keep internal planning text internal; only assistant output and tool events stream to the TUI.

### `tools/mod.rs`

- Define the common tool interface and registry.
- v1 tools are only `shell` and `fs`.
- Tool results must be structured enough for both the agent and the TUI.
- Tool errors must include actionable context.

### `tools/shell.rs`

- Execute commands from the startup directory only.
- Validate startup-directory-relative paths before execution where applicable.
- Capture exit status, stdout, and stderr.
- Stream command output when possible.
- v1 shell access is read-only; commands that would mutate files or version-control state are rejected before execution.
- Invoke `/bin/sh -lc` on both Linux and macOS.
- Inherit the parent process environment unchanged.
- Use a default timeout of 60 seconds and allow user-triggered cancellation.

### `tools/fs.rs`

- Perform file operations inside the startup directory only.
- Normalize and validate all paths before touching the filesystem.
- v1 filesystem access is read-only.
- Allowed operations are reading file contents, listing directories, and reading file metadata. Writes, deletes, renames, and directory creation are out of scope for v1.

### `llm/mod.rs`

- Define the provider interface.
- v1 only needs Ollama, but the boundary should stay clean enough for later extension.
- The interface must support:
  - normal turns
  - streaming output
  - the mechanism used for tool requests
  - structured provider errors

### `llm/ollama.rs`

- Implement the v1 provider.
- Use only the configured Ollama server address.
- Use the built-in default model name `llama3.1:8b` and fail fast with a clear startup error if that model is unavailable.
- Use Ollama's `/api/chat` endpoint with streaming enabled and the endpoint's tool-calling shape for tool requests.

### `context.rs`

- Hold in-memory conversation history for the active session only.
- Store tool observations needed for the current run.
- Never persist session data to disk in v1.
- When context approaches the model limit, summarize the oldest conversation into a compact internal summary while keeping recent raw turns.
- Retain raw tool output only for recent tool calls needed by the active conversation; summarize older tool observations before pruning them.

### `config.rs`

- Load config from:
  - CLI flags
  - environment variables
  - `~/.config/glass/config.toml`
- The only v1 config value is the Ollama server address.
- Validate config at startup and fail fast with a clear error.
- Use the CLI flag name `--ollama-url`.
- Use the environment variable name `GLASS_OLLAMA_URL`.
- Use the TOML key `ollama_url`.
- Apply precedence in the order CLI flag > environment variable > config file.
- Require the value to be a full base URL, including scheme and port when needed.

## Startup directory boundary rules

- All tool execution is scoped to the startup directory.
- No tool may intentionally read or write outside the startup directory.
- Boundary validation must happen before executing a tool.
- Reject any path or symlink whose resolved target escapes the startup directory.

## TUI rules

- The TUI is the only frontend in v1.
- Assistant output must stream live.
- Tool activity and tool output must stream live.
- Keep TUI code isolated so headless mode can be added later without rewriting the agent loop.
- Use a single scrolling transcript pane with a bottom input composer; show tool activity and tool output inline in the transcript.
- Use `Enter` to send, `Shift+Enter` to insert a newline, `PageUp`/`PageDown` to scroll the transcript, and `Ctrl+C` to quit.
- Fold long tool output inline behind a short preview and allow it to be expanded in place; do not add a separate pager in v1.

## Testing

- Unit tests only in v1.
- Prioritize tests for pure logic:
  - config parsing and validation
  - startup-directory boundary checks
  - path normalization
  - provider request and response parsing
  - agent-loop state transitions that can be tested without the TUI

## Change discipline

- Update this file whenever scope, architecture, or module ownership changes.
- When something important is unresolved, ask instead of guessing.
