# AGENTS.md v1 scope alignment design

## Problem

`AGENTS.md` still describes a v1 surface that includes a `shell` tool and leaves two implementation details under-specified:

- how context pruning is bounded
- how folded transcript sections are expanded in the TUI

This creates avoidable ambiguity before implementation.

## Goals

- Remove `shell` from the v1 tool surface and all related architecture/runtime references
- Add a configurable approximate context budget for pruning
- Define folded transcript interaction as mouse-click expansion/collapse
- Keep the core agent loop UI-agnostic while making the TUI behavior explicit

## Non-goals

- Adding any tool beyond read-only filesystem access
- Changing the Ollama-only provider direction
- Introducing persistent session history
- Adding headless-mode behavior to v1

## Recommended approach

Update `AGENTS.md` so the v1 crate and runtime model are explicitly filesystem-only. Replace shell-related requirements with `fs`-only behavior, add a second config value for context pruning, and define mouse-click interaction for folded transcript sections in the TUI.

This is the smallest spec change that resolves the current blockers without broadening v1.

## Alternatives considered

### 1. Minimal text patch

Delete the most visible shell references and append short notes for pruning and folding interaction.

**Trade-off:** Fastest, but leaves inconsistent module responsibilities and runtime steps.

### 2. Full section rewrite

Rewrite the affected areas of `AGENTS.md` end to end.

**Trade-off:** Clear, but more churn than needed for the current scope correction.

### 3. Targeted alignment update (recommended)

Edit the crate layout, runtime model, module responsibilities, config section, and TUI rules in place so they consistently describe the new v1 surface.

**Trade-off:** Slightly more editing than a minimal patch, but keeps the document internally consistent.

## Approved design

### Tool surface and crate layout

v1 tools are reduced to **filesystem-only** access.

- Remove `shell` from the crate layout
- Remove `tools/shell.rs` from the v1 path
- Keep `tools/mod.rs` and `tools/fs.rs`
- `tools/mod.rs` defines the registry and shared structured result types for filesystem operations only
- `tools/fs.rs` remains read-only and supports:
  - reading file contents
  - listing directories
  - reading file metadata

### Runtime model

The runtime model stays the same except for tool execution details.

- Start from the current working directory and use it as the startup directory
- Load v1 config
- Start one in-memory chat session
- Accept user input in the TUI
- Send session context to Ollama
- Stream assistant output to the TUI
- When the assistant requests a tool, execute the read-only filesystem tool automatically
- Stream tool status and tool output to the TUI
- Feed tool observations back into the loop
- End the session when the user quits without persisting history

### Module responsibilities

#### `tools/mod.rs`

- Define the common tool interface and registry
- v1 tools are only `fs`
- Tool results must stay structured enough for both the agent and the TUI
- Tool errors must include actionable context

#### `tools/fs.rs`

- Perform file operations inside the startup directory only
- Normalize and validate all paths before touching the filesystem
- Keep v1 filesystem access read-only
- Reject any path or symlink whose resolved target escapes the startup directory
- Support reading file contents, listing directories, and reading file metadata only

#### `context.rs`

- Hold in-memory conversation history for the active session only
- Store tool observations needed for the current run
- Never persist session data to disk in v1
- Use a configurable approximate token budget to decide when pruning/summarization begins
- Summarize older conversation and older tool observations first while preserving recent raw turns

#### `config.rs`

v1 config contains exactly two values:

- `ollama_url`
- `context_limit_tokens`

Config loading and precedence remain:

- CLI flags
- environment variables
- `~/.config/glass/config.toml`

Key names:

- CLI: `--ollama-url`
- env: `GLASS_OLLAMA_URL`
- TOML: `ollama_url`
- CLI: `--context-limit-tokens`
- env: `GLASS_CONTEXT_LIMIT_TOKENS`
- TOML: `context_limit_tokens`

Precedence stays **CLI > environment > config file**.

`context_limit_tokens` is an approximate budget for context sent to the model. The implementation may estimate token usage rather than using an exact tokenizer, but pruning must be driven by this configured limit.

### TUI behavior

The TUI remains the only frontend in v1 and keeps a single scrolling transcript pane with a bottom input composer.

- Assistant output streams live
- Tool activity and tool output stream live
- Long tool output is shown inline behind a short preview
- Folded transcript sections expand and collapse **in place on mouse click**
- Existing keyboard controls remain for send/newline/scroll/quit
- Fold state and click-hit testing stay in the TUI layer rather than the agent loop

### Testing impact

Unit-test priorities expand to include:

- config parsing and validation for `context_limit_tokens`
- context pruning behavior against the configured approximate limit
- startup-directory boundary checks and path normalization for filesystem access
- provider request/response parsing
- agent-loop state transitions testable without the TUI
- TUI-local fold state and click-hit testing logic where practical as pure units

## Notes for implementation

- `AGENTS.md` should be updated to match this design before broader implementation proceeds
- Existing `src/tools/search.rs` should remain out of the v1 path, consistent with the current spec direction
- Existing `src/tools/shell.rs` should be removed from the v1 path or retired from active use when implementation begins
