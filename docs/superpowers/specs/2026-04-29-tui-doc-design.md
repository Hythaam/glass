# TUI contributor document design

## Goal

Create a contributor-focused document in `docs/` that explains how Glass's TUI works, with emphasis on runtime flow, module responsibilities, and the interfaces where the TUI interacts with the rest of the system.

## Audience

Contributors working in the codebase.

## Scope

The document describes:

- how `main.rs` hands control to the TUI
- how `tui/app.rs` runs the terminal loop, spawns turns, and redraws the screen
- how `tui/state.rs`, `tui/transcript.rs`, and `tui/render.rs` divide state, transcript behavior, and layout logic
- how the TUI consumes `AgentEvent`, `AgentStatus`, and `HarnessCommand`
- how the TUI depends on `SessionContext`, `Provider`, and `ToolExecutor` contracts indirectly through the agent
- how transcript entries represent user text, assistant streaming, provider-exposed thinking, tool execution, and errors
- how scrolling, selection, folding, wrapping, and cursor placement work
- what `src/tui/tests.rs` covers

The document does not become a full provider or tool protocol reference. Llama.cpp and `fs` are covered only where the TUI depends on their exposed behavior.

## Proposed document structure

1. Overview
2. Runtime flow from startup to shutdown
3. Main event loop in `tui/app.rs`
4. State ownership in `tui/state.rs`
5. Transcript model in `tui/transcript.rs`
6. Layout and wrapping helpers in `tui/render.rs`
7. External interaction surfaces
8. Test coverage map

## Diagrams

Include Mermaid diagrams only where they clarify flow:

- a component diagram for TUI, agent, provider, tools, and context boundaries
- a sequence diagram for one user turn, including streamed updates and tool execution

## Writing guidelines

- Keep the tone descriptive and technical.
- Prefer concrete control-flow explanations over commentary.
- Name the owning file for each responsibility.
- Explain what data crosses each boundary.
- Keep diagrams small enough to support the prose rather than replace it.

## Implementation notes

- Write the document to `docs/tui.md`.
- Use repository terminology consistently with `AGENTS.md`.
- Keep the document concise, but complete enough that a contributor can trace input handling, redraws, transcript updates, and agent interaction without opening every file first.
