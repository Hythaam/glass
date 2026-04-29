# provider thinking transcript design

## Problem

The transcript currently renders only assistant output and tool events. If the configured llama.cpp server exposes provider-visible thinking text, Glass has no dedicated way to stream or display it in the TUI.

At the same time, `AGENTS.md` explicitly keeps Glass's internal planning private, so the design must distinguish provider-exposed thinking from the agent's hidden internal reasoning.

## Goals

- Display provider-exposed thinking in the transcript when the backend emits it
- Keep provider thinking distinct from assistant output and tool activity
- Render thinking as an expandable transcript block
- Preserve the privacy boundary around Glass internal planning
- Keep no-thinking turns unchanged
- Cover the new behavior with unit tests only

## Non-goals

- Displaying Glass internal planning or hidden reasoning
- Generating synthetic thinking locally when the provider does not emit it
- Changing normal assistant-message rendering for non-thinking content
- Adding persistence beyond the current in-memory session

## Recommended approach

Add first-class thinking stream items across the provider, agent, and transcript layers.

This keeps thinking separate from assistant output, makes the transcript model explicit, and fits the existing typed-entry architecture used for assistant, tool, and error transcript entries.

## Alternatives considered

### 1. First-class thinking stream items (recommended)

Add explicit thinking delta and completion events to the stream and transcript model.

**Trade-off:** Adds a small amount of new event plumbing, but keeps the behavior clear and typed.

### 2. Reuse assistant events with a thinking flag

Treat thinking as assistant text with an extra mode flag.

**Trade-off:** Smaller API surface, but mixes two distinct transcript concepts and complicates rendering logic.

### 3. Post-process complete thinking only

Render a completed thinking block only after a turn ends, without streaming deltas live.

**Trade-off:** Simpler transcript updates, but loses live feedback and diverges from the current streaming model.

## Approved design

### Data flow

`ProviderStreamItem` should gain explicit provider-thinking variants, such as:

- thinking delta
- thinking complete

`AgentEvent` should mirror those stream items so the TUI receives typed thinking updates just as it already receives assistant and tool events.

`TranscriptEntry` should gain a dedicated thinking entry type rather than overloading assistant entries.

This preserves the distinction between:

- provider-exposed thinking that the backend intentionally emits
- normal assistant output
- Glass internal planning, which remains private

### Transcript rendering

Thinking appears as its own transcript block.

Behavior:

- While the provider is streaming thinking, the transcript shows an in-progress thinking entry
- Once thinking is complete, the entry collapses by default
- The collapsed entry shows a short preview line and an expand/collapse hint
- Expanding the entry reveals the full provider-exposed thinking text inline in the transcript
- Any subsequent assistant response renders as a separate assistant entry below the thinking block

If a turn contains no provider-exposed thinking, no thinking transcript entry is created.

### Parsing and fallback behavior

The llama.cpp/Ollama-compatible provider implementation should only emit thinking stream items when the server response includes a distinct field for provider-visible thinking.

If the configured backend does not emit such a field:

- Glass behaves exactly as it does today
- no empty thinking block is created
- no local or synthetic thinking text is generated

This keeps the feature backend-driven and avoids inventing hidden reasoning inside Glass.

### Folding and interaction

Thinking blocks should follow the existing transcript interaction model used for foldable tool output:

- collapsed by default after completion
- expandable and collapsible in place
- keyboard and mouse interaction should work through the existing transcript selection/toggle mechanisms where practical

The thinking block remains a distinct transcript entry type even if it reuses shared folding behavior internally.

### Testing impact

Add unit coverage for:

- provider parsing of thinking chunks when present
- no-thinking provider responses leaving behavior unchanged
- agent forwarding of thinking deltas and completion events
- transcript rendering of collapsed and expanded thinking entries
- transcript fold interaction for completed thinking blocks

## Notes for implementation

- `AGENTS.md` should be updated to clarify that provider-exposed thinking may be shown in the transcript while Glass internal planning remains private
- The implementation should keep the agent loop UI-agnostic by emitting typed thinking events rather than Ratatui-specific structures
- If the provider response format for thinking is absent or ambiguous, Glass should ignore it rather than guess
