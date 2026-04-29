# inline tool arguments design

## Problem

Tool transcript entries currently show only the tool name in the header line and then stream the tool body below it. In practice, that causes tool arguments to appear as the first lines of the tool response instead of being attached to the `Tool <name>:` line where the call context belongs.

The current transcript pipeline already carries tool status and tool output as explicit entries, so the design should keep that structure and add tool arguments to the same event path instead of reconstructing them later from context.

## Goals

- Show tool-call arguments inline on the `Tool <name>:` transcript line
- Keep tool output body reserved for actual tool results
- Preserve the current folded and expanded tool-output interaction model
- Keep the transcript renderer self-contained and easy to test
- Avoid coupling transcript rendering to provider or context internals

## Non-goals

- Reformatting tool arguments into prose
- Truncating long argument payloads in this iteration
- Changing tool-output folding behavior beyond the header/body split
- Changing the provider tool-call schema

## Recommended approach

Extend the tool event and transcript entry model so tool arguments are captured once at tool start, stored directly on the tool transcript entry, and rendered inline as compact JSON on the tool header line.

This keeps the transcript state explicit, avoids hidden dependencies on conversation history, and makes the display behavior straightforward to unit test.

## Alternatives considered

### 1. Store arguments on tool transcript entries (recommended)

Add an `arguments` field to the tool-start event and the tool transcript entries, then render that field directly in the header line.

**Trade-off:** Slightly larger event and transcript structs, but the behavior is explicit and local.

### 2. Reconstruct arguments from stored context during rendering

Keep transcript entries minimal and look backward into the context history when rendering tool lines.

**Trade-off:** Reduces stored transcript data, but couples rendering to context internals and makes tests more brittle.

### 3. Store only a preformatted header string

Build the entire `Tool <name>: <args>` line in the agent and treat it as opaque transcript text.

**Trade-off:** Simplest renderer, but loses structure and makes future formatting changes harder.

## Approved design

### Data flow

`AgentEvent::ToolStarted` should carry:

- `turn_id`
- `tool_name`
- `arguments`

`arguments` is a compact JSON string derived from the tool call payload and kept exactly as displayed in the transcript.

`TranscriptEntry::ToolStatus` and `TranscriptEntry::ToolOutput` should both store that same `arguments` string. When a tool transitions from a running status entry to an output entry, the arguments string must be preserved.

### Transcript rendering

The header line becomes the canonical location for tool-call metadata.

Examples:

- running: `Tool fs: {"op":"read_file","path":"src/main.rs"} - running...`
- finished folded: `Tool fs: {"op":"read_file","path":"src/main.rs"}`
- finished expanded: same header line, followed by the tool body and the collapse hint

The body should contain only tool output. Tool arguments should no longer appear as the first lines of the tool body.

### Formatting rules

- Use compact JSON as-is for inline tool arguments
- Do not truncate the arguments string in this iteration
- Let the existing transcript line wrapping handle long header lines
- Keep folding decisions based on the tool body, not on the header line

This keeps the feature small and behaviorally predictable.

### Testing impact

Add or update unit tests to cover:

- running tool entries display inline arguments on the header line
- folded finished tool entries display inline arguments on the header line
- expanded tool entries keep inline arguments on the header line and show only tool output in the body
- transitions from tool-start to tool-output preserve the arguments string

## Notes for implementation

- The change should stay within the existing agent-event -> transcript-entry -> render flow
- No transcript rendering logic should read back into `SessionContext` to recover tool arguments
- Existing folding and selection behavior should remain unchanged aside from the new header text
