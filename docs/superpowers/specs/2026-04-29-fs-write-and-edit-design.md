# fs write and edit design

## Problem

`fs` is currently read-only and supports only `read_file` and `list_dir`, but the tool surface now needs to support in-place file modification inside the startup directory.

The existing implementation already centralizes path normalization and startup-directory escape checks in `tools/fs.rs`, so the design should extend that boundary instead of introducing a second tool or moving write logic elsewhere.

## Goals

- Add targeted file-edit support under the existing `fs` tool
- Add explicit full-file write support for replacement and file creation
- Preserve startup-directory boundary enforcement for all read and write operations
- Keep the tool interface explicit and structured for llama.cpp tool-calling
- Return clear errors for ambiguous or unsafe edits
- Cover the new behavior with unit tests only
- Update `AGENTS.md` so the documented v1 scope matches the expanded tool behavior

## Non-goals

- Adding any tool beyond `fs`
- Supporting binary file writes
- Supporting arbitrary patch-language parsing
- Allowing writes outside the startup directory
- Adding persistence beyond normal filesystem writes requested through the tool

## Recommended approach

Expand `fs` with two explicit write operations instead of overloading a single operation:

1. `write_file { path, contents }` for full replacement and creation
2. `edit_file { path, edits }` for targeted text edits on an existing file

This keeps overwrite/create behavior explicit, keeps targeted edits deterministic, and avoids relying on free-form patch parsing.

## Alternatives considered

### 1. Single flexible `edit_file`

Use one operation that accepts either full contents or a structured edit list.

**Trade-off:** Smaller surface area, but mixes two distinct behaviors under one name and makes validation more complex.

### 2. Separate `write_file` and `edit_file` (recommended)

Use one operation for overwrite/create and one for targeted replacements.

**Trade-off:** Slightly larger schema, but the behavior is clearer for both the model and the implementation.

### 3. Patch-string operation

Accept a diff-like patch string and apply it to a file.

**Trade-off:** Flexible, but harder to validate safely and more error-prone for model-generated input.

## Approved design

### Tool surface

`fs` will advertise four operations:

- `read_file`
- `list_dir`
- `write_file`
- `edit_file`

`read_file` and `list_dir` keep their current behavior.

`write_file` accepts:

- `path`: relative path inside the startup directory
- `contents`: replacement text to write as UTF-8

`edit_file` accepts:

- `path`: relative path inside the startup directory
- `edits`: ordered array of text replacements

Each edit entry contains:

- `old_text`: text that must appear in the current file
- `new_text`: replacement text

### File-write behavior

#### `write_file`

`write_file` is the explicit create/overwrite path.

- If the target file exists, replace its full contents with `contents`
- If the target file does not exist, create it
- If parent directories do not exist, create the missing directories inside the startup directory before writing
- The written data must be UTF-8 text

The operation should reuse centralized path normalization and boundary validation before creating directories or writing any bytes.

#### `edit_file`

`edit_file` only modifies an existing UTF-8 text file.

- Fail if the target file does not exist
- Read the full file as UTF-8 text
- Apply the provided edits in order
- Persist the final updated text back to the same file after all edits succeed

Each edit is deterministic:

- `old_text` must match exactly once in the current file contents at the time that edit is applied
- If `old_text` matches zero times, return a clear error
- If `old_text` matches more than once, return a clear error
- No fuzzy matching, regex matching, or best-effort replacement

This keeps `edit_file` predictable and prevents the tool from guessing when the requested change is ambiguous.

### Boundary and safety rules

All existing startup-directory rules continue to apply to writes:

- Paths must be relative
- Absolute paths are rejected
- `..` escapes are rejected
- Any symlink or resolved path that escapes the startup directory is rejected

Boundary validation must happen before creating directories, reading the old file, or writing new data.

The implementation should continue to keep the path-validation logic centralized in `tools/fs.rs` so read and write behavior share the same trust boundary.

### Tool parsing and schema changes

`FsToolArguments` and the JSON schema advertised through `ToolDefinition` should be expanded to cover the two new operations.

The schema should remain explicit enough for tool-calling:

- `read_file` and `list_dir` require `path`
- `write_file` requires `path` and `contents`
- `edit_file` requires `path` and `edits`

Inference when `op` is omitted should remain limited to the existing path-only shape used for read operations. Write operations should require an explicit `op` because their argument shapes are materially different.

### Error handling

Errors should stay explicit and actionable:

- Invalid UTF-8: reject reads and edits on non-text files with a clear message
- Missing file on `edit_file`: return a direct missing-file error
- Zero-match edit: identify which requested replacement did not match
- Multi-match edit: identify that the replacement was ambiguous
- Invalid path or boundary escape: include startup-directory context as existing errors do
- Directory creation or write failure: surface the filesystem error with the target path and startup-directory context

No silent fallbacks should turn a failed `edit_file` into a full overwrite.

### Testing impact

Add unit coverage for:

- successful `write_file` overwrite
- successful `write_file` create for a new file
- successful creation of missing parent directories inside the startup directory
- successful `edit_file` replacement on an existing file
- ordered multi-edit application
- `edit_file` failure on missing file
- `edit_file` failure on zero matches
- `edit_file` failure on multiple matches
- invalid UTF-8 rejection for `edit_file`
- boundary rejection for write paths and symlink escapes
- updated argument parsing and advertised tool schema

## Notes for implementation

- `AGENTS.md` must be updated to reflect that v1 `fs` is no longer read-only and now supports structured writes inside the startup directory
- The implementation should continue to keep the agent loop and TUI agnostic of filesystem internals
- `ToolResult` should remain structured enough for the TUI to show concise previews for successful writes and edits
