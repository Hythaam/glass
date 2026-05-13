The Glass project is a very simple agentic harness. It's primary differentiator is that it stores the agent context as a YAML file, and loads it between rounds. This allows users to manually edit context between rounds for precise control.

### Usage
When starting the CLI, the `GLASS_OPENAI_HOST` environment variable is required to set the host URL for an OpenAI-compatible chat API.
A `-s/--session <session file name>` option can be set to either load an existing session or create a new one with the given name. If the option is not set, a new session file is created with the name `glass-session-<index>.yaml`.

The `-p/--prompt <prompt string>` can be provided to immediately start a new chat round. Once the round is completed, the CLI will automatically exit.

### UX
Glass provides a basic input prompt to allow the user to enter multiline text. shift-enter can be used to add a newline, and enter is used to submit the prompt to the agent to start a chat round. The LLM response is streamed to the terminal below the submitted prompt.

### Session Files
Every time a new prompt is entered, Glass will reload the session yaml file to build the chat context with the new prompt appended to the end. For each completed response, tool call, or tool result, the session file is updated with the appropriate history.

Within the session file, each interaction is a separate document (separated by `---`). The documents include metadata, as well as a `message` parameter that contains a multiline string with the content of the interaction.

The session file contains _all_ history of the session, even components that are not sent to the chat API as context.

### Stack
The Glass project is written in Rust, with `cliclack` for input handling and `async_openai` for LLM interaction.
