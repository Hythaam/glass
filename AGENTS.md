Glass is an agentic coding agent with a focus on simplicity.

## Stack
- Ratatui/Crossterm for TUI
- Reqwest for HTTP requests
- Tokio for async execution

## Module Outline
```
coding_agent/
├── main.rs              # Entry point; wires config, runs agent loop
├── agent.rs             # Core loop: plan → act → observe → repeat
├── tools/
│   ├── mod.rs           # Tool trait + dispatch registry
│   ├── shell.rs         # Execute shell commands
│   ├── fs.rs            # Read/write files
│   └── search.rs        # Grep / symbol search
├── llm/
│   ├── mod.rs           # LLM client trait (swappable backend)
│   └── openai.rs        # OpenAI API implementation
├── context.rs           # Conversation history + token budget management
└── config.rs            # CLI args, env vars, runtime settings
```

## Configuration
Glass accepts configuration via CLI flags, environment variables, and a configuration file in `~/.config/glass/config.toml`.

The only piece of configuration currently is the URL for an ollama server to use as a basic llm provider.

