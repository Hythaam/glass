mod agent;
mod config;
mod context;
mod llm;
#[cfg(test)]
mod test_support;
mod tools;
mod tui;

use anyhow::{Context, Result};
use llm::Provider;

use crate::agent::Agent;
use crate::config::Config;
use crate::context::SessionContext;
use crate::llm::llama_cpp::LlamaCppProvider;
use crate::tools::fs::FsTool;
use crate::tui::TuiApp;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let config = Config::load(std::env::args_os())?;
    let startup_dir =
        std::env::current_dir().context("failed to determine the current working directory")?;
    let mut provider = LlamaCppProvider::new(config.ollama_url.clone());
    provider.validate().await?;

    let tools = FsTool::new(startup_dir.clone())?;
    let context = SessionContext::new(config.context_limit_tokens, config.system_prompt.clone());
    let agent = Agent::new(provider, tools, context);

    TuiApp::new(agent, startup_dir).run().await
}
