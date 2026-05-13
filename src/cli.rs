use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "glass")]
#[command(about = "A simple YAML-backed agentic CLI harness.")]
pub struct Cli {
    #[arg(short, long)]
    pub session: Option<String>,

    #[arg(short, long)]
    pub prompt: Option<String>,
}
