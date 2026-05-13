use std::env;

use anyhow::{Context, Result};
use clap::Parser;
use cliclack::{Input, intro, outro_cancel};

use crate::chat::ChatClient;
use crate::cli::Cli;
use crate::context::build_chat_messages;
use crate::session::{Session, SessionEvent};

pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    let cwd = env::current_dir().context("failed to resolve current working directory")?;
    let session = Session::resolve(cli.session.as_deref(), &cwd)?;
    let chat = ChatClient::from_env()?;

    if let Some(prompt) = cli.prompt {
        run_round(&session, &chat, prompt).await?;
        return Ok(());
    }

    intro("glass")?;
    loop {
        let prompt: String = Input::new("Prompt")
            .multiline()
            .required(false)
            .interact()?;

        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            outro_cancel("Session ended without a new prompt.")?;
            return Ok(());
        }

        run_round(&session, &chat, prompt).await?;
    }
}

async fn run_round(session: &Session, chat: &ChatClient, prompt: String) -> Result<()> {
    let sequence = session.next_sequence()?;
    let user_event = SessionEvent::user_message(sequence, prompt);
    session.append_event(&user_event)?;

    let events = session.load_events()?;
    let messages = build_chat_messages(&events);
    let assistant_output = chat.stream_chat_completion(messages).await?;

    let assistant_event = SessionEvent::assistant_message(sequence + 1, assistant_output);
    session.append_event(&assistant_event)?;
    Ok(())
}
