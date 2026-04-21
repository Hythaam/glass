use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;

use crate::api::{ApiClient, ChatMessage};
use crate::session::Session;

/// Commands sent from the TUI to the agent loop.
pub enum AgentCommand {
    /// User sent a text message to the LLM.
    Send(String),
    // Future: User typed a backslash-prefixed tool command.
    // ToolCommand(String),
    // Future: LLM requested a tool call.
    // ToolCall(String, Vec<serde_json::Value>),
}

/// Display updates sent from the agent loop to the TUI.
pub enum DisplayUpdate {
    /// A user message was recorded and should be displayed.
    UserMessage(String),
    /// A streaming token arrived from the LLM.
    StreamToken(String),
    /// The assistant's response is complete.
    AssistantMessage(String),
    /// An error occurred.
    Error(String),
}

/// The agent harness that orchestrates interactions between the TUI, LLM, and session.
pub struct Harness {
    /// Sender for commands from TUI to the agent loop.
    command_tx: mpsc::Sender<AgentCommand>,
}

impl Harness {
    /// Create a new harness with the given model name.
    /// Returns (Harness, display_rx) — the receiver is moved out for the TUI.
    pub async fn new(model: &str) -> (Self, mpsc::Receiver<DisplayUpdate>) {
        let (command_tx, command_rx) = mpsc::channel(32);
        let (display_tx, display_rx) = mpsc::channel(256);

        let api = ApiClient::new(model);
        let session = Arc::new(RwLock::new(Session::new(Some(
            "You are a helpful coding assistant.",
        ))));

        // Spawn the agent loop
        tokio::spawn(agent_loop(command_rx, display_tx, api, session));

        (Self { command_tx }, display_rx)
    }

    /// Send a user message to the agent loop.
    pub fn send(&self, msg: String) {
        // Non-blocking send; silently drop if channel is full.
        let _ = self.command_tx.try_send(AgentCommand::Send(msg));
    }
}

/// The main agent event loop. Runs in a background tokio task.
async fn agent_loop(
    mut command_rx: mpsc::Receiver<AgentCommand>,
    display_tx: mpsc::Sender<DisplayUpdate>,
    api: ApiClient,
    session: Arc<RwLock<Session>>,
) {
    while let Some(cmd) = command_rx.recv().await {
        match cmd {
            AgentCommand::Send(text) => {
                // 1. Record user message in session
                {
                    let mut sess = session.write().await;
                    sess.add_user_message(&text);
                }
                let _ = display_tx
                    .send(DisplayUpdate::UserMessage(text))
                    .await;

                // 2. Get full message history
                let messages: Vec<ChatMessage> = {
                    let sess = session.read().await;
                    sess.get_messages().to_vec()
                };

                // 3. Stream response from LLM
                let mut full_response = String::new();

                match api.send_stream_channel(&messages).await {
                    Ok(mut rx) => {
                        while let Some(token_result) = rx.recv().await {
                            match token_result {
                                Ok(token) => {
                                    if !token.is_empty() {
                                        full_response.push_str(&token);
                                        let _ = display_tx
                                            .send(DisplayUpdate::StreamToken(token))
                                            .await;
                                    }
                                }
                                Err(e) => {
                                    let _ = display_tx
                                        .send(DisplayUpdate::Error(e.to_string()))
                                        .await;
                                    break;
                                }
                            }
                        }
                        // 4. Record assistant response in session
                        if !full_response.is_empty() {
                            let mut sess = session.write().await;
                            sess.add_assistant_message(&full_response);
                            let _ = display_tx
                                .send(DisplayUpdate::AssistantMessage(
                                    full_response,
                                ))
                                .await;
                        }
                    }
                    Err(e) => {
                        let _ = display_tx
                            .send(DisplayUpdate::Error(e.to_string()))
                            .await;
                    }
                }
            }
        }
    }
}
