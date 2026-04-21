use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Span, Spans};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{Frame, Terminal};
use std::io;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use crate::harness::{DisplayUpdate, Harness};

/// A single rendered message with role prefix.
#[derive(Clone)]
struct ChatEntry {
    role: String,
    content: String,
}

/// Shared state between the TUI loop and the update drainer task.
struct SharedState {
    entries: Vec<ChatEntry>,
    /// Accumulated streamed content for the current assistant reply.
    streaming_content: String,
    scroll: usize,
    input: String,
    running: bool,
}

/// The ratatui-based chat interface.
pub struct Tui {
    harness: Harness,
    /// Channel receiver for display updates from the agent loop.
    updates_rx: mpsc::Receiver<DisplayUpdate>,
    /// Shared state for entries, scroll, input, running flag.
    state: Arc<Mutex<SharedState>>,
}

impl Tui {
    /// Create a new Tui with the given harness and display update channel.
    pub fn new(harness: Harness, updates_rx: mpsc::Receiver<DisplayUpdate>) -> Self {
        Self {
            harness,
            updates_rx,
            state: Arc::new(Mutex::new(SharedState {
                entries: Vec::new(),
                streaming_content: String::new(),
                scroll: 0,
                input: String::new(),
                running: true,
            })),
        }
    }

    /// Process all pending display updates from the channel.
    fn process_updates(&mut self) {
        let mut s = self.state.lock().unwrap();
        while let Ok(update) = self.updates_rx.try_recv() {
            match update {
                DisplayUpdate::UserMessage(text) => {
                    // New user message — finalize any in-progress stream
                    if !s.streaming_content.is_empty() {
                        let streaming = std::mem::take(&mut s.streaming_content);
                        s.entries.push(ChatEntry {
                            role: "assistant".into(),
                            content: streaming,
                        });
                    }
                    s.entries.push(ChatEntry {
                        role: "user".into(),
                        content: text,
                    });
                }
                DisplayUpdate::StreamToken(token) => {
                    // Append to the streaming buffer; create an assistant entry on first token
                    if s.streaming_content.is_empty() {
                        s.entries.push(ChatEntry {
                            role: "assistant".into(),
                            content: String::new(),
                        });
                    }
                    s.streaming_content.push_str(&token);
                    // Update the last entry with accumulated stream content
                    let snapshot = s.streaming_content.clone();
                    if let Some(last) = s.entries.last_mut() {
                        last.content = snapshot;
                    }
                }
                DisplayUpdate::AssistantMessage(text) => {
                    // Finalize streaming — update the assistant entry with the complete text
                    s.streaming_content.clear();
                    if let Some(last) = s.entries.last_mut() {
                        if last.role == "assistant" {
                            last.content = text;
                        } else {
                            s.entries.push(ChatEntry {
                                role: "assistant".into(),
                                content: text,
                            });
                        }
                    } else {
                        s.entries.push(ChatEntry {
                            role: "assistant".into(),
                            content: text,
                        });
                    }
                }
                DisplayUpdate::Error(err) => {
                    // Finalize any in-progress stream before showing error
                    if !s.streaming_content.is_empty() {
                        let streaming = std::mem::take(&mut s.streaming_content);
                        s.entries.push(ChatEntry {
                            role: "assistant".into(),
                            content: streaming,
                        });
                    }
                    s.entries.push(ChatEntry {
                        role: "error".into(),
                        content: format!("Error: {}", err),
                    });
                }
            }
            // Auto-scroll to bottom on new content
            s.scroll = s.entries.len().saturating_sub(1);
        }
    }

    /// Run the TUI event loop. Takes ownership of self.
    pub async fn run(mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        stdout.execute(EnterAlternateScreen)?;
        stdout.execute(EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        while {
            let s = self.state.lock().unwrap();
            s.running
        } {
            // Process any pending display updates
            self.process_updates();

            // Draw the UI
            terminal.draw(|frame| {
                let s = self.state.lock().unwrap();
                self.render(frame, &s);
            })?;

            // Handle input events
            if event::poll(std::time::Duration::from_millis(16))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }

                    let mut s = self.state.lock().unwrap();
                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            s.running = false;
                        }
                        KeyCode::Enter => {
                            let msg = std::mem::take(&mut s.input);
                            if !msg.is_empty() {
                                self.harness.send(msg);
                            }
                        }
                        KeyCode::Char('l') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            s.entries.clear();
                            s.scroll = 0;
                        }
                        KeyCode::Esc => {
                            s.running = false;
                        }
                        KeyCode::Char(c) => {
                            s.input.push(c);
                        }
                        KeyCode::Backspace => {
                            s.input.pop();
                        }
                        KeyCode::Up => {
                            if s.scroll > 0 {
                                s.scroll -= 1;
                            }
                        }
                        KeyCode::Down => {
                            s.scroll += 1;
                            let max_scroll = s.entries.len().saturating_sub(1);
                            if s.scroll > max_scroll {
                                s.scroll = max_scroll;
                            }
                        }
                        KeyCode::PageUp => {
                            s.scroll = s.scroll.saturating_sub(10);
                        }
                        KeyCode::PageDown => {
                            s.scroll += 10;
                            let max_scroll = s.entries.len().saturating_sub(1);
                            if s.scroll > max_scroll {
                                s.scroll = max_scroll;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Restore terminal
        disable_raw_mode()?;
        terminal.backend_mut().execute(LeaveAlternateScreen)?;
        terminal.backend_mut().execute(DisableMouseCapture)?;
        terminal.show_cursor()?;

        Ok(())
    }

    /// Render the TUI frame.
    fn render(&self, frame: &mut Frame<CrosstermBackend<io::Stdout>>, state: &SharedState) {
        let area = frame.size();
        // Always reserve the last row for input; chat fills the rest.
        let chat_area = Rect::new(0, 0, area.width, area.height.saturating_sub(3));
        let input_area = Rect::new(0, area.height.saturating_sub(3), area.width, 3);

        self.render_chat(frame, chat_area, state);
        self.render_input(frame, input_area, state);
    }

    /// Render the scrollable chat history.
    fn render_chat(
        &self,
        frame: &mut Frame<CrosstermBackend<io::Stdout>>,
        area: Rect,
        state: &SharedState,
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(
                " Chat ",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ))
            .style(Style::default().bg(Color::Black));

        // Build lines from entries, starting from scroll position
        let max_visible = area.height as usize;
        let start = state.scroll.min(state.entries.len());
        let end = (start + max_visible).min(state.entries.len());
        let visible_entries = &state.entries[start..end];

        let spans: Vec<Spans> = visible_entries
            .iter()
            .map(|entry| {
                let prefix_color = match entry.role.as_str() {
                    "user" => Color::Green,
                    "assistant" => Color::Yellow,
                    "error" => Color::Red,
                    _ => Color::White,
                };
                let prefix = format!("[{}] ", entry.role);
                Spans::from(vec![
                    Span::styled(
                        prefix,
                        Style::default()
                            .fg(prefix_color)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(&entry.content),
                ])
            })
            .collect();

        let paragraph = Paragraph::new(spans).block(block);
        frame.render_widget(paragraph, area);
    }

    /// Render the input line with cursor.
    fn render_input(
        &self,
        frame: &mut Frame<CrosstermBackend<io::Stdout>>,
        area: Rect,
        state: &SharedState,
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(
                " Input ",
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            ))
            .style(Style::default().bg(Color::Black));

        // Build the input line with a ">" prompt
        let prompt = "> ";
        let full_text = format!("{}{}", prompt, state.input);

        let paragraph = Paragraph::new(full_text).block(block);
        frame.render_widget(paragraph, area);

        // Place cursor after the prompt
        let cursor_x = area.x + 1 + (prompt.len() as u16) + (state.input.len() as u16);
        let cursor_y = area.y + 1;
        frame.set_cursor(cursor_x, cursor_y);
    }
}
