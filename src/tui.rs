use std::io::{self, Stdout};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, KeyCode, KeyEvent,
    KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Span, Spans, Text};
use ratatui::widgets::{Block, Borders, Paragraph};
use tokio::sync::{Mutex, mpsc};

use crate::agent::{Agent, AgentEvent};
use crate::llm::Provider;
use crate::tools::ToolExecutor;

const PAGE_SCROLL_LINES: u16 = 10;
const UI_POLL_INTERVAL: Duration = Duration::from_millis(25);
const TOOL_FOLD_LINE_THRESHOLD: usize = 4;
const TOOL_FOLD_CHAR_THRESHOLD: usize = 160;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiAction {
    None,
    Submit(String),
    Quit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TranscriptEntry {
    User(String),
    Assistant {
        turn_id: u64,
        text: String,
        done: bool,
    },
    ToolStatus {
        turn_id: u64,
        tool_name: String,
    },
    ToolOutput {
        turn_id: u64,
        tool_name: String,
        preview: String,
        body: String,
        folded: bool,
    },
    Error {
        turn_id: u64,
        message: String,
    },
}

impl TranscriptEntry {
    fn render_segments(&self) -> Vec<(String, Style)> {
        match self {
            Self::User(text) => vec![(
                format!("You: {text}"),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )],
            Self::Assistant { text, .. } => vec![(
                format!("Assistant: {text}"),
                Style::default().fg(Color::Green),
            )],
            Self::ToolStatus { tool_name, .. } => vec![(
                format!("Tool {tool_name}: running…"),
                Style::default().fg(Color::Yellow),
            )],
            Self::ToolOutput {
                tool_name,
                preview,
                body,
                folded,
                ..
            } => {
                if *folded {
                    vec![
                        (
                            format!("Tool {tool_name}: {preview}"),
                            Style::default().fg(Color::Yellow),
                        ),
                        (
                            "(click to expand)".into(),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]
                } else {
                    vec![
                        (
                            format!("Tool {tool_name}: {preview}"),
                            Style::default().fg(Color::Yellow),
                        ),
                        (body.clone(), Style::default().fg(Color::White)),
                        (
                            "(click to collapse)".into(),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]
                }
            }
            Self::Error { message, .. } => vec![(
                format!("Error: {message}"),
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            )],
        }
    }

    #[cfg(test)]
    fn display_text(&self) -> String {
        self.render_segments()
            .into_iter()
            .map(|(text, _)| text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn render_lines(&self, width: usize) -> Vec<Spans<'static>> {
        let width = width.max(1);
        self.render_segments()
            .into_iter()
            .flat_map(|(text, style)| {
                wrap_plain_text(&text, width)
                    .into_iter()
                    .map(move |line| Spans::from(Span::styled(line, style)))
            })
            .collect()
    }

    fn line_count(&self, width: usize) -> usize {
        self.render_lines(width).len().max(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TuiState {
    transcript: Vec<TranscriptEntry>,
    composer: String,
    scroll: u16,
    turn_in_flight: bool,
}

impl TuiState {
    pub fn composer(&self) -> &str {
        &self.composer
    }

    #[cfg(test)]
    pub fn composer_mut(&mut self) -> &mut String {
        &mut self.composer
    }

    #[cfg(test)]
    pub fn scroll(&self) -> u16 {
        self.scroll
    }

    #[cfg(test)]
    pub fn set_scroll(&mut self, scroll: u16) {
        self.scroll = scroll;
    }

    #[cfg(test)]
    pub fn transcript_text(&self) -> Vec<String> {
        self.transcript
            .iter()
            .map(TranscriptEntry::display_text)
            .collect()
    }

    #[cfg(test)]
    pub fn tool_output_is_folded(&self, transcript_index: usize) -> bool {
        matches!(
            self.transcript.get(transcript_index),
            Some(TranscriptEntry::ToolOutput { folded: true, .. })
        )
    }

    pub fn handle_key_event(&mut self, event: KeyEvent) -> TuiAction {
        match (event.code, event.modifiers) {
            (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                TuiAction::Quit
            }
            (KeyCode::Enter, modifiers) if modifiers.contains(KeyModifiers::SHIFT) => {
                self.composer.push('\n');
                TuiAction::None
            }
            (KeyCode::Enter, _) => {
                if self.turn_in_flight || self.composer.trim().is_empty() {
                    return TuiAction::None;
                }
                let input = std::mem::take(&mut self.composer);
                self.transcript.push(TranscriptEntry::User(input.clone()));
                self.turn_in_flight = true;
                self.scroll_to_bottom();
                TuiAction::Submit(input)
            }
            (KeyCode::Backspace, _) => {
                self.composer.pop();
                TuiAction::None
            }
            (KeyCode::PageUp, _) => {
                self.scroll = self.scroll.saturating_sub(PAGE_SCROLL_LINES);
                TuiAction::None
            }
            (KeyCode::PageDown, _) => {
                self.scroll = self.scroll.saturating_add(PAGE_SCROLL_LINES);
                TuiAction::None
            }
            (KeyCode::Char(character), modifiers)
                if !modifiers.contains(KeyModifiers::CONTROL)
                    && !modifiers.contains(KeyModifiers::ALT) =>
            {
                self.composer.push(character);
                TuiAction::None
            }
            (KeyCode::Tab, _) => {
                self.composer.push_str("    ");
                TuiAction::None
            }
            _ => TuiAction::None,
        }
    }

    pub fn apply_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::AssistantDelta { turn_id, text } => {
                if let Some(TranscriptEntry::Assistant {
                    turn_id: active_turn_id,
                    text: active_text,
                    ..
                }) = self.transcript.last_mut()
                    && *active_turn_id == turn_id
                {
                    active_text.push_str(&text);
                } else {
                    self.transcript.push(TranscriptEntry::Assistant {
                        turn_id,
                        text,
                        done: false,
                    });
                }
                self.scroll_to_bottom();
            }
            AgentEvent::AssistantDone { turn_id } => {
                if let Some(TranscriptEntry::Assistant {
                    turn_id: active_turn_id,
                    done,
                    ..
                }) = self.transcript.last_mut()
                    && *active_turn_id == turn_id
                {
                    *done = true;
                }
                self.turn_in_flight = false;
                self.scroll_to_bottom();
            }
            AgentEvent::ToolStarted { turn_id, tool_name } => {
                self.turn_in_flight = true;
                self.transcript
                    .push(TranscriptEntry::ToolStatus { turn_id, tool_name });
                self.scroll_to_bottom();
            }
            AgentEvent::ToolOutputDelta {
                turn_id,
                tool_name,
                text,
            } => {
                if let Some(last_entry) = self.transcript.last_mut() {
                    match last_entry {
                        TranscriptEntry::ToolOutput {
                            turn_id: active_turn_id,
                            tool_name: active_tool_name,
                            body,
                            ..
                        } if *active_turn_id == turn_id && *active_tool_name == tool_name => {
                            body.push_str(&text);
                        }
                        TranscriptEntry::ToolStatus {
                            turn_id: active_turn_id,
                            tool_name: active_tool_name,
                        } if *active_turn_id == turn_id && *active_tool_name == tool_name => {
                            *last_entry = TranscriptEntry::ToolOutput {
                                turn_id,
                                tool_name: tool_name.clone(),
                                preview: format!("{tool_name} output"),
                                body: text,
                                folded: false,
                            };
                        }
                        _ => {
                            self.transcript.push(TranscriptEntry::ToolOutput {
                                turn_id,
                                tool_name: tool_name.clone(),
                                preview: format!("{tool_name} output"),
                                body: text,
                                folded: false,
                            });
                        }
                    }
                } else {
                    self.transcript.push(TranscriptEntry::ToolOutput {
                        turn_id,
                        tool_name: tool_name.clone(),
                        preview: format!("{tool_name} output"),
                        body: text,
                        folded: false,
                    });
                }
                self.scroll_to_bottom();
            }
            AgentEvent::ToolFinished {
                turn_id,
                tool_name,
                preview,
            } => {
                if let Some(last_entry) = self.transcript.last_mut() {
                    match last_entry {
                        TranscriptEntry::ToolOutput {
                            turn_id: active_turn_id,
                            tool_name: active_tool_name,
                            preview: active_preview,
                            body,
                            folded,
                        } if *active_turn_id == turn_id && *active_tool_name == tool_name => {
                            *active_preview = preview;
                            *folded = should_fold_tool_output(body);
                        }
                        TranscriptEntry::ToolStatus {
                            turn_id: active_turn_id,
                            tool_name: active_tool_name,
                        } if *active_turn_id == turn_id && *active_tool_name == tool_name => {
                            *last_entry = TranscriptEntry::ToolOutput {
                                turn_id,
                                tool_name,
                                preview,
                                folded: false,
                                body: String::new(),
                            };
                        }
                        _ => {
                            self.transcript.push(TranscriptEntry::ToolOutput {
                                turn_id,
                                tool_name,
                                preview,
                                folded: false,
                                body: String::new(),
                            });
                        }
                    }
                } else {
                    self.transcript.push(TranscriptEntry::ToolOutput {
                        turn_id,
                        tool_name,
                        preview,
                        folded: false,
                        body: String::new(),
                    });
                }
                self.scroll_to_bottom();
            }
            AgentEvent::TurnError { turn_id, message } => {
                self.transcript
                    .push(TranscriptEntry::Error { turn_id, message });
                self.turn_in_flight = false;
                self.scroll_to_bottom();
            }
        }
    }

    pub fn toggle_tool_output_at_line(&mut self, transcript_line: usize, width: usize) -> bool {
        let mut cursor = 0usize;

        for entry in &mut self.transcript {
            let lines = entry.line_count(width);
            let contains_line = (cursor..cursor + lines).contains(&transcript_line);
            if contains_line {
                if let TranscriptEntry::ToolOutput { folded, .. } = entry {
                    *folded = !*folded;
                    return true;
                }
                return false;
            }
            cursor += lines;
        }

        false
    }

    fn render_transcript(&self, width: usize) -> Text<'static> {
        let mut lines = Vec::new();
        for entry in &self.transcript {
            lines.extend(entry.render_lines(width));
        }
        Text::from(lines)
    }

    fn max_scroll(&self, width: usize, height: usize) -> u16 {
        let total_lines: usize = self.transcript.iter().map(|entry| entry.line_count(width)).sum();
        total_lines.saturating_sub(height) as u16
    }

    fn clamp_scroll(&mut self, width: usize, height: usize) {
        self.scroll = self.scroll.min(self.max_scroll(width, height));
    }

    fn scroll_to_bottom(&mut self) {
        self.scroll = u16::MAX;
    }

    fn composer_cursor(&self, width: usize) -> (u16, u16) {
        let width = width.max(1);
        let wrapped = wrap_plain_text(self.composer(), width);
        let row = wrapped.len().saturating_sub(1) as u16;
        let col = wrapped
            .last()
            .map(|line| line.chars().count() as u16)
            .unwrap_or(0)
            .min(width as u16);
        (col, row)
    }
}

#[derive(Debug)]
enum AppEvent {
    Agent(AgentEvent),
    AgentFailed(String),
}

pub struct TuiApp<P, T> {
    agent: Arc<Mutex<Agent<P, T>>>,
    startup_dir: PathBuf,
    state: TuiState,
    transcript_area: Rect,
}

impl<P, T> TuiApp<P, T>
where
    P: Provider + Send + 'static,
    T: ToolExecutor + Send + 'static,
{
    pub fn new(agent: Agent<P, T>, startup_dir: PathBuf) -> Self {
        Self {
            agent: Arc::new(Mutex::new(agent)),
            startup_dir,
            state: TuiState::default(),
            transcript_area: Rect::default(),
        }
    }

    pub async fn run(mut self) -> Result<()> {
        let _terminal_guard = TerminalGuard::enter()?;
        let backend = CrosstermBackend::new(io::stdout());
        let mut terminal = Terminal::new(backend)?;
        let (tx, mut rx) = mpsc::unbounded_channel();

        loop {
            while let Ok(app_event) = rx.try_recv() {
                match app_event {
                    AppEvent::Agent(event) => self.state.apply_agent_event(event),
                    AppEvent::AgentFailed(message) => {
                        self.state.apply_agent_event(AgentEvent::TurnError {
                            turn_id: 0,
                            message,
                        });
                    }
                }
            }

            self.draw(&mut terminal)?;

            if event::poll(UI_POLL_INTERVAL)? {
                match event::read()? {
                    CrosstermEvent::Key(key) => match self.state.handle_key_event(key) {
                        TuiAction::None => {}
                        TuiAction::Quit => return Ok(()),
                        TuiAction::Submit(input) => {
                            self.spawn_turn(input, tx.clone());
                        }
                    },
                    CrosstermEvent::Mouse(mouse) => self.handle_mouse(mouse),
                    CrosstermEvent::Resize(_, _) => {}
                    _ => {}
                }
            }
        }
    }

    fn spawn_turn(&self, input: String, tx: mpsc::UnboundedSender<AppEvent>) {
        let agent = Arc::clone(&self.agent);
        tokio::spawn(async move {
            let result = {
                let mut agent = agent.lock().await;
                agent
                    .run_turn(&input, |event| {
                        let _ = tx.send(AppEvent::Agent(event));
                    })
                    .await
            };

            if let Err(error) = result {
                let _ = tx.send(AppEvent::AgentFailed(error.to_string()));
            }
        });
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        if mouse.kind != MouseEventKind::Down(MouseButton::Left) {
            return;
        }

        if !rect_contains(self.transcript_area, mouse.column, mouse.row) {
            return;
        }

        let width = self.transcript_area.width.saturating_sub(2) as usize;
        if width == 0 {
            return;
        }

        let line = self.state.scroll as usize
            + mouse.row.saturating_sub(self.transcript_area.y + 1) as usize;
        self.state.toggle_tool_output_at_line(line, width);
    }

    fn draw(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
        terminal.draw(|frame| {
            let size = frame.size();
            let composer_lines = self.state.composer.lines().count().max(1) as u16;
            let composer_height = (composer_lines + 2).clamp(3, 8);
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(composer_height)])
                .split(size);

            self.transcript_area = chunks[0];
            let transcript_inner_width = chunks[0].width.saturating_sub(2) as usize;
            let transcript_inner_height = chunks[0].height.saturating_sub(2) as usize;
            self.state
                .clamp_scroll(transcript_inner_width.max(1), transcript_inner_height.max(1));

            let transcript = Paragraph::new(self.state.render_transcript(transcript_inner_width))
                .block(
                    Block::default()
                        .title(format!("Glass — {}", self.startup_dir.display()))
                        .borders(Borders::ALL),
                )
                .scroll((self.state.scroll, 0));

            let composer = Paragraph::new(self.state.composer.as_str()).block(
                Block::default()
                    .title("Composer")
                    .borders(Borders::ALL)
                    .border_style(if self.state.turn_in_flight {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default()
                    }),
            );

            frame.render_widget(transcript, chunks[0]);
            frame.render_widget(composer, chunks[1]);

            let composer_inner_width = chunks[1].width.saturating_sub(2) as usize;
            if composer_inner_width > 0 {
                let (cursor_x, cursor_y) = self.state.composer_cursor(composer_inner_width);
                frame.set_cursor(
                    chunks[1].x + 1 + cursor_x.min(chunks[1].width.saturating_sub(3)),
                    chunks[1].y + 1 + cursor_y.min(chunks[1].height.saturating_sub(3)),
                );
            }
        })?;

        Ok(())
    }
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
    }
}

fn rect_contains(rect: Rect, column: u16, row: u16) -> bool {
    column >= rect.x
        && column < rect.x + rect.width
        && row >= rect.y
        && row < rect.y + rect.height
}

fn should_fold_tool_output(body: &str) -> bool {
    let line_count = body.lines().count().max(1);
    line_count >= TOOL_FOLD_LINE_THRESHOLD || body.chars().count() >= TOOL_FOLD_CHAR_THRESHOLD
}

fn wrap_plain_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut wrapped = Vec::new();

    for raw_line in text.split('\n') {
        if raw_line.is_empty() {
            wrapped.push(String::new());
            continue;
        }

        let mut current = String::new();
        let mut count = 0usize;

        for character in raw_line.chars() {
            if count == width {
                wrapped.push(std::mem::take(&mut current));
                count = 0;
            }
            current.push(character);
            count += 1;
        }

        if current.is_empty() {
            wrapped.push(String::new());
        } else {
            wrapped.push(current);
        }
    }

    if wrapped.is_empty() {
        wrapped.push(String::new());
    }

    wrapped
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    use crate::agent::AgentEvent;

    use super::{TuiAction, TuiState};

    #[test]
    fn enter_submits_nonempty_input_and_clears_composer() {
        let mut state = TuiState::default();
        state.composer_mut().push_str("hello glass");

        let action = state.handle_key_event(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(action, TuiAction::Submit("hello glass".into()));
        assert_eq!(state.composer(), "");
    }

    #[test]
    fn shift_enter_inserts_newline_without_submitting() {
        let mut state = TuiState::default();
        state.composer_mut().push_str("hello");

        let action =
            state.handle_key_event(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));

        assert_eq!(action, TuiAction::None);
        assert_eq!(state.composer(), "hello\n");
    }

    #[test]
    fn page_keys_adjust_transcript_scroll() {
        let mut state = TuiState::default();
        state.set_scroll(20);

        let page_up = state.handle_key_event(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
        assert_eq!(page_up, TuiAction::None);
        assert_eq!(state.scroll(), 10);

        let page_down =
            state.handle_key_event(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));

        assert_eq!(page_down, TuiAction::None);
        assert_eq!(state.scroll(), 20);
    }

    #[test]
    fn ctrl_c_requests_quit() {
        let mut state = TuiState::default();

        let action = state.handle_key_event(KeyEvent::new(
            KeyCode::Char('c'),
            KeyModifiers::CONTROL,
        ));

        assert_eq!(action, TuiAction::Quit);
    }

    #[test]
    fn enter_ignores_whitespace_only_input() {
        let mut state = TuiState::default();
        state.composer_mut().push_str("   \n");

        let action = state.handle_key_event(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(action, TuiAction::None);
        assert_eq!(state.composer(), "   \n");
    }

    #[test]
    fn enter_does_not_submit_while_turn_is_in_flight() {
        let mut state = TuiState::default();
        state.composer_mut().push_str("hello again");
        state.apply_agent_event(AgentEvent::ToolStarted {
            turn_id: 9,
            tool_name: "fs".into(),
        });

        let action = state.handle_key_event(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(action, TuiAction::None);
        assert_eq!(state.composer(), "hello again");
    }

    #[test]
    fn assistant_deltas_append_to_active_turn_entry() {
        let mut state = TuiState::default();

        state.apply_agent_event(AgentEvent::AssistantDelta {
            turn_id: 7,
            text: "hello".into(),
        });
        state.apply_agent_event(AgentEvent::AssistantDelta {
            turn_id: 7,
            text: " world".into(),
        });
        state.apply_agent_event(AgentEvent::AssistantDone { turn_id: 7 });

        assert_eq!(state.transcript_text(), vec!["Assistant: hello world"]);
    }

    #[test]
    fn long_tool_output_starts_folded_and_toggles_from_preview_line() {
        let mut state = TuiState::default();

        state.apply_agent_event(AgentEvent::ToolStarted {
            turn_id: 3,
            tool_name: "fs".into(),
        });
        state.apply_agent_event(AgentEvent::ToolOutputDelta {
            turn_id: 3,
            tool_name: "fs".into(),
            text: "agent.rs\ncontext.rs\nllm.rs\nmain.rs\ntools.rs\ntui.rs".into(),
        });
        state.apply_agent_event(AgentEvent::ToolFinished {
            turn_id: 3,
            tool_name: "fs".into(),
            preview: "src entries".into(),
        });

        assert!(state.transcript_text()[0].contains("src entries"));
        assert!(state.tool_output_is_folded(0));

        assert!(state.toggle_tool_output_at_line(0, 40));
        assert!(!state.tool_output_is_folded(0));
    }

    #[test]
    fn finished_tool_output_replaces_running_status() {
        let mut state = TuiState::default();

        state.apply_agent_event(AgentEvent::ToolStarted {
            turn_id: 11,
            tool_name: "fs".into(),
        });
        state.apply_agent_event(AgentEvent::ToolOutputDelta {
            turn_id: 11,
            tool_name: "fs".into(),
            text: "main.rs\n".into(),
        });
        state.apply_agent_event(AgentEvent::ToolFinished {
            turn_id: 11,
            tool_name: "fs".into(),
            preview: "1 entry".into(),
        });

        let transcript = state.transcript_text();
        assert_eq!(transcript.len(), 1);
        assert!(transcript[0].contains("Tool fs: 1 entry"));
        assert!(transcript[0].contains("main.rs"));
        assert!(!transcript[0].contains("running"));
    }

    #[test]
    fn turn_errors_are_rendered_inline() {
        let mut state = TuiState::default();

        state.apply_agent_event(AgentEvent::TurnError {
            turn_id: 5,
            message: "provider down".into(),
        });

        assert_eq!(state.transcript_text(), vec!["Error: provider down"]);
    }

    #[test]
    fn transcript_auto_scrolls_to_latest_agent_output() {
        let mut state = TuiState::default();

        for turn_id in 0..6 {
            state.apply_agent_event(AgentEvent::TurnError {
                turn_id,
                message: format!("error {turn_id}"),
            });
        }

        state.clamp_scroll(40, 2);

        assert!(state.scroll() > 0);
    }
}
