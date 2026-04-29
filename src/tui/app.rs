use std::io::{self, Stdout};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, KeyCode, MouseEvent,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::widgets::Paragraph;
use tokio::sync::{Mutex, mpsc};

use crate::agent::{Agent, AgentEvent, AgentStatus};
use crate::llm::Provider;
use crate::tools::ToolExecutor;

use super::render::{
    composer_block, composer_cursor_position, composer_inner_area, layout_chunks,
    transcript_inner_size,
};
use super::state::{TuiAction, TuiState};

const UI_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Debug)]
enum AppEvent {
    Agent(AgentEvent),
    AgentFailed(String),
    Status(AgentStatus),
}

#[derive(Debug, Clone, Copy)]
struct DrawLayout {
    transcript_area: Rect,
    composer_area: Rect,
    status_area: Rect,
    transcript_inner_width: usize,
    cursor: Option<(u16, u16)>,
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
        let mut state = TuiState::default();
        state.update_status(agent.status_snapshot());
        Self {
            agent: Arc::new(Mutex::new(agent)),
            startup_dir,
            state,
            transcript_area: Rect::default(),
        }
    }

    pub(crate) fn sync_selected_tool_output_to_view(&mut self) {
        self.state
            .sync_transcript_selection_to_view(self.transcript_area);
    }

    pub async fn run(mut self) -> Result<()> {
        let _terminal_guard = TerminalGuard::enter()?;
        let backend = CrosstermBackend::new(io::stdout());
        let mut terminal = Terminal::new(backend)?;
        let (tx, mut rx) = mpsc::unbounded_channel();

        loop {
            while let Ok(app_event) = rx.try_recv() {
                self.handle_app_event(app_event);
            }

            self.draw(&mut terminal)?;

            if event::poll(UI_POLL_INTERVAL)? {
                match event::read()? {
                    CrosstermEvent::Key(key) => {
                        let action = self.state.handle_key_event(key);
                        self.sync_selection_after_key(key.code, &action);
                        match action {
                            TuiAction::None => {}
                            TuiAction::Quit => return Ok(()),
                            TuiAction::Submit(input) => {
                                self.spawn_turn(input, tx.clone());
                            }
                        }
                    }
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
            let (result, status) = {
                let mut agent = agent.lock().await;
                let result = agent
                    .run_turn(&input, |event| {
                        let _ = tx.send(AppEvent::Agent(event));
                    })
                    .await;
                let status = agent.status_snapshot();
                (result, status)
            };

            let _ = tx.send(AppEvent::Status(status));
            if let Err(error) = result {
                let _ = tx.send(AppEvent::AgentFailed(error.to_string()));
            }
        });
    }

    pub(crate) fn handle_mouse(&mut self, mouse: MouseEvent) {
        self.state
            .handle_transcript_mouse(mouse, self.transcript_area);
    }

    fn draw(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
        terminal.draw(|frame| {
            let layout = self.compute_draw_layout(frame.size());
            self.render_draw_layout(frame, layout);
        })?;

        Ok(())
    }

    fn handle_app_event(&mut self, app_event: AppEvent) {
        match app_event {
            AppEvent::Agent(event) => self.state.apply_agent_event(event),
            AppEvent::AgentFailed(message) => {
                self.state.apply_agent_event(AgentEvent::TurnError {
                    turn_id: 0,
                    message,
                });
            }
            AppEvent::Status(status) => self.state.update_status(status),
        }
    }

    fn sync_selection_after_key(&mut self, key: KeyCode, action: &TuiAction) {
        if matches!(action, TuiAction::None)
            && matches!(key, KeyCode::Up | KeyCode::Down | KeyCode::Enter)
        {
            self.sync_selected_tool_output_to_view();
        }
    }

    fn compute_draw_layout(&mut self, size: Rect) -> DrawLayout {
        let composer_probe_width = composer_inner_area(
            &self.startup_dir,
            self.state.is_turn_in_flight(),
            Rect::new(0, 0, size.width, 3),
        )
        .width as usize;
        let composer_height = self
            .state
            .composer_height(composer_probe_width)
            .min(size.height.saturating_sub(1));
        let chunks = layout_chunks(size, composer_height);
        let transcript_area = chunks[0];
        let composer_area = chunks[1];
        let status_area = chunks[2];
        let (transcript_inner_width, transcript_inner_height) =
            transcript_inner_size(transcript_area);

        self.transcript_area = transcript_area;
        self.state.clamp_transcript_scroll(
            transcript_inner_width.max(1),
            transcript_inner_height.max(1),
        );

        let cursor = composer_cursor_position(
            self.state.composer(),
            &self.startup_dir,
            self.state.is_turn_in_flight(),
            composer_area,
        );

        DrawLayout {
            transcript_area,
            composer_area,
            status_area,
            transcript_inner_width,
            cursor,
        }
    }

    fn render_draw_layout(
        &self,
        frame: &mut ratatui::Frame<CrosstermBackend<Stdout>>,
        layout: DrawLayout,
    ) {
        let transcript =
            Paragraph::new(self.state.render_transcript(layout.transcript_inner_width))
                .scroll((self.state.transcript_scroll(), 0));
        let composer = Paragraph::new(self.state.composer()).block(composer_block(
            &self.startup_dir,
            self.state.is_turn_in_flight(),
        ));
        let status = Paragraph::new(
            self.state
                .status_line_span(layout.status_area.width as usize),
        );

        frame.render_widget(transcript, layout.transcript_area);
        frame.render_widget(composer, layout.composer_area);
        frame.render_widget(status, layout.status_area);

        if let Some((cursor_x, cursor_y)) = layout.cursor {
            frame.set_cursor(cursor_x, cursor_y);
        }
    }

    #[cfg(test)]
    pub(crate) fn state(&self) -> &TuiState {
        &self.state
    }

    #[cfg(test)]
    pub(crate) fn state_mut(&mut self) -> &mut TuiState {
        &mut self.state
    }

    #[cfg(test)]
    pub(crate) fn set_transcript_area(&mut self, area: Rect) {
        self.transcript_area = area;
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
