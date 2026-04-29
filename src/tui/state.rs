use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Span, Text};

use crate::agent::{AgentEvent, AgentStatus};

use super::render::{
    compose_status_line, rect_contains, transcript_inner_size, wrap_composer_text,
};
use super::transcript::{ToolOutputDirection, TranscriptState};

const PAGE_SCROLL_LINES: u16 = 10;
const MOUSE_WHEEL_SCROLL_LINES: u16 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TuiAction {
    None,
    Submit(String),
    Quit,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct TuiState {
    transcript: TranscriptState,
    composer: String,
    turn_in_flight: bool,
    status: AgentStatus,
}

impl TuiState {
    pub(crate) fn composer(&self) -> &str {
        &self.composer
    }

    pub(crate) fn is_turn_in_flight(&self) -> bool {
        self.turn_in_flight
    }

    #[cfg(test)]
    pub(crate) fn composer_mut(&mut self) -> &mut String {
        &mut self.composer
    }

    #[cfg(test)]
    pub(crate) fn transcript(&self) -> &TranscriptState {
        &self.transcript
    }

    #[cfg(test)]
    pub(crate) fn transcript_mut(&mut self) -> &mut TranscriptState {
        &mut self.transcript
    }

    pub(crate) fn update_status(&mut self, status: AgentStatus) {
        self.status = status;
    }

    pub(crate) fn reset_session(&mut self) {
        self.transcript = TranscriptState::default();
        self.composer.clear();
        self.turn_in_flight = false;
    }

    pub(crate) fn status_line(&self, width: usize) -> String {
        if width == 0 {
            return String::new();
        }

        let left = if self.status.model_name.is_empty() {
            "model: -".to_string()
        } else {
            self.status.model_name.clone()
        };
        let right = format!(
            "ctx {}/{} | last {}",
            self.status.context_tokens,
            self.status.context_limit,
            self.status
                .last_request_usage
                .as_ref()
                .map(|usage| format!(
                    "{}/{}/{}",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                ))
                .unwrap_or_else(|| "-".into())
        );

        compose_status_line(&left, &right, width)
    }

    pub(crate) fn status_line_span(&self, width: usize) -> Span<'static> {
        Span::styled(self.status_line(width), Style::default().fg(Color::Gray))
    }

    pub(crate) fn composer_visual_lines(&self, width: usize) -> usize {
        wrap_composer_text(self.composer(), width.max(1))
            .len()
            .max(1)
    }

    pub(crate) fn composer_height(&self, width: usize) -> u16 {
        (self.composer_visual_lines(width) as u16 + 2).max(3)
    }

    pub(crate) fn handle_key_event(&mut self, event: KeyEvent) -> TuiAction {
        match event.code {
            KeyCode::Char('c') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                TuiAction::Quit
            }
            KeyCode::Char('j') if event.modifiers.contains(KeyModifiers::CONTROL) => {
                self.insert_newline()
            }
            KeyCode::Enter if event.modifiers.contains(KeyModifiers::SHIFT) => {
                self.insert_newline()
            }
            KeyCode::Enter => self.handle_enter_key(),
            KeyCode::Backspace => {
                self.composer.pop();
                TuiAction::None
            }
            KeyCode::PageUp | KeyCode::PageDown | KeyCode::Up | KeyCode::Down => {
                self.handle_navigation_key(event.code)
            }
            KeyCode::Char(character)
                if !event.modifiers.contains(KeyModifiers::CONTROL)
                    && !event.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.composer.push(character);
                TuiAction::None
            }
            KeyCode::Tab => {
                self.composer.push_str("    ");
                TuiAction::None
            }
            _ => TuiAction::None,
        }
    }

    pub(crate) fn apply_agent_event(&mut self, event: AgentEvent) {
        match &event {
            AgentEvent::AssistantDone { .. } | AgentEvent::TurnError { .. } => {
                self.turn_in_flight = false;
            }
            AgentEvent::ToolStarted { .. } => {
                self.turn_in_flight = true;
            }
            AgentEvent::AssistantDelta { .. }
            | AgentEvent::ThinkingDelta { .. }
            | AgentEvent::ThinkingDone { .. }
            | AgentEvent::ToolOutputDelta { .. }
            | AgentEvent::ToolFinished { .. } => {}
        }

        self.transcript.apply_agent_event(event);
    }

    pub(crate) fn sync_transcript_selection_to_view(&mut self, transcript_area: Rect) {
        let (width, height) = transcript_inner_size(transcript_area);
        if width == 0 || height == 0 {
            return;
        }

        self.transcript
            .scroll_selected_tool_output_into_view(width, height);
        self.transcript.clamp_scroll(width, height);
    }

    pub(crate) fn handle_transcript_mouse(&mut self, mouse: MouseEvent, transcript_area: Rect) {
        if !rect_contains(transcript_area, mouse.column, mouse.row) {
            return;
        }

        let width = transcript_area.width as usize;
        if width == 0 {
            return;
        }

        match mouse.kind {
            MouseEventKind::ScrollUp => self.transcript.scroll_up_lines(MOUSE_WHEEL_SCROLL_LINES),
            MouseEventKind::ScrollDown => {
                self.transcript.scroll_down_lines(MOUSE_WHEEL_SCROLL_LINES)
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let line = self.transcript.scroll() as usize
                    + mouse.row.saturating_sub(transcript_area.y) as usize;
                self.transcript.toggle_tool_output_at_line(line, width);
            }
            _ => {}
        }
    }

    pub(crate) fn clamp_transcript_scroll(&mut self, width: usize, height: usize) {
        self.transcript.clamp_scroll(width, height);
    }

    pub(crate) fn render_transcript(&self, width: usize) -> Text<'static> {
        self.transcript.render(width)
    }

    pub(crate) fn transcript_scroll(&self) -> u16 {
        self.transcript.scroll()
    }

    fn insert_newline(&mut self) -> TuiAction {
        self.composer.push('\n');
        TuiAction::None
    }

    fn handle_enter_key(&mut self) -> TuiAction {
        if self.try_toggle_selected_tool_output() {
            return TuiAction::None;
        }

        self.submit_composer()
    }

    fn try_toggle_selected_tool_output(&mut self) -> bool {
        self.composer.trim().is_empty() && self.transcript.toggle_selected_tool_output()
    }

    fn submit_composer(&mut self) -> TuiAction {
        let trimmed = self.composer.trim();
        if trimmed.is_empty() || (self.turn_in_flight && trimmed != "/exit") {
            return TuiAction::None;
        }

        let input = std::mem::take(&mut self.composer);
        self.transcript.push_user_message(input.clone());
        self.turn_in_flight = true;
        TuiAction::Submit(input)
    }

    fn handle_navigation_key(&mut self, key: KeyCode) -> TuiAction {
        match key {
            KeyCode::PageUp => self.transcript.scroll_up_lines(PAGE_SCROLL_LINES),
            KeyCode::PageDown => self.transcript.scroll_down_lines(PAGE_SCROLL_LINES),
            KeyCode::Up => self
                .transcript
                .move_tool_output_selection(ToolOutputDirection::Up),
            KeyCode::Down => self
                .transcript
                .move_tool_output_selection(ToolOutputDirection::Down),
            _ => {}
        }
        TuiAction::None
    }
}
