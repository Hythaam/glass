use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Span, Spans, Text};

use crate::agent::AgentEvent;

use super::render::{should_fold_tool_output, wrap_plain_text};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolOutputDirection {
    Up,
    Down,
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
    fn render_segments(&self, selected: bool) -> Vec<(String, Style)> {
        let with_selection = |style: Style| {
            if selected {
                style.bg(Color::DarkGray)
            } else {
                style
            }
        };

        match self {
            Self::User(text) => vec![(
                format!("You: {text}"),
                with_selection(
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
            )],
            Self::Assistant { text, .. } => vec![(
                format!("Assistant: {text}"),
                with_selection(Style::default().fg(Color::Green)),
            )],
            Self::ToolStatus { tool_name, .. } => vec![(
                format!("Tool {tool_name}: running…"),
                with_selection(Style::default().fg(Color::Yellow)),
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
                            with_selection(Style::default().fg(Color::Yellow)),
                        ),
                        (
                            "(Enter or click to expand)".into(),
                            with_selection(Style::default().fg(Color::DarkGray)),
                        ),
                    ]
                } else {
                    vec![
                        (
                            format!("Tool {tool_name}: {preview}"),
                            with_selection(Style::default().fg(Color::Yellow)),
                        ),
                        (
                            body.clone(),
                            with_selection(Style::default().fg(Color::White)),
                        ),
                        (
                            "(Enter or click to collapse)".into(),
                            with_selection(Style::default().fg(Color::DarkGray)),
                        ),
                    ]
                }
            }
            Self::Error { message, .. } => vec![(
                format!("Error: {message}"),
                with_selection(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            )],
        }
    }

    #[cfg(test)]
    fn display_text(&self) -> String {
        self.render_segments(false)
            .into_iter()
            .map(|(text, _)| text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn render_lines(&self, width: usize, selected: bool) -> Vec<Spans<'static>> {
        let width = width.max(1);
        self.render_segments(selected)
            .into_iter()
            .flat_map(|(text, style)| {
                wrap_plain_text(&text, width)
                    .into_iter()
                    .map(move |line| Spans::from(Span::styled(line, style)))
            })
            .collect()
    }

    fn line_count(&self, width: usize) -> usize {
        self.render_lines(width, false).len().max(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct TranscriptState {
    entries: Vec<TranscriptEntry>,
    scroll: u16,
    selected_tool_output: Option<usize>,
}

impl TranscriptState {
    pub(crate) fn apply_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::AssistantDelta { turn_id, text } => {
                self.append_assistant_delta(turn_id, text)
            }
            AgentEvent::AssistantDone { turn_id } => self.finish_assistant_turn(turn_id),
            AgentEvent::ToolStarted { turn_id, tool_name } => self.start_tool(turn_id, tool_name),
            AgentEvent::ToolOutputDelta {
                turn_id,
                tool_name,
                text,
            } => self.append_tool_output(turn_id, tool_name, text),
            AgentEvent::ToolFinished {
                turn_id,
                tool_name,
                preview,
            } => self.finish_tool_output(turn_id, tool_name, preview),
            AgentEvent::TurnError { turn_id, message } => self.push_turn_error(turn_id, message),
        }

        self.scroll_to_bottom();
    }

    pub(crate) fn push_user_message(&mut self, input: String) {
        self.entries.push(TranscriptEntry::User(input));
        self.scroll_to_bottom();
    }

    pub(crate) fn toggle_tool_output_at_line(
        &mut self,
        transcript_line: usize,
        width: usize,
    ) -> bool {
        let mut cursor = 0usize;

        for index in 0..self.entries.len() {
            let lines = self.entries[index].line_count(width);
            let contains_line = (cursor..cursor + lines).contains(&transcript_line);
            if contains_line {
                return self.toggle_tool_output(index);
            }
            cursor += lines;
        }

        false
    }

    pub(crate) fn render(&self, width: usize) -> Text<'static> {
        let mut lines = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            let selected = self.selected_tool_output == Some(index);
            lines.extend(entry.render_lines(width, selected));
        }
        Text::from(lines)
    }

    fn max_scroll(&self, width: usize, height: usize) -> u16 {
        let total_lines: usize = self
            .entries
            .iter()
            .map(|entry| entry.line_count(width))
            .sum();
        total_lines.saturating_sub(height) as u16
    }

    pub(crate) fn clamp_scroll(&mut self, width: usize, height: usize) {
        self.scroll = self.scroll.min(self.max_scroll(width, height));
    }

    fn scroll_to_bottom(&mut self) {
        self.scroll = u16::MAX;
    }

    pub(crate) fn scroll_up_lines(&mut self, lines: u16) {
        self.scroll = self.scroll.saturating_sub(lines);
    }

    pub(crate) fn scroll_down_lines(&mut self, lines: u16) {
        self.scroll = self.scroll.saturating_add(lines);
    }

    pub(crate) fn scroll(&self) -> u16 {
        self.scroll
    }

    #[cfg(test)]
    pub(crate) fn set_scroll(&mut self, scroll: u16) {
        self.scroll = scroll;
    }

    pub(crate) fn move_tool_output_selection(&mut self, direction: ToolOutputDirection) {
        self.selected_tool_output =
            self.find_adjacent_tool_output(self.selected_tool_output, direction);
    }

    pub(crate) fn scroll_selected_tool_output_into_view(&mut self, width: usize, height: usize) {
        let Some(selected_index) = self.selected_tool_output else {
            return;
        };

        let width = width.max(1);
        let height = height.max(1);
        let mut line_start = 0usize;

        for (index, entry) in self.entries.iter().enumerate() {
            let line_count = entry.line_count(width);
            let line_end = line_start + line_count;
            if index == selected_index {
                let viewport_start = self.scroll as usize;
                let viewport_end = viewport_start + height;

                if line_start < viewport_start {
                    self.scroll = line_start as u16;
                } else if line_end > viewport_end {
                    self.scroll = line_end.saturating_sub(height) as u16;
                }
                return;
            }
            line_start = line_end;
        }
    }

    pub(crate) fn toggle_selected_tool_output(&mut self) -> bool {
        let Some(index) = self.selected_tool_output else {
            return false;
        };

        self.toggle_tool_output(index)
    }

    #[cfg(test)]
    pub(crate) fn text(&self) -> Vec<String> {
        self.entries
            .iter()
            .map(TranscriptEntry::display_text)
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn tool_output_is_folded(&self, transcript_index: usize) -> bool {
        matches!(
            self.entries.get(transcript_index),
            Some(TranscriptEntry::ToolOutput { folded: true, .. })
        )
    }

    #[cfg(test)]
    pub(crate) fn selected_tool_output(&self) -> Option<usize> {
        self.selected_tool_output
    }

    fn append_assistant_delta(&mut self, turn_id: u64, text: String) {
        if let Some(TranscriptEntry::Assistant {
            turn_id: active_turn_id,
            text: active_text,
            ..
        }) = self.entries.last_mut()
            && *active_turn_id == turn_id
        {
            active_text.push_str(&text);
            return;
        }

        self.entries.push(TranscriptEntry::Assistant {
            turn_id,
            text,
            done: false,
        });
    }

    fn finish_assistant_turn(&mut self, turn_id: u64) {
        if let Some(TranscriptEntry::Assistant {
            turn_id: active_turn_id,
            done,
            ..
        }) = self.entries.last_mut()
            && *active_turn_id == turn_id
        {
            *done = true;
        }
    }

    pub(crate) fn start_tool(&mut self, turn_id: u64, tool_name: String) {
        self.entries
            .push(TranscriptEntry::ToolStatus { turn_id, tool_name });
    }

    pub(crate) fn append_tool_output(&mut self, turn_id: u64, tool_name: String, text: String) {
        if let Some(last_entry) = self.entries.last_mut() {
            match last_entry {
                TranscriptEntry::ToolOutput {
                    turn_id: active_turn_id,
                    tool_name: active_tool_name,
                    body,
                    ..
                } if *active_turn_id == turn_id && *active_tool_name == tool_name => {
                    body.push_str(&text);
                    return;
                }
                TranscriptEntry::ToolStatus {
                    turn_id: active_turn_id,
                    tool_name: active_tool_name,
                } if *active_turn_id == turn_id && *active_tool_name == tool_name => {
                    *last_entry = Self::tool_output_entry(
                        turn_id,
                        tool_name.clone(),
                        format!("{tool_name} output"),
                        text,
                    );
                    return;
                }
                _ => {}
            }
        }

        self.entries.push(Self::tool_output_entry(
            turn_id,
            tool_name.clone(),
            format!("{tool_name} output"),
            text,
        ));
    }

    pub(crate) fn finish_tool_output(&mut self, turn_id: u64, tool_name: String, preview: String) {
        if let Some(last_entry) = self.entries.last_mut() {
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
                    return;
                }
                TranscriptEntry::ToolStatus {
                    turn_id: active_turn_id,
                    tool_name: active_tool_name,
                } if *active_turn_id == turn_id && *active_tool_name == tool_name => {
                    *last_entry =
                        Self::tool_output_entry(turn_id, tool_name, preview, String::new());
                    return;
                }
                _ => {}
            }
        }

        self.entries.push(Self::tool_output_entry(
            turn_id,
            tool_name,
            preview,
            String::new(),
        ));
    }

    fn push_turn_error(&mut self, turn_id: u64, message: String) {
        self.entries
            .push(TranscriptEntry::Error { turn_id, message });
    }

    fn tool_output_entry(
        turn_id: u64,
        tool_name: String,
        preview: String,
        body: String,
    ) -> TranscriptEntry {
        TranscriptEntry::ToolOutput {
            turn_id,
            tool_name,
            preview,
            body,
            folded: false,
        }
    }

    fn find_adjacent_tool_output(
        &self,
        current: Option<usize>,
        direction: ToolOutputDirection,
    ) -> Option<usize> {
        match (current, direction) {
            (None, _) => self.last_tool_output_index(),
            (Some(current), ToolOutputDirection::Down) => {
                self.next_tool_output_index_after(current).or(Some(current))
            }
            (Some(current), ToolOutputDirection::Up) => {
                self.previous_tool_output_index_before(current)
                    .or(Some(current))
            }
        }
    }

    fn toggle_tool_output(&mut self, index: usize) -> bool {
        match self.entries.get_mut(index) {
            Some(TranscriptEntry::ToolOutput { folded, .. }) => {
                *folded = !*folded;
                true
            }
            _ => false,
        }
    }

    fn last_tool_output_index(&self) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, entry)| match entry {
                TranscriptEntry::ToolOutput { .. } => Some(index),
                _ => None,
            })
    }

    fn next_tool_output_index_after(&self, current: usize) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .skip(current + 1)
            .find_map(|(index, entry)| match entry {
                TranscriptEntry::ToolOutput { .. } => Some(index),
                _ => None,
            })
    }

    fn previous_tool_output_index_before(&self, current: usize) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .take(current)
            .rev()
            .find_map(|(index, entry)| match entry {
                TranscriptEntry::ToolOutput { .. } => Some(index),
                _ => None,
            })
    }
}
