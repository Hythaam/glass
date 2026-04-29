use std::path::Path;

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

const STATUS_BAR_HEIGHT: u16 = 1;
const TOOL_FOLD_LINE_THRESHOLD: usize = 4;
const TOOL_FOLD_CHAR_THRESHOLD: usize = 160;

pub(crate) fn transcript_inner_size(area: Rect) -> (usize, usize) {
    (area.width as usize, area.height as usize)
}

pub(crate) fn composer_block(startup_dir: &Path, turn_in_flight: bool) -> Block<'static> {
    Block::default()
        .title(format!(" {} -", startup_dir.display()))
        .title_alignment(Alignment::Right)
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(if turn_in_flight {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::Gray)
        })
}

pub(crate) fn composer_inner_area(startup_dir: &Path, turn_in_flight: bool, area: Rect) -> Rect {
    composer_block(startup_dir, turn_in_flight).inner(area)
}

pub(crate) fn composer_paragraph(
    composer: impl Into<ratatui::text::Text<'static>>,
    startup_dir: &Path,
    turn_in_flight: bool,
) -> Paragraph<'static> {
    Paragraph::new(composer)
        .block(composer_block(startup_dir, turn_in_flight))
        .wrap(Wrap { trim: false })
}

pub(crate) fn composer_cursor_position(
    composer: &str,
    startup_dir: &Path,
    turn_in_flight: bool,
    area: Rect,
) -> Option<(u16, u16)> {
    let composer_inner = composer_inner_area(startup_dir, turn_in_flight, area);
    if composer_inner.width == 0 || composer_inner.height == 0 {
        return None;
    }

    let (cursor_x, cursor_y) = composer_cursor(composer, composer_inner.width as usize);
    Some((
        composer_inner.x + cursor_x.min(composer_inner.width.saturating_sub(1)),
        composer_inner.y + cursor_y.min(composer_inner.height.saturating_sub(1)),
    ))
}

pub(crate) fn layout_chunks(size: Rect, composer_height: u16) -> Vec<Rect> {
    Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(composer_height),
            Constraint::Length(STATUS_BAR_HEIGHT),
        ])
        .split(size)
        .to_vec()
}

pub(crate) fn compose_status_line(left: &str, right: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    let right = truncate_leftmost(right, width);
    let right_width = right.chars().count();
    if right_width >= width {
        return right;
    }

    let left_max = width.saturating_sub(right_width + 1);
    let left = truncate_rightmost(left, left_max);
    let left_width = left.chars().count();
    let spaces = width.saturating_sub(left_width + right_width).max(1);

    format!("{left}{}{right}", " ".repeat(spaces))
}

pub(crate) fn rect_contains(rect: Rect, column: u16, row: u16) -> bool {
    column >= rect.x && column < rect.x + rect.width && row >= rect.y && row < rect.y + rect.height
}

pub(crate) fn should_fold_tool_output(body: &str) -> bool {
    let line_count = body.lines().count().max(1);
    line_count >= TOOL_FOLD_LINE_THRESHOLD || body.chars().count() >= TOOL_FOLD_CHAR_THRESHOLD
}

pub(crate) fn wrap_plain_text(text: &str, width: usize) -> Vec<String> {
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

pub(crate) fn wrap_composer_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut wrapped = Vec::new();

    for raw_line in text.split('\n') {
        if raw_line.is_empty() {
            wrapped.push(String::new());
            continue;
        }

        let mut remaining = raw_line.chars().collect::<Vec<_>>();
        while !remaining.is_empty() {
            let mut line_width = 0usize;
            let mut last_word_end = 0usize;
            let mut prev_whitespace = false;
            let mut overflow = false;

            for index in 0..remaining.len() {
                let character = remaining[index];
                let is_whitespace = character.is_whitespace();
                if is_whitespace && !prev_whitespace {
                    last_word_end = index;
                }

                line_width += 1;
                if line_width > width {
                    let break_at = if last_word_end != 0 {
                        last_word_end
                    } else {
                        index
                    };
                    wrapped.push(remaining[..break_at].iter().collect());
                    let mut rest = remaining[break_at..].to_vec();
                    if let Some(non_whitespace) =
                        rest.iter().position(|character| !character.is_whitespace())
                    {
                        rest.drain(..non_whitespace);
                    } else {
                        rest.clear();
                    }
                    remaining = rest;
                    overflow = true;
                    break;
                }

                prev_whitespace = is_whitespace;
            }

            if !overflow {
                wrapped.push(remaining.iter().collect());
                remaining.clear();
            }
        }
    }

    if wrapped.is_empty() {
        wrapped.push(String::new());
    }

    wrapped
}

fn truncate_rightmost(text: &str, width: usize) -> String {
    text.chars().take(width).collect()
}

fn truncate_leftmost(text: &str, width: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= width {
        return text.to_string();
    }

    text.chars().skip(char_count - width).collect()
}

fn composer_cursor(composer: &str, width: usize) -> (u16, u16) {
    let width = width.max(1);
    let wrapped = wrap_composer_text(composer, width);
    let row = wrapped.len().saturating_sub(1) as u16;
    let col = wrapped
        .last()
        .map(|line| line.chars().count() as u16)
        .unwrap_or(0)
        .min(width as u16);
    (col, row)
}
