use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Paragraph;

use crate::agent::{AgentEvent, AgentStatus};
use crate::context::SessionContext;
use crate::llm::RequestTokenUsage;
use crate::test_support::fakes::{FakeProvider, FakeTools};

use super::{
    TranscriptState, TuiAction, TuiApp, TuiState, composer_block, composer_cursor_position,
    layout_chunks, transcript_inner_size,
};

#[test]
fn enter_and_newline_shortcuts_update_composer() {
    let cases = [
        (
            "hello glass",
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            TuiAction::Submit("hello glass".into()),
            "",
        ),
        (
            "hello",
            KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT),
            TuiAction::None,
            "hello\n",
        ),
        (
            "hello",
            KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL),
            TuiAction::None,
            "hello\n",
        ),
    ];

    for (initial, event, expected_action, expected_composer) in cases {
        let mut state = TuiState::default();
        state.composer_mut().push_str(initial);

        let action = state.handle_key_event(event);

        assert_eq!(action, expected_action);
        assert_eq!(state.composer(), expected_composer);
    }
}

#[test]
fn page_keys_adjust_transcript_scroll() {
    let mut state = TuiState::default();
    state.transcript_mut().set_scroll(20);

    let page_up = state.handle_key_event(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
    assert_eq!(page_up, TuiAction::None);
    assert_eq!(state.transcript().scroll(), 10);

    let page_down = state.handle_key_event(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));

    assert_eq!(page_down, TuiAction::None);
    assert_eq!(state.transcript().scroll(), 20);
}

#[test]
fn ctrl_c_requests_quit() {
    let mut state = TuiState::default();

    let action = state.handle_key_event(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));

    assert_eq!(action, TuiAction::Quit);
}

#[test]
fn status_line_shows_model_and_token_stats() {
    let mut state = TuiState::default();
    state.update_status(status("gemma-4", Some((19, 2, 21))));

    let line = state.status_line(48);

    assert_eq!(line.len(), 48);
    assert!(line.starts_with("gemma-4"));
    assert!(line.ends_with("ctx 128/512 | last 19/2/21"));
}

#[test]
fn layout_reserves_status_bar_below_composer() {
    let chunks = layout_chunks(Rect::new(0, 0, 80, 20), 4);

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[1].height, 4);
    assert_eq!(chunks[2].height, 1);
    assert_eq!(chunks[2].y, chunks[1].y + chunks[1].height);
}

#[test]
fn status_line_uses_light_gray_text() {
    let mut state = TuiState::default();
    state.update_status(status("gemma-4", None));

    let span = state.status_line_span(48);

    assert_eq!(span.style.fg, Some(Color::Gray));
}

#[test]
fn app_initializes_status_from_agent_snapshot() {
    let app = test_app();
    let line = app.state().status_line(40);

    assert!(line.starts_with("fake"));
    assert!(line.contains("ctx 0/512"));
}

#[test]
fn transcript_inner_size_uses_full_area_without_border() {
    assert_eq!(transcript_inner_size(Rect::new(0, 0, 40, 10)), (40, 10));
}

#[test]
fn composer_block_shows_cwd_in_upper_right_without_composer_title() {
    let top_row = top_row_text(&render_block(composer_block(
        PathBuf::from("/tmp/glass").as_path(),
        false,
    )));

    assert!(top_row.contains("/tmp/glass"));
    assert!(!top_row.contains("Composer"));
}

#[test]
fn composer_block_offsets_cwd_one_column_left() {
    let top_row = top_row_text(&render_block(composer_block(
        PathBuf::from("/tmp/glass").as_path(),
        false,
    )));

    assert!(top_row.contains(" /tmp/glass"));
}

#[test]
fn composer_block_uses_light_gray_border_when_idle() {
    let buffer = render_block(composer_block(PathBuf::from("/tmp/glass").as_path(), false));

    assert_eq!(buffer.get(0, 0).fg, Color::Gray);
}

#[test]
fn composer_visual_lines_counts_explicit_newlines() {
    let mut state = TuiState::default();
    state.composer_mut().push_str("one\ntwo\nthree");

    assert_eq!(state.composer_visual_lines(20), 3);
}

#[test]
fn composer_visual_lines_counts_soft_wrapped_single_line() {
    let mut state = TuiState::default();
    state.composer_mut().push_str("abcdefghij");

    assert_eq!(state.composer_visual_lines(4), 3);
}

#[test]
fn composer_height_grows_with_wrapped_visual_lines() {
    let mut state = TuiState::default();
    state.composer_mut().push_str("abcdefghij");

    assert_eq!(state.composer_height(4), 5);
}

#[test]
fn composer_height_is_not_capped_at_eight_rows() {
    let mut state = TuiState::default();
    state.composer_mut().push_str("abcdefghijklmnopqrstuvwxyz");

    assert_eq!(state.composer_height(3), 11);
}

#[test]
fn draw_positions_composer_cursor_using_block_inner_area() {
    let mut state = TuiState::default();
    state.composer_mut().push_str("abcdefg");
    let composer_area = Rect::new(0, 0, 8, 3);
    let backend = TestBackend::new(8, 6);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal
        .draw(|frame| {
            let composer = Paragraph::new(state.composer())
                .block(composer_block(PathBuf::from(".").as_path(), false));
            frame.render_widget(composer, composer_area);
        })
        .unwrap();

    let actual_cursor = composer_cursor_position(
        state.composer(),
        PathBuf::from(".").as_path(),
        false,
        composer_area,
    )
    .unwrap();

    let buffer = terminal.backend().buffer().clone();
    let last_character = find_symbol(&buffer, composer_area, "g").unwrap();

    assert_eq!(actual_cursor, (last_character.0 + 1, last_character.1));
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

    assert_eq!(state.transcript().text(), vec!["Assistant: hello world"]);
}

#[test]
fn transcript_state_renders_assistant_turns_directly() {
    let mut transcript = TranscriptState::default();

    transcript.apply_agent_event(AgentEvent::AssistantDelta {
        turn_id: 21,
        text: "hello".into(),
    });
    transcript.apply_agent_event(AgentEvent::AssistantDelta {
        turn_id: 21,
        text: " world".into(),
    });
    transcript.apply_agent_event(AgentEvent::AssistantDone { turn_id: 21 });

    assert_eq!(transcript.text(), vec!["Assistant: hello world"]);
}

#[test]
fn long_tool_output_starts_folded_and_toggles_from_preview_line() {
    let mut state = TuiState::default();

    emit_tool_output(&mut state, 3, "src entries", false);

    assert!(state.transcript().text()[0].contains("src entries"));
    assert!(state.transcript().tool_output_is_folded(0));

    assert!(state.transcript_mut().toggle_tool_output_at_line(0, 40));
    assert!(!state.transcript().tool_output_is_folded(0));
}

#[test]
fn down_and_enter_toggle_selected_tool_output() {
    let mut state = TuiState::default();

    emit_tool_output(&mut state, 1, "first entries", true);
    emit_tool_output(&mut state, 2, "second entries", true);

    let action = state.handle_key_event(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(action, TuiAction::None);
    assert_eq!(state.transcript().selected_tool_output(), Some(1));

    let action = state.handle_key_event(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(action, TuiAction::None);

    assert!(state.transcript().tool_output_is_folded(0));
    assert!(!state.transcript().tool_output_is_folded(1));
}

#[test]
fn enter_still_submits_composer_when_keyboard_selection_exists() {
    let mut state = TuiState::default();

    emit_tool_output(&mut state, 3, "src entries", true);
    state.handle_key_event(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(state.transcript().selected_tool_output(), Some(0));
    state.composer_mut().push_str("hello glass");

    let action = state.handle_key_event(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

    assert_eq!(action, TuiAction::Submit("hello glass".into()));
    assert!(state.transcript().tool_output_is_folded(0));
}

#[test]
fn keyboard_selection_scrolls_selected_tool_output_into_view() {
    let mut app = test_app();

    emit_tool_output(app.state_mut(), 4, "first entries", true);
    emit_tool_output(app.state_mut(), 5, "second entries", true);
    app.set_transcript_area(test_transcript_area(4));
    app.state_mut().transcript_mut().set_scroll(0);

    app.state_mut()
        .handle_key_event(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    app.sync_selected_tool_output_to_view();

    assert_eq!(app.state().transcript().selected_tool_output(), Some(1));
    assert_eq!(app.state().transcript().scroll(), 0);
}

#[test]
fn enter_toggles_selected_tool_output_while_turn_is_in_flight() {
    let mut state = TuiState::default();

    emit_tool_output(&mut state, 6, "streaming entries", false);
    state.handle_key_event(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));

    let action = state.handle_key_event(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

    assert_eq!(action, TuiAction::None);
    assert!(!state.transcript().tool_output_is_folded(0));
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

    let transcript = state.transcript().text();
    assert_eq!(transcript.len(), 1);
    assert!(transcript[0].contains("Tool fs: 1 entry"));
    assert!(transcript[0].contains("main.rs"));
    assert!(!transcript[0].contains("running"));
}

#[test]
fn append_tool_output_reuses_existing_output_entry() {
    let mut state = TuiState::default();

    state.transcript_mut().start_tool(12, "fs".into());
    state
        .transcript_mut()
        .append_tool_output(12, "fs".into(), "main.rs\n".into());
    state
        .transcript_mut()
        .append_tool_output(12, "fs".into(), "tui.rs\n".into());

    let transcript = state.transcript().text();
    assert_eq!(transcript.len(), 1);
    assert!(transcript[0].contains("main.rs"));
    assert!(transcript[0].contains("tui.rs"));
}

#[test]
fn finish_tool_output_updates_preview_and_fold_state() {
    let mut state = TuiState::default();

    state.transcript_mut().start_tool(13, "fs".into());
    state.transcript_mut().append_tool_output(
        13,
        "fs".into(),
        "agent.rs\ncontext.rs\nllm.rs\nmain.rs\ntools.rs\ntui.rs".into(),
    );
    state
        .transcript_mut()
        .finish_tool_output(13, "fs".into(), "src entries".into());

    let transcript = state.transcript().text();
    assert_eq!(transcript.len(), 1);
    assert!(transcript[0].contains("Tool fs: src entries"));
    assert!(state.transcript().tool_output_is_folded(0));
}

#[test]
fn turn_errors_are_rendered_inline() {
    let mut state = TuiState::default();

    state.apply_agent_event(AgentEvent::TurnError {
        turn_id: 5,
        message: "provider down".into(),
    });

    assert_eq!(state.transcript().text(), vec!["Error: provider down"]);
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

    state.transcript_mut().clamp_scroll(40, 2);

    assert!(state.transcript().scroll() > 0);
}

#[test]
fn mouse_wheel_scroll_behavior_depends_on_transcript_hit_test() {
    let cases = [
        (
            8,
            test_transcript_area(10),
            mouse_event(MouseEventKind::ScrollUp, 5, 5),
            5,
        ),
        (
            2,
            test_transcript_area(10),
            mouse_event(MouseEventKind::ScrollDown, 5, 5),
            5,
        ),
        (
            4,
            Rect::new(10, 10, 40, 10),
            mouse_event(MouseEventKind::ScrollDown, 5, 5),
            4,
        ),
    ];

    for (initial_scroll, area, event, expected_scroll) in cases {
        let mut app = test_app();
        app.state_mut().transcript_mut().set_scroll(initial_scroll);
        app.set_transcript_area(area);

        app.handle_mouse(event);

        assert_eq!(app.state().transcript().scroll(), expected_scroll);
    }
}

#[test]
fn left_click_still_toggles_tool_output_from_transcript() {
    let mut app = test_app();
    emit_tool_output(app.state_mut(), 3, "src entries", false);
    app.set_transcript_area(test_transcript_area(10));
    app.state_mut().transcript_mut().clamp_scroll(38, 8);

    app.handle_mouse(mouse_event(MouseEventKind::Down(MouseButton::Left), 1, 1));

    assert!(!app.state().transcript().tool_output_is_folded(0));
}

#[test]
fn tui_state_handles_transcript_mouse_without_app_reaching_into_transcript() {
    let mut state = TuiState::default();
    emit_tool_output(&mut state, 14, "src entries", true);
    state.transcript_mut().set_scroll(0);

    state.handle_transcript_mouse(
        mouse_event(MouseEventKind::Down(MouseButton::Left), 1, 1),
        test_transcript_area(10),
    );

    assert!(!state.transcript().tool_output_is_folded(0));
}

fn test_app() -> TuiApp<FakeProvider, FakeTools> {
    let provider = FakeProvider::idle();
    let tools = FakeTools::failing("unused");
    let agent = crate::agent::Agent::new(provider, tools, SessionContext::new(512, None));
    TuiApp::new(agent, PathBuf::from("."))
}

fn mouse_event(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind,
        column,
        row,
        modifiers: KeyModifiers::NONE,
    }
}

fn status(model_name: &str, usage: Option<(usize, usize, usize)>) -> AgentStatus {
    AgentStatus {
        model_name: model_name.into(),
        context_tokens: 128,
        context_limit: 512,
        last_request_usage: usage.map(|(prompt_tokens, completion_tokens, total_tokens)| {
            RequestTokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            }
        }),
    }
}

fn render_block(block: ratatui::widgets::Block<'static>) -> Buffer {
    let backend = TestBackend::new(24, 3);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            let widget = Paragraph::new("").block(block.clone());
            frame.render_widget(widget, Rect::new(0, 0, 24, 3));
        })
        .unwrap();
    terminal.backend().buffer().clone()
}

fn top_row_text(buffer: &Buffer) -> String {
    (0..24)
        .map(|x| buffer.get(x, 0).symbol.as_str())
        .collect::<String>()
}

fn find_symbol(buffer: &Buffer, area: Rect, symbol: &str) -> Option<(u16, u16)> {
    (0..area.height)
        .flat_map(|y| (0..area.width).map(move |x| (x, y)))
        .find(|(x, y)| buffer.get(*x, *y).symbol == symbol)
}

fn test_transcript_area(height: u16) -> Rect {
    Rect::new(0, 0, 40, height)
}

fn emit_tool_output(state: &mut TuiState, turn_id: u64, preview: &str, complete_turn: bool) {
    state.apply_agent_event(AgentEvent::ToolStarted {
        turn_id,
        tool_name: "fs".into(),
    });
    state.apply_agent_event(AgentEvent::ToolOutputDelta {
        turn_id,
        tool_name: "fs".into(),
        text: "agent.rs\ncontext.rs\nllm.rs\nmain.rs\ntools.rs\ntui.rs".into(),
    });
    state.apply_agent_event(AgentEvent::ToolFinished {
        turn_id,
        tool_name: "fs".into(),
        preview: preview.into(),
    });
    if complete_turn {
        state.apply_agent_event(AgentEvent::AssistantDone { turn_id });
    }
    let last_index = state.transcript().text().len() - 1;
    assert!(state.transcript().tool_output_is_folded(last_index));
}
