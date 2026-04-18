use std::io::stdout;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::text::{Span, Spans, Text};
use ratatui::style::{Style, Color};
use ratatui::Terminal;

/// Simple blocking TUI chat. Type and press Enter to send. Ctrl-C or Esc to quit.
pub fn run(client: &crate::api::ApiClient) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut messages: Vec<(String, String)> = Vec::new();
    let mut input = String::new();

    loop {
        terminal.draw(|f| {
            let size = f.size();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)].as_ref())
                .split(size);

            let mut spans_vec: Vec<Spans> = Vec::new();
            for (role, content) in &messages {
                spans_vec.push(Spans::from(Span::styled(format!("{}:", role), Style::default().fg(Color::Yellow))));
                for line in content.lines() {
                    spans_vec.push(Spans::from(Span::raw(line)));
                }
                spans_vec.push(Spans::from(Span::raw("")));
            }
            let text = Text::from(spans_vec);
            let messages_para = Paragraph::new(text)
                .block(Block::default().borders(Borders::ALL).title("Chat"))
                .wrap(Wrap { trim: true });
            f.render_widget(messages_para, chunks[0]);

            let input_para = Paragraph::new(input.as_ref()).block(Block::default().borders(Borders::ALL).title("Input"));
            f.render_widget(input_para, chunks[1]);
        })?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(KeyEvent { code, modifiers, .. }) = event::read()? {
                match code {
                    KeyCode::Char(c) => {
                        if modifiers.contains(KeyModifiers::CONTROL) && c == 'c' {
                            break;
                        } else {
                            input.push(c);
                        }
                    }
                    KeyCode::Backspace => { input.pop(); }
                    KeyCode::Enter => {
                        let user_msg = input.clone();
                        messages.push(("User".to_string(), user_msg.clone()));

                        // Send request (blocking) and append assistant reply
                        match client.send_message(&user_msg) {
                            Ok(reply) => messages.push(("Assistant".to_string(), reply)),
                            Err(e) => messages.push(("Error".to_string(), format!("Request failed: {}", e))),
                        }

                        input.clear();
                    }
                    KeyCode::Esc => break,
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
