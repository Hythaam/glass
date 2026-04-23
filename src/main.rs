mod agent;

mod config;
#[allow(dead_code)]
mod context;
#[allow(dead_code)]
mod llm;
#[allow(dead_code)]
mod tools;
#[allow(dead_code)]
mod tui;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    // Load config to validate startup settings at startup. `_config` is unused for now
    // but will be consumed by features added later; keeping this load ensures early
    // validation of CLI/env/config values.
    let _config = config::Config::load(std::env::args_os())?;
    Ok(())
}

#[cfg(test)]
fn assert_module_smoke() {
    use crate::agent::AgentEvent;
    use crate::llm::ProviderStreamItem;
    use crate::tools::ToolResult;

    let _ = AgentEvent::AssistantDelta {
        turn_id: 1,
        text: "hi".into(),
    };
    let _ = ProviderStreamItem::Done;
    let _ = ToolResult::Text {
        preview: "ok".into(),
        body: "ok".into(),
    };
}

#[cfg(test)]
#[test]
fn module_smoke_test() {
    assert_module_smoke();
}
