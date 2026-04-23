#![allow(dead_code)]

mod agent;
mod config;
mod context;
mod llm;
mod tools;
mod tui;

fn main() {}

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

#[cfg(test)]
mod tests {
    #[test]
    fn module_smoke_test() {
        super::assert_module_smoke();
    }
}
