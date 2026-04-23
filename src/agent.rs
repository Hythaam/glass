#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    AssistantDelta {
        turn_id: u64,
        text: String,
    },
    AssistantDone {
        turn_id: u64,
    },
    ToolStarted {
        turn_id: u64,
        tool_name: String,
    },
    ToolFinished {
        turn_id: u64,
        tool_name: String,
        preview: String,
        body: String,
    },
    TurnError {
        turn_id: u64,
        message: String,
    },
}
