use crate::llm::{ChatMessage, ChatRequest, ChatRole, ProviderResult, ToolCall};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    role: &'static str,
    content: String,
    tool_calls: Vec<ToolCall>,
    sequence: u64,
}

impl Turn {
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn role(&self) -> &'static str {
        self.role
    }
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolObservation {
    tool_name: String,
    body: String,
    // internal flag indicating whether this observation has been compacted
    is_compacted: bool,
    sequence: u64,
}

impl ToolObservation {
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn body(&self) -> &str {
        &self.body
    }
    /// Read-only accessor to indicate whether the observation has been compacted.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn is_compacted(&self) -> bool {
        self.is_compacted
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionContext {
    limit: usize,
    summary: Option<String>,
    recent_turns: Vec<Turn>,
    tool_observations: Vec<ToolObservation>,
    next_sequence: u64,
}

impl SessionContext {
    /// Read-only accessor for configured token limit.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn limit(&self) -> usize {
        self.limit
    }
    /// Read-only accessor for the optional summary (if any).
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }
    /// Read-only slice of recent turns.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn recent_turns(&self) -> &[Turn] {
        &self.recent_turns
    }
    /// Read-only slice of tool observations.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn tool_observations(&self) -> &[ToolObservation] {
        &self.tool_observations
    }

    pub fn chat_request(&self) -> ProviderResult<ChatRequest> {
        let mut messages = Vec::new();

        if let Some(summary) = self.summary() {
            messages.push(ChatMessage {
                role: ChatRole::System,
                content: format!("Conversation summary:\n{summary}"),
                tool_name: None,
                tool_calls: None,
            });
        }

        let mut timeline = Vec::with_capacity(self.recent_turns.len() + self.tool_observations.len());

        for turn in &self.recent_turns {
            let role = match turn.role {
                "user" => ChatRole::User,
                "assistant" => ChatRole::Assistant,
                _ => ChatRole::System,
            };
            let tool_calls = if turn.tool_calls.is_empty() {
                None
            } else {
                Some(
                    turn.tool_calls
                        .iter()
                        .map(ToolCall::as_chat_tool_call)
                        .collect::<ProviderResult<Vec<_>>>()?,
                )
            };
            timeline.push((
                turn.sequence,
                ChatMessage {
                    role,
                    content: turn.content.clone(),
                    tool_name: None,
                    tool_calls,
                },
            ));
        }

        for observation in &self.tool_observations {
            timeline.push((
                observation.sequence,
                ChatMessage {
                    role: ChatRole::Tool,
                    content: observation.body.clone(),
                    tool_name: Some(observation.tool_name.clone()),
                    tool_calls: None,
                },
            ));
        }

        timeline.sort_by_key(|(sequence, _)| *sequence);
        messages.extend(timeline.into_iter().map(|(_, message)| message));

        Ok(ChatRequest {
            messages,
            tools: Vec::new(),
        })
    }
}

const TURN_OVERHEAD_TOKENS: usize = 12;
const TOOL_OBSERVATION_OVERHEAD_TOKENS: usize = 12;
const RAW_TOOL_WINDOW: usize = 2;

impl SessionContext {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            summary: None,
            recent_turns: Vec::new(),
            tool_observations: Vec::new(),
            next_sequence: 0,
        }
    }

    pub fn push_user(&mut self, content: &str) {
        let sequence = self.reserve_sequence();
        self.recent_turns.push(Turn {
            role: "user",
            content: content.into(),
            tool_calls: Vec::new(),
            sequence,
        });
    }

    pub fn push_assistant(&mut self, content: &str) {
        let sequence = self.reserve_sequence();
        self.recent_turns.push(Turn {
            role: "assistant",
            content: content.into(),
            tool_calls: Vec::new(),
            sequence,
        });
    }

    pub fn push_assistant_tool_call(&mut self, call: &ToolCall) -> ProviderResult<()> {
        call.as_chat_tool_call()?;
        let sequence = self.reserve_sequence();
        self.recent_turns.push(Turn {
            role: "assistant",
            content: String::new(),
            tool_calls: vec![call.clone()],
            sequence,
        });
        Ok(())
    }

    pub fn push_tool_output(&mut self, tool_name: &str, body: &str) {
        let sequence = self.reserve_sequence();
        self.tool_observations.push(ToolObservation {
            tool_name: tool_name.into(),
            body: body.into(),
            is_compacted: false,
            sequence,
        });
        // Enforce recent-only raw retention: compact older tool outputs immediately
        // so only the most recent RAW_TOOL_WINDOW remain as full raw bodies.
        // compact_old_tool_outputs() is safe to call repeatedly and is intentionally
        // idempotent for already-compacted entries.
        self.compact_old_tool_outputs();
    }

    pub fn prune_if_needed(&mut self) {
        if !self.should_prune() {
            return;
        }

        // Compact older tool outputs early so that pruning can remove whole
        // observations if compaction doesn't reduce tokens enough. This reduces
        // overall token usage before we attempt to drop observations.
        // compact_old_tool_outputs() is safe to call repeatedly.
        self.compact_old_tool_outputs();
        self.prune_old_tool_observations();

        // Intentionally keep at least one recent turn (e.g. the final user message)
        // so that the agent still has context to answer. Only prune older turns
        // when there is more than one turn available.
        while self.estimated_tokens() > self.limit && self.recent_turns.len() > 1 {
            let removed = self.recent_turns.remove(0);
            self.append_turn_summary(removed);
        }

        // NOTE: a previous implementation made an extra call to prune_old_tool_observations().
        // That duplicate was removed because removing turns and compacting the summary
        // also reduces token usage. We use compact_summary() here to further shrink
        // the summary if needed.
        while self.estimated_tokens() > self.limit && self.compact_summary() {}
    }

    fn should_prune(&self) -> bool {
        let threshold = self.limit.saturating_sub(self.limit / 5).max(1);
        self.estimated_tokens() >= threshold
    }

    fn reserve_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        sequence
    }

    fn estimated_tokens(&self) -> usize {
        let summary_tokens = self.summary.as_deref().map_or(0, estimate_text_tokens);
        let turn_tokens = self
            .recent_turns
            .iter()
            .map(|turn| {
                TURN_OVERHEAD_TOKENS
                    + estimate_text_tokens(turn.role)
                    + estimate_text_tokens(&turn.content)
            })
            .sum::<usize>();
        let tool_tokens = self
            .tool_observations
            .iter()
            .map(|observation| {
                TOOL_OBSERVATION_OVERHEAD_TOKENS
                    + estimate_text_tokens(&observation.tool_name)
                    + estimate_text_tokens(&observation.body)
            })
            .sum::<usize>();

        summary_tokens + turn_tokens + tool_tokens
    }

    fn compact_old_tool_outputs(&mut self) {
        // Compact older tool outputs so they no longer keep their full bodies in
        // memory. Keep a small recent window of full bodies (RAW_TOOL_WINDOW)
        // and replace older observations with a compacted summary.
        // Safe to call multiple times; already-compacted bodies are recognized
        // and not re-compacted (idempotent behavior via the is_compacted flag).
        let n = self.tool_observations.len();
        if n == 0 {
            return;
        }
        let keep = RAW_TOOL_WINDOW.min(n);

        for index in 0..n {
            if index + keep >= n {
                // part of the recent window: keep the full body as-is.
                continue;
            }

            // Older than the recent window: replace the body with a compacted
            // summary if not already compacted.
            let obs = &mut self.tool_observations[index];
            if obs.is_compacted {
                continue;
            }
            let compacted = summarize_tool_body(&obs.body);
            obs.body = compacted;
            obs.is_compacted = true;
        }
    }

    fn append_turn_summary(&mut self, turn: Turn) {
        let next = if turn.tool_calls.is_empty() {
            format!("{}: {}", turn.role, summarize_turn_content(&turn.content))
        } else {
            let tool_names = turn
                .tool_calls
                .iter()
                .map(|call| call.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!("assistant tool call: {tool_names}")
        };
        match self.summary.as_mut() {
            Some(summary) if !summary.is_empty() => {
                summary.push('\n');
                summary.push_str(&next);
            }
            Some(summary) => summary.push_str(&next),
            None => self.summary = Some(next),
        }
    }

    fn compact_summary(&mut self) -> bool {
        let Some(summary) = self.summary.as_mut() else {
            return false;
        };
        let compacted = summarize_summary(summary);
        if compacted == *summary {
            return false;
        }
        *summary = compacted;
        true
    }

    fn prune_old_tool_observations(&mut self) {
        // Remove oldest tool observations (usually already compacted) while we're
        // still over the token budget. We keep at least one tool observation so
        // recent raw tool output remains available for follow-ups.
        while self.estimated_tokens() > self.limit && self.tool_observations.len() > 1 {
            let removed = self.tool_observations.remove(0);
            self.append_tool_summary(removed);
        }
    }

    fn append_tool_summary(&mut self, observation: ToolObservation) {
        let body = observation.body;
        // If the observation was already compacted, use its body as-is; otherwise
        // produce a compacted summary string.
        let summary_part = if observation.is_compacted {
            body
        } else {
            summarize_tool_body(&body)
        };
        let next = format!("tool {}: {}", observation.tool_name, summary_part);
        match self.summary.as_mut() {
            Some(summary) if !summary.is_empty() => {
                summary.push('\n');
                summary.push_str(&next);
            }
            Some(summary) => summary.push_str(&next),
            None => self.summary = Some(next),
        }
    }
}

fn estimate_text_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

fn summarize_tool_body(body: &str) -> String {
    // Produce a compacted preview of the tool output. This function assumes
    // callers use the is_compacted flag to avoid double-compaction.
    let preview = body
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(2)
        .collect::<Vec<_>>()
        .join(" ");

    if preview.is_empty() {
        "summary: (empty tool output)".into()
    } else {
        format!("summary: {preview}")
    }
}

fn summarize_turn_content(content: &str) -> String {
    let words = content.split_whitespace().collect::<Vec<_>>();
    if words.len() <= 4 {
        return words.join(" ");
    }

    format!("{} …", words[..4].join(" "))
}

fn summarize_summary(summary: &str) -> String {
    let trimmed = summary.trim();
    if trimmed == "older context" {
        return trimmed.into();
    }
    if trimmed.len() <= 16 {
        return "older context".into();
    }

    let target_len = (trimmed.len() / 2).max(16);
    let mut compacted = trimmed.chars().take(target_len).collect::<String>();
    if let Some((head, _)) = compacted.rsplit_once(char::is_whitespace) {
        compacted = head.to_string();
    }

    if compacted.is_empty() || compacted == trimmed {
        "older context".into()
    } else {
        format!("{compacted} …")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatRole, ToolCall};

    #[test]
    fn prunes_old_turns_into_summary() {
        let mut context = SessionContext::new(40);
        context.push_user("one two three four five six");
        context.push_assistant("alpha beta gamma delta epsilon zeta");
        context.push_user("recent");

        context.prune_if_needed();

        assert!(context.summary().unwrap_or("").contains("one two"));
        assert_eq!(context.recent_turns().last().unwrap().content(), "recent");
    }

    #[test]
    fn compacts_old_tool_output_before_recent_turns() {
        let mut context = SessionContext::new(30);
        context.push_tool_output("list", "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta");
        context.push_tool_output("stat", "recent\nraw\noutput");
        context.push_user("what changed?");

        context.prune_if_needed();

        assert!(!context.tool_observations().is_empty());
        assert!(context.summary().is_some());
        assert_eq!(context.tool_observations().len(), 1);
        assert_eq!(
            context.tool_observations().last().unwrap().body(),
            "recent\nraw\noutput"
        );
        assert_eq!(
            context.recent_turns().last().unwrap().content(),
            "what changed?"
        );
    }

    #[test]
    fn compacts_older_tool_outputs_but_keeps_recent_raw_bodies() {
        let mut context = SessionContext::new(1000);
        context.push_tool_output("t1", "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta");
        context.push_tool_output("t2", "middle\ncontent");
        context.push_tool_output("t3", "recent\nraw\noutput");
        context.push_user("what changed?");

        context.prune_if_needed();

        assert_eq!(context.tool_observations().len(), 3);
        // the oldest observation(s) should be compacted
        assert!(
            context.tool_observations()[0].is_compacted(),
            "oldest must be compacted"
        );
        // the most recent RAW_TOOL_WINDOW observations keep their full bodies
        assert_eq!(context.tool_observations()[1].body(), "middle\ncontent");
        assert_eq!(context.tool_observations()[2].body(), "recent\nraw\noutput");
    }

    #[test]
    fn keeps_only_tool_output_raw_for_follow_up_questions() {
        let mut context = SessionContext::new(40);
        context.push_tool_output("list", "a\nb\nc\nd\ne\nf");
        context.push_user("what changed?");

        context.prune_if_needed();

        assert_eq!(context.tool_observations()[0].body(), "a\nb\nc\nd\ne\nf");
    }

    #[test]
    fn summary_is_more_compact_than_removed_turns() {
        let mut context = SessionContext::new(40);
        context.push_user("one two three four five six");
        context.push_assistant("alpha beta gamma delta epsilon zeta");
        context.push_user("recent");

        context.prune_if_needed();

        let summary = context.summary().unwrap_or("");
        assert!(summary.contains("one two"));
        assert!(!summary.contains("five six"));
    }

    #[test]
    fn keeps_most_recent_tool_output_raw() {
        let mut context = SessionContext::new(10);
        context.push_tool_output("older", "a\nb\nc\nd\ne\nf");
        context.push_tool_output("recent", "one\ntwo\nthree\nfour");
        context.push_user("follow up question");

        context.prune_if_needed();

        assert!(context.summary().is_some());
        assert_eq!(
            context.tool_observations().last().unwrap().tool_name(),
            "recent"
        );
        assert_eq!(
            context.tool_observations().last().unwrap().body(),
            "one\ntwo\nthree\nfour"
        );
    }

    #[test]
    fn estimates_tokens_approximately() {
        assert_eq!(estimate_text_tokens("12345678"), 2);
    }

    #[test]
    fn single_large_turn_may_exceed_budget() {
        let mut ctx = SessionContext::new(5);
        // Push a single very long turn so its estimated tokens exceed the limit.
        ctx.push_user("a very very very very very very very long message meant to be large");
        ctx.prune_if_needed();
        // The single remaining turn is intentionally retained even if it exceeds the limit.
        assert_eq!(ctx.recent_turns().len(), 1);
        assert!(ctx.estimated_tokens() > ctx.limit());
    }

    #[test]
    fn prune_if_needed_brings_context_within_budget() {
        let mut context = SessionContext::new(20);
        context.push_user("one two three four five six seven eight");
        context.push_assistant("alpha beta gamma delta epsilon zeta eta theta");
        context.push_user("iota kappa lambda mu nu xi omicron pi");
        context.push_user("ok");

        context.prune_if_needed();

        assert!(
            context.estimated_tokens() <= context.limit(),
            "estimated={} limit={} summary={:?} recent_turns={:?}",
            context.estimated_tokens(),
            context.limit(),
            context.summary(),
            context
                .recent_turns()
                .iter()
                .map(|t| t.content())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn prunes_old_compacted_tool_observations_when_still_over_budget() {
        let mut context = SessionContext::new(24);
        context.push_tool_output("list", "a\nb\nc\nd\ne\nf");
        context.push_tool_output("stat", "one\ntwo\nthree\nfour\nfive\nsix");

        context.prune_if_needed();

        assert!(context.estimated_tokens() <= context.limit());
        assert_eq!(context.tool_observations().len(), 1);
        assert_eq!(context.tool_observations()[0].tool_name(), "stat");
    }

    #[test]
    fn compacts_tool_bodies_idempotently_with_many_observations() {
        let mut ctx = SessionContext::new(40);
        ctx.push_tool_output("t1", "one\ntwo\nthree\nfour");
        ctx.push_tool_output("t2", "a\nb\nc\nd");
        ctx.push_tool_output("t3", "alpha\nbeta\ngamma\ndelta");
        ctx.push_tool_output("t4", "recent\nraw\noutput");

        // Use the public surface: prune_if_needed() will call compaction under the hood.
        ctx.prune_if_needed();

        // At least one observation should be marked compacted.
        let any_compacted = ctx.tool_observations().iter().any(|o| o.is_compacted());
        let removed_some = ctx.tool_observations().len() < 4;
        assert!(
            any_compacted || removed_some,
            "expected compaction or pruning (len={} compacted={})",
            ctx.tool_observations().len(),
            any_compacted
        );

        // Re-run the same public action; compaction must be idempotent and not
        // produce double-prefixed summaries.
        ctx.prune_if_needed();

        for b in ctx.tool_observations().iter().map(|o| o.body()) {
            assert!(!b.contains("summary: summary:"), "double summary detected");
        }
    }

    #[test]
    fn chat_request_interleaves_assistant_tool_calls_and_tool_results() {
        let mut context = SessionContext::new(512);
        context.push_user("show src");
        context
            .push_assistant_tool_call(&ToolCall {
                name: "fs".into(),
                arguments_json: r#"{"op":"list_dir","path":"src"}"#.into(),
            })
            .unwrap();
        context.push_tool_output("fs", "agent.rs\nmain.rs");

        let request = context.chat_request().unwrap();

        assert_eq!(request.messages.len(), 3);
        assert_eq!(request.messages[0].role, ChatRole::User);
        assert_eq!(request.messages[1].role, ChatRole::Assistant);
        assert_eq!(request.messages[2].role, ChatRole::Tool);
        assert_eq!(request.messages[2].tool_name.as_deref(), Some("fs"));
        assert_eq!(
            request.messages[1]
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.first())
                .map(|call| call.function.name.as_str()),
            Some("fs")
        );
    }
}
