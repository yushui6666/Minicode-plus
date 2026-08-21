"""Turn-loop text constants and predicates for the LangGraph runtime.

These are moved verbatim from the retired agent loop (minicode/agent_loop.py)
so graph nodes can reuse the same nudge vocabulary and step predicates without
importing the retired module. Keep this module dependency-free: it must stay
importable from both the thin and kernel graph topologies.
"""

from __future__ import annotations

NUDGE_CONTINUE = (
    "Continue immediately from your <progress> update with concrete tool calls, "
    "code changes, or an explicit <final> answer only if the task is complete. "
    "Prefer taking the next concrete action over explaining what you plan to do."
)

NUDGE_AFTER_TOOL_RESULT = (
    "You have received tool results. Review them briefly, then take the next "
    "concrete action: call another tool, edit code, or give an explicit <final> "
    "answer only if the task is truly complete. Do not restate what you just saw."
)

NUDGE_AFTER_EMPTY_RESPONSE = (
    "Your last response was empty. This often happens after tool errors or when "
    "the model is uncertain. Pick the most likely next action and try it — you can "
    "adjust based on results. Call a tool, edit code, or give <final> if done."
)

NUDGE_AFTER_EMPTY_NO_TOOLS = (
    "Your last response was empty but you have not used any tools yet. Start by "
    "inspecting the relevant files (read_file, grep_files, list_files) to understand "
    "the codebase before making changes."
)

RESUME_AFTER_PAUSE = (
    "Resume from the previous pause. Continue with the next concrete tool call, "
    "code change, or <final> answer."
)

RESUME_AFTER_MAX_TOKENS = (
    "Your previous response was cut short by the token limit. Resume immediately "
    "with the next concrete action — pick up where you left off."
)


def is_empty_assistant_response(content: str) -> bool:
    return len(content.strip()) == 0


def is_recoverable_thinking_stop(
    *,
    is_empty: bool,
    stop_reason: str | None,
    ignored_block_types: list[str] | None,
) -> bool:
    if not is_empty:
        return False
    if stop_reason not in {"pause_turn", "max_tokens"}:
        return False
    return "thinking" in (ignored_block_types or [])


def should_treat_assistant_as_progress(
    *,
    kind: str | None,
    content: str,
    saw_tool_result: bool,  # noqa: ARG001 - kept for parity with the retired loop
) -> bool:
    if kind == "progress":
        return True
    if kind == "final":
        return False
    if not saw_tool_result:
        return False
    return False


def format_diagnostics(
    stop_reason: str | None,
    block_types: list[str] | None,
    ignored_block_types: list[str] | None,
) -> str:
    parts: list[str] = []
    if stop_reason:
        parts.append(f"stop_reason={stop_reason}")
    if block_types:
        parts.append(f"blocks={','.join(block_types)}")
    if ignored_block_types:
        parts.append(f"ignored={','.join(ignored_block_types)}")
    return f" Diagnostics: {'; '.join(parts)}." if parts else ""
