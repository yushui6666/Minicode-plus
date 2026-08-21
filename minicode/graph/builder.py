from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from minicode.graph.turn_text import (
    NUDGE_AFTER_EMPTY_NO_TOOLS,
    NUDGE_AFTER_EMPTY_RESPONSE,
    NUDGE_AFTER_TOOL_RESULT,
    NUDGE_CONTINUE,
    RESUME_AFTER_MAX_TOKENS,
    RESUME_AFTER_PAUSE,
    format_diagnostics,
    is_empty_assistant_response,
    is_recoverable_thinking_stop,
    should_treat_assistant_as_progress,
)
from minicode.turn_kernel import (
    build_turn_coda_summary,
    build_widening_transition_nudge,
    decide_assistant_turn,
    decide_tool_turn,
    derive_turn_step_policy,
    render_turn_policy_message,
    snapshot_to_turn_state,
    turn_state_to_snapshot,
)
from minicode.types import AgentStep, RuntimeEvent


class AgentState(TypedDict, total=False):
    """Small orchestration state shared by the graph migration slices.

    Existing MiniCode state remains outside this boundary until later migration
    phases. Nodes communicate through plain serializable values.

    The ``turn kernel`` field groups below are the flat mirror of
    ``TurnRecurrentState``; they round-trip through
    ``snapshot_to_turn_state``/``turn_state_to_snapshot`` so graph nodes reuse
    the kernel decision functions without duplicating their semantics.
    """

    # ── slice 1: orchestration core ────────────────────────────────────────
    messages: list[dict[str, Any]]
    next_action: Literal["tool", "verify", "continue", "complete"]
    tool_name: str
    tool_input: Any
    tool_result: str
    verified: bool
    status: Literal["running", "completed", "failed"]
    permission: Literal["allowed", "denied", "pending"]
    memory_context: str
    compacted: bool
    repair_requested: bool

    # ── turn kernel: budget ────────────────────────────────────────────────
    step: int
    max_steps: int | None
    profile_name: str
    widen_after_step: int | None
    widening_step_bonus: int
    # ── turn kernel: retry budget ──────────────────────────────────────────
    empty_response_retry_limit: int
    empty_response_retry_count: int
    recoverable_thinking_retry_limit: int
    recoverable_thinking_retry_count: int
    # ── turn kernel: tool observations ─────────────────────────────────────
    saw_tool_result: bool
    tool_error_count: int
    tool_observation_count: int
    successful_tool_observation_count: int
    latest_tool_result_summary: str
    tool_result_ok: bool
    tool_await_user: bool
    tool_summary: str
    # ── turn kernel: widening ──────────────────────────────────────────────
    widening_active: bool
    widening_transition_count: int
    widening_trigger_reason: str
    widening_trigger_evidence: str
    # ── turn kernel: verification ──────────────────────────────────────────
    verification_strict: bool
    verification_requires_explicit_final: bool
    verification_requires_evidence: bool
    verification_evidence_ready: bool
    verification_evidence_summary: str
    verification_last_note: str
    # ── turn kernel: previous step policy ──────────────────────────────────
    policy_phase: str
    policy_phase_index: int
    policy_remaining_steps: int | None
    policy_guidance: str
    policy_verification_focus: str
    policy_allow_widening: bool
    policy_should_compact_aggressively: bool
    # ── turn kernel: flattened model step ──────────────────────────────────
    step_type: Literal["assistant", "tool_calls"]
    step_content: str
    step_kind: str | None
    step_stop_reason: str | None
    step_block_types: list[str]
    step_ignored_block_types: list[str]
    tool_call_id: str
    # ── turn kernel: classify decision ─────────────────────────────────────
    decision_kind: Literal["progress", "retry", "fallback", "final", "tool"]
    decision_assistant_content: str
    decision_user_content: str
    decision_stop_reason: str
    decision_event_category: str
    decision_route: Literal[
        "model", "authorize", "assistant_followup", "widen", "finalize"
    ]
    # ── turn kernel: terminal ──────────────────────────────────────────────
    stop_reason: str
    stop_event_emitted: bool
    coda_summary: str


ToolExecutor = Callable[[AgentState], dict[str, Any]]
StepProvider = Callable[[AgentState], AgentStep]


@dataclass
class GraphEventSink:
    """Callback seam that lets graph nodes emit runtime events.

    The LangGraph runtime injects the production sink; tests inject fakes.
    Node event semantics mirror the retired loop's ``emit_runtime_event``:
    ``phase``/``widening`` events also drive the progress channel, ``stop``
    and ``guard`` do not.
    """

    on_runtime_event: Callable[[RuntimeEvent], None] | None = None
    on_progress_message: Callable[[str], None] | None = None
    on_assistant_message: Callable[[str], None] | None = None
    on_protect_final_answer: Callable[[str], None] | None = None

    def emit_runtime(self, event: RuntimeEvent) -> None:
        if self.on_runtime_event is not None:
            self.on_runtime_event(event)

    def progress(self, content: str) -> None:
        if self.on_progress_message is not None:
            self.on_progress_message(content)

    def assistant(self, content: str) -> None:
        if self.on_assistant_message is not None:
            self.on_assistant_message(content)

    def protect_final(self, content: str) -> None:
        if self.on_protect_final_answer is not None:
            self.on_protect_final_answer(content)


_NOOP_SINK = GraphEventSink()


def _route(state: AgentState) -> str:
    action = state.get("next_action", "complete")
    if action == "tool":
        return "authorize"
    if action == "verify":
        return "verify"
    if action == "continue":
        return "model"
    return "finalize"


def _execute_tool(state: AgentState, execute_tool: ToolExecutor) -> dict[str, Any]:
    return execute_tool(state)


def _verify(state: AgentState) -> dict[str, Any]:
    verified = bool(state.get("tool_result"))
    return {"verified": verified, "repair_requested": not verified, "next_action": "complete"}


def _finalize(_: AgentState) -> dict[str, Any]:
    return {"status": "completed"}


def _permission_route(state: AgentState) -> str:
    return "execute_tool" if state.get("permission", "allowed") == "allowed" else "finalize"


def _verification_route(state: AgentState) -> str:
    return "repair" if state.get("repair_requested") else "model"


def _model_step(state: AgentState, next_step: StepProvider) -> dict[str, Any]:
    step = next_step(state)
    if step.type == "tool_calls" and step.calls:
        call = step.calls[0]
        return {
            "next_action": "tool",
            "tool_name": call["toolName"],
            "tool_input": call["input"],
            "messages": state.get("messages", []) + [{"role": "assistant", "content": step.content}],
        }
    if step.kind == "progress" or not step.content.strip():
        return {
            "next_action": "continue",
            "messages": state.get("messages", [])
            + [{"role": "assistant_progress", "content": step.content}],
        }
    return {
        "next_action": "complete",
        "messages": state.get("messages", []) + [{"role": "assistant", "content": step.content}],
    }


# ── Turn-kernel nodes (slice 2) ─────────────────────────────────────────────
#
# Each node snapshots the flat AgentState fields into TurnRecurrentState,
# reuses the kernel function unchanged, and writes the result back as plain
# values. Counter side effects inside the kernel functions (retry counters,
# widening transitions) therefore land in the returned partial update and
# survive checkpointing.


def _step_policy_node(state: AgentState, sink: GraphEventSink) -> dict[str, Any]:
    """Loop head: budget guard, per-step policy derivation, phase events."""

    turn_state = snapshot_to_turn_state(state)
    if not turn_state.has_remaining_steps():
        message = "Reached the maximum tool step limit."
        sink.emit_runtime(
            RuntimeEvent(
                category="stop",
                message=message,
                step=turn_state.step,
                profile=turn_state.profile_name,
                stop_reason="max_steps",
            )
        )
        return {
            **turn_state_to_snapshot(turn_state),
            "stop_reason": "max_steps",
            "stop_event_emitted": True,
            "messages": state.get("messages", [])
            + [{"role": "assistant", "content": message}],
        }
    previous_policy = (
        turn_state.step_policy if turn_state.step_policy.phase_index > 0 else None
    )
    step = turn_state.begin_step()
    policy = derive_turn_step_policy(turn_state)
    message = render_turn_policy_message(
        previous_policy=previous_policy,
        current_policy=policy,
    )
    updates = turn_state_to_snapshot(turn_state)
    if message:
        sink.emit_runtime(
            RuntimeEvent(
                category="phase",
                message=message,
                step=step,
                profile=turn_state.profile_name,
                phase=policy.phase,
                verification_focus=policy.verification_focus,
            )
        )
        sink.progress(message)
    return updates


def _loop_route(state: AgentState) -> str:
    return "finalize" if state.get("stop_reason") else "model"


def _model_step_kernel(state: AgentState, next_step: StepProvider) -> dict[str, Any]:
    """Flatten one AgentStep into plain fields; keep content messages."""

    step = next_step(state)
    updates: dict[str, Any] = {
        "step_type": step.type,
        "step_content": step.content,
        "step_kind": getattr(step, "kind", None),
        "step_stop_reason": None,
        "step_block_types": [],
        "step_ignored_block_types": [],
    }
    diagnostics = getattr(step, "diagnostics", None)
    if diagnostics is not None:
        updates["step_stop_reason"] = diagnostics.stopReason
        updates["step_block_types"] = list(diagnostics.blockTypes or [])
        updates["step_ignored_block_types"] = list(diagnostics.ignoredBlockTypes or [])
    if step.type == "tool_calls" and step.calls:
        call = step.calls[0]
        updates["tool_call_id"] = call.get("id", "")
        updates["tool_name"] = call["toolName"]
        updates["tool_input"] = call["input"]
        if step.content.strip():
            if step.contentKind == "progress":
                updates["messages"] = state.get("messages", []) + [
                    {"role": "assistant_progress", "content": step.content},
                    {"role": "user", "content": NUDGE_CONTINUE},
                ]
            else:
                updates["messages"] = state.get("messages", []) + [
                    {"role": "assistant", "content": step.content}
                ]
    return updates


def _classify_step_node(state: AgentState) -> dict[str, Any]:
    """Decision hub: route tool steps to authorize, assistant steps through
    ``decide_assistant_turn`` (retry counters ride the snapshot write-back)."""

    if state.get("step_type") == "tool_calls":
        return {
            "decision_kind": "tool",
            "decision_route": "authorize",
            "decision_assistant_content": "",
            "decision_user_content": "",
            "decision_stop_reason": "",
            "decision_event_category": "",
        }

    turn_state = snapshot_to_turn_state(state)
    content = state.get("step_content", "")
    is_empty = is_empty_assistant_response(content)
    step_policy = (
        turn_state.step_policy if turn_state.step_policy.phase_index > 0 else None
    )
    decision = decide_assistant_turn(
        turn_state=turn_state,
        step_content=content,
        step_kind=state.get("step_kind"),
        stop_reason=state.get("step_stop_reason"),
        block_types=state.get("step_block_types"),
        ignored_block_types=state.get("step_ignored_block_types"),
        is_empty=is_empty,
        treat_as_progress=should_treat_assistant_as_progress(
            kind=state.get("step_kind"),
            content=content,
            saw_tool_result=turn_state.saw_tool_result,
        ),
        is_recoverable_thinking_stop=is_recoverable_thinking_stop(
            is_empty=is_empty,
            stop_reason=state.get("step_stop_reason"),
            ignored_block_types=state.get("step_ignored_block_types"),
        ),
        format_diagnostics=format_diagnostics,
        nudge_continue=NUDGE_CONTINUE,
        nudge_after_tool_result=NUDGE_AFTER_TOOL_RESULT,
        resume_after_pause=RESUME_AFTER_PAUSE,
        resume_after_max_tokens=RESUME_AFTER_MAX_TOKENS,
        nudge_after_empty_response=NUDGE_AFTER_EMPTY_RESPONSE,
        nudge_after_empty_no_tools=NUDGE_AFTER_EMPTY_NO_TOOLS,
        step_policy=step_policy,
    )
    route: Literal["assistant_followup", "widen", "finalize"]
    if decision.kind in {"progress", "retry"}:
        route = "assistant_followup"
    elif decision.kind == "fallback" and decision.stop_reason == "widen_needed":
        route = "widen"
    else:
        route = "finalize"
    return {
        **turn_state_to_snapshot(turn_state),
        "decision_kind": decision.kind,
        "decision_assistant_content": decision.assistant_content or "",
        "decision_user_content": decision.user_content or "",
        "decision_stop_reason": decision.stop_reason or "",
        "decision_event_category": decision.runtime_event_category or "",
        "decision_route": route,
    }


def _classify_route(state: AgentState) -> str:
    return state.get("decision_route", "finalize")


def _assistant_followup_node(state: AgentState, sink: GraphEventSink) -> dict[str, Any]:
    """Append progress/retry message pairs and loop back to the policy head."""

    messages = list(state.get("messages", []))
    kind = state.get("decision_kind", "progress")
    assistant_content = state.get("decision_assistant_content", "")
    user_content = state.get("decision_user_content", "")
    category = state.get("decision_event_category", "")
    if assistant_content:
        if category:
            turn_state = snapshot_to_turn_state(state)
            sink.emit_runtime(
                RuntimeEvent(
                    category=category,  # type: ignore[arg-type]
                    message=assistant_content,
                    step=turn_state.step,
                    profile=turn_state.profile_name,
                    phase=turn_state.step_policy.phase,
                    verification_focus=turn_state.step_policy.verification_focus,
                    evidence_summary=(
                        turn_state.verification_state.evidence_summary
                        or turn_state.latest_tool_result_summary
                    ),
                )
            )
        else:
            sink.progress(assistant_content)
        messages.append(
            {
                "role": "assistant_progress" if kind == "progress" else "assistant",
                "content": assistant_content,
            }
        )
    if user_content:
        messages.append({"role": "user", "content": user_content})
    return {"messages": messages}


def _widen_node(state: AgentState, sink: GraphEventSink) -> dict[str, Any]:
    """Widening transition: bump the budget once, nudge, resume the loop.

    ``activate_widening`` is idempotent — a second widen-needed decision falls
    through to a typed stop instead of looping.
    """

    turn_state = snapshot_to_turn_state(state)
    bonus = int(state.get("widening_step_bonus", 0) or 0)
    transitioned = turn_state.activate_widening(extra_steps=bonus)
    messages = list(state.get("messages", []))
    if not transitioned:
        stop_reason = state.get("decision_stop_reason") or "widen_needed"
        turn_state.set_stop_reason(stop_reason)  # type: ignore[arg-type]
        fallback_content = state.get("decision_assistant_content", "")
        if fallback_content:
            messages.append({"role": "assistant", "content": fallback_content})
        return {
            **turn_state_to_snapshot(turn_state),
            "stop_reason": stop_reason,
            "messages": messages,
        }
    widening_message = state.get("decision_assistant_content", "") or (
        "Depth stalled; switching to widened mode."
    )
    if turn_state.widening_trigger_reason:
        widening_message += f" Escalation trigger: {turn_state.widening_trigger_reason}."
    sink.emit_runtime(
        RuntimeEvent(
            category="widening",
            message=widening_message,
            step=turn_state.step,
            profile=turn_state.profile_name,
            widening_reason=turn_state.widening_trigger_reason,
            evidence_summary=turn_state.widening_trigger_evidence,
        )
    )
    sink.progress(widening_message)
    messages.append({"role": "assistant_progress", "content": widening_message})
    messages.append(
        {
            "role": "user",
            "content": build_widening_transition_nudge(
                turn_state.latest_tool_result_summary,
                widening_reason=turn_state.widening_trigger_reason,
                widening_evidence_summary=turn_state.widening_trigger_evidence,
            ),
        }
    )
    return {**turn_state_to_snapshot(turn_state), "messages": messages}


def _post_widen_route(state: AgentState) -> str:
    return "finalize" if state.get("stop_reason") else "step_policy"


def _execute_tool_kernel(state: AgentState, execute_tool: ToolExecutor) -> dict[str, Any]:
    result = execute_tool(state)
    updates: dict[str, Any] = dict(result)
    updates.setdefault("tool_result", "")
    updates.setdefault("tool_result_ok", True)
    updates.setdefault("tool_await_user", False)
    updates.setdefault("messages", state.get("messages", []))
    updates.setdefault(
        "tool_summary",
        f"{state.get('tool_name', 'tool')}: {str(updates.get('tool_result', ''))[:200]}",
    )
    return updates


def _observe_tool_node(state: AgentState, sink: GraphEventSink) -> dict[str, Any]:
    """Record the observation, honor await_user pauses."""

    turn_state = snapshot_to_turn_state(state)
    ok = bool(state.get("tool_result_ok", True))
    summary = state.get("tool_summary", "")
    turn_state.record_tool_result(ok, summary=summary or None)
    decision = decide_tool_turn(
        tool_name=state.get("tool_name", ""),
        result_output=state.get("tool_result", ""),
        await_user=bool(state.get("tool_await_user", False)),
    )
    updates = turn_state_to_snapshot(turn_state)
    if decision.kind == "await_user":
        turn_state.set_stop_reason("await_user")
        message = decision.assistant_content or state.get("tool_result", "")
        sink.emit_runtime(
            RuntimeEvent(
                category="stop",
                message=message,
                step=turn_state.step,
                profile=turn_state.profile_name,
                stop_reason="await_user",
                evidence_summary=turn_state.latest_tool_result_summary,
            )
        )
        sink.assistant(message)
        updates = turn_state_to_snapshot(turn_state)
        updates["stop_reason"] = "await_user"
        updates["stop_event_emitted"] = True
        updates["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": message}
        ]
    return updates


def _observe_route(state: AgentState) -> str:
    return "finalize" if state.get("stop_reason") == "await_user" else "verify"


def _verify_kernel(state: AgentState, has_repair: bool) -> dict[str, Any]:
    """Evidence-aware verification gate.

    Without an injected repair callback a failed gate must not kill the turn —
    the retired loop kept going after tool errors, so the graph only requests
    repair when a repair service actually exists.
    """

    ok = bool(state.get("tool_result_ok", True))
    turn_state = snapshot_to_turn_state(state)
    requires_evidence = turn_state.verification_state.requires_evidence
    evidence_ready = turn_state.verification_state.evidence_ready
    verified = ok and (not requires_evidence or evidence_ready)
    return {
        "verified": verified,
        "repair_requested": (not verified) and has_repair,
    }


def _verify_kernel_route(state: AgentState) -> str:
    return "repair" if state.get("repair_requested") else "step_policy"


def _finalize_kernel(state: AgentState, sink: GraphEventSink) -> dict[str, Any]:
    """Resolve the terminal state, emit the stop event, build the coda."""

    turn_state = snapshot_to_turn_state(state)
    messages = list(state.get("messages", []))
    kind = state.get("decision_kind", "")
    stop_reason = turn_state.stop_reason
    terminal_message = ""
    if not stop_reason:
        if kind == "final":
            stop_reason = "done"
        elif kind == "fallback":
            stop_reason = state.get("decision_stop_reason") or "blocked"
        else:
            stop_reason = "blocked"
        turn_state.set_stop_reason(stop_reason)  # type: ignore[arg-type]
    content = state.get("decision_assistant_content", "")
    if kind in {"final", "fallback"} and content:
        terminal_message = content
        messages.append({"role": "assistant", "content": content})
        sink.assistant(content)
        if kind == "final":
            sink.protect_final(content[:500])
    if not state.get("stop_event_emitted"):
        sink.emit_runtime(
            RuntimeEvent(
                category="stop",
                message=terminal_message or "Turn stopped.",
                step=turn_state.step,
                profile=turn_state.profile_name,
                phase=turn_state.step_policy.phase,
                stop_reason=stop_reason or "",
                evidence_summary=(
                    turn_state.verification_state.evidence_summary
                    or turn_state.latest_tool_result_summary
                ),
            )
        )
    coda = build_turn_coda_summary(turn_state=turn_state, context_usage=0.0)
    completed = stop_reason in {"done", "await_user"} or not stop_reason
    return {
        "status": "completed" if completed else "failed",
        "stop_reason": stop_reason or "",
        "messages": messages,
        "coda_summary": coda.result_summary,
    }


def build_model_graph(
    *,
    next_step: StepProvider,
    execute_tool: ToolExecutor,
    checkpointer: Any = None,
    authorize_tool: Callable[[AgentState], Literal["allowed", "denied", "pending"]] | None = None,
    load_context: Callable[[AgentState], str] | None = None,
    compact_context: Callable[[AgentState], dict[str, Any]] | None = None,
    repair: Callable[[AgentState], dict[str, Any]] | None = None,
    event_sink: GraphEventSink | None = None,
    turn_kernel: bool = False,
):
    """Build the model-driven graph.

    ``turn_kernel=False`` keeps the slice-1 thin topology (the escape hatch
    selected by ``runtime={"turnKernel": "thin"}``). ``turn_kernel=True``
    wires the turn-kernel topology: the loop head lives in ``step_policy``,
    assistant reactions flow through ``classify_step``/``assistant_followup``/
    ``widen``, and all kernel fields stay plain serializable values.
    """

    sink = event_sink or _NOOP_SINK
    graph = StateGraph(AgentState)
    graph.add_node("load_context", lambda state: {"memory_context": load_context(state) if load_context else ""})
    graph.add_node("compact", lambda state: (compact_context(state) if compact_context else {"compacted": False}))

    if not turn_kernel:
        graph.add_node("model", lambda state: _model_step(state, next_step))
        graph.add_node("authorize", lambda state: {"permission": authorize_tool(state) if authorize_tool else "allowed"})
        graph.add_node("execute_tool", lambda state: _execute_tool(state, execute_tool))
        graph.add_node("verify", _verify)
        graph.add_node("repair", lambda state: repair(state) if repair else {"status": "failed"})
        graph.add_node("finalize", _finalize)
        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "compact")
        graph.add_edge("compact", "model")
        graph.add_conditional_edges("model", _route)
        graph.add_edge("execute_tool", "verify")
        graph.add_conditional_edges("authorize", _permission_route)
        graph.add_conditional_edges("verify", _verification_route)
        graph.add_edge("repair", "model")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=checkpointer)

    graph.add_node("step_policy", lambda state: _step_policy_node(state, sink))
    graph.add_node("model", lambda state: _model_step_kernel(state, next_step))
    graph.add_node("classify_step", _classify_step_node)
    graph.add_node(
        "assistant_followup", lambda state: _assistant_followup_node(state, sink)
    )
    graph.add_node("widen", lambda state: _widen_node(state, sink))
    graph.add_node(
        "authorize",
        lambda state: {"permission": authorize_tool(state) if authorize_tool else "allowed"},
    )
    graph.add_node(
        "execute_tool", lambda state: _execute_tool_kernel(state, execute_tool)
    )
    graph.add_node("observe_tool", lambda state: _observe_tool_node(state, sink))
    graph.add_node("verify", lambda state: _verify_kernel(state, repair is not None))
    graph.add_node("repair", lambda state: repair(state) if repair else {"status": "failed"})
    graph.add_node("finalize", lambda state: _finalize_kernel(state, sink))
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "compact")
    graph.add_edge("compact", "step_policy")
    graph.add_conditional_edges("step_policy", _loop_route)
    graph.add_edge("model", "classify_step")
    graph.add_conditional_edges("classify_step", _classify_route)
    graph.add_conditional_edges("authorize", _permission_route)
    graph.add_edge("execute_tool", "observe_tool")
    graph.add_conditional_edges("observe_tool", _observe_route)
    graph.add_conditional_edges("verify", _verify_kernel_route)
    graph.add_edge("repair", "step_policy")
    graph.add_edge("assistant_followup", "step_policy")
    graph.add_conditional_edges("widen", _post_widen_route)
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def build_agent_graph(*, execute_tool: ToolExecutor):
    """Build the first LangGraph runtime graph without changing old entrypoints."""

    graph = StateGraph(AgentState)
    graph.add_node("route", lambda state: {})
    graph.add_node("authorize", lambda state: {"permission": "allowed"})
    graph.add_node("execute_tool", lambda state: _execute_tool(state, execute_tool))
    graph.add_node("verify", _verify)
    graph.add_node("finalize", _finalize)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _route)
    graph.add_conditional_edges("authorize", _permission_route)
    graph.add_edge("execute_tool", "verify")
    graph.add_edge("verify", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
