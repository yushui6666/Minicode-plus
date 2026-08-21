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
    progress_summary: str
    tool_result_ok: bool
    tool_await_user: bool
    tool_summary: str
    # ── turn kernel: multi-call batches (slice 3) ──────────────────────────
    step_calls: list[dict[str, Any]]
    tool_results_batch: list[dict[str, Any]]
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
    and ``guard`` do not. The tool seams carry the *deferred* callbacks for
    concurrency-safe tools whose execution ran inside the parallel phase —
    serial tools fire their callbacks at execution time in the runtime.
    """

    on_runtime_event: Callable[[RuntimeEvent], None] | None = None
    on_progress_message: Callable[[str], None] | None = None
    on_assistant_message: Callable[[str], None] | None = None
    on_protect_final_answer: Callable[[str], None] | None = None
    on_tool_start: Callable[[str, dict[str, Any]], None] | None = None
    on_tool_result: Callable[[str, str, bool], None] | None = None

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

    def tool_start(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        if self.on_tool_start is not None:
            self.on_tool_start(tool_name, tool_input)

    def tool_result(self, tool_name: str, output: str, is_error: bool) -> None:
        if self.on_tool_result is not None:
            self.on_tool_result(tool_name, output, is_error)


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


def _authorize_node(
    state: AgentState,
    authorize_tool: Callable[[AgentState], Literal["allowed", "denied", "pending"]] | None,
) -> dict[str, Any]:
    """Authorize every pending call; one denial denies the batch."""

    if authorize_tool is None:
        return {"permission": "allowed"}
    for call in state.get("step_calls", []):
        call_view: AgentState = {
            **state,
            "tool_call_id": call.get("id", ""),
            "tool_name": call["toolName"],
            "tool_input": call["input"],
        }
        if authorize_tool(call_view) != "allowed":
            return {"permission": "denied"}
    return {"permission": "allowed"}


def _model_step_kernel(state: AgentState, next_step: StepProvider, sink: GraphEventSink | None = None) -> dict[str, Any]:
    """Flatten one AgentStep into plain fields; keep content messages.

    Tool steps keep the whole ``calls`` list in ``step_calls`` — the batch
    executor runs every call (slice 3); the singular fields stay populated
    from the first call for debugging and the slice-1 demo graph.
    """

    step = next_step(state)
    updates: dict[str, Any] = {
        "step_type": step.type,
        "step_content": step.content,
        "step_kind": getattr(step, "kind", None),
        "step_stop_reason": None,
        "step_block_types": [],
        "step_ignored_block_types": [],
        "step_calls": [],
    }
    diagnostics = getattr(step, "diagnostics", None)
    if diagnostics is not None:
        updates["step_stop_reason"] = diagnostics.stopReason
        updates["step_block_types"] = list(diagnostics.blockTypes or [])
        updates["step_ignored_block_types"] = list(diagnostics.ignoredBlockTypes or [])
    if step.type == "tool_calls" and step.calls:
        updates["step_calls"] = [dict(call) for call in step.calls]
        call = step.calls[0]
        updates["tool_call_id"] = call.get("id", "")
        updates["tool_name"] = call["toolName"]
        updates["tool_input"] = call["input"]
        if step.content.strip():
            if step.contentKind == "progress":
                if sink is not None:
                    try:
                        sink.progress(step.content)
                    except Exception:
                        pass
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
    ``decide_assistant_turn`` (retry counters ride the snapshot write-back).
    Provider failures arrive as ``kind="error"`` steps and resolve to a
    typed blocked stop with the fallback message, mirroring the retired
    loop's model-API error handling."""

    if state.get("step_type") == "tool_calls":
        return {
            "decision_kind": "tool",
            "decision_route": "authorize",
            "decision_assistant_content": "",
            "decision_user_content": "",
            "decision_stop_reason": "",
            "decision_event_category": "",
        }

    if state.get("step_kind") == "error":
        return {
            "decision_kind": "fallback",
            "decision_route": "finalize",
            "decision_assistant_content": state.get("step_content", ""),
            "decision_user_content": "",
            "decision_stop_reason": "blocked",
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
            # Legacy emit_runtime_event also forwarded to progress (emit_progress=True)
            # so progress subscribers see recovery/guard messages (pause_turn, verification guard).
            try:
                sink.progress(assistant_content)
            except Exception:
                pass
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
    if not updates.get("tool_results_batch"):
        # Legacy single-call executors (and the slice-1 demo graph fakes)
        # return singular fields only — normalize into a one-entry batch so
        # the observe node sees one shape. ``tool_result`` already carries
        # the noted output in that contract.
        noted = str(updates.get("tool_result", ""))
        updates["tool_results_batch"] = [
            {
                "id": state.get("tool_call_id", ""),
                "toolName": state.get("tool_name", ""),
                "input": state.get("tool_input", {}),
                "ok": bool(updates.get("tool_result_ok", True)),
                "output": noted,
                "content": noted,
                "awaitUser": bool(updates.get("tool_await_user", False)),
                "concurrent": False,
            }
        ]
    return updates


def _observe_tool_node(state: AgentState, sink: GraphEventSink) -> dict[str, Any]:
    """Record observations for every executed call, honor await_user pauses.

    Results are processed in the model's original call order (the executor
    pre-sorts). Per result: deferred callbacks for concurrent tools, kernel
    recording, the tool decision, and the ``assistant_tool_call`` +
    ``tool_result`` message pair. The FIRST await_user decision stops the
    turn immediately — later calls' pairs are not appended (their tools
    already ran in the execution phase), mirroring the retired loop.
    """

    turn_state = snapshot_to_turn_state(state)
    messages = list(state.get("messages", []))
    batch = state.get("tool_results_batch", [])
    updates: dict[str, Any] = {}
    for entry in batch:
        tool_name = str(entry.get("toolName", ""))
        raw_output = str(entry.get("output", ""))
        noted_output = str(entry.get("content", raw_output))
        ok = bool(entry.get("ok", True))
        if entry.get("concurrent"):
            sink.tool_start(tool_name, dict(entry.get("input", {})))
        tool_summary = f"{tool_name}: {raw_output[:200]}"
        turn_state.record_tool_result(ok, summary=tool_summary or None)
        decision = decide_tool_turn(
            tool_name=tool_name,
            result_output=raw_output,
            await_user=bool(entry.get("awaitUser", False)),
        )
        if decision.progress_summary:
            turn_state.set_progress_summary(decision.progress_summary)
        if entry.get("concurrent"):
            sink.tool_result(tool_name, raw_output, not ok)
        messages.append(
            {
                "role": "assistant_tool_call",
                "toolUseId": str(entry.get("id", "") or ""),
                "toolName": tool_name,
                "input": entry.get("input", {}),
            }
        )
        messages.append(
            {
                "role": "tool_result",
                "toolUseId": str(entry.get("id", "") or ""),
                "toolName": tool_name,
                "content": noted_output,
                "isError": not ok,
            }
        )
        if decision.kind == "await_user":
            turn_state.set_stop_reason("await_user")
            message = decision.assistant_content or noted_output
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
            messages.append({"role": "assistant", "content": message})
            updates = turn_state_to_snapshot(turn_state)
            updates["stop_reason"] = "await_user"
            updates["stop_event_emitted"] = True
            updates["tool_result_ok"] = ok
            updates["tool_result"] = noted_output
            updates["tool_summary"] = tool_summary
            updates["messages"] = messages
            return updates
    updates = turn_state_to_snapshot(turn_state)
    if batch:
        last = batch[-1]
        updates["tool_result_ok"] = all(bool(e.get("ok", True)) for e in batch)
        last_output = str(last.get("content", last.get("output", "")))
        updates["tool_result"] = last_output
        updates["tool_summary"] = f"{last.get('toolName', 'tool')}: {str(last.get('output', ''))[:200]}"
    return {**updates, "messages": messages}


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
):
    """Build the model-driven graph with the turn-kernel topology.

    The loop head lives in ``step_policy``, assistant reactions flow through
    ``classify_step``/``assistant_followup``/``widen``, and all kernel fields
    stay plain serializable values. The slice-1 thin topology and its
    ``runtime={"turnKernel": "thin"}`` escape hatch were removed in slice 3.
    """

    sink = event_sink or _NOOP_SINK
    graph = StateGraph(AgentState)
    graph.add_node("load_context", lambda state: {"memory_context": load_context(state) if load_context else ""})
    graph.add_node("compact", lambda state: (compact_context(state) if compact_context else {"compacted": False}))

    graph.add_node("step_policy", lambda state: _step_policy_node(state, sink))
    graph.add_node("model", lambda state: _model_step_kernel(state, next_step, sink))
    graph.add_node("classify_step", _classify_step_node)
    graph.add_node(
        "assistant_followup", lambda state: _assistant_followup_node(state, sink)
    )
    graph.add_node("widen", lambda state: _widen_node(state, sink))
    graph.add_node(
        "authorize",
        lambda state: _authorize_node(state, authorize_tool),
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
