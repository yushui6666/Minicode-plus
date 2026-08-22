from __future__ import annotations

import concurrent.futures
import os
import inspect
import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from minicode.agent_intelligence import ToolScheduler
from minicode.config import MINI_CODE_DIR
from minicode.graph.builder import (
    AgentState,
    GraphEventSink,
    build_model_graph,
)
from minicode.runtime_profiles import resolve_runtime_profile
from minicode.tooling import ToolContext, ToolRegistry, ToolResult
from minicode.turn_events import TurnEvent
from minicode.types import AgentStep, ChatMessage, ModelAdapter, RuntimeEvent
from minicode.working_memory import protect_context
import warnings


def _execute_single_tool(
    call: dict[str, Any],
    tools: ToolRegistry,
    cwd: str,
    permissions: Any | None,
    session: Any | None,
    runtime: dict | None,
    tool_scheduler: ToolScheduler | None = None,
) -> ToolResult:
    """Execute one tool call with timeout protection and a crash safety net.

    Port of the retired loop's worker function (``agent_loop._execute_single_tool``
    minus hooks/store/metrics): any unexpected crash in the execution pipeline
    is converted to an error ``ToolResult`` instead of killing the graph turn.
    """

    tool_name = call["toolName"]
    tool_input = call["input"]
    base_timeout = int(os.environ.get("MINICODE_TOOL_TIMEOUT", "120"))
    tool_timeout = int(
        getattr(tool_scheduler, "_force_tool_timeout", base_timeout)
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                tools.execute,
                tool_name,
                tool_input,
                ToolContext(
                    cwd=cwd,
                    permissions=permissions,
                    session=session,
                    _runtime=runtime,
                ),
            )
            try:
                return future.result(timeout=tool_timeout)
            except concurrent.futures.TimeoutError:
                return ToolResult(
                    ok=False,
                    output=f"Tool '{tool_name}' timed out after {tool_timeout}s",
                )
    except Exception as exc:  # noqa: BLE001 - a tool crash must not kill the turn
        return ToolResult(
            ok=False,
            output=f"Tool '{tool_name}' crashed: {type(exc).__name__}: {exc}",
        )



def _build_authorize_from_permissions(permissions: Any, tools: Any) -> Any:
    """Build an authorize_tool callback from PermissionManager (slice 4).

    Returns None if permissions is falsy. The callback checks each pending
    tool call via PermissionManager's public checks so a batch containing a
    denied call is routed to finalize without executing (mirrors the retired
    loop's early permission gate). Fail-open on unexpected permission errors
    to keep turns robust — deny only on explicit RuntimeError/permission
    denial.
    """
    if permissions is None:
        return None

    def authorize(state) -> str:
        for call in (state.get("step_calls") or []):
            name = call.get("toolName", "")
            inp = call.get("input", {}) or {}
            try:
                # Map tool intents to permission checks
                if name in {"run_command", "run_commands"}:
                    # run_command input: {command: str, args?: list, cwd?: str}
                    # Be lenient: try multiple shapes
                    cmd = str(inp.get("command", "") or inp.get("cmd", "") or "").strip()
                    if not cmd and isinstance(inp.get("input"), dict):
                        cmd = str(inp["input"].get("command", ""))
                    args = inp.get("args") or inp.get("arguments") or []
                    if isinstance(args, str):
                        args = args.split()
                    if hasattr(permissions, "check_command_run"):
                        try:
                            permissions.check_command_run(cmd, list(args) if isinstance(args, list) else [])
                        except TypeError:
                            # Fallback to ensure_command signature
                            pass
                    elif hasattr(permissions, "ensure_command"):
                        permissions.ensure_command(cmd, list(args) if isinstance(args, list) else [], "")
                    # If check raises, deny
                elif name in {"write_file", "read_file", "edit_file", "patch_file", "list_files", "grep_files"}:
                    target = str(inp.get("path", "") or inp.get("file", "") or inp.get("target", "") or "")
                    if target:
                        intent = {"write_file": "write", "read_file": "read", "edit_file": "edit", "patch_file": "edit", "list_files": "list", "grep_files": "read"}.get(name, "read")
                        if hasattr(permissions, "check_path_access"):
                            permissions.check_path_access(target, intent)
                        elif hasattr(permissions, "ensure_path_access"):
                            permissions.ensure_path_access(target, intent)
                        elif hasattr(permissions, "check_file_write") and intent == "write":
                            permissions.check_file_write(target)
                # Other tools: allow (no permission gate)
            except RuntimeError as exc:
                # Explicit denial from PermissionManager — deny the batch
                low = str(exc).lower()
                if "denied" in low or "outside cwd" in low or "permission" in low:
                    return "denied"
                # Other RuntimeError: fail-open to avoid false denies
                continue
            except Exception:
                # Fail-open on unexpected permission plumbing errors
                continue
        return "allowed"

    return authorize

def run_graph_turn(
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    messages: list[ChatMessage],
    cwd: str,
    permissions: Any | None = None,
    session: Any | None = None,
    max_steps: int = 50,
    thread_id: str = "default",
    checkpointer: Any = None,
    authorize_tool: Any | None = None,
    load_context: Any | None = None,
    compact_context: Any | None = None,
    repair: Any | None = None,
    callbacks: Any | None = None,
    context_manager: Any | None = None,
    memory_manager: Any | None = None,
    runtime: dict | None = None,
    store: Any | None = None,
    on_tool_start: Any | None = None,
    on_tool_result: Any | None = None,
    on_assistant_message: Any | None = None,
    on_progress_message: Any | None = None,
    on_runtime_event: Any | None = None,
) -> list[ChatMessage]:
    """Run one model/tool turn through LangGraph using existing adapters.

    ``runtime={"turnKernel": "thin"}`` is accepted but ignored: the slice-1
    thin topology escape hatch was removed in migration slice 3.
    """

    profile = resolve_runtime_profile(runtime, fallback_max_steps=max_steps)
    effective_max_steps = int(profile.max_steps or max_steps)

    def emit_runtime(event: RuntimeEvent) -> None:
        # Always publish structured runtime event first
        try:
            _publish(TurnEvent.runtime_message(step=event.step, event=event))
        except Exception:
            pass
        if on_runtime_event is not None:
            try:
                on_runtime_event(event)
            except Exception:
                pass
            # Also need to ensure progress mirroring for phase/recovery? handled at node level
            return
        # Fall back to named handler if on_event not present
        if callbacks is not None:
            try:
                handler = getattr(callbacks, "on_runtime_event", None)
                if handler and not callable(getattr(callbacks, "on_event", None)):
                    handler(event)
            except Exception:
                pass

    if load_context is None and memory_manager is not None:
        def load_context(state: AgentState) -> str:
            query = next(
                (
                    str(message.get("content", ""))
                    for message in reversed(state.get("messages", []))
                    if message.get("role") == "user"
                ),
                "",
            )
            try:
                return str(memory_manager.get_relevant_context(query=query))
            except TypeError:
                return str(memory_manager.get_relevant_context())

    if compact_context is None and context_manager is not None:
        def compact_context(state: AgentState) -> dict[str, Any]:
            current_messages = list(state.get("messages", []))
            context_manager.messages = current_messages
            should_compact = getattr(context_manager, "should_auto_compact", None)
            if callable(should_compact) and should_compact():
                compacted_messages = list(context_manager.compact_messages())
                emit_runtime(
                    RuntimeEvent(
                        category="compaction",
                        message="Context was compacted before the model step.",
                        step=0,
                        profile=profile.name,
                        phase="pre-model",
                    )
                )
                return {"messages": compacted_messages, "compacted": True}
            return {"compacted": False}

    # Kernel topology: nodes own progress/assistant callbacks, so the
    # provider stays a pure AgentStep source (streaming still happens
    # inside the adapter). Model API failures degrade gracefully — the
    # retired loop caught provider exceptions and returned a typed
    # fallback message instead of crashing the turn.

    from minicode.model_fallback import call_model_with_fallback

    def _publish_stream(chunk: str) -> None:
        try:
            _publish(TurnEvent.assistant_stream_chunk(step=None, content=chunk))
        except Exception:
            pass

    def _publish_thinking(chunk: str) -> None:
        try:
            _publish(TurnEvent.thinking_chunk(step=None, content=chunk))
        except Exception:
            pass

    def next_step(state: AgentState) -> AgentStep:
        want_stream = False
        want_thinking = False
        if callbacks is not None:
            try:
                if bool(getattr(callbacks, "include_stream_chunks", False)):
                    want_stream = True
                if bool(getattr(callbacks, "include_thinking_chunks", False)):
                    want_thinking = True
            except Exception:
                pass

        step, fallback, _switched = call_model_with_fallback(
            model=model,
            messages=state.get("messages", []),
            store=store,
            runtime=runtime,
            tools=tools,
            state=state,
            want_stream=want_stream,
            want_thinking=want_thinking,
            publish_stream=_publish_stream,
            publish_thinking=_publish_thinking,
            emit_runtime=emit_runtime,
            profile_name=profile.name,
        )
        if step is not None:
            return step
        assert fallback is not None
        return AgentStep(type="assistant", content=fallback, kind="error")

    def _publish(event: TurnEvent) -> None:
        if callbacks is not None:
            try:
                on_event = getattr(callbacks, "on_event", None)
                if callable(on_event):
                    on_event(event)
                    # Also still call legacy named handlers if they exist alongside on_event? legacy's publish does return early after on_event
                    return
                # Fallback to named handlers for callbacks object that implements AgentTurnCallbacks
                kind_map = {
                    "tool_start": "on_tool_start",
                    "tool_result": "on_tool_result",
                    "assistant": "on_assistant_message",
                    "progress": "on_progress_message",
                    "runtime": "on_runtime_event",
                }
                h = getattr(callbacks, kind_map.get(event.kind, ""), None)
                if callable(h):
                    if event.kind == "tool_start":
                        h(event.tool_name, event.tool_input)
                    elif event.kind == "tool_result":
                        h(event.tool_name, event.output, event.is_error)
                    elif event.kind == "runtime" and event.runtime_event is not None:
                        h(event.runtime_event)
                    elif event.kind in {"assistant", "progress"}:
                        h(event.content)
                    return
            except Exception:
                pass

    def _fire_tool_start(tool_name: str, tool_input: dict[str, Any]) -> None:
        # Structured event first
        try:
            _publish(TurnEvent.tool_started(step=None, tool_name=tool_name, tool_input=tool_input))
        except Exception:
            pass
        if callbacks is not None:
            # Also try direct named handler if on_event not present (legacy publish already handled)
            try:
                handler = getattr(callbacks, "on_tool_start", None)
                # Avoid double-call when on_event already handled: publish returns early, so this is only for non-on_event sinks
                if handler and not callable(getattr(callbacks, "on_event", None)):
                    handler(tool_name, tool_input)
            except Exception:
                pass
        if on_tool_start is not None:
            try:
                on_tool_start(tool_name, tool_input)
            except Exception:
                pass

    def _fire_tool_result(tool_name: str, output: str, is_error: bool) -> None:
        try:
            _publish(TurnEvent.tool_finished(step=None, tool_name=tool_name, output=output, is_error=is_error))
        except Exception:
            pass
        if callbacks is not None:
            try:
                handler = getattr(callbacks, "on_tool_result", None)
                if handler and not callable(getattr(callbacks, "on_event", None)):
                    handler(tool_name, output, is_error)
            except Exception:
                pass
        if on_tool_result is not None:
            try:
                on_tool_result(tool_name, output, is_error)
            except Exception:
                pass

    def _noted_output(tool_name: str, result: ToolResult) -> str:
        if result.ok:
            return result.output
        return (
            f"{result.output}\n\n[System note: tool '{tool_name}' failed; "
            "check the input, adjust the approach, or report the blocker.]"
        )

    tool_scheduler = ToolScheduler()

    def _batch_entry(call: dict[str, Any], result: ToolResult, concurrent: bool) -> dict[str, Any]:
        return {
            "id": str(call.get("id", "") or ""),
            "toolName": call["toolName"],
            "input": call["input"],
            "ok": result.ok,
            "output": result.output,
            "content": _noted_output(call["toolName"], result),
            "awaitUser": bool(getattr(result, "awaitUser", False)),
            "concurrent": concurrent,
        }

    def execute_tool(state: AgentState) -> dict[str, Any]:
        """Execute every pending call, replicating the retired loop's
        ToolScheduler phases: parallel for concurrency-safe calls, in-order
        serial for the rest (early break on awaitUser), results re-sorted to
        the model's original call order. Message pairs are NOT appended here
        — the observe node owns them so an await_user short-circuit can skip
        later pairs."""

        calls = [
            dict(call)
            for call in (state.get("step_calls") or [])
            if call.get("toolName")
        ]
        if not calls:
            # classify_step only routes tool steps with calls; keep a legacy
            # fallback for the singular fields just in case.
            calls = [
                {
                    "id": state.get("tool_call_id", ""),
                    "toolName": state.get("tool_name", ""),
                    "input": state.get("tool_input", {}),
                }
            ]

        results: list[tuple[dict[str, Any], ToolResult]] = []
        concurrent_ids: set[int] = set()

        if len(calls) <= 1:
            call = calls[0]
            _fire_tool_start(call["toolName"], call["input"])
            result = _execute_single_tool(
                call, tools, cwd, permissions, session, runtime, tool_scheduler
            )
            _fire_tool_result(call["toolName"], result.output, not result.ok)
            results.append((call, result))
        else:
            concurrent_calls, serial_calls = tool_scheduler.schedule_calls(calls, tools)
            concurrent_ids = {id(call) for call in concurrent_calls}

            if concurrent_calls:
                step = int(state.get("step", 0) or 0)
                tool_error_count = int(state.get("tool_error_count", 0) or 0)
                max_workers = tool_scheduler.get_recommended_max_workers(
                    concurrent_calls,
                    error_rate=tool_error_count / max(step, 1),
                    avg_latency=step * 2.0,
                    recent_failures=tool_error_count,
                )
                # Cybernetic concurrency cap (FeedbackController) pokes this
                # attribute when wired — honor it when present.
                force_cap = getattr(tool_scheduler, "_force_max_workers", None)
                if force_cap:
                    max_workers = min(max_workers, int(force_cap))
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="mc-tool",
                ) as pool:
                    future_to_call = {
                        pool.submit(
                            _execute_single_tool,
                            call,
                            tools,
                            cwd,
                            permissions,
                            session,
                            runtime,
                            tool_scheduler,
                        ): call
                        for call in concurrent_calls
                    }
                    # No UI callbacks during the concurrent phase — they are
                    # deferred to the observe node in original call order.
                    for future in concurrent.futures.as_completed(future_to_call):
                        call = future_to_call[future]
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001
                            result = ToolResult(
                                ok=False,
                                output=f"Concurrent execution error: {exc}",
                            )
                        results.append((call, result))

            for call in serial_calls:
                _fire_tool_start(call["toolName"], call["input"])
                result = _execute_single_tool(
                    call, tools, cwd, permissions, session, runtime, tool_scheduler
                )
                _fire_tool_result(call["toolName"], result.output, not result.ok)
                results.append((call, result))
                # If a serial tool awaits the user, stop launching the rest —
                # already-computed results still flow through for messages.
                if result.awaitUser:
                    break

            # Pairwise conflict recording for co-failures, in both directions
            # (the retired loop recorded each ordered pair, so one co-failed
            # batch reaches the conflict threshold immediately).
            for call, result in results:
                if not result.ok:
                    for other_call, other_result in results:
                        if other_call.get("id") == call.get("id"):
                            continue
                        if not other_result.ok:
                            tool_scheduler.record_conflict(
                                call["toolName"], other_call["toolName"]
                            )

        # Results always flow back in the model's original call order.
        call_order = {call.get("id", ""): idx for idx, call in enumerate(calls)}
        results.sort(key=lambda pair: call_order.get(pair[0].get("id", ""), 999))
        batch = [
            _batch_entry(call, result, concurrent=id(call) in concurrent_ids)
            for call, result in results
        ]
        last_call, last_result = results[-1] if results else (None, ToolResult(ok=True, output=""))
        return {
            "tool_results_batch": batch,
            "tool_result": (
                _noted_output(last_call["toolName"], last_result) if last_call else ""
            ),
            "tool_result_ok": all(result.ok for _, result in results),
            "tool_await_user": any(
                bool(getattr(result, "awaitUser", False)) for _, result in results
            ),
            "tool_summary": (
                f"{last_call['toolName']}: {last_result.output[:200]}"
                if last_call
                else ""
            ),
            "messages": state.get("messages", []),
        }

    def sink_assistant(content: str) -> None:
        try:
            _publish(TurnEvent.assistant_message(step=None, content=content))
        except Exception:
            pass
        if callbacks is not None:
            try:
                handler = getattr(callbacks, "on_assistant_message", None)
                if handler and not callable(getattr(callbacks, "on_event", None)):
                    handler(content)
            except Exception:
                pass
        if on_assistant_message is not None:
            try:
                on_assistant_message(content)
            except Exception:
                pass

    def sink_progress(content: str) -> None:
        try:
            _publish(TurnEvent.progress_message(step=None, content=content))
        except Exception:
            pass
        if callbacks is not None:
            try:
                handler = getattr(callbacks, "on_progress_message", None)
                if handler and not callable(getattr(callbacks, "on_event", None)):
                    handler(content)
            except Exception:
                pass
        if on_progress_message is not None:
            try:
                on_progress_message(content)
            except Exception:
                pass

    def sink_protect_final(content: str) -> None:
        try:
            protect_context(
                content=content,
                entry_type="key_decision",
                ttl_seconds=profile.working_memory_ttl_seconds,
                importance=profile.working_memory_importance,
            )
        except Exception:  # noqa: BLE001 - working memory must never break a turn
            pass

    event_sink = GraphEventSink(
        on_runtime_event=emit_runtime,
        on_progress_message=sink_progress,
        on_assistant_message=sink_assistant,
        on_protect_final_answer=sink_protect_final,
        on_tool_start=_fire_tool_start,
        on_tool_result=_fire_tool_result,
    )

    emit_runtime(
        RuntimeEvent(
            category="phase",
            message="Agent runtime entered model phase.",
            step=0,
            profile=profile.name,
            phase="model",
        )
    )
    # Slice-4: auto-wire authorize from PermissionManager when no explicit authorizer
    if authorize_tool is None and permissions is not None:
        built = _build_authorize_from_permissions(permissions, tools)
        if built is not None:
            authorize_tool = built

    # Slice-4: checkpoint via explicit checkpointer, session, or runtime flag
    # runtime={"graphCheckpoint": True} or env MINICODE_GRAPH_CHECKPOINT=1 enables file-backed SqliteSaver
    want_checkpoint = False
    if isinstance(runtime, dict) and runtime.get("graphCheckpoint"):
        want_checkpoint = True
    if os.environ.get("MINICODE_GRAPH_CHECKPOINT", "").strip().lower() in {"1", "true", "yes"}:
        want_checkpoint = True
    checkpoint_connection: sqlite3.Connection | None = None
    effective_checkpointer = checkpointer
    effective_thread_id = thread_id
    if session is not None:
        effective_thread_id = str(getattr(session, "session_id", thread_id) or thread_id)
        if effective_checkpointer is None:
            MINI_CODE_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_connection = sqlite3.connect(
                MINI_CODE_DIR / "langgraph-checkpoints.sqlite3",
                check_same_thread=False,
            )
            effective_checkpointer = SqliteSaver(checkpoint_connection)
    elif want_checkpoint and effective_checkpointer is None:
        MINI_CODE_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_connection = sqlite3.connect(
            MINI_CODE_DIR / "langgraph-checkpoints.sqlite3",
            check_same_thread=False,
        )
        effective_checkpointer = SqliteSaver(checkpoint_connection)

    try:
        graph = build_model_graph(
            next_step=next_step,
            execute_tool=execute_tool,
            checkpointer=effective_checkpointer,
            authorize_tool=authorize_tool,
            load_context=load_context,
            compact_context=compact_context,
            repair=repair,
            event_sink=event_sink,
        )
        initial_state: dict[str, Any] = {"messages": messages, "status": "running"}
        initial_state.update(
            {
                "profile_name": profile.name,
                "max_steps": effective_max_steps,
                "widen_after_step": profile.widen_after_step,
                "widening_step_bonus": profile.widening_step_bonus,
                "empty_response_retry_limit": profile.empty_response_retry_limit,
                "recoverable_thinking_retry_limit": (
                    profile.recoverable_thinking_retry_limit
                ),
                "verification_strict": profile.strict_step_verification,
            }
        )
        # Checkpointed threads continue from the previous turn's channel
        # values — every per-turn channel must be reset here or turn 2
        # inherits turn 1's stop_reason/decision fields and finalizes
        # immediately without calling the model (slice-3 defect fix).
        # Keep this key set in sync with turn_state_to_snapshot() plus the
        # decision/step/batch fields.
        initial_state.update(
            {
                "step": 0,
                "stop_reason": None,
                "stop_event_emitted": False,
                "saw_tool_result": False,
                "tool_error_count": 0,
                "tool_observation_count": 0,
                "successful_tool_observation_count": 0,
                "empty_response_retry_count": 0,
                "recoverable_thinking_retry_count": 0,
                "widening_active": False,
                "widening_transition_count": 0,
                "widening_trigger_reason": "",
                "widening_trigger_evidence": "",
                "latest_tool_result_summary": "",
                "progress_summary": "",
                "tool_result_ok": True,
                "tool_await_user": False,
                "step_type": "assistant",
                "step_content": "",
                "step_calls": [],
                "tool_results_batch": [],
                "decision_kind": "",
                "decision_assistant_content": "",
                "decision_user_content": "",
                "decision_stop_reason": "",
                "decision_event_category": "",
                "decision_route": "model",
            }
        )
        config: dict[str, Any] | None = None
        if effective_checkpointer:
            config = {"configurable": {"thread_id": effective_thread_id}}
        else:
            config = {}
        # LangGraph defaults to 25 supersteps; a kernel tool round costs up to
        # 7 nodes (step_policy/model/classify/authorize/execute/observe/verify)
        # plus back-edges, so budget for the full step limit plus the widening
        # bonus — otherwise long turns die with GraphRecursionError long before
        # the profile budget is spent.
        config["recursion_limit"] = (
            effective_max_steps
            + int(profile.widening_step_bonus or 0)
        ) * 8 + 32
        result = graph.invoke(initial_state, config)
        # Emit TurnEvent done so structured consumers see the terminal (mirrors agent_loop coda)
        try:
            stop_reason = str(result.get("stop_reason") or result.get("status") or "done")
            _publish(TurnEvent.completed(step=int(result.get("step", 0) or 0), messages=list(result.get("messages", [])), stop_reason=stop_reason))
        except Exception:
            pass
        return result["messages"]
    finally:
        if checkpoint_connection is not None:
            checkpoint_connection.close()