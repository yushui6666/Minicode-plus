from __future__ import annotations

import concurrent.futures
import os
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
from minicode.types import AgentStep, ChatMessage, ModelAdapter, RuntimeEvent
from minicode.working_memory import protect_context


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
        try:
            return tools.execute(
                tool_name,
                tool_input,
                ToolContext(
                    cwd=cwd,
                    permissions=permissions,
                    session=session,
                    _runtime=runtime,
                ),
            )
        except Exception:  # noqa: BLE001
            return ToolResult(
                ok=False,
                output=f"Tool '{tool_name}' crashed: {type(exc).__name__}",
            )


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
        if on_runtime_event is not None:
            on_runtime_event(event)
            return
        # Fall back to a Rust-shaped callbacks object (e.g. TurnEventQueue)
        # so checkpointed TUI turns surface phase/widening/stop events.
        if callbacks is not None:
            handler = getattr(callbacks, "on_runtime_event", None)
            if handler:
                handler(event)

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
    def next_step(state: AgentState) -> AgentStep:
        try:
            return model.next(state.get("messages", []))
        except (KeyboardInterrupt, SystemExit):
            raise
        except ConnectionError as error:
            fallback = f"Network error (connection failed or dropped): {error}"
        except TimeoutError as error:
            fallback = f"Model API timeout: {error}"
        except Exception as error:  # noqa: BLE001 - provider failures must not crash the turn
            text = str(error)
            lowered = text.lower()
            # Keep the headless CI guidance for provider-channel failures
            # inside the turn fallback so run_headless can surface it without
            # relying on an exception bubbling out of the graph.
            if "no available channel" in lowered or "provider unavailable" in lowered:
                fallback = (
                    f"Provider availability failure: {text}. "
                    "Configure a fallback model or provider channel and retry."
                )
            else:
                fallback = f"Model API error ({type(error).__name__}): {text}"
        return AgentStep(type="assistant", content=fallback, kind="error")

    def _fire_tool_start(tool_name: str, tool_input: dict[str, Any]) -> None:
        if callbacks is not None:
            handler = getattr(callbacks, "on_tool_start", None)
            if handler:
                handler(tool_name, tool_input)
        if on_tool_start is not None:
            on_tool_start(tool_name, tool_input)

    def _fire_tool_result(tool_name: str, output: str, is_error: bool) -> None:
        if callbacks is not None:
            handler = getattr(callbacks, "on_tool_result", None)
            if handler:
                handler(tool_name, output, is_error)
        if on_tool_result is not None:
            on_tool_result(tool_name, output, is_error)

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
        if callbacks is not None:
            handler = getattr(callbacks, "on_assistant_message", None)
            if handler:
                handler(content)
        if on_assistant_message is not None:
            on_assistant_message(content)

    def sink_progress(content: str) -> None:
        if on_progress_message is not None:
            on_progress_message(content)

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
        return result["messages"]
    finally:
        if checkpoint_connection is not None:
            checkpoint_connection.close()
