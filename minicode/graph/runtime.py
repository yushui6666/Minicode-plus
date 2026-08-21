from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from minicode.config import MINI_CODE_DIR
from minicode.graph.builder import (
    AgentState,
    GraphEventSink,
    build_model_graph,
)
from minicode.runtime_profiles import resolve_runtime_profile
from minicode.tooling import ToolContext, ToolRegistry
from minicode.types import AgentStep, ChatMessage, ModelAdapter, RuntimeEvent
from minicode.working_memory import protect_context


def _kernel_topology_enabled(runtime: dict | None) -> bool:
    """The turn-kernel topology is the default; ``turnKernel=thin`` is the
    escape hatch back to the slice-1 topology (removed in migration slice 3)."""

    if not runtime:
        return True
    return str(runtime.get("turnKernel", "")).strip().lower() != "thin"


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
    """Run one model/tool turn through LangGraph using existing adapters."""

    profile = resolve_runtime_profile(runtime, fallback_max_steps=max_steps)
    effective_max_steps = int(profile.max_steps or max_steps)
    kernel_enabled = _kernel_topology_enabled(runtime)

    def emit_runtime(event: RuntimeEvent) -> None:
        if on_runtime_event is not None:
            on_runtime_event(event)

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

    if kernel_enabled:
        # Kernel topology: nodes own progress/assistant callbacks, so the
        # provider stays a pure AgentStep source (streaming still happens
        # inside the adapter). The thin topology keeps the slice-1 closure
        # behavior, including its immediate progress/stop event emission.
        def next_step(state: AgentState) -> AgentStep:
            return model.next(state.get("messages", []))
    else:
        steps = 0
        widened = False

        def next_step(state: AgentState) -> AgentStep:
            nonlocal steps, widened
            if steps >= effective_max_steps:
                emit_runtime(
                    RuntimeEvent(
                        category="stop",
                        message="Reached the maximum tool step limit.",
                        step=steps,
                        profile=profile.name,
                        stop_reason="max_steps",
                    )
                )
                return AgentStep(
                    type="assistant",
                    content="Reached the maximum tool step limit.",
                    kind="final",
                )
            steps += 1
            if (
                not widened
                and profile.widen_after_step is not None
                and steps >= profile.widen_after_step
            ):
                widened = True
                emit_runtime(
                    RuntimeEvent(
                        category="widening",
                        message="Runtime widened the search budget.",
                        step=steps,
                        profile=profile.name,
                        widening_reason="step-threshold",
                    )
                )
            step = model.next(state.get("messages", []))
            if step.kind == "progress":
                if on_progress_message is not None:
                    on_progress_message(step.content)
            elif step.type == "assistant" and step.content.strip():
                emit_runtime(
                    RuntimeEvent(
                        category="stop",
                        message="Agent turn completed.",
                        step=steps,
                        profile=profile.name,
                        stop_reason="done",
                    )
                )
            if callbacks is not None and step.type == "assistant":
                handler = getattr(callbacks, "on_assistant_message", None)
                if handler:
                    handler(step.content)
            if on_assistant_message is not None and step.type == "assistant":
                on_assistant_message(step.content)
            return step

    if kernel_enabled:
        def execute_tool(state: AgentState) -> dict[str, Any]:
            tool_name = state["tool_name"]
            tool_input = state.get("tool_input", {})
            tool_call_id = str(state.get("tool_call_id", "") or "")
            if callbacks is not None:
                handler = getattr(callbacks, "on_tool_start", None)
                if handler:
                    handler(tool_name, tool_input)
            if on_tool_start is not None:
                on_tool_start(tool_name, tool_input)
            result = tools.execute(
                tool_name,
                tool_input,
                ToolContext(cwd=cwd, permissions=permissions, session=session),
            )
            if callbacks is not None:
                handler = getattr(callbacks, "on_tool_result", None)
                if handler:
                    handler(tool_name, result.output, not result.ok)
            if on_tool_result is not None:
                on_tool_result(tool_name, result.output, not result.ok)
            result_output = result.output
            if not result.ok:
                result_output = (
                    f"{result.output}\n\n[System note: tool '{tool_name}' failed; "
                    "check the input, adjust the approach, or report the blocker.]"
                )
            return {
                "tool_result": result_output,
                "tool_result_ok": result.ok,
                "tool_await_user": bool(getattr(result, "awaitUser", False)),
                "tool_summary": f"{tool_name}: {result.output[:200]}",
                "messages": state.get("messages", [])
                + [
                    {
                        "role": "assistant_tool_call",
                        "toolUseId": tool_call_id,
                        "toolName": tool_name,
                        "input": tool_input,
                    },
                    {
                        "role": "tool_result",
                        "toolUseId": tool_call_id,
                        "toolName": tool_name,
                        "content": result_output,
                        "isError": not result.ok,
                    },
                ],
            }
    else:
        def execute_tool(state: AgentState) -> dict[str, Any]:
            if callbacks is not None:
                handler = getattr(callbacks, "on_tool_start", None)
                if handler:
                    handler(state["tool_name"], state.get("tool_input", {}))
            if on_tool_start is not None:
                on_tool_start(state["tool_name"], state.get("tool_input", {}))
            result = tools.execute(
                state["tool_name"],
                state.get("tool_input", {}),
                ToolContext(cwd=cwd, permissions=permissions, session=session),
            )
            if callbacks is not None:
                handler = getattr(callbacks, "on_tool_result", None)
                if handler:
                    handler(state["tool_name"], result.output, not result.ok)
            if on_tool_result is not None:
                on_tool_result(state["tool_name"], result.output, not result.ok)
            return {
                "tool_result": result.output,
                "messages": state.get("messages", [])
                + [{"role": "tool_result", "content": result.output, "toolName": state["tool_name"]}],
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
        on_progress_message=sink_progress if kernel_enabled else None,
        on_assistant_message=sink_assistant if kernel_enabled else None,
        on_protect_final_answer=sink_protect_final if kernel_enabled else None,
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
            turn_kernel=kernel_enabled,
        )
        initial_state: dict[str, Any] = {"messages": messages, "status": "running"}
        if kernel_enabled:
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
