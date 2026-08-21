from langgraph.checkpoint.memory import InMemorySaver

from minicode.graph import (
    AgentState,
    GraphEventSink,
    build_agent_graph,
    build_model_graph,
    run_graph_turn,
)
from minicode.turn_kernel import (
    TurnRecurrentState,
    TurnVerificationState,
    snapshot_to_turn_state,
    turn_state_to_snapshot,
)
from minicode.types import AgentStep, RuntimeEvent, StepDiagnostics


def test_graph_routes_tool_request_then_verifies_and_finishes():
    calls = []

    def execute_tool(state: AgentState) -> dict:
        calls.append(state["tool_name"])
        return {"tool_result": "ok", "next_action": "verify"}

    graph = build_agent_graph(execute_tool=execute_tool)
    result = graph.invoke(
        {
            "messages": [],
            "next_action": "tool",
            "tool_name": "read_file",
            "tool_result": "",
            "verified": False,
            "status": "running",
        }
    )

    assert calls == ["read_file"]
    assert result["tool_result"] == "ok"
    assert result["verified"] is True
    assert result["status"] == "completed"


def test_graph_finishes_without_tool_when_already_complete():
    graph = build_agent_graph(execute_tool=lambda _: {"tool_result": "unexpected"})

    result = graph.invoke(
        {
            "messages": [],
            "next_action": "complete",
            "tool_name": "",
            "tool_result": "",
            "verified": True,
            "status": "running",
        }
    )

    assert result["status"] == "completed"
    assert result["tool_result"] == ""


def test_model_graph_executes_model_tool_then_returns_final_answer():
    steps = iter(
        [
            AgentStep(
                type="tool_calls",
                calls=[{"id": "1", "toolName": "read_file", "input": {}}],
            ),
            AgentStep(type="assistant", content="done", kind="final"),
        ]
    )
    executed = []

    graph = build_model_graph(
        next_step=lambda state: next(steps),
        execute_tool=lambda state: executed.append(state["tool_name"]) or {"tool_result": "file"},
    )
    result = graph.invoke({"messages": [], "status": "running"})

    assert executed == ["read_file"]
    assert result["messages"][-1]["content"] == "done"
    assert result["status"] == "completed"


def test_graph_can_use_thread_checkpoint_configuration():
    graph = build_model_graph(
        next_step=lambda _: AgentStep(type="assistant", content="done", kind="final"),
        execute_tool=lambda _: {},
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"messages": [], "status": "running"},
        {"configurable": {"thread_id": "test-thread"}},
    )

    assert result["status"] == "completed"
    assert graph.get_state({"configurable": {"thread_id": "test-thread"}}).values["status"] == "completed"


# ── Turn-kernel snapshot round trip ─────────────────────────────────────────


def test_snapshot_round_trip_preserves_kernel_fields():
    turn_state = TurnRecurrentState(
        max_steps=17,
        profile_name="single-deep",
        widen_after_step=6,
        empty_response_retry_limit=3,
        recoverable_thinking_retry_limit=5,
        saw_tool_result=True,
        empty_response_retry_count=2,
        recoverable_thinking_retry_count=1,
        tool_error_count=4,
        tool_observation_count=9,
        successful_tool_observation_count=5,
        step=11,
        widening_active=True,
        widening_transition_count=1,
        widening_trigger_reason="stalled",
        widening_trigger_evidence="evidence text",
        latest_tool_result_summary="read_file: tests passed",
        stop_reason="await_user",
        verification_state=TurnVerificationState(
            strict=True,
            requires_explicit_final=True,
            requires_evidence=True,
            evidence_ready=True,
            evidence_summary="read_file: tests passed",
            last_verification_note="note",
        ),
    )
    rebuilt = snapshot_to_turn_state(turn_state_to_snapshot(turn_state))
    assert rebuilt.max_steps == 17
    assert rebuilt.profile_name == "single-deep"
    assert rebuilt.widen_after_step == 6
    assert rebuilt.empty_response_retry_count == 2
    assert rebuilt.recoverable_thinking_retry_count == 1
    assert rebuilt.saw_tool_result is True
    assert rebuilt.tool_error_count == 4
    assert rebuilt.tool_observation_count == 9
    assert rebuilt.successful_tool_observation_count == 5
    assert rebuilt.step == 11
    assert rebuilt.widening_active is True
    assert rebuilt.widening_transition_count == 1
    assert rebuilt.widening_trigger_reason == "stalled"
    assert rebuilt.widening_trigger_evidence == "evidence text"
    assert rebuilt.latest_tool_result_summary == "read_file: tests passed"
    assert rebuilt.stop_reason == "await_user"
    assert rebuilt.verification_state.strict is True
    assert rebuilt.verification_state.requires_explicit_final is True
    assert rebuilt.verification_state.requires_evidence is True
    assert rebuilt.verification_state.evidence_ready is True
    assert rebuilt.verification_state.evidence_summary == "read_file: tests passed"
    assert rebuilt.verification_state.last_verification_note == "note"
    assert rebuilt.has_remaining_steps() is turn_state.has_remaining_steps()


def test_snapshot_defaults_reconstruct_a_fresh_turn():
    rebuilt = snapshot_to_turn_state({})
    assert rebuilt.max_steps is None
    assert rebuilt.step == 0
    assert rebuilt.profile_name == "single"
    assert rebuilt.saw_tool_result is False
    assert rebuilt.stop_reason is None
    assert rebuilt.has_remaining_steps() is True


# ── Turn-kernel graph topology ──────────────────────────────────────────────


def _recording_sink():
    events: list[RuntimeEvent] = []
    progress: list[str] = []
    assistant: list[str] = []
    protected: list[str] = []
    sink = GraphEventSink(
        on_runtime_event=events.append,
        on_progress_message=progress.append,
        on_assistant_message=assistant.append,
        on_protect_final_answer=protected.append,
    )
    return sink, events, progress, assistant, protected


def _kernel_initial(**overrides):
    state = {
        "messages": [{"role": "user", "content": "do the task"}],
        "status": "running",
        "profile_name": "single",
        "max_steps": 10,
        "empty_response_retry_limit": 2,
        "recoverable_thinking_retry_limit": 3,
        "widening_step_bonus": 0,
    }
    state.update(overrides)
    return state


def _tool_step(call_id: str = "c1", tool: str = "read_file"):
    return AgentStep(
        type="tool_calls",
        calls=[{"id": call_id, "toolName": tool, "input": {"path": "x"}}],
    )


def _tool_executor(
    executed: list[str],
    *,
    output: str = "read_file: tests passed 5/5",
    ok: bool = True,
    await_user: bool = False,
):
    def execute(state: AgentState) -> dict:
        executed.append(state["tool_name"])
        messages = state.get("messages", []) + [
            {
                "role": "assistant_tool_call",
                "toolUseId": state.get("tool_call_id", ""),
                "toolName": state["tool_name"],
                "input": state.get("tool_input", {}),
            },
            {
                "role": "tool_result",
                "toolUseId": state.get("tool_call_id", ""),
                "toolName": state["tool_name"],
                "content": output,
                "isError": not ok,
            },
        ]
        return {
            "tool_result": output,
            "tool_result_ok": ok,
            "tool_await_user": await_user,
            "tool_summary": f"{state['tool_name']}: {output[:200]}",
            "messages": messages,
        }

    return execute


def _kernel_graph(steps, executed, sink, **initial_overrides):
    scripted = iter(steps)
    graph = build_model_graph(
        next_step=lambda state: next(scripted),
        execute_tool=_tool_executor(executed),
        event_sink=sink,
        turn_kernel=True,
    )
    return graph, _kernel_initial(**initial_overrides)


def _roles(result) -> list[str]:
    return [message["role"] for message in result["messages"]]


def test_kernel_retries_empty_response_then_finalizes():
    executed: list[str] = []
    sink, events, _progress, assistant, _protected = _recording_sink()
    graph, initial = _kernel_graph(
        [
            AgentStep(type="assistant", content=""),
            AgentStep(type="assistant", content=""),
            AgentStep(type="assistant", content="Done — tests passed.", kind="final"),
        ],
        executed,
        sink,
    )
    result = graph.invoke(initial)

    assert result["stop_reason"] == "done"
    assert result["status"] == "completed"
    assert result["empty_response_retry_count"] == 2
    nudges = [
        m["content"]
        for m in result["messages"]
        if m["role"] == "user" and "last response was empty" in m["content"]
    ]
    assert len(nudges) == 2
    assert assistant[-1] == "Done — tests passed."
    assert any(e.category == "stop" and e.stop_reason == "done" for e in events)


def test_kernel_recovers_recoverable_thinking_stop():
    executed: list[str] = []
    sink, events, _progress, _assistant, _protected = _recording_sink()
    graph, initial = _kernel_graph(
        [
            AgentStep(
                type="assistant",
                content="",
                diagnostics=StepDiagnostics(
                    stopReason="max_tokens",
                    ignoredBlockTypes=["thinking"],
                ),
            ),
            AgentStep(type="assistant", content="Resumed and finished.", kind="final"),
        ],
        executed,
        sink,
    )
    result = graph.invoke(initial)

    assert result["stop_reason"] == "done"
    assert result["recoverable_thinking_retry_count"] == 1
    recovery_events = [e for e in events if e.category == "recovery"]
    assert recovery_events and "max_tokens" in recovery_events[0].message
    resume_nudges = [
        m["content"]
        for m in result["messages"]
        if m["role"] == "user" and "cut short by the token limit" in m["content"]
    ]
    assert len(resume_nudges) == 1


def test_kernel_activates_widening_with_step_bonus():
    executed: list[str] = []
    sink, events, _progress, _assistant, _protected = _recording_sink()
    graph, initial = _kernel_graph(
        [
            AgentStep(type="assistant", content=""),
            AgentStep(type="assistant", content=""),
            AgentStep(type="assistant", content=""),
            AgentStep(type="assistant", content="Wider approach done.", kind="final"),
        ],
        executed,
        sink,
        max_steps=6,
        widen_after_step=2,
        widening_step_bonus=4,
    )
    result = graph.invoke(initial)

    assert result["widening_active"] is True
    assert result["widening_transition_count"] == 1
    assert result["max_steps"] == 10
    widening_events = [e for e in events if e.category == "widening"]
    assert widening_events, "expected a widening runtime event"
    assert widening_events[0].widening_reason
    widening_nudges = [
        m["content"]
        for m in result["messages"]
        if m["role"] == "user" and "widened mode" in m["content"]
    ]
    assert len(widening_nudges) == 1
    assert result["stop_reason"] == "done"


def test_kernel_widening_is_idempotent():
    executed: list[str] = []
    sink, events, _progress, _assistant, _protected = _recording_sink()
    graph, initial = _kernel_graph(
        [AgentStep(type="assistant", content="") for _ in range(6)],
        executed,
        sink,
        max_steps=6,
        widen_after_step=2,
        widening_step_bonus=4,
    )
    result = graph.invoke(initial)

    assert result["widening_transition_count"] == 1
    assert len([e for e in events if e.category == "widening"]) == 1
    assert result["stop_reason"] == "blocked"
    assert result["status"] == "failed"


def test_kernel_withholds_final_without_evidence():
    executed: list[str] = []
    sink, events, _progress, _assistant, _protected = _recording_sink()
    graph, initial = _kernel_graph(
        [
            _tool_step(),
            AgentStep(type="assistant", content="Still working on it.", kind="progress"),
            AgentStep(type="assistant", content="Almost there.", kind="progress"),
            AgentStep(type="assistant", content="Done.", kind="final"),
            AgentStep(
                type="assistant",
                content="Done — tests passed 5/5 and the output was verified.",
                kind="final",
            ),
        ],
        executed,
        sink,
        max_steps=8,
        verification_strict=True,
    )
    result = graph.invoke(initial)

    guard_events = [e for e in events if e.category == "guard"]
    assert guard_events, "expected the verification guard to withhold the answer"
    assert "Verification guard" in guard_events[0].message
    evidence_nudges = [
        m["content"]
        for m in result["messages"]
        if m["role"] == "user" and "strict verification mode" in m["content"]
    ]
    assert len(evidence_nudges) == 1
    finals = [m for m in result["messages"] if m["role"] == "assistant"]
    assert finals[-1]["content"].startswith("Done — tests passed")
    assert result["stop_reason"] == "done"


def test_kernel_stops_at_max_steps_with_reason():
    executed: list[str] = []
    sink, events, _progress, _assistant, _protected = _recording_sink()
    graph, initial = _kernel_graph(
        [_tool_step(f"c{i}") for i in range(5)],
        executed,
        sink,
        max_steps=3,
    )
    result = graph.invoke(initial)

    assert executed == ["read_file", "read_file", "read_file"]
    assert result["stop_reason"] == "max_steps"
    assert result["status"] == "failed"
    assert "Reached the maximum tool step limit." in [
        m["content"] for m in result["messages"] if m["role"] == "assistant"
    ]
    assert any(e.category == "stop" and e.stop_reason == "max_steps" for e in events)
    assert "max step budget" in result["coda_summary"]


def test_kernel_phase_events_progress_explore_execute_verify():
    executed: list[str] = []
    sink, events, _progress, _assistant, _protected = _recording_sink()
    graph, initial = _kernel_graph(
        [_tool_step(f"c{i}") for i in range(6)]
        + [AgentStep(type="assistant", content="Verified and done.", kind="final")],
        executed,
        sink,
        max_steps=10,
    )
    result = graph.invoke(initial)

    phases = [e.phase for e in events if e.category == "phase" and e.step > 0]
    assert phases == ["explore", "execute", "verify"]
    assert result["stop_reason"] == "done"


def test_kernel_await_user_tool_pauses_turn():
    executed: list[str] = []
    sink, events, _progress, assistant, _protected = _recording_sink()
    scripted = iter([_tool_step()])
    graph = build_model_graph(
        next_step=lambda state: next(scripted),
        execute_tool=_tool_executor(
            executed, output="read_file: needs your input", await_user=True
        ),
        event_sink=sink,
        turn_kernel=True,
    )
    result = graph.invoke(_kernel_initial())

    assert result["stop_reason"] == "await_user"
    assert result["status"] == "completed"
    assert assistant == ["read_file: needs your input"]
    assert any(
        e.category == "stop" and e.stop_reason == "await_user" for e in events
    )
    assert _roles(result)[-1] == "assistant"


def test_kernel_state_is_checkpoint_serializable():
    executed: list[str] = []
    sink, _events, _progress, _assistant, _protected = _recording_sink()
    scripted = iter([_tool_step(), AgentStep(type="assistant", content="Done.", kind="final")])
    graph = build_model_graph(
        next_step=lambda state: next(scripted),
        execute_tool=_tool_executor(executed),
        event_sink=sink,
        turn_kernel=True,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "kernel-thread"}}
    result = graph.invoke(_kernel_initial(), config)

    snapshot = graph.get_state(config).values
    assert isinstance(snapshot["step"], int)
    assert isinstance(snapshot["stop_reason"], str)
    scalar_keys = [
        "step",
        "max_steps",
        "empty_response_retry_count",
        "recoverable_thinking_retry_count",
        "tool_error_count",
        "widening_active",
        "latest_tool_result_summary",
        "verification_evidence_summary",
        "stop_reason",
    ]
    for key in scalar_keys:
        assert snapshot[key] is None or isinstance(snapshot[key], (int, bool, str)), key
    assert isinstance(snapshot["messages"], list)
    assert result["stop_reason"] == "done"


def test_thin_topology_escape_hatch_preserves_slice1_behavior():
    executed: list[str] = []
    sink, events, _progress, _assistant, _protected = _recording_sink()
    scripted = iter(
        [
            AgentStep(type="assistant", content=""),
            AgentStep(type="assistant", content="done", kind="final"),
        ]
    )
    graph = build_model_graph(
        next_step=lambda state: next(scripted),
        execute_tool=_tool_executor(executed),
        event_sink=sink,
        turn_kernel=False,
    )
    result = graph.invoke({"messages": [], "status": "running"})

    assert {"role": "assistant_progress", "content": ""} in result["messages"]
    assert result["status"] == "completed"
    assert events == []  # the thin topology never touches the event sink


def test_single_profile_keeps_widening_dormant():
    executed: list[str] = []
    sink, events, _progress, _assistant, _protected = _recording_sink()
    graph, initial = _kernel_graph(
        [AgentStep(type="assistant", content="") for _ in range(4)],
        executed,
        sink,
    )
    result = graph.invoke(initial)

    assert result["widening_active"] is False
    assert not [e for e in events if e.category in {"widening", "guard"}]
    assert result["stop_reason"] == "blocked"


def test_recursion_limit_allows_long_tool_turns():
    from minicode.tooling import ToolDefinition, ToolRegistry, ToolResult

    class _ScriptedModel:
        model_id = "scripted"

        def __init__(self, steps):
            self._steps = iter(steps)
            self.calls = 0

        def next(self, messages, on_stream_chunk=None, store=None):
            self.calls += 1
            return next(self._steps)

    model = _ScriptedModel(
        [_tool_step(f"c{i}") for i in range(8)]
        + [AgentStep(type="assistant", content="all done", kind="final")]
    )

    def run(data, context):
        return ToolResult(ok=True, output="ok")

    registry = ToolRegistry(
        [
            ToolDefinition(
                name="read_file",
                description="scripted",
                input_schema={"type": "object"},
                validator=lambda data: data,
                run=run,
            )
        ]
    )
    messages = run_graph_turn(
        model=model,
        tools=registry,
        messages=[{"role": "user", "content": "keep going"}],
        cwd=".",
        max_steps=10,
    )

    # 8 tool rounds plus the final answer must survive the default-25
    # superstep budget: the runtime raises recursion_limit accordingly.
    assert model.calls == 9
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "all done"
