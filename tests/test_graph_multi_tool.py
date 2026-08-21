"""Multi-tool-call execution through the LangGraph turn kernel (slice 3).

The retired agent loop executed every tool call of a model step through the
ToolScheduler (concurrent phase for concurrency-safe calls, in-order serial
for the rest). These tests pin the graph runtime's zero-drift port of those
semantics plus the slice-3 defect fixes (message pairs per call, checkpoint
reset across turns, TurnEventQueue adapters).
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from minicode.graph import build_model_graph, run_graph_turn
from minicode.tooling import (
    ToolCapability,
    ToolDefinition,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)
from minicode.turn_events import TurnEventQueue
from minicode.types import AgentStep


class ScriptedModel:
    model_id = "scripted"

    def __init__(self, steps):
        self._steps = iter(steps)
        self.calls = 0

    def next(self, messages, on_stream_chunk=None, store=None):
        self.calls += 1
        return next(self._steps)


def _calls_step(pairs):
    """One tool_calls step: pairs of (id, toolName)."""
    return AgentStep(
        type="tool_calls",
        calls=[
            {"id": call_id, "toolName": tool, "input": {"n": call_id}}
            for call_id, tool in pairs
        ],
    )


def _final_step(content="all done"):
    return AgentStep(type="assistant", content=content, kind="final")


def _registry(specs: dict[str, dict[str, Any]]):
    """Build a registry: {name: {"run": fn, "safe": bool}}."""

    definitions = []
    for name, spec in specs.items():
        capabilities = (
            {ToolCapability.CONCURRENCY_SAFE} if spec.get("safe") else set()
        )
        definitions.append(
            ToolDefinition(
                name=name,
                description="test tool",
                input_schema={"type": "object"},
                validator=lambda data: data,
                run=spec["run"],
                metadata=ToolMetadata(
                    name=name,
                    description="test tool",
                    capabilities=capabilities,
                ),
            )
        )
    return ToolRegistry(definitions)


def _ok_run(output: str = "ok"):
    return lambda data, context: ToolResult(ok=True, output=f"{output}:{data['n']}")


def _roles(messages) -> list[str]:
    return [message["role"] for message in messages]


def _tool_pairs(messages) -> list[tuple[str, str, bool]]:
    return [
        (m["toolName"], m["content"], m["isError"])
        for m in messages
        if m["role"] == "tool_result"
    ]


# ── 1. all-concurrent batch ─────────────────────────────────────────────────


def test_multi_call_all_concurrent_executes_every_call_in_original_order():
    executed: list[str] = []
    specs = {
        name: {"run": lambda data, ctx, name=name: executed.append(name)
               or ToolResult(ok=True, output=f"{name}:{data['n']}"), "safe": True}
        for name in ("read_a", "read_b", "read_c")
    }
    model = ScriptedModel(
        [
            _calls_step([("c3", "read_c"), ("c1", "read_a"), ("c2", "read_b")]),
            _final_step(),
        ]
    )
    messages = run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=5,
    )

    assert sorted(executed) == ["read_a", "read_b", "read_c"]
    # Pairs keep the model's original call order, not completion order.
    assert [pair[0] for pair in _tool_pairs(messages)] == ["read_c", "read_a", "read_b"]
    assert _roles(messages)[-1] == "assistant"


# ── 2. mixed concurrent + serial ────────────────────────────────────────────


def test_multi_call_mixed_concurrent_and_serial_preserves_order():
    executed: list[str] = []
    specs = {
        "read_a": {"run": _ok_run("a"), "safe": True},
        "read_b": {"run": _ok_run("b"), "safe": True},
        "write_c": {"run": _ok_run("c"), "safe": False},
    }
    model = ScriptedModel(
        [
            _calls_step([("c1", "read_a"), ("c2", "write_c"), ("c3", "read_b")]),
            _final_step(),
        ]
    )
    messages = run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=5,
    )

    assert [pair[0] for pair in _tool_pairs(messages)] == ["read_a", "write_c", "read_b"]
    assert messages[-1]["content"] == "all done"


# ── 3. await_user in a serial position ──────────────────────────────────────


def test_multi_call_await_user_skips_later_serial_calls():
    executed: list[str] = []
    events: list[Any] = []

    def _run(name, await_user=False):
        def run(data, ctx):
            executed.append(name)
            return ToolResult(ok=True, output=f"{name}:{data['n']}", awaitUser=await_user)

        return run

    specs = {
        "serial_a": {"run": _run("serial_a"), "safe": False},
        "ask_b": {"run": _run("ask_b", await_user=True), "safe": False},
        "serial_c": {"run": _run("serial_c"), "safe": False},
    }
    model = ScriptedModel(
        [
            _calls_step([("c1", "serial_a"), ("c2", "ask_b"), ("c3", "serial_c")]),
            _final_step("never reached"),
        ]
    )
    messages = run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=5,
        on_runtime_event=events.append,
    )

    # The serial phase stops launching after the awaitUser call.
    assert executed == ["serial_a", "ask_b"]
    # First await_user wins: pairs for c1/c2, none for c3, assistant stop last.
    assert [pair[0] for pair in _tool_pairs(messages)] == ["serial_a", "ask_b"]
    assert _roles(messages)[-1] == "assistant"
    assert messages[-1]["content"] == "ask_b:c2"
    stop_events = [e for e in events if e.category == "stop"]
    assert len(stop_events) == 1
    assert stop_events[0].stop_reason == "await_user"


# ── 4. await_user inside the concurrent batch ───────────────────────────────


def test_multi_call_await_user_in_concurrent_batch_drops_later_pairs():
    executed: list[str] = []

    def _run(name, await_user=False):
        def run(data, ctx):
            executed.append(name)
            return ToolResult(ok=True, output=f"{name}:{data['n']}", awaitUser=await_user)

        return run

    specs = {
        "read_a": {"run": _run("read_a"), "safe": True},
        "ask_b": {"run": _run("ask_b", await_user=True), "safe": True},
        "read_c": {"run": _run("read_c"), "safe": True},
    }
    model = ScriptedModel(
        [
            _calls_step([("c1", "read_a"), ("c2", "ask_b"), ("c3", "read_c")]),
            _final_step("never reached"),
        ]
    )
    messages = run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=5,
    )

    # Concurrent phase ran everything; observe short-circuits at c2.
    assert sorted(executed) == ["ask_b", "read_a", "read_c"]
    assert [pair[0] for pair in _tool_pairs(messages)] == ["read_a", "ask_b"]
    assert _roles(messages)[-1] == "assistant"


# ── 5. per-call timeout ─────────────────────────────────────────────────────


def test_multi_call_timeout_yields_error_result_and_turn_continues(monkeypatch):
    monkeypatch.setenv("MINICODE_TOOL_TIMEOUT", "1")

    def slow(data, ctx):
        time.sleep(3)
        return ToolResult(ok=True, output="too late")

    specs = {"slow_tool": {"run": slow, "safe": True}}
    model = ScriptedModel(
        [_calls_step([("c1", "slow_tool")]), _final_step("recovered")]
    )
    messages = run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=3,
    )

    pairs = _tool_pairs(messages)
    assert len(pairs) == 1
    assert pairs[0][2] is True  # isError
    assert "timed out" in pairs[0][1]
    assert messages[-1]["content"] == "recovered"


# ── 6. conflict recording serializes the next batch ────────────────────────


def test_multi_call_co_failure_records_conflict_and_serializes_next_batch():
    windows: dict[str, list[tuple[float, float]]] = {"fail_a": [], "fail_b": []}

    def failing(name):
        def run(data, ctx):
            start = time.monotonic()
            time.sleep(0.25)
            windows[name].append((start, time.monotonic()))
            return ToolResult(ok=False, output=f"{name} failed")

        return run

    specs = {
        "fail_a": {"run": failing("fail_a"), "safe": True},
        "fail_b": {"run": failing("fail_b"), "safe": True},
    }
    model = ScriptedModel(
        [
            _calls_step([("c1", "fail_a"), ("c2", "fail_b")]),
            _calls_step([("c3", "fail_a"), ("c4", "fail_b")]),
            _final_step(),
        ]
    )
    messages = run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=6,
    )

    assert len(_tool_pairs(messages)) == 4
    assert len(windows["fail_a"]) == 2
    assert len(windows["fail_b"]) == 2
    a1, a2 = windows["fail_a"]
    b1, b2 = windows["fail_b"]
    # Batch 1: no conflict history yet — both run concurrently (overlap).
    assert a1[0] < b1[1] and b1[0] < a1[1]
    # One co-failed batch double-records the pair (both directions) and the
    # frozenset key reaches the threshold of 2 — in batch 2 the conflicted
    # tool waits for the concurrent phase, so the windows no longer overlap
    # (the pair head stays concurrent, as in the retired loop).
    assert b2[0] >= a2[1] - 0.05


# ── 7. budget semantics ─────────────────────────────────────────────────────


def test_multi_call_costs_exactly_one_step_in_the_budget():
    specs = {
        name: {"run": _ok_run(name), "safe": True}
        for name in ("read_a", "read_b", "read_c")
    }
    model = ScriptedModel(
        [
            _calls_step([("c1", "read_a"), ("c2", "read_b"), ("c3", "read_c")]),
            _final_step(),
        ]
    )
    run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=2,
    )

    # N calls = one model step: 3-call batch + final fits a 2-step budget.
    assert model.calls == 2


# ── 8. single-call regression ───────────────────────────────────────────────


def test_multi_call_single_call_regression_equivalence():
    fired: list[tuple[str, str]] = []

    def run(data, ctx):
        return ToolResult(ok=True, output="solo")

    specs = {"read_a": {"run": run, "safe": True}}
    model = ScriptedModel([_calls_step([("c1", "read_a")]), _final_step()])
    messages = run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=3,
        on_tool_start=lambda name, data: fired.append(("start", name)),
        on_tool_result=lambda name, out, err: fired.append(("result", name)),
    )

    assert fired == [("start", "read_a"), ("result", "read_a")]
    pairs = _tool_pairs(messages)
    assert len(pairs) == 1
    assert pairs[0] == ("read_a", "solo", False)
    assert messages[-1]["content"] == "all done"


# ── 9. aggregate ok feeds the verify gate ──────────────────────────────────


def test_multi_call_aggregate_tool_result_ok_feeds_verify_gate():
    specs = {
        "good_a": {"run": _ok_run("good"), "safe": True},
        "bad_b": {
            "run": lambda data, ctx: ToolResult(ok=False, output="bad_b exploded"),
            "safe": True,
        },
    }
    model = ScriptedModel(
        [_calls_step([("c1", "good_a"), ("c2", "bad_b")]), _final_step()]
    )
    messages = run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=4,
    )

    pairs = _tool_pairs(messages)
    assert [pair[2] for pair in pairs] == [False, True]
    assert "System note" in pairs[1][1]  # failure rides a system note
    # No repair callback exists, so a failed gate must not kill the turn.
    assert messages[-1]["content"] == "all done"


# ── 10. authorize denies the batch ──────────────────────────────────────────


def test_multi_call_authorize_denies_batch_when_any_call_denied():
    executed: list[str] = []
    sink_events: list[Any] = []
    steps = iter([_calls_step([("c1", "read_a"), ("c2", "read_b")])])

    def execute_tool(state):
        executed.append(state.get("tool_name"))
        return {"tool_results_batch": [], "tool_result": "", "tool_result_ok": True}

    def deny_second(state) -> str:
        return "denied" if state.get("tool_name") == "read_b" else "allowed"

    graph = build_model_graph(
        next_step=lambda state: next(steps),
        execute_tool=execute_tool,
        authorize_tool=deny_second,
        event_sink=None,
    )
    result = graph.invoke(
        {"messages": [], "status": "running", "max_steps": 3},
        {"recursion_limit": 64},
    )

    assert executed == []  # denial happens before any execution
    assert result["permission"] == "denied"
    assert result["stop_reason"] == "blocked"
    assert not sink_events


# ── 11. deferred concurrent callbacks fire in original order ───────────────


def test_deferred_concurrent_callbacks_fire_in_original_order():
    fired: list[tuple[str, str]] = []
    specs = {
        name: {"run": _ok_run(name), "safe": True}
        for name in ("read_a", "read_b", "read_c")
    }
    model = ScriptedModel(
        [
            _calls_step([("c3", "read_c"), ("c1", "read_a"), ("c2", "read_b")]),
            _final_step(),
        ]
    )
    run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=3,
        on_tool_start=lambda name, data: fired.append(("start", name)),
        on_tool_result=lambda name, out, err: fired.append(("result", name)),
    )

    # Deferred to the observe node, so callbacks land in original call order.
    assert fired == [
        ("start", "read_c"),
        ("result", "read_c"),
        ("start", "read_a"),
        ("result", "read_a"),
        ("start", "read_b"),
        ("result", "read_b"),
    ]


# ── 12. TurnEventQueue adapters publish mid-turn events ────────────────────


def test_turn_event_queue_adapter_publishes_mid_turn_events():
    queue = TurnEventQueue()
    specs = {"read_a": {"run": _ok_run("queue"), "safe": True}}
    model = ScriptedModel([_calls_step([("c1", "read_a")]), _final_step("queued done")])
    run_graph_turn(
        model=model,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=3,
        callbacks=queue,
    )

    kinds: list[str] = []
    queue.drain(lambda event: kinds.append(event.kind))
    assert "tool_start" in kinds
    assert "tool_result" in kinds
    # phase + stop RuntimeEvents reach the queue via the fallback seam.
    assert "runtime" in kinds
    # The final assistant message flows through the adapter too.
    assert "assistant" in kinds


# ── 13. checkpointed second turn starts fresh ───────────────────────────────


class _FakeSession:
    session_id = "two-turn-thread"


def test_second_turn_on_checkpointed_thread_starts_fresh():
    saver = InMemorySaver()
    specs = {"read_a": {"run": _ok_run("fresh"), "safe": True}}

    first = ScriptedModel([_calls_step([("c1", "read_a")]), _final_step("first answer")])
    messages_one = run_graph_turn(
        model=first,
        tools=_registry(specs),
        messages=[{"role": "user", "content": "q1"}],
        cwd=".",
        max_steps=3,
        session=_FakeSession(),
        checkpointer=saver,
    )

    second = ScriptedModel([_final_step("second answer")])
    messages_two = run_graph_turn(
        model=second,
        tools=_registry(specs),
        messages=messages_one + [{"role": "user", "content": "q2"}],
        cwd=".",
        max_steps=3,
        session=_FakeSession(),
        checkpointer=saver,
    )

    # Turn 1's stop_reason must not leak: turn 2 really calls the model.
    assert second.calls == 1
    assert messages_two[-1]["content"] == "second answer"
    assert messages_two[-1]["role"] == "assistant"


# ── 15. model API failures degrade gracefully ──────────────────────────────


def test_model_connection_error_returns_typed_fallback_message():
    class _Boom:
        model_id = "boom"

        def next(self, messages, on_stream_chunk=None, store=None):
            raise ConnectionError("Simulated network failure")

    messages = run_graph_turn(
        model=_Boom(),
        tools=_registry({"read_a": {"run": _ok_run("x"), "safe": True}}),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=3,
    )

    assert _roles(messages) == ["user", "assistant"]
    assert "network error" in messages[-1]["content"].lower()
    assert "connection" in messages[-1]["content"].lower()


def test_model_timeout_error_returns_typed_fallback_message():
    class _Slow:
        model_id = "slow"

        def next(self, messages, on_stream_chunk=None, store=None):
            raise TimeoutError("Request timed out after 60s")

    messages = run_graph_turn(
        model=_Slow(),
        tools=_registry({"read_a": {"run": _ok_run("x"), "safe": True}}),
        messages=[{"role": "user", "content": "go"}],
        cwd=".",
        max_steps=3,
    )

    assert "timeout" in messages[-1]["content"].lower()


# ── 14. progress summary round trip ─────────────────────────────────────────


def test_observe_progress_summary_round_trips_into_coda():
    specs = {"read_a": {"run": _ok_run("progress"), "safe": True}}
    steps = iter([_calls_step([("c1", "read_a")]), _final_step()])
    graph = build_model_graph(
        next_step=lambda state: next(steps),
        execute_tool=lambda state: {
            "tool_results_batch": [
                {
                    "id": state.get("tool_call_id", "c1"),
                    "toolName": state.get("tool_name", "read_a"),
                    "input": state.get("tool_input", {}),
                    "ok": True,
                    "output": "progress:c1",
                    "content": "progress:c1",
                    "awaitUser": False,
                    "concurrent": False,
                }
            ],
            "tool_result": "progress:c1",
            "tool_result_ok": True,
            "messages": state.get("messages", []),
        },
    )
    result = graph.invoke(
        {"messages": [], "status": "running", "max_steps": 3},
        {"recursion_limit": 64},
    )

    assert result["progress_summary"] == "processed tool result from read_a"
    assert result["stop_reason"] == "done"
