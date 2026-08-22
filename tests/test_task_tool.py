"""Tests for the sub-agent ``task`` tool (multi-agent M1).

Covers the three M1 invariants added on top of the original tool:
- recursion is depth-capped (nested spawns rejected, tool stripped at cap)
- the parent turn's tool registry and model adapter are reused, not rebuilt
- long-running sub-agents get a declared timeout instead of the 120s default
"""
from __future__ import annotations

import pytest

from minicode.tooling import (
    ToolContext,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    resolve_tool_timeout,
)
from minicode.tools.task import _resolve_max_depth, _run, _validate, task_tool
from minicode.types import AgentStep

import os

_CWD = os.path.dirname(os.path.abspath(__file__))


def _echo_tool() -> ToolDefinition:
    def validate(input_data):
        return {"path": str(input_data.get("path", ""))}

    def run(input_data, context):
        return ToolResult(ok=True, output=f"echo:{input_data['path']}")

    return ToolDefinition(
        name="echo_file",
        description="echo the given path",
        input_schema={"type": "object", "properties": {}},
        validator=validate,
        run=run,
    )


class ScriptedAdapter:
    """Model adapter returning a fixed sequence of AgentStep.

    Records the message list seen at every next() call so tests can
    assert what the sub-agent actually observed (e.g. the tool_result
    of a rejected nested spawn).
    """

    def __init__(self, steps):
        self._steps = list(steps)
        self.seen_messages = []

    def next(self, messages, **kwargs):
        self.seen_messages.append(list(messages))
        if self._steps:
            return self._steps.pop(0)
        return AgentStep(type="assistant", content="fallback final", kind="final")


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_validate_rejects_unknown_agent_type():
    with pytest.raises(ValueError, match="agent_type"):
        _validate({"description": "do thing", "agent_type": "nope"})


def test_validate_defaults_prompt_and_agent_type():
    parsed = _validate({"description": "explore the graph runtime"})
    assert parsed["prompt"] == "explore the graph runtime"
    assert parsed["agent_type"] == "general"


# ---------------------------------------------------------------------------
# recursion guard
# ---------------------------------------------------------------------------

def test_run_rejects_spawn_at_depth_cap():
    context = ToolContext(
        cwd=_CWD,
        _runtime={"model": "fake", "subagentDepth": 1},
    )

    result = _run({"description": "nested", "agent_type": "general"}, context)

    assert result.ok is False
    assert "depth limit reached" in result.output


def test_subagent_cannot_spawn_grandchild():
    # The sub-agent tries to spawn a nested task first; the spawn tool is
    # stripped from its registry (depth 0 -> 1 with max depth 1), so the
    # call fails as an unknown tool and the sub-agent finishes normally.
    adapter = ScriptedAdapter(
        [
            AgentStep(
                type="tool_calls",
                calls=[{"id": "1", "toolName": "task", "input": {"description": "nested"}}],
            ),
            AgentStep(type="assistant", content="SUB_DONE", kind="final"),
        ]
    )
    context = ToolContext(
        cwd=_CWD,
        _runtime={
            "model": "fake",
            "subagentDepth": 0,
            "toolRegistry": ToolRegistry([task_tool, _echo_tool()]),
            "modelAdapter": adapter,
        },
    )

    result = _run({"description": "outer", "agent_type": "general"}, context)

    assert result.ok is True
    assert "SUB_DONE" in result.output
    # Exactly two model steps: the spawn attempt + the final answer. A
    # recursion would have consumed more steps or hung.
    assert len(adapter.seen_messages) == 2
    # The second model step must have observed a failed task call —
    # proof the grandchild never ran.
    tool_results = [
        m
        for messages in adapter.seen_messages
        for m in messages
        if m.get("role") == "tool_result"
    ]
    assert tool_results, "expected the nested task call to produce a tool_result"
    assert "task" in tool_results[-1].get("content", "")


# ---------------------------------------------------------------------------
# parent handle reuse
# ---------------------------------------------------------------------------

def test_run_reuses_injected_registry_and_model(monkeypatch):
    def _forbid(*args, **kwargs):
        raise AssertionError("registry/adapter must be reused, not rebuilt")

    monkeypatch.setattr("minicode.tools.create_default_tool_registry", _forbid)
    monkeypatch.setattr("minicode.model_registry.create_model_adapter", _forbid)

    adapter = ScriptedAdapter(
        [AgentStep(type="assistant", content="REUSED", kind="final")]
    )
    context = ToolContext(
        cwd=_CWD,
        _runtime={
            "model": "fake",
            "toolRegistry": ToolRegistry([_echo_tool()]),
            "modelAdapter": adapter,
        },
    )

    result = _run({"description": "quick", "agent_type": "general"}, context)

    assert result.ok is True
    assert "REUSED" in result.output


# ---------------------------------------------------------------------------
# timeout resolution
# ---------------------------------------------------------------------------

def test_resolve_tool_timeout_priority(monkeypatch):
    monkeypatch.delenv("MINICODE_TOOL_TIMEOUT", raising=False)
    registry = ToolRegistry([task_tool])

    # 1. Declared per-tool budget wins for the sub-agent tool.
    assert resolve_tool_timeout("task", registry, None) == 600
    # 2. Unknown tools fall back to the default.
    assert resolve_tool_timeout("echo_file", registry, None) == 120
    # 3. Env var raises the default for undeclared tools.
    monkeypatch.setenv("MINICODE_TOOL_TIMEOUT", "333")
    assert resolve_tool_timeout("echo_file", registry, None) == 333
    # 4. Scheduler force overrides undeclared tools, not declared ones.
    class StubScheduler:
        _force_tool_timeout = 45

    assert resolve_tool_timeout("echo_file", registry, StubScheduler()) == 45
    assert resolve_tool_timeout("task", registry, StubScheduler()) == 600


def test_task_tool_declares_long_timeout():
    assert task_tool.timeout_seconds == 600


def test_resolve_max_depth_env_parsing(monkeypatch):
    monkeypatch.delenv("MINICODE_SUBAGENT_MAX_DEPTH", raising=False)
    assert _resolve_max_depth() == 1
    monkeypatch.setenv("MINICODE_SUBAGENT_MAX_DEPTH", "2")
    assert _resolve_max_depth() == 2
    monkeypatch.setenv("MINICODE_SUBAGENT_MAX_DEPTH", "0")
    assert _resolve_max_depth() == 1  # clamped
    monkeypatch.setenv("MINICODE_SUBAGENT_MAX_DEPTH", "bogus")
    assert _resolve_max_depth() == 1

# ---------------------------------------------------------------------------
# M2: concurrency fan-out + event bubbling
# ---------------------------------------------------------------------------

def test_task_tool_is_concurrency_safe():
    assert task_tool.is_concurrency_safe is True


def test_scheduler_batches_multiple_task_calls_concurrently():
    from minicode.agent_intelligence import ToolScheduler

    registry = ToolRegistry([task_tool])
    calls = [
        {"id": "1", "toolName": "task", "input": {"description": "a"}},
        {"id": "2", "toolName": "task", "input": {"description": "b"}},
    ]
    concurrent, serial = ToolScheduler().schedule_calls(calls, registry)

    assert len(concurrent) == 2
    assert serial == []


class _FakeParentQueue:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def test_events_bubble_to_parent_queue():
    adapter = ScriptedAdapter(
        [
            AgentStep(
                type="tool_calls",
                calls=[{"id": "1", "toolName": "echo_file", "input": {"path": "x"}}],
            ),
            AgentStep(type="assistant", content="BUBBLED", kind="final"),
        ]
    )
    parent = _FakeParentQueue()
    context = ToolContext(
        cwd=_CWD,
        _runtime={
            "model": "fake",
            "toolRegistry": ToolRegistry([_echo_tool()]),
            "modelAdapter": adapter,
            "turnCallbacks": parent,
        },
    )

    result = _run({"description": "bubble"}, context)

    assert result.ok is True
    forwarded = [e for e in parent.events if e.kind == "progress"]
    assert any(
        "[sub:General]" in e.content and "echo_file" in e.content
        for e in forwarded
    ), [e.content for e in forwarded]
    # Thinking/stream chunks must not be forwarded
    assert all(e.kind == "progress" for e in parent.events)


def test_subagent_runtime_strips_checkpoint_flag(monkeypatch):
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("minicode.tools.task.run_graph_turn", _capture)
    adapter = ScriptedAdapter(
        [AgentStep(type="assistant", content="x", kind="final")]
    )
    context = ToolContext(
        cwd=_CWD,
        _runtime={
            "model": "fake",
            "graphCheckpoint": True,
            "toolRegistry": ToolRegistry([_echo_tool()]),
            "modelAdapter": adapter,
        },
    )

    result = _run({"description": "no-checkpoint"}, context)

    assert result.ok is True
    sub_runtime = captured["runtime"]
    assert "graphCheckpoint" not in sub_runtime
    assert sub_runtime["subagentDepth"] == 1


def test_resolve_max_concurrency_env(monkeypatch):
    from minicode.tools.task import _resolve_max_concurrency

    monkeypatch.delenv("MINICODE_SUBAGENT_MAX_CONCURRENCY", raising=False)
    assert _resolve_max_concurrency() == 4
    monkeypatch.setenv("MINICODE_SUBAGENT_MAX_CONCURRENCY", "8")
    assert _resolve_max_concurrency() == 8
    monkeypatch.setenv("MINICODE_SUBAGENT_MAX_CONCURRENCY", "0")
    assert _resolve_max_concurrency() == 1  # clamped
    monkeypatch.setenv("MINICODE_SUBAGENT_MAX_CONCURRENCY", "bogus")
    assert _resolve_max_concurrency() == 4
