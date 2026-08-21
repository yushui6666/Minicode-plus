"""Slice 4: authorize wiring, checkpoint resume, and agent_loop shim."""

from __future__ import annotations

import os
import warnings
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from minicode.graph import build_model_graph, run_graph_turn
from minicode.permissions import PermissionManager
from minicode.tooling import ToolCapability, ToolDefinition, ToolMetadata, ToolRegistry, ToolResult
from minicode.types import AgentStep


class ScriptedModel:
    model_id = "scripted"
    def __init__(self, steps):
        self._steps = iter(steps)
    def next(self, messages, on_stream_chunk=None, store=None):
        return next(self._steps)


def _final(content="done"):
    return AgentStep(type="assistant", content=content, kind="final")


def _calls_step(pairs):
    return AgentStep(type="tool_calls", calls=[{"id": cid, "toolName": tool, "input": {"path": "/tmp/x", "command": "rm"}} for cid, tool in pairs])


def _registry_with_tools():
    def _run(data, context):
        return ToolResult(ok=True, output="ok")
    defs = [
        ToolDefinition(name="read_file", description="r", input_schema={"type":"object"}, validator=lambda d: d, run=_run, metadata=ToolMetadata(name="read_file", description="r", capabilities=set())),
        ToolDefinition(name="run_command", description="r", input_schema={"type":"object"}, validator=lambda d: d, run=_run, metadata=ToolMetadata(name="run_command", description="r", capabilities=set())),
    ]
    return ToolRegistry(defs)


class DenyAllPermissions:
    """Minimal stub that denies every tool call via explicit RuntimeError."""
    workspace_root = "/tmp"
    def check_path_access(self, path, intent):
        raise RuntimeError("Access denied for path outside cwd: " + path)
    def check_command_run(self, cmd, args):
        raise RuntimeError("Access denied: permission denied for run_command")
    def check_file_write(self, path):
        raise RuntimeError("Access denied for path outside cwd: " + path)


def test_authorize_denies_batch_without_executing():
    executed = []
    def _run(data, context):
        executed.append(data)
        return ToolResult(ok=True, output="should not run")
    registry = ToolRegistry([
        ToolDefinition(name="read_file", description="r", input_schema={"type":"object"}, validator=lambda d: d, run=_run, metadata=ToolMetadata(name="read_file", description="r", capabilities=set())),
    ])
    model = ScriptedModel([
        AgentStep(type="tool_calls", calls=[{"id":"1","toolName":"read_file","input":{"path":"/etc/passwd"}}]),
        _final(),
    ])
    perms = DenyAllPermissions()
    # run_graph_turn should auto-build authorize from permissions and deny
    messages = [{"role":"user","content":"hi"}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_graph_turn(model=model, tools=registry, messages=messages, cwd="/tmp", permissions=perms, max_steps=5)
    # No tool execution should have happened; batch denied → finalize
    assert executed == []
    # Result should contain a deny notice or just not have tool_result; but at least not contain tool execution
    assert isinstance(result, list)
    # The graph should have stopped without tool_result message for denied batch
    roles = [m.get("role") for m in result]
    # Either no tool_result or final assistant message
    assert "tool_result" not in roles or result[-1]["role"] == "assistant"


def test_checkpoint_resumes_across_threads():
    # Use explicit InMemorySaver to verify cross-thread isolation and per-thread resume
    saver = InMemorySaver()
    # First turn on thread-a
    model_a = ScriptedModel([AgentStep(type="tool_calls", calls=[{"id":"c1","toolName":"read_file","input":{"path":"/tmp/a"}}]), _final("done-a")])
    def _run(data, context):
        return ToolResult(ok=True, output=f"ok-{data.get('path','')}")
    registry = ToolRegistry([
        ToolDefinition(name="read_file", description="r", input_schema={"type":"object"}, validator=lambda d: d, run=_run, metadata=ToolMetadata(name="read_file", description="r", capabilities=set())),
    ])
    # Need a run that goes through graph with checkpointer
    from minicode.graph.builder import build_model_graph
    from minicode.types import AgentStep as AS
    def _scripted_next(state):
        # first call returns tool, second returns final
        if state.get("step", 0) == 0:
            return AS(type="tool_calls", calls=[{"id":"c1","toolName":"read_file","input":{"path":"/tmp/a"}}])
        return AS(type="assistant", content="done-a", kind="final")
    graph = build_model_graph(next_step=_scripted_next, execute_tool=lambda s: {"tool_results_batch": [{"id":"c1","toolName":"read_file","input":{"path":"/tmp/a"},"ok":True,"output":"ok","content":"ok","awaitUser":False,"concurrent":False}], "messages": s.get("messages",[])}, checkpointer=saver)
    # Invoke thread-a first turn
    cfg_a = {"configurable": {"thread_id": "thread-a"}}
    res_a = graph.invoke({"messages":[{"role":"user","content":"hi"}],"status":"running"}, cfg_a)
    assert res_a["stop_reason"] in {"done", None} or res_a.get("status") == "completed"
    # Second turn on same thread should reset per-turn channels (slice-3 fix) and still call model
    # Use run_graph_turn with checkpointer to test second turn on same thread_id
    model_b = ScriptedModel([_final("done-b")])
    messages_b = res_a["messages"] + [{"role":"user","content":"second"}]
    result_b = run_graph_turn(model=model_b, tools=registry, messages=messages_b, cwd="/tmp", checkpointer=saver, thread_id="thread-a", max_steps=5)
    assert any(m.get("content") == "done-b" for m in result_b)
    # Different thread should start fresh (isolated)
    model_c = ScriptedModel([_final("done-c")])
    result_c = run_graph_turn(model=model_c, tools=registry, messages=[{"role":"user","content":"hi"}], cwd="/tmp", checkpointer=saver, thread_id="thread-b", max_steps=5)
    assert any(m.get("content") == "done-c" for m in result_c)


def test_agent_loop_shim_delegates_to_graph():
    # Shim should emit DeprecationWarning and produce same result as graph (opt-in via env)
    from minicode.agent_loop import run_agent_turn
    from minicode.tooling import ToolRegistry, ToolDefinition, ToolMetadata, ToolResult
    def _run(data, context):
        return ToolResult(ok=True, output="shim-ok")
    registry = ToolRegistry([
        ToolDefinition(name="read_file", description="r", input_schema={"type":"object"}, validator=lambda d: d, run=_run, metadata=ToolMetadata(name="read_file", description="r", capabilities=set())),
    ])
    model = ScriptedModel([
        AgentStep(type="tool_calls", calls=[{"id":"1","toolName":"read_file","input":{"path":"/tmp/x"}}]),
        _final("shim-done"),
    ])
    messages = [{"role":"user","content":"hi"}]
    os.environ["MINICODE_USE_GRAPH"] = "1"
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_agent_turn(model=model, tools=registry, messages=messages, cwd="/tmp", max_steps=5)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
    finally:
        os.environ.pop("MINICODE_USE_GRAPH", None)
    assert any(m.get("content") == "shim-done" for m in result)

    # Legacy fallback is default (no env) — also warns
    # Legacy path (default) should still produce result and warn
    model2 = ScriptedModel([_final("legacy-done")])
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        result2 = run_agent_turn(model=model2, tools=registry, messages=messages, cwd="/tmp", max_steps=5)
        assert any(issubclass(x.category, DeprecationWarning) for x in w2)
    assert any(m.get("content") == "legacy-done" for m in result2)
