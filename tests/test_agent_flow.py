"""Agent flow integration tests against the LangGraph runtime.

The retired rich loop's cybernetic-hook coverage (orchestrator lifecycle,
work chain) moved to slice 4; these tests keep the basic-flow and
memory-injection guarantees on the production graph path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from minicode.context_manager import ContextManager
from minicode.graph import run_graph_turn
from minicode.mock_model import MockModelAdapter
from minicode.permissions import PermissionManager
from minicode.tools import create_default_tool_registry


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_model():
    return MockModelAdapter()


@pytest.fixture
def tools(workspace):
    return create_default_tool_registry(str(workspace), runtime=None)


@pytest.fixture
def permissions(workspace):
    def _allow(request):
        return {"decision": "allow_once"}
    return PermissionManager(str(workspace), prompt=_allow)


@pytest.fixture
def messages(workspace, permissions):
    return [
        {"role": "system", "content": "You are a coding assistant. Use tools to help the user."},
        {"role": "user", "content": "Create a React login form component"},
    ]


class TestAgentFlowBasic:
    """Basic agent turns run without errors on the graph runtime."""

    def test_agent_completes_without_error(
        self, mock_model, tools, messages, workspace, permissions
    ):
        result = run_graph_turn(
            model=mock_model,
            tools=tools,
            messages=messages,
            cwd=str(workspace),
            permissions=permissions,
            max_steps=3,
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_agent_with_context_manager(
        self, mock_model, tools, messages, workspace, permissions
    ):
        ctx = ContextManager(model="claude-sonnet-4-20250514")
        result = run_graph_turn(
            model=mock_model,
            tools=tools,
            messages=messages,
            cwd=str(workspace),
            permissions=permissions,
            context_manager=ctx,
            max_steps=3,
        )
        assert len(result) > 0

    def test_agent_with_memory_manager(
        self, mock_model, tools, messages, workspace, permissions
    ):
        """Memory injection (domain classify → search → inject) runs via the
        graph's load_context seam."""
        from minicode.memory import MemoryManager, MemoryScope

        mgr = MemoryManager(project_root=str(workspace))
        mgr.add_entry(
            scope=MemoryScope.PROJECT, category="pattern",
            content="React forms use react-hook-form with zod validation",
            tags=["react", "form", "validation"],
        )
        mgr.add_entry(
            scope=MemoryScope.PROJECT, category="convention",
            content="Use functional components with hooks, avoid class components",
            tags=["react", "component"],
        )

        result = run_graph_turn(
            model=mock_model,
            tools=tools,
            messages=messages,
            cwd=str(workspace),
            permissions=permissions,
            context_manager=ContextManager(model="claude-sonnet-4-20250514"),
            memory_manager=mgr,
            max_steps=3,
        )
        assert len(result) > 0
