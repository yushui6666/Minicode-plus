"""Task tool — spawn a sub-agent to handle complex multi-step tasks.

Inspired by Claude Code's Task tool which launches an independent agent loop
with its own context window, isolated from the main conversation.

The sub-agent runs a full agent loop (model + tools) with:
- Its own system prompt tailored to the task type
- A filtered tool set based on the agent type
- A turn limit to prevent runaway execution
- Result summarized back into the parent context
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, TypedDict, cast

from minicode.graph import run_graph_turn
from minicode.tooling import (
    ToolCapability,
    ToolDefinition,
    ToolMetadata,
    ToolResult,
)
from minicode.turn_events import TurnEvent
from minicode.types import ChatMessage


# ---------------------------------------------------------------------------
# Agent type definitions
# ---------------------------------------------------------------------------

class AgentDef(TypedDict):
    name: str
    description: str
    system_prompt: str
    allowed_tools: set[str] | None
    max_turns: int


AGENT_TYPES: dict[str, AgentDef] = {
    "explore": {
        "name": "Explore",
        "description": "Fast, read-only agent for codebase exploration and search",
        "system_prompt": (
            "You are an exploration agent. Your job is to quickly search and "
            "understand codebases. You should be fast and focused on finding "
            "relevant files and understanding structure. "
            "You can only use read-only tools. "
            "When done, provide a concise summary of your findings."
        ),
        "allowed_tools": {"read_file", "list_files", "grep_files", "file_tree", "find_symbols", "find_references", "get_ast_info"},
        "max_turns": 5,
    },
    "plan": {
        "name": "Plan",
        "description": "Thorough agent for gathering context and understanding code",
        "system_prompt": (
            "You are a planning agent. Your job is to thoroughly understand "
            "the codebase and task before acting. Read multiple files, trace "
            "code paths, and build a complete mental model. "
            "You can only use read-only tools. "
            "When done, provide a detailed analysis with actionable recommendations."
        ),
        "allowed_tools": {"read_file", "list_files", "grep_files", "file_tree", "find_symbols", "find_references", "get_ast_info", "code_review"},
        "max_turns": 8,
    },
    "general": {
        "name": "General",
        "description": "Full-featured agent for complex multi-step tasks",
        "system_prompt": (
            "You are a general-purpose coding agent. You can read, write, "
            "and modify code. Follow best practices and explain your changes. "
            "Break complex tasks into smaller steps. "
            "When done, provide a summary of what you did and any important findings."
        ),
        "allowed_tools": None,  # None = all tools allowed
        "max_turns": 15,
    },
}


def _validate(input_data: dict) -> dict:
    description = input_data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required")
    
    agent_type = input_data.get("agent_type", "general")
    if agent_type not in AGENT_TYPES:
        valid = ", ".join(AGENT_TYPES.keys())
        raise ValueError(f"agent_type must be one of: {valid}. Got: {agent_type}")
    
    return {
        "description": description.strip(),
        "agent_type": agent_type,
        "prompt": input_data.get("prompt", description.strip()),
    }


def _resolve_max_depth() -> int:
    """Maximum sub-agent generation depth (1 = sub-agents cannot spawn)."""
    try:
        return max(1, int(os.environ.get("MINICODE_SUBAGENT_MAX_DEPTH", "1")))
    except ValueError:
        return 1



def _resolve_max_concurrency() -> int:
    """Global cap on simultaneously running sub-agents (fan-out width)."""
    try:
        return max(1, int(os.environ.get("MINICODE_SUBAGENT_MAX_CONCURRENCY", "4")))
    except ValueError:
        return 4


_SUBAGENT_SLOT_LOCK = threading.Lock()
_SUBAGENT_SLOTS: threading.Semaphore | None = None


def _get_subagent_slots() -> threading.Semaphore:
    """Lazily build the process-wide sub-agent slot semaphore."""
    global _SUBAGENT_SLOTS
    with _SUBAGENT_SLOT_LOCK:
        if _SUBAGENT_SLOTS is None:
            _SUBAGENT_SLOTS = threading.Semaphore(_resolve_max_concurrency())
        return _SUBAGENT_SLOTS


class _SubagentEventForwarder:
    """Bubbles a sub-agent's turn events up to the parent event queue.

    The parent turn is blocked inside the task tool while the sub-agent
    runs; forwarding a curated subset (phase progress, tool starts, tool
    failures) as prefixed progress messages keeps the TUI live instead of
    silent for the whole sub-turn. Thinking/stream chunks are deliberately
    NOT forwarded — they would interleave confusingly with the parent's
    own (already finished) streams.
    """

    def __init__(self, parent_callbacks: Any, label: str) -> None:
        self._parent = parent_callbacks
        self._prefix = f"[sub:{label}]"
        # The graph runtime inspects these flags to decide whether to
        # request stream/thinking callbacks from the model adapter; the
        # sub-agent keeps both channels off.
        self.include_stream_chunks = False
        self.include_thinking_chunks = False

    def on_event(self, event: TurnEvent) -> None:
        forward: str | None = None
        if event.kind == "progress":
            forward = event.content
        elif event.kind == "tool_start":
            forward = f"\u25b6 {event.tool_name}"
        elif event.kind == "tool_result" and event.is_error:
            forward = f"\u2717 {event.tool_name} failed"
        if forward:
            self._parent.on_event(
                TurnEvent.progress_message(step=None, content=f"{self._prefix} {forward}")
            )


def _run(input_data: dict, context) -> ToolResult:
    """Execute a sub-agent task.
    
    This creates an isolated agent loop with:
    - Its own message history (system + task prompt)
    - Filtered tools based on agent type
    - A turn limit
    - Result summarized for the parent context
    """
    from minicode.model_registry import create_model_adapter
    from minicode.permissions import PermissionManager
    from minicode.tools import create_default_tool_registry
    
    agent_type = input_data.get("agent_type", "general")
    agent_def = AGENT_TYPES[agent_type]
    task_prompt = input_data.get("prompt") or input_data.get("description", "")

    # Runtime comes from the parent turn's ToolContext when available (the
    # graph runtime injects reuse handles there), else fall back to config.
    runtime = getattr(context, "_runtime", None) or None

    if not runtime:
        try:
            from minicode.config import load_runtime_config
            runtime = load_runtime_config(context.cwd)
        except Exception:
            runtime = None

    if not runtime:
        return ToolResult(
            ok=False,
            output="Cannot run sub-agent: no model configuration available. Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL."
        )

    # Recursion guard: depth counts spawned generations (0 = top-level
    # turn). The sub-agent receives depth+1 in its runtime, so a nested
    # 'task' call is rejected — and the tool stripped from its registry —
    # once the configured cap is reached.
    depth = int(runtime.get("subagentDepth", 0) or 0)
    max_depth = _resolve_max_depth()
    if depth >= max_depth:
        return ToolResult(
            ok=False,
            output=(
                f"Sub-agent spawn rejected: depth limit reached "
                f"(depth={depth}, max={max_depth}). "
                "Complete the task directly with your own tools."
            ),
        )

    # Tool registry: reuse the parent turn's registry when injected (the
    # graph runtime provides it) instead of redoing skill discovery and
    # MCP server connections per spawn.
    parent_registry = runtime.get("toolRegistry")
    if parent_registry is not None:
        full_tools = parent_registry
    else:
        full_tools = create_default_tool_registry(context.cwd, runtime=runtime)

    allowed = agent_def["allowed_tools"]
    strip_spawn_tool = depth + 1 >= max_depth
    filtered_tools = [
        t
        for t in full_tools.list()
        if (allowed is None or t.name in allowed)
        and (not strip_spawn_tool or t.name != "task")
    ]
    from minicode.tooling import ToolRegistry
    tools = ToolRegistry(filtered_tools)

    # Model: reuse the parent turn's adapter when injected (keeps mock
    # mode and provider selection identical to the parent); fall back to
    # creating one from config.
    model = runtime.get("modelAdapter")
    if model is None or not hasattr(model, "next"):
        model = create_model_adapter(
            model=runtime.get("model", ""),
            tools=tools,
            runtime=runtime,
        )
    
    # Create isolated permissions (no prompts — auto-deny writes for read-only agents)
    if agent_def["allowed_tools"] is not None:
        # Read-only agent: create permission manager that denies writes
        sub_permissions = PermissionManager(context.cwd, prompt=None)
    else:
        # General agent: inherit parent's permission prompt handler
        sub_permissions = PermissionManager(context.cwd, prompt=getattr(context.permissions, 'prompt', None))
    
    # Build isolated message list
    sub_messages: list[ChatMessage] = cast(list[ChatMessage], [
        {
            "role": "system",
            "content": agent_def["system_prompt"]
            + f"\n\nCurrent cwd: {context.cwd}"
            + "\n\nIMPORTANT: When you have completed your task, end with <final> and provide your findings."
            + " Do not ask the user questions — work autonomously with the tools available."
            + " Be concise and focused."
        },
        {
            "role": "user",
            "content": task_prompt,
        },
    ])
    
    # Run the sub-agent loop. The sub runtime carries depth+1 and the
    # reuse handles so a nested spawn attempt is rejected at the cap.
    # graphCheckpoint is stripped: sub-agents stay checkpoint-free so a
    # parallel fan-out never contends on one sqlite checkpointer file.
    start_time = time.time()
    max_turns = agent_def["max_turns"]
    sub_runtime = {
        key: value for key, value in runtime.items() if key != "graphCheckpoint"
    }
    sub_runtime.update(
        {
            "subagentDepth": depth + 1,
            "toolRegistry": tools,
            "modelAdapter": model,
        }
    )

    # Event bubbling: forward a curated subset of the sub-agent's turn
    # events to the parent queue as prefixed progress messages.
    parent_callbacks = runtime.get("turnCallbacks")
    forwarder = None
    if parent_callbacks is not None and callable(getattr(parent_callbacks, "on_event", None)):
        forwarder = _SubagentEventForwarder(parent_callbacks, agent_def["name"])

    try:
        with _get_subagent_slots():
            result_messages = run_graph_turn(
                model=model,
                tools=tools,
                messages=sub_messages,
                cwd=context.cwd,
                permissions=sub_permissions,
                max_steps=max_turns,
                runtime=sub_runtime,
                callbacks=forwarder,
            )
    except Exception as e:
        return ToolResult(
            ok=False,
            output=f"Sub-agent ({agent_def['name']}) failed: {type(e).__name__}: {e}"
        )
    
    elapsed = time.time() - start_time
    
    # Extract the final assistant message as the result
    final_message = None
    for msg in reversed(result_messages):
        if msg.get("role") == "assistant" and msg.get("content", "").strip():
            final_message = msg["content"]
            break
    
    if not final_message:
        final_message = "(sub-agent completed without a final message)"
    
    # Build summary
    tool_calls_count = sum(1 for m in result_messages if m.get("role") == "assistant_tool_call")
    user_messages_count = sum(1 for m in result_messages if m.get("role") == "user")
    
    header = (
        f"[Sub-agent {agent_def['name']} completed]\n"
        f"  Type: {agent_type}\n"
        f"  Turns: {user_messages_count} (tool calls: {tool_calls_count})\n"
        f"  Duration: {elapsed:.1f}s\n"
        f"  Max turns: {max_turns}\n"
    )
    
    # Truncate very long results
    result_text = final_message
    MAX_RESULT_LEN = 8000
    if len(result_text) > MAX_RESULT_LEN:
        result_text = result_text[:MAX_RESULT_LEN] + f"\n\n... (truncated, {len(final_message)} chars total)"
    
    return ToolResult(ok=True, output=header + "\n" + result_text)


task_tool = ToolDefinition(
    name="task",
    description=(
        "Launch a sub-agent to handle a complex task autonomously. "
        "The sub-agent runs in its own isolated context with a turn limit. "
        "Use 'explore' for fast read-only codebase exploration, "
        "'plan' for thorough analysis, or 'general' for full-featured multi-step work. "
        "The sub-agent's final result is returned to you."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short 3-5 word description of the task",
            },
            "prompt": {
                "type": "string",
                "description": "Full task description for the sub-agent. If not provided, uses 'description'.",
            },
            "agent_type": {
                "type": "string",
                "enum": ["explore", "plan", "general"],
                "description": "Type of sub-agent: 'explore' (fast, read-only), 'plan' (thorough, read-only), 'general' (full tools, default)",
            },
        },
        "required": ["description"],
    },
    validator=_validate,
    run=_run,
    # Concurrency-safe: a model batch of N task calls fans out in the
    # scheduler's parallel phase, bounded by the slot semaphore above.
    metadata=ToolMetadata(
        name="task",
        description="Sub-agent spawn (parallel fan-out, slot-capped)",
        capabilities={ToolCapability.CONCURRENCY_SAFE},
    ),
    # One sub-agent turn easily runs for minutes; the generic 120s
    # default would kill it. resolve_tool_timeout treats this as
    # authoritative over the env default and scheduler caps.
    timeout_seconds=600,
)
