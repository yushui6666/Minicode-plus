from __future__ import annotations

import concurrent.futures
import inspect
import os
import re
import warnings
import time
from pathlib import Path
from typing import Any, Callable

from minicode.config import describe_fallback_guidance, describe_provider_channel
from minicode.context_manager import ContextManager, estimate_message_tokens
from minicode.logging_config import get_logger
from minicode.model_registry import detect_provider
from minicode.permissions import PermissionManager
from minicode.state import Store, AppState, increment_tool_calls, set_busy, set_idle
from minicode.tooling import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    resolve_tool_timeout,
)
from minicode.types import (
    AgentStep,
    ChatMessage,
    ModelAdapter,
    RuntimeEvent,
    RuntimeEventCategory,
)

# Hooks integration
from minicode.hooks import HookEvent, fire_hook_sync

# Intelligence integration
from minicode.agent_metrics import AgentMetricsCollector
from minicode.agent_intelligence import ErrorClassifier, NudgeGenerator, ToolScheduler
from minicode.working_memory import get_working_memory, protect_context

# Work chain integration
from minicode.intent_parser import parse_intent
from minicode.task_object import build_task, TaskObject, TaskState
from minicode.task_graph import TaskGraph, TaskState as GraphTaskState
from minicode.pipeline_engine import get_pipeline_engine
from minicode.capability_registry import get_registry, CapabilityDomain
from minicode.layered_context import ContextBuilder, LayeredContext
from minicode.decision_audit import get_auditor, DecisionOutcome
from minicode.runtime_profiles import resolve_runtime_profile

# 工程控制论集成
from minicode.cybernetic_orchestrator import CyberneticOrchestrator
from minicode.cybernetic_supervisor import save_supervisor_report
from minicode.feedforward_controller import FeedforwardController

# 高级控制论模块
from minicode.state_observer import MeasurementVector
from minicode.self_healing_engine import SelfHealingEngine

# 任务进度控制
from minicode.progress_controller import ProgressSignal, ProgressAction

# 记忆注入和模型选择控制
from minicode.memory_injector import MemoryInjectionSignal, MemoryInjector
from minicode.model_registry import ModelSelectionSignal

# 智能路由与自省 (Phase 3 导入)
from minicode.smart_router import TaskOutcome

# 上下文管理集成 (Claude Code-style + Engineering Cybernetics)
from minicode.context_compactor import (
    ContextCompactor,
    AutoCompactConfig,
)
from minicode.context_cybernetics import ContextCyberneticsOrchestrator
from minicode.cost_control import CostControlLoop
from minicode.micro_compact import MicroCompactor
from minicode.circuit_breaker import CompactionCircuitBreaker
from minicode.memory import MemoryManager
from minicode.turn_kernel import (
    TurnPreludeState,
    TurnRecurrentState,
    TurnVerificationState,
    build_stable_task_pack,
    build_turn_coda_summary,
    build_widening_transition_nudge,
    decide_tool_turn,
    decide_assistant_turn,
    derive_turn_step_policy,
    finalize_work_chain_task,
    render_turn_policy_message,
)
from minicode.turn_events import AgentTurnCallbacks, AgentTurnEventSink, TurnEvent

logger = get_logger("agent_loop")

# 甯搁噺锛氶伩鍏嶉噸澶嶇殑鎻愮ず鏂囨湰
NUDGE_CONTINUE = (
    "Continue immediately from your <progress> update with concrete tool calls, "
    "code changes, or an explicit <final> answer only if the task is complete. "
    "Prefer taking the next concrete action over explaining what you plan to do."
)

NUDGE_AFTER_TOOL_RESULT = (
    "You have received tool results. Review them briefly, then take the next "
    "concrete action: call another tool, edit code, or give an explicit <final> "
    "answer only if the task is truly complete. Do not restate what you just saw."
)

NUDGE_AFTER_EMPTY_RESPONSE = (
    "Your last response was empty. This often happens after tool errors or when "
    "the model is uncertain. Pick the most likely next action and try it — you can "
    "adjust based on results. Call a tool, edit code, or give <final> if done."
)

NUDGE_AFTER_EMPTY_NO_TOOLS = (
    "Your last response was empty but you have not used any tools yet. Start by "
    "inspecting the relevant files (read_file, grep_files, list_files) to understand "
    "the codebase before making changes."
)

RESUME_AFTER_PAUSE = (
    "Resume from the previous pause. Continue with the next concrete tool call, "
    "code change, or <final> answer."
)

RESUME_AFTER_MAX_TOKENS = (
    "Your previous response was cut short by the token limit. Resume immediately "
    "with the next concrete action — pick up where you left off."
)


STABLE_TASK_STATE_MARKER = "[Stable task state]"
_MODEL_FALLBACK_ERROR_HINTS = (
    "no available channel",
    "temporarily unavailable",
    "service unavailable",
    "please try again later",
    "capacity exceeded",
    "overloaded",
    "high demand",
    "503",
    "502",
    "500",
    "connection refused",
    "connection reset",
    "timed out",
    "timeout",
)
_MODEL_FALLBACK_BLOCK_HINTS = (
    "unauthorized",
    "forbidden",
    "invalid api key",
    "authentication",
    "bad request",
    "invalid_request",
    "validation",
    "tool schema",
    "context length",
)


def _upsert_stable_task_state_message(
    messages: list[ChatMessage],
    stable_text: str,
) -> list[ChatMessage]:
    filtered = [
        message
        for message in messages
        if not (
            message.get("role") == "system"
            and str(message.get("content", "")).startswith(STABLE_TASK_STATE_MARKER)
        )
    ]
    filtered.append(
        {
            "role": "system",
            "content": f"{STABLE_TASK_STATE_MARKER}\n{stable_text}",
        }
    )
    return filtered


def _should_attempt_model_fallback(error_message: str) -> bool:
    normalized = error_message.lower()
    if any(marker in normalized for marker in _MODEL_FALLBACK_BLOCK_HINTS):
        return False
    return any(marker in normalized for marker in _MODEL_FALLBACK_ERROR_HINTS)


def _looks_like_provider_availability_error(error_message: str) -> bool:
    normalized = error_message.lower()
    return any(
        marker in normalized
        for marker in (
            "no available channel",
            "temporarily unavailable",
            "service unavailable",
            "please try again later",
            "capacity exceeded",
            "overloaded",
            "high demand",
            "503",
            "502",
            "500",
        )
    )


def _summarize_model_api_failure(
    *,
    error_type: str,
    error: Exception,
    active_model_id: str = "",
    fallback_errors: list[str] | None = None,
    runtime: dict[str, Any] | None = None,
) -> str:
    fallback_errors = fallback_errors or []
    if fallback_errors:
        combined = " ".join(fallback_errors)
        if (
            "no viable fallback models were available" in combined.lower()
            and any(_looks_like_provider_availability_error(item) for item in fallback_errors + [str(error)])
        ):
            runtime = runtime or {}
            guidance_model = (
                str(runtime.get("configuredModel", "")).strip()
                or str(runtime.get("model", "")).strip()
                or active_model_id
                or "the active model"
            )
            model_label = guidance_model or active_model_id or "the active model"
            provider = detect_provider(guidance_model, runtime).value if guidance_model else "unknown"
            channel = describe_provider_channel(runtime, provider)
            guidance = describe_fallback_guidance(
                runtime,
                provider_name=provider,
                current_model=guidance_model,
            )
            guidance_suffix = f" Next step: {guidance[0]}" if guidance else ""
            return (
                f"Provider availability failure: {model_label} failed and all viable fallback models were unavailable. "
                f"Remaining blocker is upstream provider/channel availability, not a local retry loop. "
                f"Active channel: {channel}. Last error ({error_type}): {error}{guidance_suffix}"
            )
    return f"Model API error ({error_type}): {error}"


def _extract_model_id_from_provider_error(error: Exception) -> str:
    message = str(error)
    match = re.search(r"model\s+([^\s]+)\s+under\s+group", message, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _infer_active_model_id(
    model: ModelAdapter,
    runtime: dict[str, Any] | None,
    error: Exception | None = None,
) -> str:
    explicit = str(getattr(model, "model_id", "") or "").strip()
    if explicit:
        return explicit
    runtime_model = str((runtime or {}).get("model", "") or "").strip()
    if runtime_model:
        return runtime_model
    if error is not None:
        return _extract_model_id_from_provider_error(error)
    return ""


def _is_empty_assistant_response(content: str) -> bool:
    return len(content.strip()) == 0


def _extract_task_description(messages: list[ChatMessage]) -> str:
    """Extract the original task description from messages."""
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            content = str(msg["content"])
            if not content.startswith("Continue") and not content.startswith("Your last"):
                return content[:500]
    return "Unknown task"


def _build_work_chain_task(messages: list[ChatMessage]) -> tuple[TaskObject | None, dict]:
    """Build TaskObject from conversation messages and return it with metadata."""
    raw_input = _extract_task_description(messages)
    if raw_input == "Unknown task":
        return None, {}
    intent = parse_intent(raw_input)
    task = build_task(intent, raw_input)
    metadata = {
        "intent_type": intent.intent_type.value,
        "action_type": intent.action_type.value,
        "confidence": intent.confidence,
        "entities": intent.entities,
        "complexity": intent.complexity_hint,
    }
    logger.info(
        "Work chain: intent=%s action=%s confidence=%.2f complexity=%s",
        intent.intent_type.value, intent.action_type.value,
        intent.confidence, intent.complexity_hint,
    )
    return task, metadata


def _build_layered_context(
    messages: list[ChatMessage],
    system_prompt: str = "",
    project_context: str = "",
    task: TaskObject | None = None,
) -> tuple[LayeredContext, ContextBuilder]:
    """Build layered context from conversation and task."""
    context = LayeredContext()
    builder = ContextBuilder(context)
    if system_prompt:
        builder.set_system_prompt(system_prompt)
    if project_context:
        builder.add_project_memory(project_context)
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            builder.add_session_message(role, content)
    if task:
        scratchpad = (
            f"Task: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Constraints: {len(task.constraints)}\n"
            f"Expected outputs: {len(task.expected_outputs)}"
        )
        builder.add_scratchpad(scratchpad)
    return context, builder


def _register_tool_capabilities(tools: ToolRegistry) -> None:
    """Register existing tools as capabilities in the registry."""
    registry = get_registry()
    if registry.list_all():
        return
    for tool_name in tools.list_all():
        try:
            from minicode.capability_registry import CapabilityMetadata, CapabilityScope
            tool_def = tools.find(tool_name)
            if not tool_def:
                continue
            domain = CapabilityDomain.UNKNOWN
            if "file" in tool_name or "write" in tool_name or "read" in tool_name:
                domain = CapabilityDomain.FILE
            elif "search" in tool_name or "grep" in tool_name:
                domain = CapabilityDomain.SEARCH
            elif "web" in tool_name or "http" in tool_name or "fetch" in tool_name:
                domain = CapabilityDomain.WEB
            elif "command" in tool_name or "run" in tool_name or "exec" in tool_name:
                domain = CapabilityDomain.EXECUTION
            elif "code" in tool_name or "diff" in tool_name or "review" in tool_name:
                domain = CapabilityDomain.CODE
            elif "memory" in tool_name:
                domain = CapabilityDomain.MEMORY
            scope = CapabilityScope.READONLY
            if any(k in tool_name for k in ("write", "modify", "edit", "delete", "create")):
                scope = CapabilityScope.WRITE
            if any(k in tool_name for k in ("command", "exec", "run")):
                scope = CapabilityScope.DESTRUCTIVE
            if any(k in tool_name for k in ("web", "fetch", "http")):
                scope = CapabilityScope.EXTERNAL
            metadata = CapabilityMetadata(
                name=tool_name, domain=domain, scope=scope,
                description=tool_def.description or f"Tool: {tool_name}",
                tags=["tool", tool_name],
            )
            registry.register(metadata, lambda **kw: tools.execute(tool_name, kw, ToolContext(cwd=str(Path.cwd()))), None)
        except Exception as e:
            logger.debug("Failed to register tool %s as capability: %s", tool_name, e)


def _execute_single_tool(
    call: dict,
    tools: ToolRegistry,
    cwd: str,
    permissions: Any | None,
    session: Any | None,
    runtime: dict | None,
    store: Any | None,
    step: int,
    on_tool_start: Callable[[str, dict], None] | None,
    on_tool_result: Callable[[str, str, bool], None] | None,
    tool_scheduler: Any | None = None,
) -> ToolResult:
    """Execute a single tool call with hooks, state updates, and crash protection.
    
    Used both for serial execution and as a worker function for concurrent execution.
    When running concurrently (store/on_tool_start/on_tool_result are None),
    hooks and UI callbacks are deferred to the result processing phase.
    
    Includes a global exception safety net: any unexpected crash in the tool
    execution pipeline (hooks, state updates, etc.) is caught and converted
    to an error ToolResult, preventing the entire agent loop from crashing.
    """
    tool_name = call["toolName"]
    tool_input = call["input"]
    
    try:
        # Pre-tool hooks and UI (only for serial execution)
        if on_tool_start:
            on_tool_start(tool_name, tool_input)
        
        if store:
            store.set_state(set_busy(tool_name))
        
        # Execute the tool with timeout protection
        import concurrent.futures
        TOOL_TIMEOUT = resolve_tool_timeout(tool_name, tools, tool_scheduler)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    tools.execute,
                    tool_name, tool_input,
                    ToolContext(cwd=cwd, permissions=permissions, session=session, _runtime=runtime),
                )
                result = future.result(timeout=TOOL_TIMEOUT)
        except concurrent.futures.TimeoutError:
            result = ToolResult(
                ok=False,
                output=f"Tool '{tool_name}' timed out after {TOOL_TIMEOUT}s",
            )
        except Exception:
            result = tools.execute(
                tool_name, tool_input,
                ToolContext(cwd=cwd, permissions=permissions, session=session, _runtime=runtime),
            )  # Fallback: direct execution
        
        # Post-tool state updates (only for serial execution)
        if store:
            store.set_state(increment_tool_calls())
            store.set_state(set_idle())
        
        if on_tool_result:
            on_tool_result(tool_name, result.output, not result.ok)
        
        return result
    
    except (KeyboardInterrupt, SystemExit):
        # Always propagate these
        raise
    except Exception as exc:  # noqa: BLE001
        # Global safety net: catch ANY unexpected error in the tool execution
        # pipeline (hooks, state updates, permission checks, etc.) and convert
        # it to an error result. This prevents a single tool crash from
        # cascading into a full session failure.
        import traceback
        tb_excerpt = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]).strip()
        error_type = type(exc).__name__
        
        logger.error("Tool execution pipeline crashed (%s): %s", error_type, exc)
        
        # Ensure state is reset even on crash
        if store:
            try:
                store.set_state(set_idle())
            except Exception:
                pass
        
        return ToolResult(
            ok=False,
            output=f"[{error_type}] Tool execution pipeline crashed: {exc}\n"
                   f"Traceback:\n{tb_excerpt}"
        )


def _format_diagnostics(stop_reason: str | None, block_types: list[str] | None, ignored_block_types: list[str] | None) -> str:
    parts: list[str] = []
    if stop_reason:
        parts.append(f"stop_reason={stop_reason}")
    if block_types:
        parts.append(f"blocks={','.join(block_types)}")
    if ignored_block_types:
        parts.append(f"ignored={','.join(ignored_block_types)}")
    return f" Diagnostics: {'; '.join(parts)}." if parts else ""


def _is_recoverable_thinking_stop(*, is_empty: bool, stop_reason: str | None, ignored_block_types: list[str] | None) -> bool:
    if not is_empty:
        return False
    if stop_reason not in {"pause_turn", "max_tokens"}:
        return False
    return "thinking" in (ignored_block_types or [])


def _should_treat_assistant_as_progress(*, kind: str | None, content: str, saw_tool_result: bool) -> bool:
    if kind == "progress":
        return True
    if kind == "final":
        return False
    if not saw_tool_result:
        return False
    return False


# ── Preemptive context guard (Claude Code-style blocking limit) ─────────────

def _is_at_blocking_limit(
    token_count: int,
    context_window: int,
    *,
    effective_window_ratio: float = 0.90,
    min_reserve_tokens: int = 3_000,
) -> bool:
    """Preemptively block API calls when context is nearly full.

    Inspired by Claude Code's `isAtBlockingLimit` — prevents 413 errors
    by refusing to send a request when the context window is too full.

    The effective window is model context_window * effective_window_ratio,
    minus min_reserve_tokens (reserved for the assistant response).

    Returns True if the request would likely trigger a 413.
    """
    effective_window = int(context_window * effective_window_ratio)
    blocking_limit = max(1, effective_window - min_reserve_tokens)
    return token_count >= blocking_limit


def _compute_effective_blocking_limit(
    context_window: int,
    *,
    effective_window_ratio: float = 0.90,
    min_reserve_tokens: int = 3_000,
) -> int:
    effective_window = int(context_window * effective_window_ratio)
    return max(1, effective_window - min_reserve_tokens)


def _try_compact_with_breaker(
    breaker: CompactionCircuitBreaker,
    compact_fn: Callable[[], tuple[list, bool]],
    current_messages: list,
    logger_fn: Callable[..., None],
) -> tuple[list, bool]:
    """Run compaction through the circuit breaker.

    Returns (messages, effective) — messages is the (possibly compacted)
    list, effective indicates whether compaction actually changed anything.
    """
    if not breaker.is_allowed():
        logger_fn("Compaction blocked by circuit breaker (consecutive failures)")
        return current_messages, False
    try:
        result_messages, effective = compact_fn()
        if effective:
            breaker.record_success()
        return result_messages, effective
    except Exception as exc:
        breaker.record_failure()
        bs = breaker.get_state()
        logger_fn("Compaction failed (breaker=%d/%d): %s",
                  bs.consecutive_failures, breaker.config.failure_threshold, exc)
        return current_messages, False


def _model_next(
    model: ModelAdapter,
    messages: list[ChatMessage],
    *,
    on_stream_chunk: Callable[[str], None] | None,
    on_thinking_chunk: Callable[[str], None] | None = None,
    store: Store[AppState] | None,
) -> AgentStep:
    """Call provider adapters with store/thinking support while preserving test doubles."""
    kwargs: dict[str, Any] = {"on_stream_chunk": on_stream_chunk}

    try:
        sig = inspect.signature(model.next)
        param_names = set(sig.parameters.keys())
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_kwargs or "on_thinking_delta" in param_names:
            kwargs["on_thinking_delta"] = on_thinking_chunk
        if has_kwargs or "store" in param_names:
            kwargs["store"] = store
    except (TypeError, ValueError):
        # Can't inspect signature (e.g. some mock objects) — be conservative
        pass

    return model.next(messages, **kwargs)


def _apply_control_signal(
    *,
    control_signal: Any,
    system_state: Any,
    max_steps: int | None,
    tool_scheduler: ToolScheduler,
    context_compactor: ContextCompactor | None,
    model_switcher: Any | None,
    feedback_controller: Any | None = None,
) -> int | None:
    """Apply FeedbackController output to live runtime knobs."""
    if not control_signal or control_signal.confidence <= 0.6:
        return max_steps

    if (
        control_signal.limit_max_steps
        and max_steps is not None
        and control_signal.limit_max_steps < max_steps
    ):
        logger.info(
            "FeedbackController: limiting max_steps %d -> %d",
            max_steps, control_signal.limit_max_steps,
        )
        max_steps = control_signal.limit_max_steps

    if control_signal.adjust_token_budget != 1.0:
        if (
            context_compactor
            and hasattr(context_compactor, "_tool_budget")
            and context_compactor._tool_budget
        ):
            tb = context_compactor._tool_budget
            new_budget = max(
                1000,
                int(
                    getattr(tb, "budget_per_message", 2000)
                    * control_signal.adjust_token_budget
                ),
            )
            if hasattr(tb, "budget_per_message"):
                tb.budget_per_message = new_budget
            logger.info(
                "FeedbackController: token budget adjusted to %d (mult=%.2f)",
                new_budget, control_signal.adjust_token_budget,
            )

    if control_signal.reduce_parallelism:
        if hasattr(tool_scheduler, "_force_max_workers"):
            tool_scheduler._force_max_workers = min(
                getattr(tool_scheduler, "_force_max_workers", 2) or 2,
                2,
            )
        logger.info(
            "FeedbackController: reduce_parallelism -> max_workers=2 "
            "(oscillation=%.2f)",
            control_signal.oscillation_index,
        )

    if control_signal.adjust_concurrency != 0:
        cap = max(1, 4 + control_signal.adjust_concurrency)
        if hasattr(tool_scheduler, "_force_max_workers"):
            tool_scheduler._force_max_workers = cap
        logger.info(
            "FeedbackController: adjust_concurrency=%+d -> max_workers=%d",
            control_signal.adjust_concurrency, cap,
        )

    if control_signal.increase_model_level:
        logger.info(
            "FeedbackController: model upgrade recommended (errors=%.2f perf=%.2f)",
            system_state.error_frequency,
            system_state.performance_score(),
        )
        if model_switcher:
            if hasattr(model_switcher, '_pending_upgrade'):
                model_switcher._pending_upgrade = True  # type: ignore[attr-defined]

    if control_signal.decrease_model_level:
        logger.info(
            "FeedbackController: model downgrade recommended (efficiency=%.2f)",
            system_state.token_efficiency,
        )

    if control_signal.suggest_memory_persistence:
        logger.info("FeedbackController: persisting working memory")
        if context_compactor and hasattr(context_compactor, "_tool_budget"):
            try:
                if hasattr(context_compactor._tool_budget, "flush"):
                    context_compactor._tool_budget.flush()
            except Exception:
                pass

    if control_signal.recommend_skill_update:
        logger.info(
            "FeedbackController: skill update recommended (pattern=%.2f)",
            system_state.pattern_reuse_rate,
        )
        # Queue skill update for next maintenance cycle
        if not hasattr(tool_scheduler, "_pending_skill_update"):
            tool_scheduler._pending_skill_update = True  # type: ignore[attr-defined]
        logger.info("FeedbackController: skill update queued for next maintenance cycle")

    if control_signal.reduce_tool_timeout:
        new_timeout = max(5.0, control_signal.reduce_tool_timeout)
        if hasattr(tool_scheduler, "_force_tool_timeout"):
            tool_scheduler._force_tool_timeout = new_timeout  # type: ignore[attr-defined]
        logger.info(
            "FeedbackController: tool timeout reduced to %.1fs (high error rate)",
            new_timeout,
        )
    elif hasattr(tool_scheduler, '_force_tool_timeout'):
        # Reset timeout when signal no longer active
        if hasattr(tool_scheduler, '_force_tool_timeout'):
            del tool_scheduler._force_tool_timeout

    if control_signal.increase_nudge_frequency:
        tool_scheduler._force_nudge_frequency = True  # type: ignore[attr-defined]
        logger.info(
            "FeedbackController: nudge frequency increased (stability=%.2f)",
            system_state.stability_score(),
        )
    elif hasattr(tool_scheduler, '_force_nudge_frequency'):
        if hasattr(tool_scheduler, '_force_nudge_frequency'):
            del tool_scheduler._force_nudge_frequency

    if control_signal.promote_pattern:
        if feedback_controller:
            feedback_controller.record_pattern_effectiveness(
                control_signal.promote_pattern, True
            )
            logger.info(
                "FeedbackController: pattern promoted '%s'",
                control_signal.promote_pattern,
            )

    if control_signal.force_compaction and context_compactor:
        try:
            compacted = context_compactor.compact_messages() if hasattr(context_compactor, 'compact_messages') else False
            logger.info(
                "FeedbackController: forced compaction completed (%d messages)",
                len(compacted) if compacted else 0,
            )
        except Exception as exc:
            logger.warning("FeedbackController: forced compaction failed: %s", exc)

    return max_steps


def run_agent_turn(
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    messages: list[ChatMessage],
    cwd: str,
    permissions: PermissionManager | None = None,
    session: Any | None = None,
    store: Store[AppState] | None = None,
    max_steps: int = 50,
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str, bool], None] | None = None,
    on_assistant_message: Callable[[str], None] | None = None,
    on_progress_message: Callable[[str], None] | None = None,
    on_runtime_event: Callable[[RuntimeEvent], None] | None = None,
    on_assistant_stream_chunk: Callable[[str], None] | None = None,
    on_thinking_chunk: Callable[[str], None] | None = None,
    context_manager: ContextManager | None = None,
    memory_manager: MemoryManager | None = None,
    runtime: dict | None = None,
    metrics_collector: AgentMetricsCollector | None = None,
    system_prompt: str = "",
    project_context: str = "",
    enable_work_chain: bool = True,
    callbacks: AgentTurnCallbacks | AgentTurnEventSink | None = None,
) -> list[ChatMessage]:
    """Thin deprecation shim — the hand-written loop was removed in slice5.

    The graph is now the single orchestration path. ``MINICODE_USE_GRAPH=0``
    / ``runtime={\"useGraph\": False}`` is retained only as a no-op escape
    hatch that warns and still delegates to the graph (legacy loop deleted).
    """
    import warnings as _warnings

    _warnings.warn(
        "minicode.agent_loop.run_agent_turn is deprecated; use minicode.graph.run_graph_turn",
        DeprecationWarning,
        stacklevel=2,
    )
    # Preserve the env/runtime opt-out surface as a warning-only hatch.
    _env_val = os.environ.get("MINICODE_USE_GRAPH", "").strip().lower()
    want_graph = True
    if _env_val in {"0", "false", "no", "off"}:
        want_graph = False
    elif _env_val in {"1", "true", "yes"}:
        want_graph = True
    if isinstance(runtime, dict) and "useGraph" in runtime:
        want_graph = bool(runtime.get("useGraph"))
    if not want_graph:
        _warnings.warn(
            "legacy agent_loop removed; MINICODE_USE_GRAPH=0 / useGraph=False is now a no-op and still delegates to the graph",
            UserWarning,
            stacklevel=2,
        )
    from minicode.graph import run_graph_turn as _graph_run

    return _graph_run(
        model=model,
        tools=tools,
        messages=messages,
        cwd=cwd,
        permissions=permissions,
        session=session,
        max_steps=max_steps,
        thread_id=str(getattr(session, "session_id", "default") if session is not None else "default"),
        authorize_tool=None,
        load_context=None,
        compact_context=None,
        repair=None,
        callbacks=callbacks,
        context_manager=context_manager,
        memory_manager=memory_manager,
        runtime=runtime,
        store=store,
        on_tool_start=on_tool_start,
        on_tool_result=on_tool_result,
        on_assistant_message=on_assistant_message,
        on_progress_message=on_progress_message,
        on_runtime_event=on_runtime_event,
    )
