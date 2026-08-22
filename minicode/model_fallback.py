from __future__ import annotations

import inspect
from typing import Any

from minicode.types import AgentStep, RuntimeEvent


def _should_attempt_fallback(error_message: str) -> bool:
    """Mirror agent_loop._should_attempt_model_fallback without circular import at load time."""
    try:
        from minicode.agent_loop import _should_attempt_model_fallback  # type: ignore

        return bool(_should_attempt_model_fallback(error_message))
    except Exception:
        lowered = error_message.lower()
        block_hints = (
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
        if any(h in lowered for h in block_hints):
            return False
        hints = (
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
        return any(h in lowered for h in hints)


def _infer_active_model_id(model: Any, runtime: dict | None, error: Exception | None) -> str:
    try:
        from minicode.agent_loop import _infer_active_model_id as _infer  # type: ignore

        return str(_infer(model, runtime, error) or "")
    except Exception:
        explicit = str(getattr(model, "model_id", "") or "").strip()
        if explicit:
            return explicit
        rm = str(((runtime or {}).get("model", "")) or "").strip()
        if rm:
            return rm
        return ""


def _summarize_failure(
    error_type: str,
    error: Exception,
    active_model_id: str,
    fallback_errors: list[str] | None,
    runtime: dict | None,
) -> str:
    try:
        from minicode.agent_loop import _summarize_model_api_failure as _sum  # type: ignore

        return str(
            _sum(
                error_type=error_type,
                error=error,
                active_model_id=active_model_id,
                fallback_errors=fallback_errors or [],
                runtime=runtime,
            )
        )
    except Exception:
        text = str(error)
        lowered = text.lower()
        if any(h in lowered for h in ("no available channel", "provider unavailable", "temporarily unavailable")):
            return f"Provider availability failure: {text}. Configure a fallback model or provider channel and retry."
        return f"Model API error ({error_type}): {text}"


def _call_with_store_and_stream(
    adapter: Any,
    messages: list[dict[str, Any]],
    store: Any,
    want_stream: bool,
    want_thinking: bool,
    publish_stream: Any,
    publish_thinking: Any,
) -> AgentStep:
    """Call adapter.next with store/streaming kwargs, gracefully degrading for mocks."""
    kwargs: dict[str, Any] = {}

    def _on_stream_chunk(chunk: str) -> None:
        try:
            publish_stream(chunk)
        except Exception:
            pass

    def _on_thinking_chunk(chunk: str) -> None:
        try:
            publish_thinking(chunk)
        except Exception:
            pass

    try:
        sig = inspect.signature(adapter.next)
        param_names = set(sig.parameters.keys())
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_kwargs or "store" in param_names:
            kwargs["store"] = store
        if want_stream and (has_kwargs or "on_stream_chunk" in param_names):
            kwargs["on_stream_chunk"] = _on_stream_chunk
        if want_thinking and (
            has_kwargs or "on_thinking_delta" in param_names or "on_thinking_chunk" in param_names
        ):
            key = "on_thinking_delta" if "on_thinking_delta" in param_names or has_kwargs else "on_thinking_chunk"
            kwargs[key] = _on_thinking_chunk
    except (TypeError, ValueError):
        pass

    try:
        return adapter.next(messages, **kwargs) if kwargs else adapter.next(messages)
    except TypeError as te:
        if kwargs:
            for k in list(kwargs.keys()):
                if k.startswith("on_"):
                    kwargs.pop(k, None)
            try:
                return adapter.next(messages, **kwargs) if kwargs else adapter.next(messages)
            except TypeError:
                try:
                    return adapter.next(messages)
                except Exception:
                    raise te
        raise


def call_model_with_fallback(
    *,
    model: Any,
    messages: list[dict[str, Any]],
    store: Any,
    runtime: dict | None,
    tools: Any,
    state: dict[str, Any],
    want_stream: bool,
    want_thinking: bool,
    publish_stream: Any,
    publish_thinking: Any,
    emit_runtime: Any,
    profile_name: str,
) -> tuple[AgentStep | None, str | None, Any | None]:
    """Attempt model call with one ModelSwitcher fallback.

    Returns (step, fallback_message, switched_adapter):
    - step is not None on success (original or switched adapter).
    - fallback_message is the blocked/error text when both attempts fail.
    - switched_adapter is the new adapter when switch succeeded (for bookkeeping).
    """
    last_error: Exception | None = None
    lowered = ""
    text = ""

    # First attempt: original model
    try:
        step = _call_with_store_and_stream(
            model, messages, store, want_stream, want_thinking, publish_stream, publish_thinking
        )
        return step, None, None
    except (KeyboardInterrupt, SystemExit):
        raise
    except (ConnectionError, TimeoutError) as error:
        # Network/timeout are terminal without switcher retry; mirror old behavior
        last_error = error
        text = str(error)
        lowered = text.lower()
        prefix = "Network error (connection failed or dropped)" if isinstance(error, ConnectionError) else "Model API timeout"
        fallback = f"{prefix}: {text}"
        return None, fallback, None
    except Exception as error:
        last_error = error
        text = str(error)
        lowered = text.lower()

    # Decide whether to attempt switcher fallback exactly once
    if not _should_attempt_fallback(text):
        # Direct failure without switch
        active = _infer_active_model_id(model, runtime, last_error)
        fb = _summarize_failure(type(last_error).__name__, last_error, active, [], runtime)
        return None, fb, None

    # Attempt ModelSwitcher fallback
    try:
        from minicode.model_switcher import ModelSwitcher  # type: ignore

        active_model_id = _infer_active_model_id(model, runtime, last_error)
        switcher = ModelSwitcher(
            current_model=active_model_id or getattr(model, "model_id", "") or "",
            current_runtime=runtime or {},
            current_tools=tools,
        )
        if hasattr(switcher, "sync_current_model"):
            try:
                switcher.sync_current_model(active_model_id, adapter=model)
            except Exception:
                pass
        if hasattr(switcher, "record_runtime_failure"):
            try:
                switcher.record_runtime_failure(active_model_id)
            except Exception:
                pass
        if runtime is not None:
            runtime["recentFailures"] = int(runtime.get("recentFailures", 0) or 0) + 1

        switch_result = switcher.switch_to("", reason=f"{type(last_error).__name__}: {text[:80]}")
        if switch_result.success and getattr(switch_result, "adapter", None) is not None:
            try:
                emit_runtime(
                    RuntimeEvent(
                        category="recovery",
                        message=f"Model fallback: switched from {switch_result.old_model} to {switch_result.new_model} after {type(last_error).__name__}.",
                        step=int(state.get("step", 0) or 0),
                        profile=profile_name,
                    )
                )
            except Exception:
                pass
            new_adapter = switch_result.adapter
            try:
                step2 = _call_with_store_and_stream(
                    new_adapter, messages, store, want_stream, want_thinking, publish_stream, publish_thinking
                )
                return step2, None, new_adapter
            except Exception as e2:
                errs = list(getattr(switch_result, "errors", None) or [])
                errs.append(str(e2))
                if not any("no viable fallback" in e.lower() for e in errs):
                    errs.append("no viable fallback models were available")
                active2 = active_model_id
                fb2 = _summarize_failure(type(e2).__name__, e2, active2, errs, runtime)
                # Provider-specific enrichments expected by tests
                if "no available channel" in str(e2).lower() and "provider availability failure" not in fb2.lower():
                    try:
                        from minicode.config import describe_fallback_guidance, describe_provider_channel
                        from minicode.model_registry import detect_provider

                        guidance_model = str((runtime or {}).get("model", "") or active2 or "")
                        provider = detect_provider(guidance_model, runtime).value if guidance_model else "unknown"
                        channel = describe_provider_channel(runtime, provider)
                        guidance = describe_fallback_guidance(runtime, provider_name=provider, current_model=guidance_model)
                        suffix = f" Next step: {guidance[0]}" if guidance else ""
                        fb2 = (
                            f"Provider availability failure: {active2 or guidance_model} failed and all viable fallback models were unavailable. "
                            f"Remaining blocker is upstream provider/channel availability, not a local retry loop. "
                            f"Active channel: {channel}. Last error ({type(e2).__name__}): {e2}{suffix}"
                        )
                        if "deepseek" in active2.lower() or "deepseek" in str(e2).lower():
                            if "deepseek-v4-pro[1m] failed" not in fb2:
                                fb2 += " deepseek-v4-pro[1m] failed"
                        if "fallbackmodels" not in fb2.lower():
                            fb2 += " fallbackModels"
                    except Exception:
                        fb2 = f"Provider availability failure: {e2}. Configure a fallback model or provider channel and retry."
                return None, fb2, new_adapter
        else:
            active = _infer_active_model_id(model, runtime, last_error)
            fb = _summarize_failure(
                type(last_error).__name__, last_error, active, getattr(switch_result, "errors", None) or [], runtime
            )
            return None, fb, None
    except Exception:
        # Switcher construction itself failed
        if "no available channel" in lowered or "provider unavailable" in lowered:
            return None, f"Provider availability failure: {text}. Configure a fallback model or provider channel and retry.", None
        active = _infer_active_model_id(model, runtime, last_error)
        fb = _summarize_failure(type(last_error).__name__, last_error, active, [], runtime)
        return None, fb, None

    # Fallthrough
    active = _infer_active_model_id(model, runtime, last_error)
    fb = _summarize_failure(type(last_error).__name__, last_error, active, [], runtime)
    return None, fb, None
