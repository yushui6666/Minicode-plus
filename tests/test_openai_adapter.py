from __future__ import annotations

import io
import json
import urllib.error
from http.client import HTTPMessage
from typing import Any

import pytest

from minicode.openai_adapter import (
    DEFAULT_OPENAI_USER_AGENT,
    OpenAIModelAdapter,
)


class _DummyTools:
    def list(self) -> list[object]:
        return []


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body


def _runtime() -> dict[str, str]:
    return {
        "model": "gpt5.5",
        "openaiBaseUrl": "https://www.cctq.ai",
        "openaiApiKey": "test-key",
    }


def test_openai_adapter_sets_compatible_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["user_agent"] = request.get_header("User-agent")
        return _FakeResponse({"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    adapter = OpenAIModelAdapter(_runtime(), _DummyTools())

    step = adapter.next([{"role": "user", "content": "Reply with exactly OK."}])

    assert step.content == "OK"
    assert captured["user_agent"] == DEFAULT_OPENAI_USER_AGENT


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("https://www.cctq.ai", "https://www.cctq.ai/v1/chat/completions"),
        ("https://www.cctq.ai/", "https://www.cctq.ai/v1/chat/completions"),
        ("https://www.cctq.ai/v1", "https://www.cctq.ai/v1/chat/completions"),
        ("https://www.cctq.ai/v1/", "https://www.cctq.ai/v1/chat/completions"),
        ("https://www.cctq.ai/v1/chat/completions", "https://www.cctq.ai/v1/chat/completions"),
    ],
)
def test_openai_adapter_accepts_provider_root_api_base_or_full_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    expected_url: str,
) -> None:
    captured: dict[str, str] = {}

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        captured["url"] = request.full_url
        return _FakeResponse({"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    runtime = {**_runtime(), "openaiBaseUrl": base_url}
    adapter = OpenAIModelAdapter(runtime, _DummyTools())

    step = adapter.next([{"role": "user", "content": "Reply with exactly OK."}])

    assert step.content == "OK"
    assert captured["url"] == expected_url


def test_openai_adapter_surfaces_non_json_http_error_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            hdrs=HTTPMessage(),
            fp=io.BytesIO(b"error code: 1010"),
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    adapter = OpenAIModelAdapter(_runtime(), _DummyTools())

    with pytest.raises(RuntimeError, match="error code: 1010"):
        adapter.next([{"role": "user", "content": "Reply with exactly OK."}])


def test_openai_adapter_does_not_retry_permanent_503_model_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        calls["count"] += 1
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Unavailable",
            hdrs=HTTPMessage(),
            fp=io.BytesIO(
                json.dumps(
                    {
                        "error": {
                            "code": "model_not_found",
                            "message": "No available channel for model gpt5.5 under group Codex-Sale",
                        }
                    }
                ).encode("utf-8")
            ),
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    adapter = OpenAIModelAdapter(_runtime(), _DummyTools())

    with pytest.raises(RuntimeError, match="No available channel for model gpt5.5"):
        adapter.next([{"role": "user", "content": "Reply with exactly OK."}])

    assert calls["count"] == 1

def test_openai_adapter_non_streaming_forwards_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thinking: list[str] = []

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "OK",
                            "reasoning_content": "let me think step by step",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    adapter = OpenAIModelAdapter(_runtime(), _DummyTools())

    step = adapter.next(
        [{"role": "user", "content": "Reply with exactly OK."}],
        on_thinking_delta=thinking.append,
    )

    assert step.content == "OK"
    assert thinking == ["let me think step by step"]


class _SSEStream:
    def __init__(self, events) -> None:
        self._lines = [
            ("data: " + json.dumps(event)).encode("utf-8") for event in events
        ]
        self._lines.append(b"data: [DONE]")

    def __iter__(self):
        return iter(self._lines)


def test_openai_adapter_streaming_forwards_reasoning_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"choices": [{"delta": {"reasoning_content": "think "}}]},
        {"choices": [{"delta": {"reasoning_content": "harder"}}]},
        {"choices": [{"delta": {"content": "OK"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    thinking: list[str] = []
    streamed: list[str] = []

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        return _SSEStream(events)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    adapter = OpenAIModelAdapter(_runtime(), _DummyTools())

    step = adapter.next(
        [{"role": "user", "content": "Reply with exactly OK."}],
        on_stream_chunk=streamed.append,
        on_thinking_delta=thinking.append,
    )

    assert step.content == "OK"
    assert thinking == ["think ", "harder"]
    assert streamed == ["OK"]


def test_openai_adapter_streaming_accepts_openrouter_reasoning_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"choices": [{"delta": {"reasoning": "openrouter-style"}}]},
        {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]},
    ]
    thinking: list[str] = []

    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        return _SSEStream(events)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    adapter = OpenAIModelAdapter(_runtime(), _DummyTools())

    step = adapter.next(
        [{"role": "user", "content": "hi"}],
        on_stream_chunk=lambda _chunk: None,
        on_thinking_delta=thinking.append,
    )

    assert step.content == "done"
    assert thinking == ["openrouter-style"]


def test_openai_adapter_thinking_callback_errors_do_not_kill_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_urlopen(request, timeout=0):  # noqa: ANN001
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": "OK", "reasoning_content": "boom"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    adapter = OpenAIModelAdapter(_runtime(), _DummyTools())

    def _explode(chunk: str) -> None:
        raise RuntimeError("UI callback bug")

    step = adapter.next(
        [{"role": "user", "content": "Reply with exactly OK."}],
        on_thinking_delta=_explode,
    )

    assert step.content == "OK"
