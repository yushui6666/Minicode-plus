"""Wiring fixes for the memory system.

Covers the four re-wired seams:
1. MemoryInjectionController consumes 'pid_adjustment' (tuner -> decision).
2. MemoryPipeline.on_failure stashes recovery notes consumed by inject().
3. write() with an error trace triggers failure recovery automatically.
4. write() runs periodic maintain() every ten writes.
5. update_control_state() flows stability into real injection decisions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minicode.memory import MemoryManager, MemoryScope
from minicode.memory_injector import (
    MemoryInjectionController,
    MemoryInjectionSignal,
)
from minicode.memory_pipeline import MemoryPipeline


@pytest.fixture
def memory(tmp_path: Path) -> MemoryManager:
    mm = MemoryManager(project_root=tmp_path)
    mm.add_entry(
        MemoryScope.PROJECT,
        "testing",
        "Fix pytest fixture error by adding tmp_path fixture to the test",
        ["pytest", "testing"],
    )
    return mm


def _pipeline(mm: MemoryManager) -> MemoryPipeline:
    pipe = MemoryPipeline(mm)
    pipe.initialize(
        model_adapter=None,
        enable_reranker=False,
        enable_vector=False,
        enable_graph=False,
    )
    return pipe


class TestPidTrim:
    """The tuner -> decision loop: decide() must consume pid_adjustment."""

    def _decide(self, **kwargs):
        ctrl = MemoryInjectionController()
        signal = MemoryInjectionSignal(context_usage=0.5, retrieval_quality=0.5)
        return ctrl.decide(
            signal,
            base_max_memories=3,
            base_min_relevance=0.3,
            base_max_tokens=200,
            **kwargs,
        )

    def test_no_adjustment_is_standard(self):
        decision = self._decide()
        assert decision.max_memories == 3
        assert decision.min_relevance == pytest.approx(0.3)

    def test_positive_trim_loosens(self):
        plain = self._decide()
        boosted = self._decide(pid_adjustment=1.0)
        assert boosted.max_memories == plain.max_memories + 2
        assert boosted.min_relevance < plain.min_relevance
        assert any("adaptive PID trim" in r for r in boosted.reasons)

    def test_negative_trim_tightens(self):
        plain = self._decide()
        tight = self._decide(pid_adjustment=-1.0)
        assert tight.max_memories == plain.max_memories - 2
        assert tight.min_relevance > plain.min_relevance
        assert any("adaptive PID trim" in r for r in tight.reasons)

    def test_adjustment_is_clamped(self):
        loose = self._decide(pid_adjustment=99.0)
        edge = self._decide(pid_adjustment=1.0)
        assert loose.max_memories == edge.max_memories


class TestControlStateFlow:
    """update_control_state -> inject -> decision carries the trim."""

    def test_stability_score_flows_into_decision(self, memory: MemoryManager):
        pipe = _pipeline(memory)
        assert pipe._current_pid_adjustment() is None

        pipe.update_control_state(stability_score=1.0)
        assert pipe._current_pid_adjustment() == pytest.approx(1.0)

        pipe.inject("any task", [], [{"role": "system", "content": "s"}])
        decision = pipe._injector.last_decision
        assert decision is not None
        assert any("adaptive PID trim" in r for r in decision.reasons)

    def test_scores_are_clamped(self, memory: MemoryManager):
        pipe = _pipeline(memory)
        pipe.update_control_state(stability_score=42.0)
        assert pipe._current_pid_adjustment() == pytest.approx(1.0)


class TestFailureRecovery:
    """on_failure stashes notes; the next successful inject() consumes them."""

    def test_explicit_on_failure_then_inject(self, memory: MemoryManager):
        pipe = _pipeline(memory)
        recovered = pipe.on_failure("pytest fixture error", "pytest")
        assert recovered, "expected at least one recovery memory"

        messages = [{"role": "system", "content": "sys"}]
        out = pipe.inject("write another test", [], messages)
        joined = "".join(str(m.get("content", "")) for m in out)
        assert "Failure Recovery Notes" in joined

        # Notes are consumed exactly once: disable cooldown, reinject.
        pipe._injector._injection_cooldown = 0
        out2 = pipe.inject("yet another task", [], [{"role": "system", "content": "s"}])
        joined2 = "".join(str(m.get("content", "")) for m in out2)
        assert "Failure Recovery Notes" not in joined2

    def test_write_with_error_trace_triggers_recovery(self, memory: MemoryManager):
        pipe = _pipeline(memory)
        pipe.write(
            "fix login bug",
            [
                {"type": "tool_call", "count": 2},
                {"type": "error", "message": "pytest fixture error", "tool": "pytest"},
            ],
        )
        assert pipe._pending_recovery, "error trace should stash recovery notes"


class TestPeriodicMaintain:
    """write() de-single-points maintenance: every 10 writes -> maintain()."""

    def test_maintain_runs_every_ten_writes(self, memory: MemoryManager, monkeypatch):
        pipe = _pipeline(memory)
        calls: list[int] = []
        monkeypatch.setattr(pipe, "maintain", lambda force=False: calls.append(1))

        for i in range(9):
            pipe.write(f"task {i}", [])
        assert calls == []

        pipe.write("task 9", [])
        assert len(calls) == 1
