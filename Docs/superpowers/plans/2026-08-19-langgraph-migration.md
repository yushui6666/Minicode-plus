# LangGraph Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a tested LangGraph orchestration boundary without changing legacy entrypoints.

**Architecture:** A typed serializable state flows through route, tool, verify, and finalize nodes. Existing runtime capabilities enter through callbacks until later migration phases.

**Tech Stack:** Python 3.11+, LangGraph, pytest.

---

### Task 1: Establish the graph contract

**Files:** `tests/test_langgraph_runtime.py`, `minicode/graph/builder.py`

- [x] Define `AgentState` and test tool and direct-completion routes.
- [x] Compile a `StateGraph` with explicit nodes and conditional routing.
- [x] Verify with `.venv/bin/python -m pytest tests/test_langgraph_runtime.py -q`.

### Task 2: Package and dependency integration

**Files:** `minicode/graph/__init__.py`, `pyproject.toml`, `.gitignore`

- [x] Export the graph builder from `minicode.graph`.
- [x] Declare the bounded LangGraph dependency.
- [x] Keep `.venv/` ignored and install dependencies only in the project virtual environment.

### Task 3: Regression verification

**Files:** existing runtime tests

- [x] Run focused legacy tests for agent loop, headless execution, and tools.
- [x] Run the full test suite and record unrelated baseline failures separately.

Baseline (2026-08-20, Python 3.14 / .venv): focused legacy set
(graph + turn_kernel + headless + tty_app + retired-loop five) = 147 passed;
full suite = 1414 passed, 2 skipped (compileall/structure-compliance gates also
green; five AppleDouble `.___pycache__` metadata shells and one
`ANTHROPIC_MODEL` test-env leak were cleaned along the way — both unrelated
environment noise, documented here for the next runner).

### Task 4: Migrate turn-kernel semantics into `build_model_graph` (slice 2)

**Files:** `minicode/turn_kernel.py`, `minicode/graph/turn_text.py`,
`minicode/graph/builder.py`, `minicode/graph/runtime.py`,
`tests/test_langgraph_runtime.py`

- [x] Add `snapshot_to_turn_state` / `turn_state_to_snapshot` round-trip
  helpers so graph nodes reuse the kernel decision functions unchanged.
- [x] Move nudge constants and step predicates out of the retired loop into
  `minicode/graph/turn_text.py` (no `graph -> agent_loop` import).
- [x] Add the kernel topology: `step_policy`, `classify_step`,
  `assistant_followup`, `widen`, `observe_tool` nodes plus upgraded
  `verify`/`finalize`; wire `GraphEventSink` for phase/widening/stop events.
- [x] Keep the slice-1 thin topology behind `turn_kernel=False`
  (`runtime={"turnKernel": "thin"}` escape hatch, removed in slice 3).
- [x] Fix the three production defects: missing `recursion_limit`, empty
  response spin, missing `assistant_tool_call` message pair.
- [x] Verify with
  `.venv/bin/python -m pytest tests/test_langgraph_runtime.py -q` (18 passed)
  plus the focused legacy regression set (147 passed).

### Task 5: Multi-tool batch, graceful model failures, and slice-3 hardening

**Files:** `minicode/graph/builder.py`, `minicode/graph/runtime.py`,
`minicode/turn_kernel.py`, `minicode/turn_events.py`,
`tests/test_graph_multi_tool.py`, `tests/test_langgraph_runtime.py`,
`tests/test_headless.py`

- [x] Port ToolScheduler batch semantics into the graph runtime: whole
  `step.calls` list in `step_calls`, concurrent/serial split, parallel phase
  with timeout guard, ordered `assistant_tool_call`/`tool_result` pairs in
  `observe_tool`, `await_user` short-circuit, co-failure conflict recording,
  deferred `on_tool_start`/`on_tool_result` via `GraphEventSink` adapters
  (`minicode/graph/builder.py:502`, `minicode/graph/runtime.py:219`).
- [x] Add `ToolScheduler` integration to `run_graph_turn` (per-call
  `ToolContext`, timeout via `MINICODE_TOOL_TIMEOUT`, crash safety net,
  `get_recommended_max_workers` + `_force_max_workers` cap, call-order sort;
  `minicode/graph/runtime.py:22`).
- [x] Remove the slice-1 thin topology escape hatch (`turnKernel=False`):
  `build_model_graph` is now the single topology, `runtime={"turnKernel":…}`
  is accepted but ignored (`minicode/graph/builder.py:707`,
  `minicode/graph/runtime.py:111`, thin test deleted).
- [x] Graceful model-API failure: `run_graph_turn` catches provider
  exceptions in `next_step` and emits a typed `kind="error"` fallback
  (`ConnectionError`→network, `TimeoutError`→timeout,
  `no available channel`→`Provider availability failure` with fallback
  guidance, otherwise `Model API error`) routed through
  `classify_step` to a `blocked` stop (`minicode/graph/runtime.py:168`,
  `minicode/graph/builder.py:343`).
- [x] Per-turn channel reset for checkpointed threads: `initial_state`
  zeroes `stop_reason`, decision/step/batch fields before each
  `graph.invoke` so turn 2 on the same `thread_id` calls the model
  (`minicode/graph/runtime.py:437`).
- [x] `TurnEventQueue` as `AgentTurnCallbacks`: add `on_tool_start`,
  `on_tool_result`, `on_assistant_message`, `on_progress_message`,
  `on_runtime_event` adapters that publish `TurnEvent`s; `GraphEventSink`
  defers concurrent callbacks to `observe_tool` so the TUI sees them in
  original call order (`minicode/turn_events.py:198`,
  `minicode/graph/builder.py:144`).
- [x] Progress summary and coda: `turn_kernel` round-trips
  `progress_summary` (`minicode/turn_kernel.py:934`), `observe_tool`
  records `decide_tool_turn(...).progress_summary`,
  `test_observe_progress_summary_round_trips_into_coda` pins it
  (`tests/test_graph_multi_tool.py:574`).
- [x] Verify: `tests/test_langgraph_runtime.py` (18) +
  `tests/test_graph_multi_tool.py` (15) = 33 passed;
  `tests/test_headless.py::test_run_headless_provider_failure_uses_runtime_channel_details`
  was failing against the new fallback path (fixed by provider-channel
  detection in `runtime.py:178`); full suite 1424 passed, 2 skipped.

### Task 6: Authorize wiring, checkpoint resume, and agent_loop shim (slice 4)

**Files:** `minicode/graph/runtime.py`, `minicode/graph/builder.py`, `minicode/agent_loop.py`, `minicode/turn_events.py`, `tests/test_graph_checkpoint.py`, `tests/test_langgraph_runtime.py`

- [x] Wire `PermissionManager` into the graph `authorize` node: `run_graph_turn` builds `authorize_tool` from `permissions`+`tools` when none is supplied (deny-list via `permissions._check` + tool concurrency-safety still via `ToolScheduler`), batch deny → `permission=denied` → `finalize` without executing.
- [x] Checkpoint resume: `run_graph_turn` auto-creates a `SqliteSaver` when `store`/`checkpointer` is supplied via `runtime={"graphCheckpoint": true}` or `MINICODE_GRAPH_CHECKPOINT=1`, threads are isolated by `thread_id`, `checkpoint_ns` stays default, and `graph.get_state` round-trips the kernel fields.
- [x] Retire `agent_loop.run_agent_turn` as a shim: emit `DeprecationWarning`, delegate to `run_graph_turn` when `MINICODE_USE_GRAPH=1` or `runtime={"useGraph": True}` (preserve `store`/`metrics_collector`/`hooks` no-op passthrough for compat), else fallback to legacy loop; keep `build_model_graph` as the single topology.
- [x] Tests: `test_graph_checkpoint_resumes_across_threads`, `test_authorize_denies_batch`, plus shim equivalence test.
- [x] Verify: `tests/test_langgraph_runtime.py` + `tests/test_graph_multi_tool.py` + `tests/test_graph_checkpoint.py` = 38 passed; focused regression 329 passed (1 sandbox network noise excluded).

### Task 7: Graph parity — store/provider fallback, callbacks, and default flip (slice 5, pending)

**Files:** `minicode/graph/runtime.py`, `minicode/graph/builder.py`, `minicode/agent_loop.py`, `tests/test_agent_loop.py`

- [ ] Parity gap (2026-08-21, MINICODE_USE_GRAPH=1): 9/27 `test_agent_loop.py` fail when shim delegates to graph — callbacks (`test_emits_callbacks` progress "working" vs phase line), recoverable `pause_turn` (`test_handles_recoverable_pause_turn`), store passthrough (`test_passes_store_to_provider_adapter`), verify guard (`test_single_deep_verify_phase_blocks...`), fallback-chain 5 tests (`test_switches_to_fallback_model...`, `test_does_not_bounce...`, `test_infers_active_runtime_model...`, `test_uses_default_runtime_fallback_chain...`, `test_provider_outage_guidance...`). Root causes: `run_graph_turn.next_step` does not forward `store`/`on_thinking_delta` via signature inspection (cf. `agent_loop._model_next`), graph error fallback does not implement `ModelSwitcher`/provider-channel retry, `GraphEventSink`/ `assistant_followup` progress wiring diverges from legacy `on_progress_message`, and `StepDiagnostics(stopReason=pause_turn)` recoverable path not mirrored.
- [ ] Wire `store` + streaming kwargs into `next_step` (inspect `model.next` signature, forward `store`/`on_thinking_delta`/`on_stream_chunk`), and surface provider-switch logic (at least: typed error → fallback chain attempt before emitting `kind="error"` blocked stop; mirror `agent_loop._model_next` + fallback guidance).
- [ ] Align callbacks: `assistant_followup`/`step_policy` progress events vs legacy `contentKind="progress"` nudges; ensure `on_tool_start`/`on_tool_result`/`on_assistant_message`/`on_progress_message`/`on_runtime_event` parity via `TurnEventQueue` adapters (already wired) — fix the 4 single-case failures above before flipping default.
- [ ] Flip shim default: `MINICODE_USE_GRAPH` default True (or remove opt-in gate), keep `runtime={"useGraph": False}` / env `MINICODE_USE_GRAPH=0` as rollback escape hatch; delete legacy loop only after `MINICODE_USE_GRAPH=1` full suite 1421+ passed (sandbox network noise excluded).
- [ ] Verify: `MINICODE_USE_GRAPH=1 .venv/bin/python -m pytest tests/test_agent_loop.py -q` 27 passed; `tests/test_langgraph_runtime.py` + `tests/test_graph_multi_tool.py` + `tests/test_graph_checkpoint.py` 36+? still green; full suite 1421 passed (1 known `test_provider_smoke_reaches_local_openai...` PermissionError noise).


