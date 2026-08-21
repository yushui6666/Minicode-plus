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
