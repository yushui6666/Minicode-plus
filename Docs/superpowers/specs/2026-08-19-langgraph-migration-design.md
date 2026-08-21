# LangGraph Migration Design

## Goal

Introduce LangGraph as a new orchestration boundary while keeping the existing
MiniCode CLI and runtime behavior unchanged during the first migration slice.

## Scope

The first slice defines a small serializable `AgentState` and a compiled graph
with routing, tool execution, verification, and finalization nodes. Existing
model adapters, tools, memory, permissions, TUI, and TypeScript code remain
unchanged and are migrated behind explicit adapters later.

Slice 2 (2026-08-20) migrates the turn-kernel control semantics into
`build_model_graph`: step-policy phases, assistant decision routing
(progress / retry / fallback / final), empty-response and recoverable-thinking
retries, the widening state machine, the strict-verification evidence guard,
and typed stop reasons. Kernel decision functions are reused unchanged from
`minicode/turn_kernel.py` through snapshot adapters
(`snapshot_to_turn_state` / `turn_state_to_snapshot`), so every kernel field
inside `AgentState` stays a plain serializable value.

## Data flow

Slice 1: `START -> route -> execute_tool -> verify -> finalize -> END` for tool
work; completed input routes directly from `route` to `finalize`.

Slice 2 (turn-kernel topology, the default for `run_graph_turn`):

```
START -> load_context -> compact -> step_policy
step_policy -[budget exhausted]-> finalize(max_steps)
step_policy -> model -> classify_step
classify_step -[tool]-> authorize -[allowed]-> execute_tool -> observe_tool
authorize -[denied]-> finalize
observe_tool -[await_user]-> finalize
observe_tool -> verify -> repair? -> step_policy
classify_step -[progress/retry/guard]-> assistant_followup -> step_policy
classify_step -[fallback+widen_needed]-> widen -> step_policy
classify_step -[final / fallback]-> finalize -> END
```

All back-edges converge on `step_policy` (the retired loop's `while
turn_state.has_remaining_steps()` head), so budget checks, policy derivation,
and phase events run exactly once per step.

## Fixed defects carried by slice 2

- `run_graph_turn` now passes an explicit `recursion_limit`
  (`(max_steps + widening bonus) * 8 + 32`); previously the LangGraph default
  of 25 supersteps ended long turns with `GraphRecursionError` (the
  interactive CLI crashed outright).
- Empty assistant responses no longer append empty progress messages and spin:
  they retry with nudges up to the profile limit, then stop with a typed
  reason (`blocked` / `verification_failed` / `widen_needed`).
- Tool rounds now append the `assistant_tool_call` + `tool_result` message
  pair (with `isError`), fixing sub-agent step accounting in
  `minicode/tools/task.py`.

## Compatibility and testing

The graph is exposed through `minicode.graph.build_agent_graph` and does not
replace an existing CLI entrypoint. Tests use a real compiled graph and a small
tool executor callback, with no provider credentials or network calls.

`build_model_graph(turn_kernel=False)` (selected by
`runtime={"turnKernel": "thin"}`) keeps the slice-1 thin topology as a
construction-time escape hatch; the flag is scheduled for removal in slice 3.
Kernel strength follows `RuntimeProfile` fields: the `single` profile keeps
widening and the strict evidence guard dormant, `single-deep` enables them.
Caller-visible changes: transcripts gain phase progress lines and nudge
messages; headless empty-response scenarios return a typed fallback message
instead of spinning; `tools/task.py` reports real `assistant_tool_call` counts.
