## Why

The current Ascend 910B report proves that Leyline executes and preserves every admitted completion target, but it cannot qualify the implementation because cache comparison, raw-logit, cache-off, and rollback evidence are either absent or cannot be joined without rerunning requests. The retained-context corpus also has only one baseline-discriminating case, so the report cannot determine whether Leyline's tendency to follow honest-edited output is a numerical defect, a workload limitation, or the model's actual retained-context behavior.

## What Changes

- Add an offline finalization flow that joins an immutable correctness run with cache, raw-logit, cache-off, rollback, and environment evidence and recomputes qualification without issuing new model requests.
- Preserve all repetitions, request IDs, cache salts, checkpoint identity, and arm-level results when combining staged connector-on and cache-off runs.
- Extend opt-in raw full-vocabulary logit capture from the first token to bounded configured decode steps, then correlate the logits that select the first divergent token with full/honest/Leyline prefixes.
- Add a guarded 910B failure-injection workflow and a machine-verifiable rollback report instead of relying on a manually asserted `rollback_passed` boolean.
- Expand retained-context diagnostics across multiple task families and require an empirical full-versus-honest first-token distinction before a diagnostic is considered informative.
- Add a repository Markdown runbook covering host preparation, commands, artifacts, pass/fail gates, storage precautions, and the final decision tree.

## Capabilities

### New Capabilities

- `leyline-qualification-finalization`: Immutable, provenance-checked joining of correctness, numerical cache, raw-logit, cache-off, rollback, and environment evidence into one final qualification report.
- `leyline-divergence-diagnostics`: Bounded decode-step logit capture and an empirically informative retained-context workload family for distinguishing numerical, workload-level, task-level, and autoregressive divergence.
- `leyline-rollback-validation`: Explicitly guarded device failure injection and externally verifiable rollback/cleanup evidence on Ascend 910B.

### Modified Capabilities

None. The repository has no synchronized main OpenSpec capabilities yet; this change closes gaps left by the active validation changes through new qualification capabilities.

## Impact

- Validation code under `benchmarks/leyline/`, including report comparison/finalization, staged report merging, runner configuration, workloads, schemas, and the 910B runbook.
- Opt-in diagnostic code in the Ascend model runner and Leyline connector for bounded decode-step logits and guarded failure injection.
- Focused unit tests under `tests/ut/distributed/kv_transfer/leyline/`.
- Ascend 910B qualification artifacts and execution procedure; normal serving behavior remains unchanged unless the explicit diagnostic environment and request controls are enabled.
