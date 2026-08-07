## Why

The latest Ascend 910B validation classified all 17 cases as connector failures because none of the workloads produced a reusable 128-token target block backed by resident source blocks. The corpus validator currently checks schema and semantic targets but not connector feasibility, so a syntactically valid suite can complete without executing a single Leyline cache transform.

## What Changes

- Add a tokenizer- and mapping-aware feasibility planner that predicts recordable source blocks, reusable target range, transformed token count, and explicit rejection reasons before issuing requests.
- Reject qualifying workloads that cannot transform at least one complete block; retain an explicit diagnostic mode for intentionally non-transformable cases.
- Rebuild the base-completion corpus so admitted semantic, position-stress, counterfactual, and deleted-evidence cases preserve enough surviving context and resident source coverage to execute Leyline.
- Add a minimal transform smoke gate that must prove record, application, TP/layer completeness, positive transformed tokens, no fallback, and required capture evidence before the full suite runs.
- Separate gate-decision stability from exact generated-token stability in the report.
- Requalify or replace unstable completion targets such as `route-label` and `route-neutral-variants` instead of treating observed accidental tokens as new oracles.

## Capabilities

### New Capabilities

- `leyline-workload-feasibility`: Defines static connector-feasibility planning, transformable corpus admission, smoke gating, and execution-aware reporting for Leyline validation workloads.

### Modified Capabilities

None. The related qualification contracts exist only in active changes and have not been archived into main specs.

## Impact

- Workload planning, validation, reporting, and runner configuration under `benchmarks/leyline/`.
- Base-completion workload data and deterministic tokenizer-aware filler generation.
- Focused unit tests for block-aligned source/target mapping, infeasible diagnostics, smoke gating, and report stability semantics.
- Ascend 910B validation procedure: the full qualification matrix will stop early unless a real cache transform is first demonstrated.
- No non-Leyline serving API or production connector behavior changes are required.
