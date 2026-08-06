## Why

The latest Ascend 910B run proves that the Leyline path executes, but its one-token reference gate accepts a shared newline while full, honest-edited, and Leyline generations otherwise diverge. The run also omits device-cache and raw-logit evidence, and the connector currently conflates record and amortize completion state, so the evidence is not strong enough to distinguish workload failure, implementation error, and expected Leyline approximation.

## What Changes

- Replace instruction-heavy repeated raw workloads with a larger, completion-native base-model corpus whose expected comparison point is a meaningful non-whitespace token.
- Add balanced workload families for admissible irrelevant deletions, boundary and position-shift coverage, counterfactual variants, deleted-only evidence diagnostics, and negative controls, with enough cases to report per-family rather than anecdotal results.
- Keep a separate Chat/structured corpus and require instruction preflight plus full, honest-edited, and counterfactual baseline qualification before making semantic claims about Leyline.
- Harden reference admission so whitespace-only or trivially short agreement remains diagnostic and cannot be presented as behavioral correctness.
- Require raw first-token distribution evidence and complete per-layer/per-TP-rank cKV/Kpe numerical evidence before overall qualification.
- Correct the record/amortize lifecycle and evidence contract so only record directives create source sessions, amortize completion does not leak or silently replace sessions, and the harness verifies the record arm separately from transform application.
- Require all expected MLA layers on every TP rank to transform successfully, expose completeness in the response/report, and fail closed on partial transformation.
- Record imported module provenance and reconcile repository, distribution, CANN, model, and tokenizer identities in 910B reports.

## Capabilities

### New Capabilities

- `leyline-validation-qualification`: Defines the expanded base and Chat workload suites, meaningful admission rules, token/logit diagnostics, device-cache qualification, reporting, and 910B acceptance sequence.
- `leyline-transaction-integrity`: Defines record/amortize session lifecycle, full layer/rank transformation completeness, failure behavior, and execution evidence semantics.

### Modified Capabilities

None. No main specs have been archived yet; this change introduces the hardened qualification contracts as new capabilities while superseding weaker assumptions in the active validation changes.

## Impact

- Validation harness and reports under `benchmarks/leyline/`, including workload schema/data, runner examples, raw-logit and cache-capture joins, and environment collection.
- Leyline connector lifecycle, worker result aggregation, response metadata, and focused unit/integration tests under `vllm_ascend/distributed/kv_transfer/leyline/`.
- Ascend 910B qualification procedure and result artifacts; existing schema-v2 runs remain historical diagnostic evidence and are not upgraded to passing qualification results.
- No serving API is changed for requests that do not opt into the Leyline namespace. The meaning of Leyline `recorded` evidence is tightened and may require consumers of experimental metadata to update.
