## 1. Shared Evidence and Qualification Model

- [x] 1.1 Add a schema-versioned run identity shared by correctness, cache, raw-logit, cache-off, rollback, and finalized reports
- [x] 1.2 Add stable evidence keys for expanded case, counterfactual variant, repetition, arm, request, rank, and decode step
- [x] 1.3 Extract environment, execution, numerical, rollback, baseline, target, and suffix classification into a pure evaluator reusable by online and offline reports
- [x] 1.4 Add artifact hashing and source-manifest helpers that preserve immutable input paths, sizes, and SHA-256 values
- [x] 1.5 Distinguish missing evidence, measured gate failure, invalid provenance, task limitation, and accepted autoregressive divergence in the shared result schema

## 2. Offline Finalization and Cache-Off Joining

- [x] 2.1 Implement `benchmarks/leyline/finalize_validation.py` without any inference client dependency or endpoint calls
- [x] 2.2 Validate corpus, evaluation contract, checkpoint/tokenizer hashes, imported modules, commits, runtime topology, block size, TP ranks, and run identity across inputs
- [x] 2.3 Preserve every connector-on repetition and reject auxiliary request IDs that cannot be traced to the immutable source report or an explicitly linked diagnostic rerun
- [x] 2.4 Align cache-off cases, variants, and all repetitions while permitting only the documented connector/prefix-cache topology difference
- [x] 2.5 Update `merge_reports.py` to use the shared repetition-preserving join path or retain it as a compatible wrapper without lossy aggregate-arm merging
- [x] 2.6 Derive numerical and rollback gates only from validated reports and stop manual prerequisite booleans from qualifying correctness
- [x] 2.7 Emit `joined-qualification.json` with source hashes, gate details, case classifications, retained-context summary, and the first determining gate

## 3. Bounded Decode-Step Raw Logits

- [x] 3.1 Extend validation request IDs with opaque run, case, arm, and repetition provenance while preserving unique request behavior
- [x] 3.2 Extend raw-logit capture from first-token-only to explicit finite decode steps and include decode-step provenance in filenames and metadata
- [x] 3.3 Add run/case request selection plus configurable artifact-count, byte, and maximum-step safety limits
- [x] 3.4 Preserve backward compatibility for existing first-token capture artifacts and label legacy captures accurately
- [x] 3.5 Extend `compare_logits.py` to load step-scoped captures, validate rank/request/step alignment, and compare full-to-honest and full-to-Leyline distributions
- [x] 3.6 Implement `plan_divergence_capture.py` to select reproducible short-prefix cases and generate a bounded targeted capture plan
- [x] 3.7 Add targeted diagnostic execution support to `run_validation.py` without changing ordinary qualification corpus semantics
- [x] 3.8 Require targeted reruns to reproduce the source prefix and divergence index before their logits are marked correlatable
- [x] 3.9 Report selected-token margins, top-k overlap, maximum absolute difference, cosine similarity, Jensen-Shannon divergence, counts, bytes, and incomplete-budget reasons per step

## 4. Informative Retained-Context Workloads

- [x] 4.1 Add at least four retained-context candidates across codeword lookup, removed mapping, state transition, and code/symbol completion families
- [x] 4.2 Ensure each candidate places deleted evidence before a block-aligned transformable surviving region and evaluates its query in the normal-prefill tail
- [x] 4.3 Add diagnostic preflight requiring stable full target, stable honest non-target, and different full/honest first tokens across all repetitions
- [x] 4.4 Classify non-discriminating candidates as `diagnostic_uninformative` without hiding their outputs or changing admitted success rates
- [x] 4.5 Require at least four informative diagnostics across at least three families before retained-context coverage is complete
- [x] 4.6 Report Leyline token, prefix, first-divergence, and raw-logit distance relative to both full and honest for every informative diagnostic
- [x] 4.7 Add schema, tokenizer-boundary, exact-deletion, block-feasibility, family-count, informative-admission, and diagnostic-separation tests

## 5. Guarded Device Rollback Validation

- [x] 5.1 Add a validation-only server environment gate for Leyline fault injection that is disabled by default
- [x] 5.2 Add a request-scoped, allow-listed post-write failpoint for one transaction, rank, and layer with no arbitrary code or path input
- [x] 5.3 Propagate injected worker failure through TP aggregation while recording reached rank/layer and preventing `applied=true`
- [x] 5.4 Expose non-sensitive rollback counters for touched-block invalidation, normal-prefill recovery, session/inflight/pending cleanup, and transaction-owned references
- [x] 5.5 Implement `benchmarks/leyline/run_rollback_validation.py` with honest control, successful record, injected amortize, target comparison, and schema-versioned evidence output
- [x] 5.6 Make the rollback runner fail unless injection happens after a destination write, fallback is honest, cleanup counters reach zero, and the target agrees with the control
- [x] 5.7 Reject request injection on a normal server and prove normal serving is unchanged when the environment gate is absent

## 6. Configuration and Operator Documentation

- [x] 6.1 Add example runner configuration for run identity, device capture, first-token logits, targeted divergence capture, cache-off, and rollback validation
- [x] 6.2 Publish `benchmarks/leyline/VALIDATION_910B.md` from the change-local `verification-guide.md` and update commands to match the implemented CLI exactly
- [x] 6.3 Document separate connector-on, cache-off, failure-injection, and capture-disabled performance server lifecycles
- [x] 6.4 Document artifact permissions, size budgeting, source-hash publication, large NPZ/raw-vector retention, and secret-handling precautions
- [x] 6.5 Add `jq` checks and a decision table for every environment, execution, numerical, rollback, baseline, target, retained-context, and suffix gate

## 7. Local Verification

- [x] 7.1 Add golden finalizer tests for a complete pass and every ordered determining gate
- [x] 7.2 Add finalizer failure tests for mixed identities, mutated source evidence, unknown requests, missing repetitions, missing layer/rank/delta evidence, and unsupported schema versions
- [x] 7.3 Add cache-off join tests proving all repetitions and counterfactual variants are retained
- [x] 7.4 Add raw-logit tests for selected steps, request filters, budgets, rank completeness, legacy first-token compatibility, and non-reproducing divergence reruns
- [x] 7.5 Add rollback tests for disabled opt-in, allowed post-write injection, TP partial failure, block invalidation, recomputation, idempotent cleanup, and evidence generation
- [ ] 7.6 Run the focused Leyline suites plus repository formatting, lint, and static checks on `leyline-mla-ascend`

## 8. Ascend 910B Qualification

- [ ] 8.1 Synchronize a clean exact commit to the 910B host and collect reconciled environment/model/tokenizer/runtime identity
- [ ] 8.2 Run the connector-on base corpus for three isolated repetitions with TP4 device capture and step-zero raw logits
- [ ] 8.3 Produce `cache-comparison.json` covering all 27 layers, 4 ranks, native RoPE, cKV, Kpe, block transitions, and deltas 0/1/127/128/129/1024
- [ ] 8.4 Produce the first-token raw-logit comparison for every admitted and informative diagnostic request/rank
- [ ] 8.5 Run bounded targeted divergence capture and mark only reproduced prefix/divergence evidence correlatable
- [ ] 8.6 Run the full cache-off corpus for three repetitions on the identity-compatible no-connector/no-prefix-cache server
- [ ] 8.7 Run guarded post-write failure injection on a dedicated concurrency-one server and produce `rollback-report.json`
- [ ] 8.8 Confirm at least four informative retained-context diagnostics across three families or publish explicit insufficient-coverage evidence
- [ ] 8.9 Finalize all immutable artifacts into `joined-qualification.json` and verify every case names its determining gate
- [ ] 8.10 Run warm performance only if environment, execution, numerical, rollback, baseline, and Leyline target gates pass with all diagnostic hooks disabled
