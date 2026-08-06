## 1. Transaction Lifecycle and Completeness

- [x] 1.1 Restrict source-session creation to `record` directives and make amortize finish cleanup idempotent without re-recording the edited request
- [x] 1.2 Update Leyline execution evidence to join `record.recorded` with amortize application metadata instead of requiring `recorded=true` on the amortize response
- [x] 1.3 Add success, worker-failure, cancellation, fallback, and shutdown tests that assert session, inflight-plan, pending-metadata, and block-reference cleanup
- [x] 1.4 Derive the expected MLA layer set from the active model/runtime contract and fail a worker transform when any expected layer is absent, incompatible, skipped, or failed
- [x] 1.5 Extend per-rank and aggregated worker results with expected/transformed layer counts, expected/successful rank counts, missing layer/rank evidence, and `transform_complete`
- [x] 1.6 Fail closed on partial TP-wide transforms, invalidate every destination block in the plan, and verify that all affected tokens return to normal-prefill accounting

## 2. Completion-Target Evaluation Contract

- [x] 2.1 Add validated `completion_target` runner configuration while retaining `structured_json` and making `reference_prefix` explicitly diagnostic-only
- [x] 2.2 Extend prompt planning to derive exact target suffix token IDs from canonical `prompt + expected_completion` tokenization and reject whitespace-only, empty, or boundary-replacing targets
- [x] 2.3 Implement base admission requiring full, honest-edited, and all counterfactual arms to generate the declared target before Leyline is evaluated
- [x] 2.4 Revise case and top-level gates so diagnostic reference matches cannot emit semantic or overall correctness acceptance
- [x] 2.5 Extend the workload schema with corpus version, evaluation mode, family, claim type, expected completion, removed variants, and requested token-position coverage
- [x] 2.6 Add a corpus validator for minimum family counts, unique case identifiers, compatible fields, and explicit expected targets/oracles
- [x] 2.7 Preserve existing schema-v2 workloads through an explicit legacy diagnostic mode without silently upgrading their evidence meaning

## 3. Expanded Base-Model Workloads

- [x] 3.1 Add tokenizer-aware deterministic filler construction that achieves requested deletion lengths and proves the exact full-minus-span invariant
- [x] 3.2 Add the six completion-native admissible cases: route label, inventory reorder, incident severity, feature state, build language, and deterministic Python continuation
- [x] 3.3 Add four admitted numerical stress cases covering deletion lengths 1, 127, 128, 129, a cross-block span, delta-zero prefix rows, and a configured long shift
- [x] 3.4 Add route-neutral and code-comment counterfactual cases with at least three removed-text variants each
- [x] 3.5 Add two deleted-only mechanism diagnostics and two negative controls, with report families excluded from admissible success rates
- [x] 3.6 Version the new sixteen-case base corpus separately from the historical repeated-JSON workload and document every case's intended claim and expected target
- [x] 3.7 Add schema, tokenizer-boundary, family-count, exact-span, and baseline-admission tests for the complete base corpus

## 4. Independent Chat Semantic Workloads

- [x] 4.1 Add six concise structured Chat cases covering routing, inventory, access policy, counterfactual incident response, deleted-secret diagnosis, and missing-evidence control
- [x] 4.2 Ensure every Chat case renders canonical full and edited prompts through the pinned tokenizer template and proves one exact token deletion
- [x] 4.3 Require the same-endpoint structured preflight plus category-appropriate full, honest-edited, and counterfactual baselines before Leyline semantic evaluation
- [x] 4.4 Add a pinned Chat runner example and tests proving that template presence alone cannot admit weights that fail structured preflight

## 5. Logit and Cache Numerical Evidence

- [x] 5.1 Complete the internal first-target-token raw-logit capture and store full-vocabulary vectors outside summary JSON with request, model, rank, and sampling provenance
- [x] 5.2 Add raw-logit comparison summaries for selected token, margin, top-k overlap, maximum absolute difference, cosine similarity, and normalized distribution distance
- [x] 5.3 Capture the active native runtime RoPE inverse-frequency buffer or equivalent cos/sin table independently from the connector-derived frequency vector
- [x] 5.4 Make the offline comparator fail on connector/native RoPE provenance mismatch and use the native source for FP32 Kpe delta-rotation expectations
- [x] 5.5 Extend bounded device capture to require first/last rows, block transitions, and available deltas 0, 1, 127, 128, 129, and the configured long shift
- [x] 5.6 Join every expected layer/rank manifest into the correctness report and fail numerical qualification on missing captures, cKV bitwise mismatch, Kpe threshold failure, or insufficient delta coverage

## 6. Reporting, Repetition, and Environment Identity

- [x] 6.1 Run at least three isolated greedy repetitions per required case with fresh request IDs and cache salts, preserving individual results and reporting stability
- [x] 6.2 Report requested and achieved deletion spans, position deltas, transformed block range, normal-prefill tail, target token IDs, common prefix, and first divergence for every arm
- [x] 6.3 Implement the ordered final classification for invalid environment, connector failure, numerical failure, invalid baseline, Leyline target limitation, and accepted target with later divergence
- [x] 6.4 Record `vllm` and `vllm_ascend` imported module paths alongside distribution versions and repository commits, and block qualification when provenance cannot be reconciled
- [x] 6.5 Expand CANN identity collection to find the active installation's available version evidence and retain unresolved version state as an explicit qualification blocker
- [x] 6.6 Add actual cache-off control configuration and retain cold-start transform cost separately from repeated warm correctness and later performance results

## 7. Local Verification

- [x] 7.1 Add focused unit tests for completion-target planning, whitespace rejection, diagnostic-only reference gates, counterfactual admission, and legacy report behavior
- [x] 7.2 Add connector tests for record/amortize evidence separation, all-layer/all-rank success, missing-layer failure, partial-write rollback, and reference-count cleanup
- [x] 7.3 Add comparator tests that inject slot, rotation-sign, layout, native-frequency, missing-layer, missing-rank, and required-delta failures
- [ ] 7.4 Run the focused Leyline test suites, formatting, lint, and static checks on the current `leyline-mla-ascend` branch

## 8. Ascend 910B Qualification

- [ ] 8.1 Synchronize the exact clean checkout to the 910B host and verify imported module paths, repository commits, distributions, CANN, runtime topology, and model/tokenizer hashes
- [ ] 8.2 Run three repetitions of the sixteen-case base corpus and qualify full, honest-edited, counterfactual, cache-off, and normal-path baselines before interpreting Leyline targets
- [ ] 8.3 Run TP4 device capture for every expected layer/rank and compare cKV, Kpe, native RoPE provenance, block transitions, and required position deltas
- [ ] 8.4 Run raw-logit capture for admitted base cases and correlate first-target distribution differences with generated common-prefix divergence
- [ ] 8.5 Run the six-case pinned Chat corpus through structured preflight, baseline admission, Leyline semantic evaluation, and diagnostic-family separation
- [ ] 8.6 Publish a joined qualification report that states which evidence gate determines each failure and whether any remaining suffix divergence is numerical, workload-level, task-level, or autoregressive-only
- [ ] 8.7 Run warm performance measurements only after applicable environment, execution, numerical, rollback, baseline, and Leyline target gates pass with all capture hooks disabled
