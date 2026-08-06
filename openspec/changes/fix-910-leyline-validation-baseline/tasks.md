## 1. Prompt and Request Contract

- [x] 1.1 Add validated `reference_prefix` and `structured_json` evaluation configuration with explicit prompt-format compatibility checks
- [x] 1.2 Implement canonical raw and chat-template prompt planning that fails unless the full prompt becomes the edited prompt through one exact token deletion
- [x] 1.3 Send preformatted token IDs with `add_special_tokens=false` and `return_token_ids=true`, and retain returned generation IDs in every arm result
- [x] 1.4 Request configurable top-N first-token log probabilities for diagnostic runs and retain candidate token IDs, scores, and explicit evidence type
- [x] 1.5 Add unit tests for raw deletion, canonical chat deletion, missing templates, token-boundary replacement rejection, and request/response token-score fields

## 2. Checkpoint Identity and Baseline Preflight

- [x] 2.1 Extend 910 environment collection with resolved model/tokenizer paths, revisions, selected configuration fields, chat-template presence, and streamed SHA-256 artifact manifests
- [x] 2.2 Add the short deterministic structured-JSON preflight using the configured tokenizer, prompt format, endpoint, and greedy decoding settings
- [x] 2.3 Fail closed for semantic admission and performance on a missing or failed structured preflight while preserving explicitly requested token-score and cache diagnostics
- [x] 2.4 Add unit tests for local checkpoint manifests and structured preflight pass, parse failure, endpoint failure, and skip behavior in reference mode

## 3. Evaluation and Report Semantics

- [x] 3.1 Implement full↔honest, full↔Leyline, and honest↔Leyline first-token agreement, common-prefix length, first-divergence, top-k overlap, and top-1/top-2 margin diagnostics
- [x] 3.2 Add an opt-in internal raw first-token logit capture with provenance and keep it distinct from API log-probability evidence
- [x] 3.3 Implement category-aware structured baseline gates for admissible, counterfactual, mechanism-diagnostic, and negative-control cases
- [x] 3.4 Keep diagnostic results reportable after baseline failure while requiring baseline admission plus recorded/applied/positive-transform/no-fallback evidence for Leyline acceptance
- [x] 3.5 Emit schema-v2 correctness reports with separate semantic/reference admission, evaluation target, prompt format, preflight, checkpoint identity, pairwise token-score diagnostics, and Leyline execution fields
- [x] 3.6 Update staged report merging to reject schema-v1 or mixed-contract inputs and recompute schema-v2 gates without changing their meaning
- [x] 3.7 Prevent performance mode from running when the selected baseline, Leyline execution, numerical, or rollback gates fail, and report the blocking reasons
- [x] 3.8 Add focused tests for both evaluation modes, every workload category, diagnostic-only reporting, fallback rejection, schema-v2 output, merge compatibility, and performance gating

## 4. Device Cache Capture Diagnostics

- [x] 4.1 Add opt-in worker capture around each selected 910 Leyline layer transform, cloning source rows before the write and reading destination rows after NPU synchronization
- [x] 4.2 Write rank-scoped capture artifacts and a request manifest containing layer/rank completeness, slot mapping, positions, inverse frequencies, dtype, block size, and synchronization provenance
- [x] 4.3 Extend offline cache comparison with per-layer/per-rank cKV bitwise failures, independent FP32 Kpe reference errors, required position-delta coverage, and aggregate max/mean/percentile statistics
- [x] 4.4 Add tests for disabled capture, bounded row selection, source/destination aliasing, multi-rank manifest merging, missing layer/rank failure, cKV mismatch, and Kpe tolerance failure

## 5. 910 Configuration and Documentation

- [x] 5.1 Make the DeepSeek-V2-Lite base runner example explicitly select raw `reference_prefix` evaluation and add a separate DeepSeek-V2-Lite-Chat `structured_json` example
- [x] 5.2 Document the parallel 910 workflow: shared identity checks, evaluation-baseline repair, immediate token/cache diagnostics, evidence joining, then performance
- [x] 5.3 Document that `/v1/chat/completions` is a diagnostic option while the canonical harness applies the chat template client-side and submits explicit token IDs to `/v1/completions`
- [x] 5.4 Document schema-v1 incompatibility and the distinction between API log probabilities, raw logits, reference agreement, semantic correctness, and cache numerical correctness

## 6. Verification on the 910 Branch and Host

- [ ] 6.1 Run focused harness, score-diagnostic, and cache-capture unit tests plus formatting and static checks on the current `leyline-mla-ascend` branch
- [ ] 6.2 Synchronize that exact checkout to the 910 environment and confirm repository imports, installed package versions, CANN version, and model/tokenizer manifests agree with the report
- [ ] 6.3 Start the diagnostic track immediately with local DeepSeek-V2-Lite full, honest-edited, and Leyline arms, recording first-token IDs, top-token scores, pairwise common prefixes, and execution metadata without semantic claims
- [ ] 6.4 Capture representative cache rows from every required layer and TP rank during the diagnostic run and evaluate cKV bitwise equality plus Kpe FP32 rotation error regardless of structured-oracle status
- [ ] 6.5 In parallel, run a pinned DeepSeek-V2-Lite-Chat checkpoint through the structured preflight and require the full and category-appropriate honest/counterfactual baselines to behave as specified
- [ ] 6.6 After the evaluation track passes, apply the structured oracle to Leyline and join semantic/reference, execution, numerical, and rollback evidence for final correctness acceptance
- [ ] 6.7 Run warm performance measurements only after every applicable correctness gate passes and all score/cache capture hooks are disabled; report cold-start costs separately
