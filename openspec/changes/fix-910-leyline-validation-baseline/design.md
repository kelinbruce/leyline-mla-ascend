## Context

The current 910 validation harness builds one raw token sequence, sends it to `/v1/completions`, parses the first JSON object from decoded output, and applies that oracle to every checkpoint. The first hardware report used `/opt/foundation_model/DeepSeek-V2-Lite`, recorded both revisions only as `local`, and produced continuation-style text in every full baseline. It also recorded repository and installed-package identities that did not clearly describe the same source checkout.

The normal edited arms were mutually identical, which is useful evidence that the baseline failure was not introduced by APC or the disabled connector path. It does not establish Leyline correctness: Leyline output diverged from full and honest baselines. Cache-level numerical capture and first-token diagnostics are independent of structured-oracle validity and can proceed while the model/prompt baseline is being corrected.

This change is implemented on the existing 910B `leyline-mla-ascend` branch. The connector and BF16 cache transformation remain unchanged.

## Goals / Non-Goals

**Goals:**

- Make every correctness report explicit about whether it tests base-model reference behavior or instruction-model semantic correctness.
- Establish checkpoint identity and a short prompt-format baseline before spending time on the repeated long-context workload.
- Preserve the Leyline edit as an exact token deletion under raw and chat-template formatting.
- Compare actual server generation token IDs and first-token score evidence across full, honest-edited, and Leyline arms.
- Capture and independently evaluate the device cKV/Kpe rows transformed on every selected layer and TP rank.
- Keep diagnostic evidence available when semantic admission fails while preventing it from being mislabeled as accepted correctness.
- Make staged 910 correctness reports merge without changing their evaluation meaning.

**Non-Goals:**

- Implement, enable, or qualify Leyline on Ascend 310.
- Change the 910 MLA cache transformation, rotation algorithm, scheduler protocol, or fallback behavior.
- Treat instruction-following success as numerical cache proof.
- Treat base-model token agreement as semantic-oracle validation.
- Optimize transformation latency or run performance before the corrected baseline is accepted.

## Decisions

### 1. Separate the two scientific claims

The runner configuration gains an explicit evaluation object:

- `reference_prefix` is the default for `DeepSeek-V2-Lite` base checkpoints. The full arm's returned generation token IDs are the reference, and a configurable prefix length controls comparison. Reports label admission as reference-only and keep semantic admission false.
- `structured_json` is for an instruction-tuned checkpoint such as `DeepSeek-V2-Lite-Chat`. Parsed output is compared with the declared oracle. This mode requires `prompt_format=chat_template` and a successful structured preflight.

An unknown mode, a prefix length below one, `structured_json` without a usable chat template, or another incompatible configuration fails before the workload runs. Merely renaming a local base-model directory to a Chat model name is not treated as identity evidence.

The alternative of changing every run to `/v1/chat/completions` was rejected. Leyline needs stable, client-visible token IDs and deletion indices; client-side formatting followed by a token-ID completion request keeps those indices under harness control. The chat endpoint remains useful as a manual diagnostic probe, not the canonical correctness path.

### 2. Preserve canonical prompt formatting and prove exact deletion

Raw mode continues to produce full and edited token sequences with one declared removed interval.

Chat mode renders the same single user message twice through `apply_chat_template(..., add_generation_prompt=True, tokenize=False)`: once with the removed text and once without it. Both rendered strings are tokenized without additional special tokens. The planner finds the common prefix and required length difference and accepts the plan only when deleting one contiguous interval from the canonical full token sequence produces the canonical edited sequence exactly.

This approach preserves the tokenizer's normal boundary behavior. If byte-pair merging turns the edit into a token replacement rather than one deletion, the case fails closed; the workload must add stable textual boundaries instead of silently constructing a non-canonical segmented tokenization.

Every request sets `add_special_tokens=false`, because BOS and role tokens are already present after prompt construction. Unit tests verify `full[:delete_start] + full[delete_end:] == edited` for raw and chat formats.

### 3. Use returned token IDs and first-token scores as behavioral evidence

Completion requests set `return_token_ids=true`. The runner stores the returned generation IDs alongside decoded text and uses them directly in `reference_prefix` mode. Missing or short ID lists fail the corresponding gate.

Diagnostic requests also ask the serving API for a configurable top-N set of token log probabilities. For each arm the report records the selected first token, candidate token IDs and log probabilities, and the top-1/top-2 margin. Because log-softmax subtracts the same normalization constant from every candidate, the top-1/top-2 log-probability difference is also their logit margin; the report still labels the source as API log probabilities rather than raw logits.

If full-vocabulary raw logits are needed, an opt-in internal capture hook records them before sampling and marks the layer/runtime provenance. API log probabilities and internally captured raw logits remain separate evidence types. The ordinary validation path does not require the high-volume raw-logit capture.

Decoded text remains in the report for analysis, and JSON extraction remains limited to `structured_json` evaluation. The preflight and semantic workload use greedy decoding without grammar-forced JSON so constrained decoding cannot disguise a checkpoint that does not follow the instruction.

The report computes full↔honest, full↔Leyline, and honest↔Leyline comparisons. Each pair records first-token agreement, actual common-prefix length, first divergent position, top-k overlap, and available score evidence at the divergence. The configured pass threshold may begin at one token for paper-style first-token experiments, while longer prefixes remain available for stricter behavioral comparison.

### 4. Capture device cache transformation evidence independently

Device capture is opt-in and diagnostic-only. The worker connector captures the rows selected by the Leyline slot mapping immediately before and after each per-layer `transform_mla_cache` call. Source rows are cloned before any destination write; destination rows are copied only after the layer operation is complete and the NPU stream has been synchronized.

Each capture includes request ID, session ID, layer name, TP rank, source and destination slots, source cKV/Kpe, actual destination cKV/Kpe, old/new positions, inverse frequencies, dtype, block size, and synchronization metadata. Each TP process writes a separate rank-scoped artifact to avoid concurrent writers corrupting one NPZ. A manifest links rank/layer artifacts back to one request and records whether the expected layer/rank set is complete.

Capture selection supports a bounded row budget and required position deltas so debug evidence does not copy the complete model cache to host memory. The first qualification run covers every layer and TP rank while sampling representative rows including deltas 0, 1, 127, 128, 129, and 1024 where the workload provides them.

The offline comparison path checks source cKV against the actual destination bitwise and evaluates actual Kpe against the independent FP32 delta-rotation reference. It reports failures by layer, rank, slot, and position delta plus aggregate maximum, mean, percentile absolute error, and relative error. Numerical reports remain valid diagnostics even when structured semantic preflight fails; they cannot alone produce semantic or overall Leyline acceptance.

### 5. Add checkpoint identity and prompt preflight

Environment collection resolves model and tokenizer paths and records the configured revision, selected configuration fields, chat-template presence, and SHA-256 manifests for configuration, tokenizer, index, and weight artifacts. Hashing is streamed and occurs during evidence collection rather than during every request.

Before `structured_json` cases, the runner submits a short prompt such as `Return only this JSON: {"ok":true}` through the same tokenizer, prompt format, endpoint, and decoding settings. The report stores the prompt token IDs, text, generated token IDs, parsed value, and pass/fail result. Failure prevents semantic admission, Leyline semantic acceptance, and performance execution.

Reference mode does not require instruction following. Its request path is still required to return generation token IDs, which the first full arm establishes.

### 6. Separate diagnostic collection from acceptance

For `structured_json`:

- admissible and counterfactual-admissible cases require full, honest-edited, and all declared counterfactuals to match the oracle;
- mechanism-diagnostic and negative-control cases require the full arm to establish the oracle but do not treat an expected honest-edited failure as an admissible success;
- a Leyline semantic result is considered only after the applicable baseline behavior is established.

For `reference_prefix`, the full arm establishes the reference. Honest-edited and counterfactual arms determine reference-only admission. This is a behavioral/mechanism experiment and does not consume the JSON oracle as correctness evidence.

First-token scores, pairwise prefixes, and device cache captures may be collected whenever the exact checkout and runtime identity are known and the Leyline request reports its actual execution outcome. A failed semantic preflight or baseline marks them diagnostic-only rather than suppressing collection.

Leyline acceptance additionally requires the applicable semantic or reference baseline, returned generation IDs, and response metadata showing `recorded=true`, `applied=true`, positive transformed tokens, and no fallback reason. Matching output after honest recomputation is not Leyline success. Numerical failure prevents overall correctness acceptance even if the selected semantic or reference target matches.

### 7. Version and merge reports by contract

Correctness reports move to schema version 2. Each case includes its evaluation mode, prompt format, match target, configured reference length, preflight state, semantic/reference admission, pairwise token-score diagnostics, and Leyline execution evidence. Top-level checkpoint evidence identifies the actual model and tokenizer artifacts, while numerical reports link rank/layer capture provenance without embedding high-volume arrays into the correctness JSON.

The merge tool requires all schema-v2 inputs for a case to agree on evaluation mode, prompt format, checkpoint identity, and preflight evidence before combining arms. Schema-v1 or mixed-contract inputs fail with an explicit compatibility error rather than receiving inferred schema-v2 meaning.

### 8. Run evaluation repair and mechanism diagnostics in parallel

After repository/runtime identity is confirmed, work proceeds on two independent tracks:

- Evaluation track: identify the checkpoint, add base/reference and Chat/structured contracts, run the short structured preflight, and establish full/honest/counterfactual admission before using the oracle against Leyline.
- Diagnostic track: immediately run full, honest-edited, and Leyline arms with returned token IDs and top-token scores, capture per-layer/per-rank cKV/Kpe transformations, and compare them with independent references without making a semantic claim.

The tracks join only for acceptance. Semantic acceptance requires a valid structured baseline; reference acceptance requires its declared full-arm token target; overall Leyline correctness additionally requires positive execution, numerical, and rollback evidence. Performance remains strictly last.

Validation-only ideas from other local branches may be used as reference, but commits containing 310 runtime work are not cherry-picked wholesale into the 910 branch.

## Risks / Trade-offs

- **[A one-token reference can be trivial]** → Record actual common-prefix length and make the pass prefix configurable; do not describe one-token success as full-generation equivalence.
- **[A chat template may change token boundaries around the edit]** → Validate canonical full-minus-span equality and fail closed so the workload can introduce stable boundaries.
- **[A local directory name can misidentify base weights as Chat weights]** → Record content-based manifests and prompt-preflight evidence rather than trusting the served model name.
- **[Weight hashing adds environment-collection time]** → Stream each file once and report hashing time separately; correctness evidence takes priority over startup convenience.
- **[Forced JSON decoding could mask instruction failure]** → Keep preflight and semantic validation unconstrained and use parsing only after generation.
- **[Small BF16 differences can cause long autoregressive divergence]** → Preserve returned token IDs and common-prefix diagnostics; continue to treat device cache comparison as the decisive numerical gate.
- **[API log probabilities may be mistaken for raw logits]** → Record an explicit evidence type and require an internal hook before labeling data as raw logits.
- **[Device capture can perturb latency and expose cached content]** → Keep capture opt-in, bounded, rank-scoped, excluded from performance runs, permission-restricted, and subject to explicit cleanup after analysis.
- **[Incomplete rank or layer capture can hide a localized error]** → Emit an expected-versus-observed manifest and fail numerical qualification when any required rank or layer is missing.
- **[Schema-v1 consumers will not read schema-v2 reports]** → Fail explicitly and retain old reports as immutable historical evidence rather than silently migrating their meaning.

## Migration Plan

1. Add focused harness tests and schema-v2 report generation without changing connector behavior.
2. Add separate base and Chat example configurations; make the base example select `reference_prefix` explicitly.
3. Update merge and documentation paths to require schema-v2 inputs.
4. Recollect the 910 environment, then start structured-preflight work and base-model token/cache diagnostics in parallel.
5. Join semantic/reference, execution, numerical, and rollback evidence for correctness acceptance.
6. Run performance only after every applicable correctness gate passes and with all capture hooks disabled.
7. Roll back by using the previous harness revision for historical schema-v1 reproduction; no runtime rollback is required because capture is opt-in and the transformation algorithm is unchanged.
