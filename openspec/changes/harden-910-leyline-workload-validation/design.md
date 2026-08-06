## Context

See `proposal.md` for motivation. The current 910B evidence uses one raw endpoint and a `reference_prefix=1` gate. Its two accepted cases share only a leading newline between full and honest-edited, device and raw-logit captures are disabled, and the report cannot decide whether later divergence comes from the workload, the cache transform, or Leyline's retained cKV approximation.

The existing implementation already has useful pieces: exact token deletion planning, pairwise generated-token diagnostics, rank-scoped cache capture, an offline FP32 rotation comparator, model/tokenizer hashing, and typed fallback metadata. This change tightens their scientific contract instead of replacing the harness.

## Goals / Non-Goals

**Goals:**

- Make base-model qualification use natural continuation behavior with an explicit meaningful target.
- Expand the corpus enough to separate task-family behavior from block/position mechanism coverage.
- Make one 910B report decide among transform error, invalid baseline, semantic failure, and later autoregressive divergence.
- Make `applied=true` prove a complete, leak-free TP-wide transformation rather than any non-zero number of transformed layers.

**Non-Goals:**

- Guarantee token-for-token equality between full, honest-edited, and Leyline generations.
- Claim secure forgetting; copied cKV may retain information from deleted history.
- Tune performance before correctness, rollback, and evidence completeness pass.
- Change non-Leyline serving behavior or add a general-purpose benchmark framework.

## Decisions

### 1. Add `completion_target` and make `reference_prefix` diagnostic-only

Raw base checkpoints are next-token models, so the qualifying base contract will be a declared short continuation rather than JSON instruction following. Each case supplies `expected_completion`; the planner tokenizes both `prompt` and `prompt + expected_completion` and requires the latter to equal the former plus a non-empty suffix. The suffix must decode to non-whitespace text and defaults to one stable token.

Full, honest-edited, and every counterfactual must generate that suffix before the case is admitted. Leyline is compared with the same target only after admission. Generated common prefixes remain useful diagnostics but cannot turn an undeclared newline into success.

Alternatives considered:

- Increasing `reference_tokens` from one to eight would correctly reject the present cases but would still use an uncontrolled full generation as the oracle.
- Stripping whitespace from arbitrary output text is easy but loses the exact tokenizer boundary needed by the deletion experiment.
- Forced decoding would hide model behavior; the suite continues to use greedy unconstrained generation.

### 2. Split the canonical workload into a 16-case base corpus and a 6-case Chat corpus

The initial base corpus will contain these distinct cases:

| Family | Case | Completion form | Target/coverage |
|---|---|---|---|
| admissible | `route-label` | few-shot `health,budget -> label` | `B` |
| admissible | `inventory-reorder-label` | few-shot `available,reorder_point -> label` | `Y` |
| admissible | `incident-severity-label` | few-shot counters to `L/M/H` | `H` |
| admissible | `feature-state-label` | few-shot enabled/blocked state | `D` |
| admissible | `build-language-label` | manifest/file pattern to language label | `P` |
| admissible | `python-square-next-token` | deterministic code continuation | `x` |
| position stress | `delete-delta-1` | admitted label completion | exact 1-token deletion |
| position stress | `delete-delta-127` | admitted label completion | just below block size |
| position stress | `delete-delta-128` | admitted label completion | one full block |
| position stress | `delete-delta-129-and-long` | admitted label completion plus variant | cross-block and long-shift coverage |
| counterfactual | `route-neutral-variants` | same route target with at least three neutral removals | invariance |
| counterfactual | `code-comment-variants` | same code token with at least three comment removals | invariance |
| diagnostic | `deleted-codeword` | deleted mapping supplies target | retained-context evidence |
| diagnostic | `deleted-label-example` | removed few-shot row is required | retained-example evidence |
| negative | `missing-route-label` | surviving text has no supported label | non-admissible control |
| negative | `contradictory-deleted-only-label` | only deleted span supports full target | non-admissible control |

For example, `route-label` uses pattern continuation rather than an instruction:

```text
health=ok,budget=open -> A
health=degraded,budget=exhausted -> B
health=stopped,budget=closed -> C
<neutral removable history>
health=degraded,budget=exhausted ->
```

The expected continuation is ` B` or `B` depending on the tokenizer-verified prompt boundary. Other label cases use the same completion principle but different fields and mappings; they do not repeat the same prose with renamed nouns. The code case ends with `return x *` and expects the tokenizer-stable continuation ` x`.

Stress cases use deterministic tokenizer-aware filler generation. The builder adds candidate filler units, retokenizes the canonical complete strings, and accepts a case only when the requested deleted token length and exact full-minus-span invariant are achieved. Delta-zero rows come from at least one complete block before the deletion. The cross-block case supplies variants to cover 129 and a configured long shift such as 1024 without requiring every ordinary semantic case to be long.

The six Chat cases cover structured routing, inventory, access-policy, counterfactual incident response, deleted-secret diagnosis, and missing-evidence negative control. They reuse neither the base prompts nor the base target mode. Repetition is limited to what is needed to cross a cache boundary; semantic records are concise and use stable delimiters around the removable span.

Alternatives considered:

- Keeping the existing four repeated JSON cases would preserve historical comparison but would not establish a base-model baseline.
- Generating hundreds of random prompts would increase sample count without providing controlled causal or block-boundary coverage.

### 3. Extend the workload schema with claim and coverage metadata

Each case gains explicit fields for evaluation mode, family, claim type, expected completion or structured oracle, removed variants, and optional token-position requirements. A corpus validator checks family minimums before requests begin. The report stores requested and achieved deletion length, boundaries, observed deltas, transformed range, and target token IDs.

Existing schema-v2 workload files remain readable only in an explicit legacy diagnostic mode. They are not silently assigned completion targets or upgraded to qualification evidence.

### 4. Join generated targets with raw distribution diagnostics

API top-logprobs remain a lightweight default. Qualification runs additionally enable the internal raw-logit hook at the first declared target token and store one full-vocabulary vector per arm/case outside the main JSON. The summary reports selected token, margin, top-k overlap, maximum absolute logit difference, cosine similarity, and a normalized distribution distance for full-to-honest and full-to-Leyline.

Raw logits diagnose sensitivity; they do not replace cache evidence. The corpus is run greedily with isolated cache salts. At least three repeated correctness runs are retained to expose nondeterministic service or cache behavior before aggregating family results.

### 5. Make the numerical reference independent of connector frequency construction

The current capture stores the connector-provided inverse frequency and the comparator reuses it. That catches rotation kernel or slot errors but can pass if the connector and comparator share the same wrong frequency derivation.

Qualification therefore records native runtime RoPE provenance separately: the active rotary module's inverse-frequency buffer or an equivalent native cos/sin table, its shape/dtype/hash, and the connector frequency vector. The offline comparator first checks the two frequency sources, then constructs FP32 delta rotation from the native source. cKV remains a bitwise copy requirement.

Every expected layer/rank pair must appear in the manifest. Row capture stays bounded but must include observed delta-zero rows, the requested boundary deltas, first/last transformed rows, and representative source/destination block transitions. Capture artifacts remain permission-restricted and are disabled for performance runs.

### 6. Separate record evidence from amortize evidence and fix cleanup

The finish hook records source blocks only for `RECORD`. `AMORTIZE` consumes the source session and reports transform state without recording the edited prompt. The harness joins `record.recorded` with the amortize response's `applied`, positive token count, no-fallback status, and completeness metadata.

Session and plan ownership is tested across success, worker failure, cancellation, invalid directive, and shutdown. The one-shot session is released by the TP-wide completion path; later finish hooks must be idempotent and must not recreate it.

### 7. Derive and enforce expected layer/rank completeness

Expected layers come from the active model contract and registered MLA layer names, not from the subset that happens to pass a shape filter. For DeepSeek-V2-Lite this is 27 MLA layers. Each worker result carries expected and transformed layer counts plus rank identity. Aggregation fails if any rank is absent, any expected layer is skipped, counts disagree, or a worker fails.

`applied=true` is set only after complete aggregation. On failure, every destination block touched by the plan is invalidated and all transformed tokens are added to normal prefill accounting before generation can use the cache.

### 8. Use an explicit qualification decision tree

The final report classifies outcomes in this order:

1. Environment/import mismatch → invalid run.
2. Missing record, incomplete transform, or rollback failure → connector failure.
3. cKV/Kpe/native-frequency failure → numerical implementation failure.
4. Full/honest/counterfactual target failure → invalid workload baseline.
5. Baseline pass but Leyline target failure → Leyline task-level limitation for that admitted family.
6. Target and numerical pass with later suffix divergence → accepted target behavior with autoregressive divergence diagnostic.

This ordering prevents both “the algorithm is broken” and “this is simply Leyline's inherent effect” from being claimed before the evidence distinguishes them.

### 9. Record actual import provenance

Environment collection imports `vllm` and `vllm_ascend` in the same runtime used by the service and records module paths, distribution versions, repository roots/commits, Python executable, and CANN version sources. A mismatch is allowed in a diagnostic manifest but blocks qualification until explained, for example by proving an editable install points at the recorded checkout.

## Risks / Trade-offs

- **[Completion labels may still be weak for one checkpoint]** → Require tokenizer-stable declared targets, six distinct admissible families, counterfactuals, and empirical full/honest admission before Leyline scoring.
- **[Tokenizer-aware exact-length filler can be slow]** → Cache tokenized filler units and bound search; fail with requested/nearest lengths rather than accepting an approximate span.
- **[Raw logits and all-layer capture consume storage and perturb latency]** → Keep artifacts out of the summary JSON, bound captured rows, restrict permissions, and run performance only with hooks disabled.
- **[Native RoPE state may not be directly accessible in every runtime]** → Fail numerical qualification with explicit missing provenance instead of falling back to the connector's own frequency vector.
- **[Stricter layer completeness may expose unsupported cache entries]** → Fail closed to honest prefill and report exact missing layer/rank names.
- **[More cases increase 910B runtime]** → Use the 16-case base corpus for qualification, the 6-case Chat corpus only on the pinned Chat service, and keep larger exploratory corpora outside the required gate.

## Migration Plan

1. Add schema and harness support for `completion_target` while keeping old reference reports readable as legacy diagnostics.
2. Fix transaction lifecycle and completeness metadata, then verify rollback and reference cleanup locally.
3. Add and validate the 16 base and 6 Chat cases without changing the historical workload file in place; assign versioned corpus identifiers.
4. Add native RoPE provenance, raw-logit joins, and all-layer/rank numerical qualification.
5. Run focused local tests and synchronize the exact checkout to the 910B host.
6. Run base baseline qualification first, then diagnostic cache/logit capture, then the Chat semantic track.
7. Enable warm performance only after all applicable gates pass with capture hooks disabled.

Rollback is configuration-based: retain the existing non-Leyline and honest-prefill paths, disable the experimental connector, and treat new reports as diagnostic if any new gate fails. Historical results remain immutable under their original corpus and contract versions.
