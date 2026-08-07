## Context

See `proposal.md` for motivation. The production connector records only complete source blocks and transforms only complete target blocks. The validation harness already constructs exact token deletion plans, but corpus admission currently checks only schema, family counts, prompt mode, and semantic target metadata. As a result, the latest 910B corpus passed preflight while thirteen cases had no complete source block and four stress cases had an edited prompt shorter than one block.

The benchmark must predict connector behavior without allocating device blocks. It also has to preserve exact tokenizer-dependent deletion lengths and completion targets for the DeepSeek-V2-Lite base checkpoint.

## Goals / Non-Goals

**Goals:**

- Make “this case can execute Leyline” a deterministic, inspectable precondition.
- Keep workload construction tokenizer-aware and exact at deletion and completion boundaries.
- Ensure the expensive 910B matrix starts only after one real transform has produced the required execution evidence.
- Make reports say whether execution metadata or generated tokens are stable.

**Non-Goals:**

- Change production connector mapping, lifecycle, or cache kernels.
- Predict local APC hits created by unrelated requests; qualification continues to isolate prompts with cache salts and expects zero local APC tokens.
- Make feasibility imply semantic correctness or numerical correctness.
- Require local development machines to reproduce the NPU smoke request.

## Decisions

### 1. Represent feasibility as a first-class immutable result

Add a pure `TransformFeasibility` result beside `PromptPlan`. It contains lengths, block size, resident source block indices, local computed tokens, maximum target tokens, reusable range, predicted transform count, post-deletion shifted transform count, blocking source block, and a stable reason.

The planner uses the same definitions as production:

- resident source blocks are `range(len(full) // block_size)`;
- generation reserves the final prompt token, so maximum target tokens are `len(edited) - 1`;
- local computed tokens must be block aligned;
- deletion mapping and longest reusable end follow the production mapping contract.

Qualification additionally requires the reusable range to extend beyond the deletion start. This prevents an unchanged delta-zero block before the removed span from being counted as evidence that cache rotation executed.

The benchmark will keep the arithmetic in a dependency-light helper covered by parity tests against representative production mapping cases. Importing the full connector into the standalone runner was rejected because it pulls runtime and device dependencies into corpus validation.

### 2. Validate feasibility after tokenizer planning and before network requests

Schema validation remains a fast structural check. After the tokenizer is loaded, the runner builds every canonical and counterfactual prompt plan, computes feasibility, and fails the entire qualification run before sending an HTTP request if an execution-required plan is infeasible.

This ordering avoids partial reports where earlier cases ran before a later impossible workload was discovered. Diagnostic cases may explicitly set `execution_expectation: diagnostic`; the default is `required` for all qualifying base cases.

### 3. Expand a surviving filler region, not the deleted span

Add optional workload construction fields for a deterministic `surviving_filler_unit` and a bounded maximum repeat count. The planner inserts the repeated unit at the start of the surviving region, retokenizes the canonical full and edited strings, and chooses the smallest repeat count satisfying both the exact deletion constraint and the configured minimum transformed tokens.

This makes the edited prompt long enough and, crucially, leaves sufficient text after the deletion for shifted source blocks to become complete. Growing only the deleted filler was rejected because it recreates the observed failure: a long record prompt followed by a 39-token edited prompt.

Canonical reports store the selected repeat count and achieved feasibility metrics so the host result remains reproducible for the pinned tokenizer.

### 4. Use one designated corpus case as a fail-fast smoke

Runner configuration identifies a smoke case and enables the gate for qualification runs. The smoke executes the normal record and amortize arms once, then applies the same execution gate used by final classification. Required layer/rank counts come from the response metadata. When device capture is required, the gate also checks the joined manifest after the request.

The full matrix may rerun the smoke case with normal repetition settings; the first execution is retained separately as startup evidence. Reusing a normal corpus case avoids maintaining a semantically unrelated synthetic endpoint path.

Local unit tests validate gate evaluation with fixtures. Actual NPU execution remains part of the documented 910B procedure.

### 5. Preserve separate semantic and numerical gates

Feasibility is recorded under workload planning, execution proof under the smoke and connector gates, and semantic target admission under the existing full/honest/counterfactual logic. Numerical capture remains a later gate. The classification order does not allow a feasible plan to mask connector fallback or a successful transform to mask a failed baseline.

### 6. Split stability into two booleans

`execution_stable` compares the existing decision/fallback/transformed/completeness tuple. `generation_stable` compares normalized output token ID sequences for every arm and counterfactual across repetitions. The legacy `stable` field remains as an alias for execution stability for schema-v2 compatibility, while new summaries and qualification messages name the two dimensions explicitly.

### 7. Requalify unstable route cases through data changes, not oracle relabeling

`route-label` and `route-neutral-variants` will be rewritten into a more closed completion pattern and combined with surviving filler. Their declared target is retained only if local/tokenizer checks and the next 910B baseline run support it. The implementation does not replace `B` with the accidentally observed `C` or `D`; if deterministic local qualification is impossible, the cases remain baseline diagnostics and are excluded from Leyline semantic claims.

## Risks / Trade-offs

- **[Standalone feasibility arithmetic drifts from connector behavior]** → Add parity fixtures for boundary, cross-block, local-APC, and partially resident source cases; keep formulas and stable reasons documented next to the helper.
- **[Long neutral filler changes base-model continuation]** → Put compact task-defining examples and the query after the filler, require full/honest/counterfactual baseline admission, and choose the smallest feasible repeat count.
- **[Tokenizer search increases startup time]** → Bound repeat search, memoize token counts within one run, and fail with the nearest achieved metrics rather than searching indefinitely.
- **[Smoke adds duplicate requests]** → Limit it to one repetition and skip the remaining matrix entirely on failure, which saves substantially more time on broken runs.
- **[Legacy consumers read only `stable`]** → Preserve it as execution stability and add new fields without changing old report parsing.

## Migration Plan

1. Add feasibility data structures, planner, workload fields, and unit tests while retaining legacy diagnostic workload support.
2. Rebuild the base corpus with tokenizer-aware surviving filler and designate a smoke case in the runner example configuration.
3. Add fail-fast corpus feasibility and smoke evaluation, then update reporting and documentation.
4. Run local unit/schema/OpenSpec validation and push the exact commit to the 910B branch.
5. On 910B, run the smoke-enabled base suite. Proceed to the full matrix only if record, application, completeness, and required capture evidence pass.

Rollback is configuration-based: disable the new smoke gate only for legacy diagnostics, or select the historical workload explicitly. Qualification mode never downgrades an infeasible case into a passing result.
