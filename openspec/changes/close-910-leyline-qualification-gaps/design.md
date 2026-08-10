## Context

See `proposal.md` for motivation. The active validation changes already provide completion-target admission, TP-wide layer/rank completeness, bounded device capture, an FP32 cache comparator, first-token raw-logit capture, and separate cache-off configuration. The latest 910B bundle nevertheless contains only the connector-on correctness report and smoke manifest evidence: raw logits were disabled, cache NPZ/comparison artifacts were not published, cache-off was not run, and rollback is represented only by a false configuration prerequisite.

The current report assembly is also ordered incorrectly for immutable evidence. `run_validation.py` can read numerical and logit reports while building `correctness.json`, but those reports require request IDs and captures created by that same run. Rerunning validation to attach the reports creates new request IDs, so a scientifically valid join needs a post-run finalization stage.

## Goals / Non-Goals

**Goals:**

- Produce one final report from immutable staged evidence with no inference calls during finalization.
- Make missing evidence, failed evidence, and measured semantic limitations distinct outcomes.
- Diagnose the distribution that selected the first divergent token without enabling unbounded full-generation capture.
- Prove device rollback after a real partial write using an opt-in validation server.
- Make retained-context claims depend on empirically discriminating baselines across several task families.
- Give operators a copy-paste-oriented 910B runbook with artifact and gate checks.

**Non-Goals:**

- Require token-for-token equality with full-context generation.
- Make raw-logit similarity a substitute for cKV/Kpe numerical correctness.
- Enable fault injection on production servers or through an untrusted request alone.
- Store full prompt text, cache contents, or unrestricted raw logits in the summary report.
- Qualify Chat workloads when only the base checkpoint is available.

## Decisions

### 1. Introduce an immutable evidence bundle and offline finalizer

Each qualification attempt receives a generated `run_id` and a result directory containing source artifacts plus a manifest of SHA-256 hashes. A new finalization command consumes, but never rewrites, the connector-on correctness report, environment manifest, cache comparison, raw-logit comparison, cache-off report, rollback report, and optional targeted-divergence report. It writes a separate `joined-qualification.json`.

The finalizer aligns evidence using stable keys:

```text
run identity
  └── corpus / expanded case / counterfactual variant / repetition
        └── arm / request ID / rank / decode step
```

It validates identities before calculating gates, then reuses one shared qualification function so online and offline classifications cannot drift.

Alternatives considered:

- Rerun `run_validation.py --numerical-report ...` after comparison: rejected because new requests and request IDs break provenance.
- Modify `correctness.json` in place: rejected because it destroys the distinction between source and derived evidence.
- Treat files in the same directory as implicitly related: rejected because stale or mixed checkpoint artifacts could be joined accidentally.

### 2. Treat cache-off as a staged full report, not one synthetic arm

The cache-off server is mutually exclusive with the connector-on topology, so it remains a separate run. The finalizer verifies that the only allowed runtime difference is connector/prefix-cache configuration and aligns all three repetitions instead of copying a representative `arms.cache_off` value into each case.

`merge_reports.py` will either be extended to preserve repetitions and qualification provenance or reduced to a compatibility wrapper over the same evidence-join library. The final source artifacts remain linked even when a convenience merged view is emitted.

### 3. Capture selected decode steps with structured request IDs

The harness will use a validation request-ID format containing an opaque run ID plus escaped case, arm, and repetition identifiers. Raw-logit capture remains environment-gated and additionally filters on run/case selection. It writes one artifact per request/rank/step and refuses unbounded capture.

The normal qualification run captures step zero. A targeted diagnostic run may capture steps `0..N` only for cases whose source report has a short common prefix. The comparator accepts the targeted result as an explanation of the source run only when the generated prefix and first-divergence index reproduce.

This two-pass approach bounds storage. Capturing 64 full-vocabulary vectors for every arm, repetition, case, and TP rank would be disproportionately large and would perturb the workload.

Alternatives considered:

- Capture only API top-10 log probabilities: insufficient for max-absolute, cosine, and Jensen-Shannon comparisons.
- Capture all steps for all requests: rejected for storage and synchronization overhead.
- Compare logits after prefixes have already diverged: retained only as labeled conditional-distribution evidence, never as a same-prefix numerical comparison.

### 4. Preflight retained-context diagnostics for informativeness

The corpus will contain at least four candidate diagnostics across at least three families. Candidate layouts follow the Leyline information path:

```text
deleted evidence R
        ↓ attention before deletion
transformable surviving cache region S
        ↓ transformed cKV/Kpe
normal-prefill query Q
```

Examples will include codeword lookup, removed few-shot mapping, state transition, and code/symbol reference forms. A case counts toward diagnostic coverage only when all repetitions establish a stable full target and a different stable honest first token. Cases such as the current `deleted-label-example`, where both baselines select the target, remain visible but are labeled uninformative.

Leyline is evaluated only after this diagnostic admission. Its token and distribution distances are reported relative to both baselines; retained-context diagnostics never enter admissible deletion success rates.

### 5. Add a validation-only post-write failpoint

Fault injection requires two controls:

1. A server-start environment switch enables validation failpoints.
2. A request directive names an allowed rank/layer and post-write stage.

Without the server switch, the request field is rejected. The failpoint is transaction-scoped, consumed once, and cannot select arbitrary code or paths. The connector exposes only non-sensitive diagnostic counters needed to prove touched-block invalidation, recomputation, and cleanup.

The rollback runner records a normal honest control, a successful record request, and an injected amortize request. It verifies that a partial worker result becomes TP-wide failure, `applied` stays false, generation reaches the declared target through recomputation, and transaction-owned state is zero afterward.

Alternatives considered:

- Infer rollback from unit tests: retained as local coverage but insufficient for 910B device qualification.
- Trigger failure with an invalid request: rejected because it does not exercise partial destination writes.
- Make `rollback_passed` a manual operator assertion: rejected as non-verifiable evidence.

### 6. Publish a staged 910B runbook

Implementation will publish `benchmarks/leyline/VALIDATION_910B.md`. The change-local `verification-guide.md` captures the required sequence now and acts as the source for the repository runbook. The guide separates correctness servers from cache-off and failure-injection servers, requires capture hooks to be disabled for performance, and lists every expected output artifact and `jq` gate check.

## Risks / Trade-offs

- **[Raw vectors still consume substantial storage]** → Require case filters, finite step sets, file/byte budgets, restricted permissions, and a cleanup policy after summaries are preserved.
- **[Fault injection could affect unrelated requests]** → Use a dedicated validation server, two explicit opt-ins, transaction-scoped failpoints, and concurrency one.
- **[A targeted rerun may not reproduce divergence]** → Require prefix/divergence reproduction and label non-reproducing captures as non-correlatable.
- **[Strict identity checking may reject editable installations]** → Record imported module paths and allow only explicitly reconciled checkout/distribution identity.
- **[Four diagnostic cases may still be checkpoint-sensitive]** → Preflight every candidate empirically and report insufficient informative coverage instead of relabeling targets from one run.
- **[Finalizer and online runner classifications may drift]** → Extract one pure qualification evaluator used by both paths and cover it with golden report tests.

## Migration Plan

1. Extract shared evidence identity and qualification evaluation without changing existing schema-v2 source reports.
2. Add offline finalization and repetition-preserving cache-off joins, then cover mixed-identity and missing-evidence failures.
3. Extend request IDs and bounded decode-step raw-logit capture while retaining first-token filename compatibility where practical.
4. Add informative retained-context cases and corpus validation without changing admitted success-rate semantics.
5. Add validation-only failure injection, rollback runner, and cleanup evidence tests.
6. Publish the repository 910B runbook and run focused local verification.
7. Collect fresh 910B staged evidence and finalize it into a new result directory; historical reports remain immutable diagnostics.

Rollback of the implementation is configuration-based: leave raw-logit and fault-injection environment switches unset and continue using the existing first-token/source-report behavior. The new finalizer is additive and does not rewrite historical files.
