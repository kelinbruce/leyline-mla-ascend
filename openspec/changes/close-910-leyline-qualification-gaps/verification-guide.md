# Ascend 910B Leyline Qualification Guide

> This change-local guide defines the expected host procedure after the change is implemented. The apply phase will publish the maintained repository copy at `benchmarks/leyline/VALIDATION_910B.md` and keep its command-line interface synchronized with the implemented scripts.

## 1. What this run must prove

A qualifying run must answer all of the following independently:

| Gate | Required evidence | Pass condition |
|---|---|---|
| Environment | `environment.json` | Imported source, commits, CANN, TP topology, model and tokenizer hashes agree |
| Workload feasibility | source correctness report | Every required case predicts a positive block-aligned transform and required deltas are constructible |
| Connector execution | record/amortize responses | Recorded, applied, no fallback, complete 27 layers on all 4 ranks |
| Cache numerical correctness | `cache-comparison.json` | cKV bitwise equal; Kpe tolerance passes; native/connector RoPE agree; no missing layer/rank/delta |
| Baselines | connector-on and cache-off reports | Full, honest, counterfactual, patched, vanilla, and cache-off target decisions are complete and stable |
| Leyline target | source correctness report | Every admitted case produces its declared target in all repetitions |
| Raw-logit diagnostics | first-token and divergence comparison reports | All requested request/rank/step vectors are present and provenance-aligned |
| Rollback | `rollback-report.json` | Partial device write fails closed, recomputes honestly, and leaves zero transaction state |
| Retained context | diagnostic report | At least four informative cases across three families; Leyline is compared with both baselines |
| Finalization | `joined-qualification.json` | No identity conflicts or missing required evidence; determining gate reported for every case |

Complete suffix equality is not a pass condition. A later suffix divergence is acceptable only after environment, execution, numerical, rollback, baseline, and declared target gates pass.

## 2. Host preparation

Use a clean checkout of the exact pushed commit. Do not mix the source report from one commit with capture or rollback artifacts from another.

```bash
cd /opt/leyline/vllm-ascend
git status --short
git rev-parse HEAD
python3 -c 'import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)'
```

Confirm four visible 910B devices and the intended CANN environment:

```bash
npu-smi info
python3 -c 'import torch; print(torch.__version__); print(torch.npu.device_count())'
```

Create a new directory; never reuse a previous run directory:

```bash
export LEYLINE_RUN_ID="schema_v3_$(date -u +%Y%m%d_%H%M%S)"
export LEYLINE_RUN_DIR="/opt/leyline/vllm-ascend/results/leyline/${LEYLINE_RUN_ID}"
mkdir -p "${LEYLINE_RUN_DIR}/cache-captures" "${LEYLINE_RUN_DIR}/raw-logits"
chmod 700 "${LEYLINE_RUN_DIR}" "${LEYLINE_RUN_DIR}/cache-captures" "${LEYLINE_RUN_DIR}/raw-logits"
```

Copy and edit the runner/runtime examples into the run directory. Pin model/tokenizer paths or revisions and use a unique run ID in validation configuration.

```bash
cp benchmarks/leyline/runner_config.example.json "${LEYLINE_RUN_DIR}/runner.connector.json"
cp benchmarks/leyline/runner_config.cache_off.example.json "${LEYLINE_RUN_DIR}/runner.cache_off.json"
cp benchmarks/leyline/runtime_config.example.json "${LEYLINE_RUN_DIR}/runtime.json"
```

Before starting, verify that the configured model, tokenizer, block size 128, TP4 topology, endpoint, capture directories, and repetition count 3 are correct.

## 3. Local focused checks before using NPUs

Run the focused validation tests from the same checkout:

```bash
python3 -m pytest -q \
  tests/ut/distributed/kv_transfer/leyline/test_core.py \
  tests/ut/distributed/kv_transfer/leyline/test_connector.py \
  tests/ut/distributed/kv_transfer/leyline/test_capture_diagnostics.py \
  tests/ut/distributed/kv_transfer/leyline/test_validation_harness.py
```

Run the repository formatting, lint, and static commands required by the branch before collecting evidence. Do not qualify a commit different from the one checked here.

## 4. Collect environment identity

With the runtime environment activated:

```bash
python3 benchmarks/leyline/collect_environment.py \
  --model /opt/foundation_model/DeepSeek-V2-Lite \
  --model-revision local \
  --tokenizer /opt/foundation_model/DeepSeek-V2-Lite \
  --tokenizer-revision local \
  --runtime-config "${LEYLINE_RUN_DIR}/runtime.json" \
  --output "${LEYLINE_RUN_DIR}/environment.json"
```

Inspect blockers before continuing:

```bash
jq '{repositories, imported_modules, cann_installation, model, runtime, blockers}' \
  "${LEYLINE_RUN_DIR}/environment.json"
```

Stop if the imported module paths do not point at the intended checkout, the repository is dirty for unexplained reasons, model/tokenizer hashes are missing, or CANN identity is unresolved.

## 5. Connector-on correctness and device capture

Start a dedicated TP4 correctness server with prefix caching, the Leyline connector, eager execution, and capture enabled. Capture hooks perturb latency, so this server must not be used for performance results.

```bash
export VLLM_ASCEND_LEYLINE_CAPTURE_DIR="${LEYLINE_RUN_DIR}/cache-captures"
export VLLM_ASCEND_LEYLINE_CAPTURE_MAX_ROWS=64
export VLLM_ASCEND_LEYLINE_CAPTURE_REQUIRED_DELTAS=0,1,127,128,129,1024
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_DIR="${LEYLINE_RUN_DIR}/raw-logits"
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_STEPS=0
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_RUN_ID="${LEYLINE_RUN_ID}"
```

Start the server using the command in `benchmarks/leyline/README.md`, with:

```text
DeepSeek-V2-Lite base checkpoint
TP=4, DCP=1, PCP=1
block-size=128
prefix caching enabled
chunked prefill enabled
enforce eager
LeylineConnector with recompute failure policy
```

After the health check passes, run the base workload:

```bash
python3 benchmarks/leyline/run_validation.py \
  --config "${LEYLINE_RUN_DIR}/runner.connector.json" \
  --workloads benchmarks/leyline/workloads.base.json \
  --environment "${LEYLINE_RUN_DIR}/environment.json" \
  --output "${LEYLINE_RUN_DIR}/correctness-source.json"
```

Check the fail-fast gates:

```bash
jq '{
  feasibility: .workload_feasibility.passed,
  smoke: .smoke_gate.passed,
  cases: (.cases | length),
  environment_blockers: .qualification.environment_blockers
}' "${LEYLINE_RUN_DIR}/correctness-source.json"
```

Expected:

```text
feasibility=true
smoke=true
17 expanded base cases (or the new documented corpus count)
environment_blockers=[]
```

Also verify every Leyline repetition reports:

```text
recorded=true on the paired record request
applied=true
fallback_reason=null
transform_complete=true
expected_layers=27 and transformed_layers=27
expected_ranks=4 and successful_ranks=4
transformed_tokens > 0
local_apc_tokens=0 for isolated Leyline runs
```

## 6. Compare cKV, Kpe, and RoPE

Do not publish only manifest counts. The NPZ files must be compared on the 910B host:

```bash
python3 benchmarks/leyline/compare_cache.py \
  "${LEYLINE_RUN_DIR}/cache-captures" \
  --expected-ranks 0,1,2,3 \
  --required-deltas 0,1,127,128,129,1024 \
  --output "${LEYLINE_RUN_DIR}/cache-comparison.json"
```

Inspect the result:

```bash
jq '{
  passed,
  observed_deltas,
  missing_deltas,
  missing_layer_ranks,
  aggregate,
  atol,
  rtol,
  frequency_atol
}' "${LEYLINE_RUN_DIR}/cache-comparison.json"
```

Pass requires:

```text
passed=true
108 layer/rank pairs for 27 layers × 4 ranks
missing_deltas=[]
missing_layer_ranks=[]
ckv_failed_captures=0
kpe_failed_captures=0
frequency_failed_captures=0
```

Any cKV mismatch, missing pair, native RoPE mismatch, or Kpe threshold failure is a numerical implementation failure. Stop semantic interpretation until it is fixed.

## 7. Compare first-token raw logits

Join step-zero full-vocabulary captures with the immutable source report:

```bash
python3 benchmarks/leyline/compare_logits.py \
  --correctness "${LEYLINE_RUN_DIR}/correctness-source.json" \
  --captures "${LEYLINE_RUN_DIR}/raw-logits" \
  --output "${LEYLINE_RUN_DIR}/logit-comparison.first-token.json"
```

Verify that every admitted full/honest/Leyline request and required rank is present. Review selected token, top-1/top-2 margin, top-k overlap, maximum absolute difference, cosine similarity, and Jensen-Shannon divergence. These values diagnose sensitivity; they do not replace the cache numerical gate.

## 8. Capture logits at the first divergence

Use the implemented divergence selector to list cases with a reproducible full/Leyline common prefix shorter than the generation length. Select only those cases and bound the maximum decode step, normally 32:

```bash
python3 benchmarks/leyline/plan_divergence_capture.py \
  --correctness "${LEYLINE_RUN_DIR}/correctness-source.json" \
  --max-step 32 \
  --output "${LEYLINE_RUN_DIR}/divergence-plan.json"
```

Restart the correctness server with raw-logit steps and case filters from that plan, run the targeted diagnostic workload, and write a separate report:

```bash
python3 benchmarks/leyline/run_validation.py \
  --config "${LEYLINE_RUN_DIR}/runner.connector.json" \
  --workloads benchmarks/leyline/workloads.base.json \
  --environment "${LEYLINE_RUN_DIR}/environment.json" \
  --diagnostic-plan "${LEYLINE_RUN_DIR}/divergence-plan.json" \
  --output "${LEYLINE_RUN_DIR}/divergence-correctness.json"

python3 benchmarks/leyline/compare_logits.py \
  --correctness "${LEYLINE_RUN_DIR}/divergence-correctness.json" \
  --captures "${LEYLINE_RUN_DIR}/raw-logits" \
  --divergence-plan "${LEYLINE_RUN_DIR}/divergence-plan.json" \
  --output "${LEYLINE_RUN_DIR}/logit-comparison.divergence.json"
```

Only interpret a divergence comparison when the diagnostic rerun reproduces the original common prefix and divergence index. Otherwise it must be marked non-correlatable.

## 9. Run cache-off baselines

Stop the connector-on server. Unset all capture variables and start a TP4 server with no KV connector and prefix caching disabled:

```bash
unset VLLM_ASCEND_LEYLINE_CAPTURE_DIR
unset VLLM_ASCEND_LEYLINE_RAW_LOGITS_DIR
unset VLLM_ASCEND_LEYLINE_RAW_LOGITS_STEPS
unset VLLM_ASCEND_LEYLINE_RAW_LOGITS_RUN_ID
```

Run the same corpus for three isolated repetitions:

```bash
python3 benchmarks/leyline/run_validation.py \
  --config "${LEYLINE_RUN_DIR}/runner.cache_off.json" \
  --workloads benchmarks/leyline/workloads.base.json \
  --environment "${LEYLINE_RUN_DIR}/environment.json" \
  --output "${LEYLINE_RUN_DIR}/correctness-cache-off.json"
```

Do not accept a single representative cache-off output. Every required case, variant, and repetition must be present and target decisions must be stable.

## 10. Run device rollback/failure injection

Stop the cache-off server. Start a dedicated connector server with validation failure injection enabled and concurrency one. Never enable this setting on a shared or production server.

```bash
export VLLM_ASCEND_LEYLINE_FAULT_INJECTION=validation-only
```

Run the rollback validator with a post-write failpoint on one selected layer/rank:

```bash
python3 benchmarks/leyline/run_rollback_validation.py \
  --config "${LEYLINE_RUN_DIR}/runner.connector.json" \
  --environment "${LEYLINE_RUN_DIR}/environment.json" \
  --case inventory-reorder-label \
  --fail-rank 1 \
  --fail-after-layer 8 \
  --output "${LEYLINE_RUN_DIR}/rollback-report.json"
```

Pass requires all of the following:

```text
injection_reached=true after at least one destination write
applied=false
fallback_reason=transform_failed
touched destination blocks invalidated
affected tokens recomputed through normal prefill
declared target agrees with honest/cache-off control
remaining session/inflight/pending/reference counters are zero
```

Stop the server and unset the opt-in immediately:

```bash
unset VLLM_ASCEND_LEYLINE_FAULT_INJECTION
```

## 11. Validate retained-context diagnostics

The diagnostic preflight must identify at least four informative cases across at least three families. An informative case requires, for every repetition:

```text
full matches the declared target
honest-edited does not match it
full and honest-edited first tokens differ
both baseline decisions are stable
```

Inspect the diagnostic summary:

```bash
jq '.retained_context_diagnostics | {
  informative_cases,
  informative_families,
  insufficient_coverage,
  cases
}' "${LEYLINE_RUN_DIR}/correctness-source.json"
```

For each informative case, compare Leyline with both baselines at the token and raw-logit levels. A Leyline result that follows honest-edited is a valid negative mechanism result, not an admitted-task failure.

## 12. Finalize immutable evidence

Finalize only after all staged reports exist:

```bash
python3 benchmarks/leyline/finalize_validation.py \
  --correctness "${LEYLINE_RUN_DIR}/correctness-source.json" \
  --environment "${LEYLINE_RUN_DIR}/environment.json" \
  --cache-comparison "${LEYLINE_RUN_DIR}/cache-comparison.json" \
  --first-token-logits "${LEYLINE_RUN_DIR}/logit-comparison.first-token.json" \
  --divergence-logits "${LEYLINE_RUN_DIR}/logit-comparison.divergence.json" \
  --cache-off "${LEYLINE_RUN_DIR}/correctness-cache-off.json" \
  --rollback "${LEYLINE_RUN_DIR}/rollback-report.json" \
  --output "${LEYLINE_RUN_DIR}/joined-qualification.json"
```

The finalizer must not contact an endpoint. It records source hashes and fails on mixed identities, missing repetitions, unknown request IDs, incomplete captures, or unsupported schema versions.

Inspect the final gates:

```bash
jq '{
  passed: .qualification.passed,
  determining_gate: .qualification.determining_gate,
  environment: .qualification.gates.environment,
  execution: .qualification.gates.execution,
  numerical: .qualification.gates.numerical,
  rollback: .qualification.gates.rollback,
  baselines: .qualification.gates.baselines,
  leyline_target: .qualification.gates.leyline_target,
  retained_context: .qualification.gates.retained_context,
  cases: .qualification.case_classifications
}' "${LEYLINE_RUN_DIR}/joined-qualification.json"
```

## 13. Decision rules

Use the first applicable result:

1. Environment mismatch → invalid run; fix provenance and rerun.
2. Record/transform incomplete → connector execution failure.
3. cKV/Kpe/native RoPE failure → numerical implementation failure.
4. Rollback failure → transaction integrity failure.
5. Full/honest/counterfactual/cache-off failure → invalid or unstable workload baseline.
6. Baselines pass but Leyline target fails → Leyline task-level limitation for that family.
7. Numerical and target gates pass, suffix later diverges → accepted target with autoregressive divergence.
8. Informative retained diagnostics follow honest → no retained-context benefit demonstrated for those cases; do not relabel this as admissible deletion failure.

## 14. Performance is a separate final run

Only after correctness qualification passes, start a fresh server with all cache/raw-logit/fault-injection hooks disabled. Record cold start separately, warm up, then run the documented concurrency matrix. Capture-enabled transformation timings are not performance evidence.

## 15. Artifacts to publish

Publish small reproducible evidence:

```text
environment.json
runtime.json
runner.connector.json
runner.cache_off.json
correctness-source.json
correctness-cache-off.json
cache-comparison.json
logit-comparison.first-token.json
divergence-plan.json
divergence-correctness.json
logit-comparison.divergence.json
rollback-report.json
joined-qualification.json
request trace and bounded HTTP debug records
```

Cache NPZ and raw-logit vectors can be large and may contain model-sensitive data. Keep them permission-restricted on the host or approved artifact storage; publish manifests, hashes, aggregate comparisons, and the documented retention location. Never publish prompt/cache contents that include secrets.
