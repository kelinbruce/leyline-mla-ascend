# Ascend 910B Leyline Qualification

This runbook produces immutable evidence for Leyline correctness. It separates connector execution, cache numerical correctness, baselines, rollback, task targets, retained-context diagnostics, and later autoregressive divergence.

## Required gates

| Gate | Artifact | Pass condition |
|---|---|---|
| Environment | `environment.connector.json`, `environment.cache-off.json` | Imports, commits, CANN, TP4 topology, model/tokenizer hashes agree |
| Execution | `correctness-source.json` | record + applied, no fallback, 27/27 layers, 4/4 ranks |
| Numerical | `cache-comparison.json` | cKV bitwise equal, Kpe allclose, native RoPE agreement, complete deltas and layer/ranks |
| Baselines | connector and cache-off reports | Full, honest, counterfactual, patched, vanilla, and cache-off targets stable for 3 repetitions |
| Logits | two logit comparison reports | Requested request/rank/step vectors complete and provenance-aligned |
| Rollback | `rollback-report.json` | Injected partial write fails closed, recomputes honestly, cleanup reaches zero |
| Leyline target | source report | Every admitted target passes in all repetitions |
| Final | `joined-qualification.json` | All required gates pass and every case names its determining gate |

Exact 64-token suffix equality is not required. Later divergence is called autoregressive-only only after numerical, rollback, baseline, and target gates pass.

## 1. Prepare a clean result directory

```bash
cd /opt/leyline/vllm-ascend
git status --short
git rev-parse HEAD
python3 -c 'import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)'
npu-smi info

export LEYLINE_RUN_ID="schema_v3_$(date -u +%Y%m%d_%H%M%S)"
export LEYLINE_RUN_DIR="/opt/leyline/vllm-ascend/results/leyline/${LEYLINE_RUN_ID}"
mkdir -p "${LEYLINE_RUN_DIR}/cache-captures" "${LEYLINE_RUN_DIR}/raw-logits"
chmod 700 "${LEYLINE_RUN_DIR}" "${LEYLINE_RUN_DIR}/cache-captures" "${LEYLINE_RUN_DIR}/raw-logits"

cp benchmarks/leyline/configs/runner_config.example.json "${LEYLINE_RUN_DIR}/runner.connector.json"
cp benchmarks/leyline/configs/runner_config.cache_off.example.json "${LEYLINE_RUN_DIR}/runner.cache-off.json"
cp benchmarks/leyline/configs/runner_config.rollback.example.json "${LEYLINE_RUN_DIR}/runner.rollback.json"
cp benchmarks/leyline/configs/runtime_config.example.json "${LEYLINE_RUN_DIR}/runtime.connector.json"
cp benchmarks/leyline/configs/runtime_config.cache_off.example.json "${LEYLINE_RUN_DIR}/runtime.cache-off.json"
```

Edit the copied files. Pin the local model/tokenizer, set `run_id`, use block size 128, TP4, three correctness repetitions, and the actual endpoints. Create a cache-off runtime manifest with only these intended differences:

```text
enable_prefix_caching=false
kv_connector/kv_role/kv_connector_module_path/kv_load_failure_policy absent
```

Run the focused tests and branch-required format/lint/static checks before using NPUs:

```bash
python3 -m pytest -q \
  tests/ut/distributed/kv_transfer/leyline/test_core.py \
  tests/ut/distributed/kv_transfer/leyline/test_connector.py \
  tests/ut/distributed/kv_transfer/leyline/test_capture_diagnostics.py \
  tests/ut/distributed/kv_transfer/leyline/test_validation_harness.py \
  tests/ut/distributed/kv_transfer/leyline/test_qualification_finalizer.py
```

## 2. Collect connector-on identity

```bash
python3 benchmarks/leyline/scripts/collect_environment.py \
  --model /opt/foundation_model/DeepSeek-V2-Lite \
  --model-revision local \
  --tokenizer /opt/foundation_model/DeepSeek-V2-Lite \
  --tokenizer-revision local \
  --runtime-config "${LEYLINE_RUN_DIR}/runtime.connector.json" \
  --output "${LEYLINE_RUN_DIR}/environment.connector.json"
```

Stop if imports point outside the recorded repositories, commits are unexplained, CANN is unresolved, or model/tokenizer hashes are missing.

## 3. Connector-on correctness, cache, and first-token logits

Start a dedicated correctness server with TP4, DCP1, PCP1, block size 128, prefix caching, chunked prefill, eager execution, LeylineConnector, and recompute failure policy.

Before starting it:

```bash
export VLLM_ASCEND_LEYLINE_CAPTURE_DIR="${LEYLINE_RUN_DIR}/cache-captures"
export VLLM_ASCEND_LEYLINE_CAPTURE_MAX_ROWS=64
export VLLM_ASCEND_LEYLINE_CAPTURE_REQUIRED_DELTAS=0,1,127,128,129,1024
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_DIR="${LEYLINE_RUN_DIR}/raw-logits"
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_STEPS=0
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_RUN_ID="${LEYLINE_RUN_ID}"
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_ARMS=full,honest_edited,leyline
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_STEP=32
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_FILES=4096
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_BYTES=8589934592
```

Run the corpus:

```bash
python3 benchmarks/leyline/scripts/run_validation.py \
  --config "${LEYLINE_RUN_DIR}/runner.connector.json" \
  --workloads benchmarks/leyline/workloads/workloads.base.json \
  --environment "${LEYLINE_RUN_DIR}/environment.connector.json" \
  --output "${LEYLINE_RUN_DIR}/correctness-source.json"
```

The source report is expected to remain numerically/rollback unqualified until offline evidence is finalized. Check the gates that are already measurable:

```bash
jq '{feasibility:.workload_feasibility.passed,
     smoke:.smoke_gate.passed,
     cases:(.cases|length),
     environment_blockers:.qualification.environment_blockers,
     retained:.retained_context_diagnostics}' \
  "${LEYLINE_RUN_DIR}/correctness-source.json"
```

Every required Leyline repetition must show `recorded=true` on its paired record, `applied=true`, `fallback_reason=null`, `transform_complete=true`, 27/27 layers, 4/4 ranks, positive block-aligned transformed tokens, and consistent token accounting.

## 4. Compare cKV, Kpe, and native RoPE

```bash
python3 benchmarks/leyline/scripts/compare_cache.py \
  "${LEYLINE_RUN_DIR}/cache-captures" \
  --expected-ranks 0,1,2,3 \
  --required-deltas 0,1,127,128,129,1024 \
  --output "${LEYLINE_RUN_DIR}/cache-comparison.json"

jq '{passed, observed_deltas, missing_deltas, missing_layer_ranks, aggregate}' \
  "${LEYLINE_RUN_DIR}/cache-comparison.json"
```

Required: 27×4 layer/rank coverage, no missing deltas/pairs, zero cKV, Kpe, and frequency failures. A failure here is a numerical implementation failure; do not interpret later text divergence first.

## 5. Compare first-token logits

```bash
python3 benchmarks/leyline/scripts/compare_logits.py \
  --correctness "${LEYLINE_RUN_DIR}/correctness-source.json" \
  --captures "${LEYLINE_RUN_DIR}/raw-logits" \
  --output "${LEYLINE_RUN_DIR}/logit-comparison.first-token.json"
```

Check `complete=true`, no missing rank pairs, and no capture-budget status failure. Max-absolute difference, cosine similarity, Jensen-Shannon divergence, top-k overlap, and margins are diagnostics, not substitutes for the cache gate.

## 6. Target first-divergence logits

```bash
python3 benchmarks/leyline/scripts/plan_divergence_capture.py \
  --correctness "${LEYLINE_RUN_DIR}/correctness-source.json" \
  --max-step 32 \
  --output "${LEYLINE_RUN_DIR}/divergence-plan.json"
```

Read `target_run_id`, `capture_steps`, and `case_ids` from the plan. Restart the correctness server with:

```bash
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_RUN_ID="$(jq -r .target_run_id "${LEYLINE_RUN_DIR}/divergence-plan.json")"
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_STEPS="$(jq -r '.capture_steps|join(",")' "${LEYLINE_RUN_DIR}/divergence-plan.json")"
export VLLM_ASCEND_LEYLINE_RAW_LOGITS_CASES="$(jq -r '.case_ids|join(",")' "${LEYLINE_RUN_DIR}/divergence-plan.json")"
```

Then run and compare:

```bash
python3 benchmarks/leyline/scripts/run_validation.py \
  --config "${LEYLINE_RUN_DIR}/runner.connector.json" \
  --workloads benchmarks/leyline/workloads/workloads.base.json \
  --environment "${LEYLINE_RUN_DIR}/environment.connector.json" \
  --diagnostic-plan "${LEYLINE_RUN_DIR}/divergence-plan.json" \
  --output "${LEYLINE_RUN_DIR}/divergence-correctness.json"

python3 benchmarks/leyline/scripts/compare_logits.py \
  --correctness "${LEYLINE_RUN_DIR}/divergence-correctness.json" \
  --captures "${LEYLINE_RUN_DIR}/raw-logits" \
  --divergence-plan "${LEYLINE_RUN_DIR}/divergence-plan.json" \
  --output "${LEYLINE_RUN_DIR}/logit-comparison.divergence.json"
```

Only entries with `correlatable=true` reproduced the source prefix/divergence and may explain it.

## 7. Cache-off baseline

Stop connector-on serving, unset capture variables, and start the same TP4 model without a KV connector and with prefix caching disabled:

```bash
unset VLLM_ASCEND_LEYLINE_CAPTURE_DIR
unset VLLM_ASCEND_LEYLINE_RAW_LOGITS_DIR
unset VLLM_ASCEND_LEYLINE_RAW_LOGITS_STEPS
unset VLLM_ASCEND_LEYLINE_RAW_LOGITS_RUN_ID
unset VLLM_ASCEND_LEYLINE_RAW_LOGITS_CASES
unset VLLM_ASCEND_LEYLINE_RAW_LOGITS_ARMS
```

Collect the cache-off environment from its own runtime file, then run all cases for three repetitions:

```bash
python3 benchmarks/leyline/scripts/collect_environment.py \
  --model /opt/foundation_model/DeepSeek-V2-Lite \
  --model-revision local \
  --tokenizer /opt/foundation_model/DeepSeek-V2-Lite \
  --tokenizer-revision local \
  --runtime-config "${LEYLINE_RUN_DIR}/runtime.cache-off.json" \
  --output "${LEYLINE_RUN_DIR}/environment.cache-off.json"

python3 benchmarks/leyline/scripts/run_validation.py \
  --config "${LEYLINE_RUN_DIR}/runner.cache-off.json" \
  --workloads benchmarks/leyline/workloads/workloads.base.json \
  --environment "${LEYLINE_RUN_DIR}/environment.cache-off.json" \
  --output "${LEYLINE_RUN_DIR}/correctness-cache-off.json"
```

The finalizer requires every case, counterfactual variant, and repetition; it rejects mixed checkpoint/topology identity while allowing only the intended connector/prefix-cache difference.

## 8. Rollback after a partial device write

Use a dedicated concurrency-one connector server. Never enable this on shared serving:

```bash
export VLLM_ASCEND_LEYLINE_FAULT_INJECTION=validation-only
```

```bash
python3 benchmarks/leyline/scripts/run_rollback_validation.py \
  --config "${LEYLINE_RUN_DIR}/runner.rollback.json" \
  --environment "${LEYLINE_RUN_DIR}/environment.connector.json" \
  --case inventory-reorder-label \
  --fail-rank 1 \
  --fail-after-layer 8 \
  --output "${LEYLINE_RUN_DIR}/rollback-report.json"

unset VLLM_ASCEND_LEYLINE_FAULT_INJECTION
```

Required conditions include: injection reached after a destination write, `applied=false`, `transform_failed`, touched blocks invalidated, full honest-prefill accounting, honest and fallback targets passing, and every cleanup counter equal to zero.

## 9. Finalize immutable evidence

```bash
python3 benchmarks/leyline/scripts/finalize_validation.py \
  --correctness "${LEYLINE_RUN_DIR}/correctness-source.json" \
  --environment "${LEYLINE_RUN_DIR}/environment.connector.json" \
  --cache-comparison "${LEYLINE_RUN_DIR}/cache-comparison.json" \
  --first-token-logits "${LEYLINE_RUN_DIR}/logit-comparison.first-token.json" \
  --divergence-logits "${LEYLINE_RUN_DIR}/logit-comparison.divergence.json" \
  --cache-off "${LEYLINE_RUN_DIR}/correctness-cache-off.json" \
  --rollback "${LEYLINE_RUN_DIR}/rollback-report.json" \
  --output "${LEYLINE_RUN_DIR}/joined-qualification.json"
```

The finalizer makes no HTTP calls and never rewrites source evidence. It hashes every input and exits nonzero when qualification fails, while still writing the report.

```bash
jq '{passed:.qualification.passed,
     determining_gate:.qualification.determining_gate,
     gates:.qualification.gates,
     provenance_errors:.qualification.provenance_errors,
     cases:.qualification.case_classifications,
     retained:.retained_context_diagnostics}' \
  "${LEYLINE_RUN_DIR}/joined-qualification.json"
```

Decision order:

1. Environment/provenance mismatch → invalid run.
2. Record/transform incomplete → connector failure.
3. cKV/Kpe/native RoPE failure → numerical implementation failure.
4. Rollback failure → transaction-integrity failure.
5. Full/honest/counterfactual/cache-off failure → invalid workload baseline.
6. Baselines pass but Leyline target fails → task-level Leyline limitation.
7. Numerical and target gates pass, suffix later differs → accepted target with autoregressive divergence.
8. Informative retained diagnostics follow honest → no retained-context benefit demonstrated for those cases; this is not an admitted-task failure.

## 10. Performance and artifact handling

Run performance only after qualification passes, on a fresh server with cache, raw-logit, and fault-injection hooks unset. Also set both diagnostic capture flags and `smoke_gate.require_device_capture` to `false` in a dedicated performance runner configuration. Warm up first and report cold/warm data separately.

Publish the small JSON/config/trace artifacts and hashes. Cache NPZ and raw vectors are permission-restricted, large, and potentially model-sensitive; retain them on approved storage and publish their manifests, hashes, comparison summaries, and retention location. Never publish prompt/cache material containing secrets.
