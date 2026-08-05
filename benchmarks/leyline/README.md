# Leyline MLA validation

This harness treats positional cache transformation and semantic admissibility as separate gates. It also separates base-model reference agreement from instruction-model semantic correctness; these are different claims and must not be mixed in one result.

## Evaluation modes

`runner_config.example.json` targets the paper-compatible `DeepSeek-V2-Lite` base checkpoint. Its `reference_prefix` mode uses the full arm's first generated token ID as the reference and checks the honest-edited, counterfactual, and Leyline arms against it. Increase `evaluation.reference_tokens` for a stricter common-prefix comparison. This mode tests deterministic behavior agreement only; `semantic_oracle_validated` and `semantic_admitted` are always false and the JSON oracle is not used to claim task correctness.

`runner_config.chat.example.json` targets `DeepSeek-V2-Lite-Chat`, renders the tokenizer's chat template while preserving the exact token deletion, and uses `structured_json` mode. A case is semantic evidence only when the full prompt, honestly re-prefilled edited prompt, and every approved counterfactual produce the declared JSON oracle. Do not use `structured_json` with the base checkpoint: a base model is trained for text continuation and is not expected to obey the JSON instruction.

Both modes submit explicit token-ID arrays with `add_special_tokens=false`; this prevents the server from inserting a second BOS token and changing deletion indices. The server is asked to return generated token IDs so reference comparison does not depend on lossy decode/re-encode behavior.

## 910B server modes

Pin the model and tokenizer revisions before collecting results. The correctness server uses TP4 on four 32 GB 910B devices and the v0 constraints enforced by the connector:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --revision PIN_MODEL_COMMIT \
  --tokenizer-revision PIN_TOKENIZER_COMMIT \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 1 \
  --prefill-context-parallel-size 1 \
  --block-size 128 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enforce-eager \
  --enable-per-request-metrics \
  --kv-transfer-config '{"kv_connector":"LeylineConnector","kv_role":"kv_both","kv_connector_module_path":"vllm_ascend.distributed.kv_transfer.leyline.connector","kv_load_failure_policy":"recompute"}'
```

The `full`, `honest_edited`, `patched_disabled`, `vanilla_apc`, and `leyline` arms may point to this one server. Requests without a Leyline directive exercise the patched-disabled path. `cache_off` requires a separate run after restarting without the KV connector and with `--no-enable-prefix-caching`; this avoids trying to fit two TP4 engines on the same four devices. Merge staged reports with `merge_reports.py`.

To test the structured oracle instead, launch the same runtime with `--model deepseek-ai/DeepSeek-V2-Lite-Chat` (and its pinned tokenizer/revision), then use `runner_config.chat.example.json`. That combination is the fix for the original “model does not execute the test instruction” symptom. It is a separate experiment from the base-checkpoint reference comparison and should be reported separately.

## Correctness and semantic gates

Copy `runner_config.example.json`, replace both revisions and endpoints, then run:

```bash
python3 benchmarks/leyline/collect_environment.py \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --model-revision PIN_MODEL_COMMIT \
  --tokenizer deepseek-ai/DeepSeek-V2-Lite \
  --tokenizer-revision PIN_TOKENIZER_COMMIT \
  --runtime-config benchmarks/leyline/runtime_config.json \
  --output results/leyline/environment.json

python3 benchmarks/leyline/run_validation.py \
  --config benchmarks/leyline/runner_config.json \
  --environment results/leyline/environment.json \
  --output results/leyline/correctness.json
```

The harness tokenizes the prefix, removed span, and suffix separately, so the declared edit is exactly one deletion and cannot be changed into a replacement by tokenizer boundary merging. In chat-template mode, reserved markers locate the deletion inside the rendered user message and are removed before tokenization. A unique `cache_salt` prevents accidental APC contamination between honest baselines; Leyline record/amortize pairs deliberately share one salt.

## Ascend 310P status

The requested 310P source baseline is vLLM-Ascend commit `05e095a20` (paired by its container files with vLLM v0.23.0 and the 310P CANN 9.1 beta image). This branch explicitly selects a 310P MLA backend, delegates MLA cache allocation to the logical cKV/Kpe path, and allowlists only the local Leyline connector. The attention implementation remains fail-closed until its FP16 operators are qualified on the target device; source integration is not hardware support.

Run the operator probe before model startup:

```bash
for rank in 0 1 2 3; do
  ASCEND_RT_VISIBLE_DEVICES=${rank} python3 benchmarks/leyline/probe_310p_mla_ops.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --model-revision PIN_MODEL_COMMIT \
    --tokenizer-revision PIN_TOKENIZER_COMMIT \
    --tensor-parallel-size 4 --rank ${rank} \
    --output results/leyline/310p-probe-rank${rank}.json
done

python3 benchmarks/leyline/qualify_310p.py \
  --probe results/leyline/310p-probe-rank0.json \
  --probe results/leyline/310p-probe-rank1.json \
  --probe results/leyline/310p-probe-rank2.json \
  --probe results/leyline/310p-probe-rank3.json \
  --numerical results/leyline/cache-comparison-310p.json \
  --e2e results/leyline/correctness-310p.json \
  --rollback results/leyline/rollback-310p.json \
  --output benchmarks/leyline/310p_qualification.json

python3 benchmarks/leyline/check_310p_capability.py \
  --runtime-config benchmarks/leyline/runtime_config.310p.example.json \
  --require-supported
```

The preflight remains false while `AscendMLAImpl310` is the unqualified placeholder or the baseline-matched hardware record is absent/failed. A passing primitive probe is used to select and implement the FP16 prefill, decode, and chunked-cache operations; it does not by itself qualify end-to-end model inference.

## Numerical gate

Capture mapped source rows and device-produced destination rows in an NPZ with arrays `source_ckv`, `actual_ckv`, `source_kpe`, `actual_kpe`, `old_positions`, `new_positions`, and `inv_freq`. Include position deltas 0, 1, 127, 128, 129, and 1024, then run:

```bash
python3 benchmarks/leyline/compare_cache.py results/leyline/cache-capture.npz \
  --output results/leyline/cache-comparison.json
```

cKV must be bitwise identical. Kpe is checked against the independent FP32, unit-magnitude YaRN delta reference; the tolerance should be tightened to the measured envelope of one native BF16 rotation after the first device capture.

For 310P FP16, capture `honest_kpe`, `rank_ids`, and `layer_ids` in addition to the arrays above. First run `compare_cache_fp16.py` without thresholds to obtain calibration metrics; that run intentionally fails qualification. After reviewing captures from every layer and TP rank, rerun with explicit analytical and native-recompute tolerances:

```bash
python3 benchmarks/leyline/compare_cache_fp16.py results/leyline/cache-capture-310p.npz \
  --expected-layers EXPECTED_LAYER_COUNT \
  --reference-atol CALIBRATED_ATOL --reference-rtol CALIBRATED_RTOL \
  --native-atol CALIBRATED_ATOL --native-rtol CALIBRATED_RTOL \
  --output results/leyline/cache-comparison-310p.json
```

## Performance gate

Run performance only after numerical, rollback, and semantic gates pass:

```bash
python3 benchmarks/leyline/run_validation.py \
  --config benchmarks/leyline/runner_config.json \
  --environment results/leyline/environment.json \
  --performance --concurrency 1,4,8,16 --repetitions 3 \
  --output results/leyline/performance.json
```

For Leyline, the harness records a distinct pinned source session for every concurrent request before releasing the batch. It reports server TTFT percentiles, client latency, output throughput, transformed and actual-prefill token counts, transformation time, fallback reasons, and sampled NPU memory. Use TP8 only as an optional scaling comparison after TP4 is accepted.
