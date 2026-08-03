# Leyline MLA validation on Ascend 910B

This harness treats positional cache transformation and semantic admissibility as separate gates. A case is valid AMORTIZE evidence only when the full prompt, an honestly re-prefilled edited prompt, and every approved counterfactual of the removed span produce the declared structured oracle. A case where only Leyline retains deleted information is reported as a mechanism diagnostic and is never counted as valid evidence.

## Server modes

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

Prompts are sent as explicit token-ID arrays. The harness tokenizes the prefix, removed span, and suffix separately, so the declared edit is exactly one deletion and cannot be changed into a replacement by tokenizer boundary merging. A unique `cache_salt` prevents accidental APC contamination between honest baselines; Leyline record/amortize pairs deliberately share one salt.

## Numerical gate

Capture mapped source rows and device-produced destination rows in an NPZ with arrays `source_ckv`, `actual_ckv`, `source_kpe`, `actual_kpe`, `old_positions`, `new_positions`, and `inv_freq`. Include position deltas 0, 1, 127, 128, 129, and 1024, then run:

```bash
python3 benchmarks/leyline/compare_cache.py results/leyline/cache-capture.npz \
  --output results/leyline/cache-comparison.json
```

cKV must be bitwise identical. Kpe is checked against the independent FP32, unit-magnitude YaRN delta reference; the tolerance should be tightened to the measured envelope of one native BF16 rotation after the first device capture.

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
