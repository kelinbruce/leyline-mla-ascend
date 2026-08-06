# Leyline MLA validation on Ascend 910B

This schema-v2 harness treats evaluation-baseline validity, Leyline execution, and cache numerical correctness as separate gates. It supports three explicit contracts:

- `completion_target` with `prompt_format=raw` for the DeepSeek-V2-Lite base checkpoint. Each case declares a tokenizer-stable non-whitespace continuation target, and full, honest-edited, and counterfactual baselines must produce it before Leyline is evaluated.
- `structured_json` with `prompt_format=chat_template` for DeepSeek-V2-Lite-Chat. A short instruction-following preflight must pass before any semantic admission or performance run.
- `reference_prefix` remains available for legacy or exploratory diagnostics, but it cannot produce semantic or overall correctness acceptance.

Schema-v1 reports cannot be merged with schema-v2 reports. Keep old results as historical evidence rather than assigning them the new semantics.

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

The `full`, `honest_edited`, `patched_disabled`, `vanilla_apc`, and `leyline` arms may point to this one server. Requests without a Leyline directive exercise the patched-disabled path. `cache_off` requires a separate run after restarting without the KV connector and with `--no-enable-prefix-caching`; use `runner_config.cache_off.example.json` and merge staged reports with `merge_reports.py`.

## Correctness and semantic gates

Use `runner_config.example.json` for the base checkpoint or `runner_config.chat.example.json` for the Chat checkpoint. Replace both revisions and endpoints, then run:

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
  --workloads benchmarks/leyline/workloads.base.json \
  --environment results/leyline/environment.json \
  --output results/leyline/correctness.json
```

Prompts are sent to `/v1/completions` as explicit token-ID arrays with `add_special_tokens=false`. Raw and chat-template modes each encode their canonical full and edited strings, then fail unless one contiguous token deletion turns full into edited. Base completion targets are derived by tokenizing the prompt both with and without the declared suffix; whitespace-only and boundary-replacing targets fail closed. The versioned base corpus contains sixteen definitions across six admissible task families, four position-stress cases, two counterfactual cases, two mechanism diagnostics, and two negative controls. One stress definition expands to a 1024-token deletion variant. The Chat corpus contains six independent structured cases.

The recommended 910 workflow has two parallel tracks after checkpoint/runtime identity is collected:

1. Repair the evaluation baseline: run base/reference or Chat/structured preflight, then establish the applicable full, honest-edited, and counterfactual gates.
2. Start mechanism diagnostics immediately: collect returned first-token IDs, top-N API log probabilities, common-prefix/divergence evidence, and per-layer/per-rank cache captures without making a semantic claim.

Join the tracks only for acceptance. Leyline additionally requires `recorded=true` on the paired record response, `applied=true` and `transform_complete=true` on amortize, positive transformed tokens, no fallback, numerical success, and rollback success. A unique `cache_salt` prevents accidental APC contamination between honest baselines; Leyline record/amortize pairs deliberately share one salt. Qualification configurations run every correctness case at least three times.

API top-token values are log probabilities. Their top-1/top-2 difference equals the corresponding logit margin, but they are still labeled `api_logprobs`. To collect full-vocabulary logits before grammar processing and sampling, set `VLLM_ASCEND_LEYLINE_RAW_LOGITS_DIR`; those rank-scoped artifacts are separately labeled `internal_raw_logits` with worker provenance and use the request ID retained in the correctness report. Raw-logit capture is limited to the non-speculative validation path.

## Numerical gate

Enable worker capture only for correctness diagnostics:

```bash
export VLLM_ASCEND_LEYLINE_CAPTURE_DIR=results/leyline/cache-captures
export VLLM_ASCEND_LEYLINE_CAPTURE_MAX_ROWS=64
export VLLM_ASCEND_LEYLINE_CAPTURE_REQUIRED_DELTAS=0,1,127,128,129,1024
```

Each TP rank writes permission-restricted per-layer NPZ files plus a rank manifest. Source rows are cloned before the transform and destination rows are read after NPU synchronization. Captures include both connector-derived and native runtime RoPE inverse frequencies; a mismatch fails before Kpe is evaluated. Include position deltas 0, 1, 127, 128, 129, and 1024 in the workload/captures, then run:

```bash
python3 benchmarks/leyline/compare_cache.py results/leyline/cache-captures \
  --expected-layers model.layers.0.self_attn,model.layers.1.self_attn \
  --expected-ranks 0,1,2,3 \
  --output results/leyline/cache-comparison.json
```

cKV must be bitwise identical. Kpe is checked against the independent FP32, unit-magnitude YaRN delta reference, with failures and aggregate errors reported per layer/rank. Missing required layer/rank pairs or position deltas fail the numerical gate.

Join internal raw-logit captures with the correctness report using:

```bash
python3 benchmarks/leyline/compare_logits.py \
  --correctness results/leyline/correctness.json \
  --captures results/leyline/raw-logits \
  --output results/leyline/logit-comparison.json
```

The comparison reports selected tokens, margins, top-k overlap, maximum absolute difference, cosine similarity, and Jensen-Shannon divergence. It remains distributional diagnostic evidence and does not replace the cache numerical gate.

## Performance gate

Run performance only after the selected semantic/reference baseline, Leyline execution, numerical, and rollback gates pass. Set both `performance_prerequisites` values to true only from recorded evidence, and disable both capture environment variables before starting:

```bash
python3 benchmarks/leyline/run_validation.py \
  --config benchmarks/leyline/runner_config.json \
  --environment results/leyline/environment.json \
  --performance --concurrency 1,4,8,16 --repetitions 3 \
  --output results/leyline/performance.json
```

For Leyline, the harness records a distinct pinned source session for every concurrent request before releasing the batch. It reports server TTFT percentiles, client latency, output throughput, transformed and actual-prefill token counts, transformation time, fallback reasons, and sampled NPU memory. Use TP8 only as an optional scaling comparison after TP4 is accepted.
