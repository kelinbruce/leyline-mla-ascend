# Leyline MLA validation on Ascend 910B

For the complete staged qualification procedure, including immutable offline
finalization, cache-off repetitions, decode-step logits, and guarded rollback
injection, see [VALIDATION_910B.md](VALIDATION_910B.md).

## Layout

- `scripts/`: executable collection, validation, comparison, and finalization
  entry points.
- `configs/`: version-controlled configuration templates; copy and edit these
  into a timestamped run directory before an NPU run.
- `workloads/`: versioned workload corpora and their JSON schema.
- `common/`: shared evidence and report helpers, not direct commands.

Write new run artifacts under `results/leyline/<run-id>/`. Historical schema-v2
evidence is retained separately under `results/leyline/historical/` and must
not be used as current qualification evidence.

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

The `full`, `honest_edited`, `patched_disabled`, `vanilla_apc`, and `leyline` arms may point to this one server. Requests without a Leyline directive exercise the patched-disabled path. `cache_off` requires a separate run after restarting without the KV connector and with `--no-enable-prefix-caching`; use `configs/runner_config.cache_off.example.json` and merge staged reports with `scripts/merge_reports.py`.

## Correctness and semantic gates

Use `configs/runner_config.example.json` for the base checkpoint or `configs/runner_config.chat.example.json` for the Chat checkpoint. Replace both revisions and endpoints, then run:

```bash
python3 benchmarks/leyline/scripts/collect_environment.py \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --model-revision PIN_MODEL_COMMIT \
  --tokenizer deepseek-ai/DeepSeek-V2-Lite \
  --tokenizer-revision PIN_TOKENIZER_COMMIT \
  --runtime-config benchmarks/leyline/configs/runtime_config.example.json \
  --output results/leyline/environment.json

python3 benchmarks/leyline/scripts/run_validation.py \
  --config benchmarks/leyline/configs/runner_config.example.json \
  --workloads benchmarks/leyline/workloads/workloads.base.json \
  --environment results/leyline/environment.json \
  --output results/leyline/correctness.json
```

Qualification now has two fail-fast stages before the full repeated matrix:

1. The tokenizer-aware workload planner calculates complete source blocks and the longest reusable target range for every canonical and counterfactual prompt. A case with `execution_expectation=required` must predict at least `minimum_transform_tokens`; otherwise the process exits before making any HTTP request and prints the case, variant, stable reason, predicted count, and blocking source block.
2. The configured `smoke_gate.case_id` runs once. The matrix proceeds only when the paired request reports `recorded=true`, `applied=true`, a positive block-aligned transformed-token count, no fallback, complete expected layer/rank counts, and any capture evidence required by the active configuration.

The report stores both stages under `workload_feasibility` and `smoke_gate`. If the smoke request fails, `cases` is empty so a fallback response cannot be mistaken for Leyline semantic evidence. Inspect the gate before reviewing case outputs:

```bash
jq '{feasibility: .workload_feasibility.passed,
     smoke: .smoke_gate | {case_id, passed, conditions, metadata}}' \
  results/leyline/correctness.json
```

When smoke capture is required, set `diagnostics.device_capture_enabled=true`, set `diagnostics.device_capture_dir` to the same directory as `VLLM_ASCEND_LEYLINE_CAPTURE_DIR`, and set `smoke_gate.require_device_capture=true`. The smoke gate joins manifests by the amortize request ID and requires one complete manifest for every expected rank.

Legacy schema-v1/reference diagnostics remain available only with `legacy_diagnostic=true`; that explicit mode skips feasibility admission. Disabling `smoke_gate` is useful for historical diagnostics but does not produce qualification evidence.

Prompts are sent to `/v1/completions` as explicit token-ID arrays with `add_special_tokens=false`. Raw and chat-template modes each encode their canonical full and edited strings, then fail unless one contiguous token deletion turns full into edited. Base completion targets are derived by tokenizing the prompt both with and without the declared suffix; whitespace-only and boundary-replacing targets fail closed. The versioned base corpus contains sixteen definitions across six admissible task families, four position-stress cases, two counterfactual cases, two mechanism diagnostics, and two negative controls. One stress definition expands to a 1024-token deletion variant. The Chat corpus contains six independent structured cases.

For a transformable workload, length must remain after deletion. Set a deterministic `surviving_filler_unit`, a bounded `surviving_filler_max_repeat`, and a block-aligned `minimum_transform_tokens`. The planner chooses the smallest tokenizer-specific repeat count that preserves the exact deletion and completion boundary while making mapped source blocks resident. Do not make a case long only by increasing `removed_repeat`: that can record source blocks while leaving the edited prompt with no complete target block, which yields `no_reusable_blocks` at runtime.

The `route-label` workloads use a closed lookup continuation and repeat the target row immediately before the query. Do not relabel a target from a single observed run; full, honest-edited, and every counterfactual must still empirically qualify the declared target on the pinned 910B checkpoint.

The recommended 910 workflow has two parallel tracks after checkpoint/runtime identity is collected:

1. Repair the evaluation baseline: run base/reference or Chat/structured preflight, then establish the applicable full, honest-edited, and counterfactual gates.
2. Start mechanism diagnostics immediately: collect returned first-token IDs, top-N API log probabilities, common-prefix/divergence evidence, and per-layer/per-rank cache captures without making a semantic claim.

Join the tracks only for acceptance. Leyline additionally requires `recorded=true` on the paired record response, `applied=true` and `transform_complete=true` on amortize, positive transformed tokens, no fallback, numerical success, and rollback success. A unique `cache_salt` prevents accidental APC contamination between honest baselines; Leyline record/amortize pairs deliberately share one salt. Qualification configurations run every correctness case at least three times.

Repetition stability is reported in two dimensions. `execution_stable` compares gates, fallback reason, transformed-token count, and completion state; `generation_stable` compares exact output token-ID sequences for every arm and counterfactual. The legacy `stable` field remains an alias of `execution_stable` and must not be interpreted as exact text stability.

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
python3 benchmarks/leyline/scripts/compare_cache.py results/leyline/cache-captures \
  --expected-layers model.layers.0.self_attn,model.layers.1.self_attn \
  --expected-ranks 0,1,2,3 \
  --output results/leyline/cache-comparison.json
```

cKV must be bitwise identical. Kpe is checked against the independent FP32, unit-magnitude YaRN delta reference, with failures and aggregate errors reported per layer/rank. Missing required layer/rank pairs or position deltas fail the numerical gate.

Join internal raw-logit captures with the correctness report using:

```bash
python3 benchmarks/leyline/scripts/compare_logits.py \
  --correctness results/leyline/correctness.json \
  --captures results/leyline/raw-logits \
  --output results/leyline/logit-comparison.json
```

The comparison reports selected tokens, margins, top-k overlap, maximum absolute difference, cosine similarity, and Jensen-Shannon divergence. It remains distributional diagnostic evidence and does not replace the cache numerical gate.

## Performance gate

Run performance only after the selected semantic/reference baseline, Leyline execution, numerical, and rollback gates pass. Set both `performance_prerequisites` values to true only from recorded evidence, and disable both capture environment variables before starting:

```bash
python3 benchmarks/leyline/scripts/run_validation.py \
  --config benchmarks/leyline/configs/runner_config.example.json \
  --environment results/leyline/environment.json \
  --performance --concurrency 1,4,8,16 --repetitions 3 \
  --output results/leyline/performance.json
```

For Leyline, the harness records a distinct pinned source session for every concurrent request before releasing the batch. It reports server TTFT percentiles, client latency, output throughput, transformed and actual-prefill token counts, transformation time, fallback reasons, and sampled NPU memory. Use TP8 only as an optional scaling comparison after TP4 is accepted.
