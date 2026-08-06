## Why

The available Ascend 310P environment cannot start DeepSeek-V2-Lite through vLLM because the official 310P runtime rejects MLA and the required fused MLA operators do not support the device. With no 910 environment available, Leyline's full, honest-edited, APC, disabled, and transformed request arms need a correctness-first 310P MLA path that can run the existing service-based validation harness.

## What Changes

- Replace the fail-closed 310P MLA placeholder with a validation-only FP16 implementation built from basic Torch/NPU tensor operations.
- Support DeepSeek-V2-Lite TP4 eager inference with a logical 128-token cKV/Kpe cache, including cache write, dense prefill, paged decode, and cache-hit tail attention.
- Keep the implementation deliberately narrow: batch size one, short contexts, no graph mode, quantization, speculation, DCP/PCP, pipeline parallelism, concurrency, or performance claim.
- Register the logical MLA caches with the existing local Leyline connector so the current `/v1/completions` validation harness can exercise every arm without silently falling back.
- Add deterministic source-level and CPU-reference tests plus a 310P launch/validation recipe that records whether Leyline actually transformed tokens.

## Capabilities

### New Capabilities

- `310p-validation-mla`: Start DeepSeek-V2-Lite through vLLM on Ascend 310P using a restricted correctness-first MLA backend and exercise the existing Leyline validation arms.

### Modified Capabilities

None.

## Impact

- `vllm_ascend/_310p/attention/mla_v1.py`: validation-only MLA cache and attention operations.
- `vllm_ascend/_310p/model_runner_310p.py`: cache registration and restricted runtime integration.
- `vllm_ascend/_310p/mla_runtime.py`: validation limits and fail-closed configuration checks.
- `vllm_ascend/distributed/kv_transfer/leyline/`: reuse of the existing connector lifecycle and metrics.
- `benchmarks/leyline/`: 310P runtime configuration, launch instructions, and arm acceptance checks.
- `tests/ut/_310p/`: CPU-reference and source-level coverage for the restricted implementation.

This change is experimental validation infrastructure only and does not advertise general or production-grade 310P MLA support.
