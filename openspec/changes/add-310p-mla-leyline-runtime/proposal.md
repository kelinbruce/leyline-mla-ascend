## Why

The pinned 310P baseline selects a GQA attention backend for MLA requests and then rejects both MLA and KV transfer during KV-cache initialization. Removing those guards alone would bind DeepSeek MLA layers to an incompatible cache layout. In addition, the existing Ascend MLA implementation uses a BF16-only prefill path, while the target 310P runtime and Leyline validation require FP16.

We need a fail-closed 310P runtime that first establishes native DeepSeek MLA correctness, then admits the local Leyline connector, and finally qualifies the FP16 cache transformation on real hardware.

## What Changes

- Add an explicit 310P MLA backend selection path so MLA requests never silently fall back to the ordinary 310P GQA backend.
- Add 310P allocation and binding for logical DeepSeek MLA cKV and Kpe cache tensors.
- Add a 310P-specific MLA implementation boundary and device-capability checks for FP16 prefill, decode, chunked-prefill, cache-write, and cache-load operations.
- Reuse the inherited vLLM KV Connector lifecycle, but allowlist only the local Leyline connector for the initial 310P runtime.
- Extend Leyline eligibility from the existing 910B BF16 matrix to a separately qualified 310P FP16 matrix.
- Add FP16 numerical capture and comparison against both an independent FP32 rotation reference and native honest-recompute cache rows.
- Keep graph mode, speculative decoding, quantized KV, DCP/PCP, sparse attention, hybrid Mamba/MLA, and arbitrary external connectors unsupported in the initial runtime.

## Capabilities

### New Capabilities

- `mla-attention-310p`: Execute DeepSeek MLA prefill, decode, and chunked-prefill with a compatible FP16 cKV/Kpe cache on Ascend 310P.
- `leyline-kv-connector-310p`: Transform and transactionally publish local MLA cache blocks through the vLLM KV Connector lifecycle on Ascend 310P.
- `leyline-fp16-validation`: Qualify the FP16 Leyline transformation numerically and end to end on the pinned 310P environment.

### Modified Capabilities

None.

## Impact

- `vllm_ascend/_310p/attention/`: new MLA backend and metadata/operation adaptations.
- `vllm_ascend/_310p/model_runner_310p.py`: MLA cache allocation, binding, and connector registration.
- `vllm_ascend/platform.py`: explicit 310P MLA backend routing with no fallback.
- `vllm_ascend/distributed/kv_transfer/leyline/`: 310P FP16 eligibility and transactional worker behavior.
- `vllm_ascend/ops/` or `csrc/`: a 310P transform/attention operator only if the hardware capability spike shows that torch-npu primitives are insufficient.
- `tests/ut/_310p/`, `tests/e2e/pull_request/four_card/_310p/`, and `benchmarks/leyline/`: source tests, device tests, captures, and qualification reports.
- The implementation remains experimental until a hardware qualification record matches vLLM-Ascend `05e095a20`, vLLM v0.23.0, CANN 9.1.0-beta.1-310p, the model revision, and the tested topology.
