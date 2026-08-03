## Why

Leyline has been validated on CUDA with Qwen3 GQA models, but the available model sizes did not exercise the paper's MLA cache transformation. We need a reproducible Ascend 910B implementation and experiment suite that tests both the positional cache transformation and the semantic precondition under which AMORTIZE is valid.

## What Changes

- Add DeepSeek-V2-Lite MLA cache amortization for vLLM-Ascend on 910B, initially targeting BF16, TP4, DCP1, eager execution, and 128-token cache blocks.
- Add a cache transformation that copies the 512-dimensional latent cKV unchanged while delta-rotating the 64-dimensional Kpe component and repacking tokens into destination cache blocks.
- Add scheduler-to-worker transaction semantics so transformed blocks are published to automatic prefix caching only after every tensor-parallel rank completes successfully; otherwise the request falls back to honest re-prefill.
- Add structural eligibility checks and explicit fallback for unsupported execution modes, missing source blocks, incompatible cache identities, and invalid edit plans.
- Add numerical, semantic, rollback, and performance validation. Semantic acceptance will require the edited prompt to remain task-equivalent under honest re-prefill; examples that depend on information retained only in old suffix KV will be treated as diagnostic or negative controls rather than valid AMORTIZE evidence.

## Capabilities

### New Capabilities

- `mla-cache-amortization`: Transform and transactionally reuse DeepSeek MLA KV cache after an AMORTIZE edit on Ascend.
- `amortize-validation`: Classify valid AMORTIZE workloads and report numerical, semantic, fallback, and performance evidence against explicit baselines.

### Modified Capabilities

None.

## Impact

- `vllm-ascend`: new MLA cache transformation operation, model-runner integration, Ascend-specific validation, and NPU tests.
- `vllm`: request/scheduler metadata and KV block lifecycle integration where existing delayed-publication or KV-connector hooks are insufficient.
- `ascend`: OpenSpec artifacts, experiment harnesses, workload fixtures, and reproducible launch/configuration scripts.
- Runtime scope is initially limited to DeepSeek-V2-Lite text inference on 910B with BF16 KV cache, TP4, DCP1, eager mode, APC enabled, and block size 128.
