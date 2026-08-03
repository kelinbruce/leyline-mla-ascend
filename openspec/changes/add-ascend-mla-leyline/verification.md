## Verification status

Last updated: 2026-08-03 (Asia/Shanghai).

### Pinned development baseline

- vLLM: `752a3a504485790a2e8491cacbb35c137339ad34`, local branch `leyline-mla-ascend`, no changes required for v0.
- vLLM-Ascend: `0ac41db46bc73e0338bce2052d5a22d441b36d9a`, local branch `leyline-mla-ascend`, Leyline changes remain uncommitted for review.
- Target container package reported by the user: `vllm_ascend 0.19.1rc2.dev1172+g0ac41db46` from editable checkout `/vllm-workspace/vllm-ascend`.
- The supplied `npu-smi` snapshot reports version `25.0.rc1` and eight 910B4 devices with 32768 MB HBM each. Devices 0-3 and 4-7 were occupied by two four-process jobs at roughly 26 GB process memory per device, so the snapshot does not establish a free TP4 partition; residency must be checked again immediately before testing.

### Local checks completed

- OpenSpec strict validation passes with 17 of 23 implementation tasks complete.
- Every new Python file passes AST/bytecode compilation with an isolated pycache.
- `git diff --check`, the repository 120-column limit check, JSON parsing, and shell syntax checks pass.
- A standard-library smoke test verifies prompt construction is an exact token deletion and verifies structured result/metric aggregation.
- A NumPy CPU smoke test covers cross-block mapping and FP32 DeepSeek YaRN delta rotation at zero, small, boundary, and long-context deltas while preserving the cached magnitude factor.
- Connector lifecycle tests were executed through lightweight local dependency stubs and cover successful completion, missing source blocks, unsupported runtime fallback, TP-wide failure aggregation, block-reference release, and honest re-prefill accounting.

The local host does not provide `torch`, `torch-npu`, `vllm`, `pytest`, or `ruff`. Therefore the repository-native test and lint commands remain part of task 5.1 and must run inside the VA environment; the lightweight checks above are not represented as a substitute for that gate.

### Hardware gates pending

The local Codex environment has neither Docker access nor an Ascend device, so tasks 5.2 through 5.5 cannot be executed here. `tools/sync_leyline_to_container.sh` copies only the reviewed Leyline files into the editable VA checkout and prints the resulting container worktree. The container run must then:

1. Verify `vllm`, `vllm-ascend`, `torch`, `torch-npu`, CANN, driver, model, and tokenizer revisions.
2. Run repository lint and CPU tests.
3. Capture BF16 MLA rows on one 910B and pass `compare_cache.py` for every layer before any semantic claim.
4. Run TP4 rollback, patched-disabled, and end-to-end structured-oracle gates.
5. Admit only cases whose full, honest-edited, and counterfactual arms agree, then run TP4 concurrency 1/4/8/16. TP8 is optional.

### Known limitations and deferred work

- v0 accepts only one contiguous deletion and transforms complete 128-token destination blocks.
- It is text-only, BF16, TP4, DCP1/PCP1, eager, APC-enabled, non-speculative, non-quantized DeepSeek-V2-Lite inference.
- It does not infer semantic irrelevance and is not a secure deletion or forgetting mechanism.
- The initial worker operation uses correctness-first torch/torch-npu tensor operations plus a device synchronization. A fused Ascend operation is intentionally deferred until the numerical and semantic gates pass and transformation time shows it is necessary.
