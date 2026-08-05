## 1. Runtime Contract and Capability Probe

- [x] 1.1 Add source-level tests requiring explicit 310P MLA backend selection and rejecting unknown MLA/sparse combinations
- [x] 1.2 Add a 310P device probe for FP16 MLA cache write/load, prefill, decode, chunked context, transform primitives, and synchronization
- [x] 1.3 Record the exact source, vLLM, CANN, torch-npu, model, topology, dtype, and cache layout in the qualification artifact

## 2. 310P MLA Backend and Cache

- [x] 2.1 Add `AscendMLABackend310` with a 128-token logical cKV/Kpe cache contract
- [x] 2.2 Add `AscendMLAAttentionSpec` allocation and binding to `NPUModelRunner310`
- [ ] 2.3 Implement or adapt FP16 prefill, decode, chunked-prefill, cache-write, and cache-load operations according to probe results
- [ ] 2.4 Add synthetic metadata, cache-layout, slot-mapping, and attention correctness tests
- [ ] 2.5 Run DeepSeek-V2-Lite FP16 greedy correctness without any KV Connector

## 3. Leyline KV Connector on 310P

- [x] 3.1 Replace the blanket KV-transfer rejection with a strict local Leyline allowlist and fail-closed runtime validation
- [ ] 3.2 Register the bound cKV/Kpe tensors and exercise the inherited no-forward connector lifecycle
- [x] 3.3 Add a distinct 310P FP16 runtime matrix without weakening the existing 910B BF16 constraints
- [ ] 3.4 Verify TP-wide completion, source pin release, invalid-block aggregation, delayed publication, and honest-prefill fallback

## 4. FP16 Numerical Qualification

- [ ] 4.1 Add 310P captures for source cKV/Kpe, transformed rows, honest-recompute rows, positions, and inverse frequencies
- [x] 4.2 Compare cKV bitwise and Kpe against independent FP32 and native-recompute references
- [ ] 4.3 Cover deltas 0, 1, 127, 128, 129, 1024 and near-maximum supported context positions
- [ ] 4.4 Calibrate and document FP16 maximum and percentile error thresholds from device evidence

## 5. End-to-End and Enablement

- [ ] 5.1 Run full, honest-edited, patched-disabled, and Leyline arms using base-model reference-prefix evaluation
- [ ] 5.2 Run failure injection for every layer/rank boundary and verify transaction rollback
- [ ] 5.3 Optionally run structured-oracle experiments with an instruction-tuned checkpoint
- [ ] 5.4 Run concurrency and performance only after correctness gates pass
- [ ] 5.5 Mark the runtime qualified only when a baseline-matched hardware record passes every required gate
