## 1. Development Baseline

- [x] 1.1 Clone vLLM and vLLM-Ascend locally, pin them to the container SHAs, and create `leyline-mla-ascend` branches
- [x] 1.2 Add the Leyline connector module and test package structure in vLLM-Ascend without enabling runtime behavior
- [x] 1.3 Document the v0 `kv_transfer_params.leyline` record/amortize schema and fallback reason vocabulary

## 2. Cache Mapping and Rotation

- [x] 2.1 Implement validated single-deletion parsing and edited-token/source-token compatibility checks
- [x] 2.2 Implement full-block reusable-prefix calculation and flat source/destination slot mapping across 128-token pages
- [x] 2.3 Implement FP32 unit-magnitude DeepSeek YaRN delta cos/sin generation and an independent Kpe rotation reference
- [x] 2.4 Implement the cache transformation that copies cKV and rotates Kpe without unsafe in-place source/destination aliasing
- [x] 2.5 Add CPU unit tests for invalid edits, block crossings, partial tails, zero delta, boundary deltas, and YaRN magnitude preservation

## 3. Local KV Connector Integration

- [x] 3.1 Implement scheduler-side source-session recording, compatibility identity validation, and APC source-block resolution
- [x] 3.2 Implement source block pin/release lifecycle and asynchronous external-token match calculation
- [x] 3.3 Implement serializable scheduler-to-worker transformation metadata and destination block binding
- [x] 3.4 Implement worker-side MLA cache registration, per-layer transformation, completion reporting, and invalid-block reporting
- [x] 3.5 Add connector lifecycle tests covering success, missing blocks, unsupported modes, TP aggregation, rollback, and honest-prefill fallback

## 4. Validation Harness

- [x] 4.1 Add workload schema and fixtures for valid AMORTIZE, counterfactual, mechanism-diagnostic, and negative-control cases
- [x] 4.2 Add full, honest-edited, patched-disabled, and Leyline execution arms with deterministic structured outcome comparison
- [x] 4.3 Add numerical cache comparison and environment-manifest collection
- [x] 4.4 Add performance result collection for transformed/prefill tokens, transform latency, TTFT percentiles, throughput, and NPU memory

## 5. Verification

- [ ] 5.1 Run formatting, static checks, and CPU unit tests in the local repositories
- [ ] 5.2 Synchronize the pinned branches into the VA container and verify package/runtime versions with the corrected version command
- [ ] 5.3 Run synthetic BF16 MLA transformation tests on one 910B and validate every layer against the FP32 reference
- [ ] 5.4 Run DeepSeek-V2-Lite TP4 end-to-end correctness, rollback, and patched-disabled tests on 4x32GB 910B
- [ ] 5.5 Run the semantic admissibility matrix and only then the concurrency/performance matrix on TP4, retaining TP8 as an optional scaling comparison
- [ ] 5.6 Record results, limitations, exact environment, and any deferred optimized-op work in the change artifacts
