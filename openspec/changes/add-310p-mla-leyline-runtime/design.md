## Context

The implementation branch contains the requested vLLM-Ascend baseline `05e095a202bdcfef4da61168eae34bfd3b99da13` plus the local Leyline prototype. The 310P container pairs this source with vLLM v0.23.0 and CANN 9.1.0-beta.1-310p.

The ordinary Ascend MLA backend exposes two logical paged tensors per layer: latent cKV and positional Kpe. The 310P runner instead allocates ordinary GQA K/V tensors in a FRACTAL_NZ-oriented shape and rejects MLA and KV transfer. The existing Leyline connector expects logical four-dimensional `[blocks, block, heads, dim]` cKV/Kpe tensors and currently accepts only BF16 TP4.

The inherited vLLM model-runner mixin already binds connector metadata, calls `start_load_kv`, collects completion and invalid block IDs, and returns worker metadata. The 310P integration should reuse that lifecycle rather than introduce a second scheduler protocol.

## Goals / Non-Goals

**Goals:**

- Establish a correct FP16 DeepSeek-V2-Lite MLA baseline on 310P before enabling Leyline.
- Preserve an explicit logical cKV/Kpe cache contract at the connector boundary.
- Fail closed for unqualified operators, layouts, connectors, and execution modes.
- Preserve TP-wide delayed publication and honest-prefill rollback.
- Produce numerical evidence tied to the exact device environment.

**Non-Goals:**

- Performance parity with 910B in the first implementation.
- Graph mode, speculative decoding, quantization, sparse attention, DCP/PCP, pipeline parallelism, or hybrid Mamba/MLA.
- General 310P support for arbitrary KV Connectors.
- Treating base-model instruction following as a backend correctness oracle.

## Decisions

### 1. Select a distinct 310P MLA backend

The platform SHALL map `(use_mla=True, use_sparse=False)` to `AscendMLABackend310`. Unknown 310P attention combinations SHALL raise an explanatory error instead of falling back to GQA.

The backend owns 310P-specific operation selection and capability validation. It may reuse device-independent MLA metadata and projection logic, but SHALL NOT inherit the BF16-only prefill behavior as evidence of FP16 support.

### 2. Keep a logical ND cache boundary

The initial connector-visible cache contract is:

```text
cKV: [num_blocks, 128, 1, 512], float16
Kpe: [num_blocks, 128, 1, 64],  float16
```

If an attention operator requires another physical format, the backend SHALL use an explicit layout adapter and physical slot mapping. A reshape that changes the meaning of a physical slot is not permitted.

### 3. Gate implementation on a device capability probe

Before the runtime is advertised, the exact 310P environment SHALL exercise FP16 RMSNorm/RoPE cache writes, prefill attention, paged decode, chunked context loads, cache gather/scatter, FP32 trigonometry, and TP synchronization.

If native latent MLA operations are unavailable, an AscendC implementation is required. Materializing full K/V from cKV may be used as a diagnostic correctness prototype but is not the production cache design.

### 4. Prove normal MLA inference before connector integration

The first end-to-end gate runs DeepSeek-V2-Lite with no KV Connector. It compares the same prompts against a trusted implementation or captured reference using greedy token IDs and selected logits. This avoids conflating an MLA backend error with a Leyline transformation error.

### 5. Allowlist the local Leyline connector

The 310P runner SHALL continue rejecting generic KV transfer. It may admit a connector only when its configured class/module identifies the local Leyline connector and every 310P runtime constraint passes.

Cache tensors are registered through the inherited initialization lifecycle. The asynchronous no-forward path is reused for transformation and completion reporting.

### 6. Use FP32 phase math with FP16 cache storage

The Leyline transform SHALL clone all source rows before destination writes, copy cKV without arithmetic, calculate delta phases and Kpe rotation in FP32, and cast only the final Kpe rows to FP16.

Device results SHALL be compared to both:

1. an independent CPU FP32 analytical reference; and
2. Kpe rows produced by honest native recomputation at the destination positions.

The existing BF16 `2e-2` tolerances are not inherited. The FP16 envelope is calibrated from device captures and records maximum and percentile error.

### 7. Preserve two-phase publication and rollback

Destination cache hashes remain unpublished until every TP rank reports success. Any rank or layer failure reports all transformed destination block IDs as invalid, releases source references, and resumes through honest prefill. Failure injection is a release gate.

### 8. Separate three correctness questions

- Backend correctness compares identical prompts across implementations.
- Leyline mechanism correctness compares honest-edited and Leyline-edited execution.
- AMORTIZE semantic admissibility compares full and honest-edited task outcomes.

Base checkpoints use token/reference agreement. Structured JSON oracle validation is reserved for an instruction-tuned checkpoint.

## Risks / Trade-offs

- **Native 310P FP16 latent MLA support is not established by source inspection.** The hardware probe is a hard gate and may result in AscendC work.
- **Logical and physical cache layouts may differ.** An adapter adds complexity but protects block/slot correctness.
- **Some torch operations are absent or inaccurate on 310P.** Every transform primitive is probed and replaceable behind a 310P operation boundary.
- **Synchronous device completion increases TTFT.** Correctness and transactional safety take priority; overlap is a later optimization.
- **TP4 may not be the only viable topology.** v0 keeps TP4 to match the existing Leyline contract until additional topologies are qualified.

## Migration Plan

1. Land source-level backend selection, cache-contract tests, connector allowlisting, and a fail-closed capability probe.
2. Run the probe on the pinned 310P image and record supported operations/layouts.
3. Implement or select the required 310P MLA operations and pass synthetic attention tests.
4. Pass DeepSeek-V2-Lite FP16 inference without a connector.
5. Enable Leyline FP16, pass numerical capture and rollback tests, then run end-to-end validation.
6. Enable performance runs only after every correctness gate passes.

Rollback is performed by disabling the connector and the experimental 310P MLA capability; ordinary 310P GQA behavior remains unchanged.

## Open Questions

- Which CANN 9.1.0-beta.1-310p operators support the required FP16 latent shapes and paged layouts?
- Does the preferred attention path consume the logical ND cache directly, or require a separate physical representation?
- What FP16 Kpe error envelope matches honest native RoPE across the supported context range?
- Does DeepSeek-V2-Lite fit and remain stable under the intended TP4 memory/concurrency settings on the target host?
