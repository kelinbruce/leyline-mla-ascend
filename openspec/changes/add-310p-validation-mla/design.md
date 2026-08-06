## Context

See `proposal.md` for motivation and `specs/310p-validation-mla/spec.md` for the observable contract. The branch already selects a distinct 310P MLA backend, allocates logical cKV/Kpe tensors through the parent MLA cache path, and allowlists the local Leyline connector. The backend implementation itself is a fail-closed placeholder. Hardware probes show that FP16 matmul and cache gather/scatter execute on all four target devices, while `KvRmsNormRopeCache` and `FusedInferAttentionScore` reject the 310P SoC. `AtbPagedCacheLoad` is also unsuitable as a validation dependency.

The parent MLA implementation already owns model projections, metadata construction, output projection, and connector wait/save hooks. The validation backend can inherit those pieces and replace only the unsupported RoPE/cache and attention operations.

## Goals / Non-Goals

**Goals:**

- Reuse the existing vLLM MLA metadata, projection, cache allocation, and Leyline connector lifecycle.
- Run deterministic DeepSeek-V2-Lite FP16 TP4 requests through the completions API.
- Favor transparent tensor math and explicit reference tests over throughput.
- Ensure a successful Leyline arm proves that cache transformation was applied.

**Non-Goals:**

- Supporting batches larger than one, concurrent requests, long contexts, or arbitrary DeepSeek-family configurations.
- Using ACL graph, fused MLA operators, ATB paged-cache load, quantization, speculation, DCP/PCP, PP, or hybrid attention.
- Providing a performant or upstream-ready 310P MLA implementation.
- Replacing the existing 910/A2/A3 MLA path.

## Decisions

### 1. Override the parent MLA implementation at the device-operation boundary

`AscendMLAImpl310` will call the parent constructor and reuse its projections and forward orchestration. It will override cache write/RoPE, prefill, cache-hit context, and decode methods. Pure tensor helpers will live in a small 310P validation module so their math can run on CPU in unit tests without constructing vLLM model objects.

Alternative: implement a separate model runner or patch the DeepSeek model. Rejected because it would duplicate scheduler metadata and bypass the connector hooks that the validation must exercise.

### 2. Compose cache writes from transparent tensor operations

For each token, split the projected latent tensor into 512-dimensional cKV and 64-dimensional Kpe input. Compute RMSNorm for cKV with FP32 variance, apply the layer weight, and cast to FP16. Reproduce the existing interleaved DeepSeek RoPE convention by rearranging even/odd Kpe coordinates, applying half-split rotation with the provided cos/sin rows, and casting to FP16. Clone computed rows and publish them with `index_copy_` into logical cache slots.

The same helper serves prefill and decode, returning the normalized cKV and rotated Kpe rows needed by the parent projection flow.

Alternative: call `npu_interleave_rope` while replacing only the fused cache-write operator. Rejected as the primary path because operator presence has not established 310P execution support; it may remain a diagnostic comparison.

### 3. Use dense attention over explicitly gathered logical rows

The validation helper uses the probe-qualified FP16 matrix multiply, converts
each score component to FP32 before addition and scaling, and keeps softmax in
FP32:

```text
score = scale * (q_nope @ k_nope.T + q_pe @ k_pe.T)
probability = softmax(mask(score), dim=-1)
output = probability @ value
```

Normal prefill uses the already expanded per-head K/V tensors produced by the parent projection flow and a causal mask derived from cumulative query lengths. Because the validation envelope permits one request, the initial implementation rejects multi-sequence metadata rather than guessing at padding semantics.

Decode gathers logical cKV/Kpe rows from the paged cache using the request block table and context length, computes absorbed latent attention directly against cKV, and passes the latent result through the parent's value up-projection.

Alternative: materialize and retain full K/V in the 310P GQA cache. Rejected because it changes the Leyline cache contract, increases memory, and requires a second cache representation after every transform.

### 4. Fold cache-hit context and new tail into one dense calculation

For a prefix-cache or Leyline hit, gather the cached logical rows addressed by the prefill block table, append the current tail rows already produced by cache write, expand cached cKV through the existing KV up-projection, and compute attention for the tail with an offset causal mask. This replaces the parent's fused split-context calculation and avoids `AtbPagedCacheLoad` and log-sum-exp merging.

The path will initially accept one prefill request and one contiguous cached prefix. Unsupported mixed decode/prefill or multi-request layouts fail with an explicit validation-only error.

### 5. Keep connector registration in the normal runner lifecycle

The 310P runner will use the parent MLA cache allocation and binding. Any missing cache registration hook needed by the local connector will be added at the same lifecycle point used by the normal runner; no second scheduler protocol will be introduced. The arm report is accepted only when the amortize request reports `applied`, a positive transformed-token count, and no fallback reason.

### 6. Separate local proof from device qualification

Unit tests will cover RMSNorm/RoPE/cache mapping, paged-row gathering, causal masks, prefill output, and absorbed decode output on CPU. Source-level tests will keep the strict runtime envelope. A checked-in launch recipe will run the existing service harness on 310P and record the exact environment. Local tests do not mark the device run complete.

## Risks / Trade-offs

- **[Base metadata has more states than the validation backend accepts]** → Reject mixed or multi-request states with explicit errors and configure `max_num_seqs=1`.
- **[Dense attention is quadratic]** → Cap the documented validation context and generate only a small number of tokens.
- **[FP16 NPU results may differ from CPU]** → Accumulate scores and softmax in FP32, record explicit tolerances, and compare generated token IDs separately.
- **[RoPE layout could diverge from the fused operator]** → Base the helper on the repository's existing interleave precision reference and compare honest and transformed cache rows on device.
- **[A Leyline request may silently recompute]** → Treat missing `applied`, zero transformed tokens, or any fallback reason as validation failure.
- **[310P-only changes could affect A2/A3]** → Keep all overrides in the 310P backend and retain the parent implementation unchanged.

## Migration Plan

1. Land pure validation helpers and CPU-reference tests with the runtime still fail-closed.
2. Replace the 310P placeholder and pass source-level construction tests.
3. Enable the restricted launch configuration and bind logical caches to the connector.
4. On the 310P host, launch DeepSeek-V2-Lite without a connector and verify full/honest prefill and decode.
5. Relaunch with the Leyline connector and run all service arms, requiring positive transform evidence.
6. Roll back by restoring the fail-closed placeholder or using an unsupported configuration; the A2/A3 backend is unaffected.

## Open Questions

- The exact short-context cap can be lowered after observing 310P memory and latency without changing the interface or task structure.
- DeepSeek-V2-Lite-Chat may be substituted for the base checkpoint when a strict structured semantic oracle is available; the base checkpoint remains limited to token-prefix behavior agreement.
