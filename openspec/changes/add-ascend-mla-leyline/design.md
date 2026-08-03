## Context

The implementation targets the clean container checkouts reproduced locally at vLLM `752a3a504485790a2e8491cacbb35c137339ad34` and vLLM-Ascend `0ac41db46bc73e0338bce2052d5a22d441b36d9a`. The Ascend MLA backend stores each layer as separate paged cKV and Kpe tensors, writes them through physical slot mappings, and supports a 128-token kernel block size. See `proposal.md` for motivation and the capability specs for observable behavior.

The vLLM checkout already provides an external KV Connector API, request-scoped `kv_transfer_params`, delayed cache-block publication for asynchronous loads, worker completion aggregation, invalid-block reporting, and scheduler recovery. These facilities match the required transaction better than a parallel Leyline-specific scheduler protocol.

## Goals / Non-Goals

**Goals:**

- Keep the initial engine integration in vLLM-Ascend and use stable extension points in the pinned vLLM checkout.
- Establish a simple, independently testable MLA transformation before optimizing it for 910B.
- Preserve prefix-cache atomicity and provide deterministic fallback.
- Produce evidence that distinguishes positional mechanism correctness from semantic workload admissibility.

**Non-Goals:**

- Automatic inference of whether removed content is semantically irrelevant.
- Secure forgetting or removal of information already encoded into surviving cKV.
- Non-empty replacement, multiple edit spans, quantized KV, graph mode, speculative decoding, PD disaggregation, DCP/PCP, multimodal input, or simultaneous use with another KV Connector in v0.
- Peak KV-memory reduction; transformed destination blocks require independent storage.

## Decisions

### 1. Implement v0 as an external local KV Connector

Add a connector module in vLLM-Ascend and select it through `kv_connector_module_path`. The scheduler-side connector discovers source APC blocks and constructs a transformation plan; the worker-side connector operates directly on registered local MLA cache tensors.

This reuses vLLM's existing lifecycle:

```text
edited request + kv_transfer_params
              │
              ▼
scheduler connector validates source and reports async external tokens
              │
              ▼
allocate destination blocks with delayed hash publication
              │
              ▼
worker connector copies cKV and rotates Kpe on every TP rank
              │
              ▼
aggregated completion / invalid block IDs
              │
       ┌──────┴──────┐
       ▼             ▼
 publish hashes   release + honest prefill
```

Alternative considered: add Leyline fields to `SchedulerOutput` and a new transaction state directly in upstream vLLM. That offers more flexible multi-stage edits but duplicates existing connector behavior and expands the initial patch surface. It remains a future option if non-empty replacement is required.

### 2. Carry directives through existing KV transfer parameters

The request uses a namespaced `leyline` object within `kv_transfer_params`. A record request provides a session identifier; an amortize request provides the same identifier and a token-index deletion span. The connector records only the source token IDs, block-hash chain, and compatibility identity needed to locate APC blocks.

The connector validates that the edited prompt equals the recorded source with the declared span removed, allowing additional prompt tokens only after the edited source prefix. Character offsets are not accepted because tokenization must be unambiguous.

### 3. Restrict v0 to one deletion and reuse only complete target blocks

A deletion maps each target token to the same source position before the span and to `target_position + deleted_length` after it. The connector transforms the longest destination prefix after ordinary local APC hits for which every source row is resident and the destination ends on a complete 128-token block. The remaining tail is prefetched normally after asynchronous completion.

This restriction makes the external-token interface sufficient without scheduler changes. Non-empty replacement would require normal computation in the middle of an external prefix load and is therefore a fallback in v0.

### 4. Separate mapping/reference logic from the NPU operation

The plan builder produces flat physical source and destination slot IDs plus old and new logical positions. A device-independent reference validates mapping and rotation. The initial NPU path may use existing torch-npu tensor operations; optimization or fusion is allowed only after reference and end-to-end equivalence tests pass.

For each layer and token:

```text
cKV_dst = cKV_src
Kpe_dst = R(theta_new - theta_old) Kpe_src
```

DeepSeek's cached cos/sin tables include YaRN `mscale`. Delta angles are therefore generated from FP32 inverse frequencies with unit magnitude; cached mscaled cos/sin values are not reapplied as a second rotation.

### 5. Use asynchronous-load semantics as a two-phase commit

The scheduler connector returns an asynchronous external-token match even if the worker implementation performs the transformation synchronously within its no-forward connector step. This causes vLLM to reserve destination blocks with delayed caching and place the request in its remote-KV waiting state. The worker reports completion only after all local layers succeed. Existing executor aggregation supplies the TP-wide completion boundary.

Source blocks are touched before destination allocation and released after completion or rollback. Invalid destination block IDs are reported through the connector failure interface so vLLM discards them before the request resumes.

### 6. Treat semantic admissibility as an experiment property

The runtime performs structural and compatibility checks only. The validation harness constructs full (`F`), honest edited re-prefill (`R`), and Leyline (`L`) arms. A workload enters the valid set only when `F` and `R` agree on the oracle task outcome and approved counterfactuals do not alter it. `L` must then preserve the task outcome; distributional distances are diagnostic.

The paper-style case where `R` loses necessary information but `L` retains it remains a mechanism positive control, not valid AMORTIZE evidence.

## Risks / Trade-offs

- **[Connector API is experimental]** → Pin both repository SHAs, isolate compatibility shims, and test connector lifecycle behavior directly.
- **[Source APC blocks may be evicted]** → Resolve and touch every required source block before reporting an external hit; otherwise return zero and prefill normally.
- **[A TP rank may fail after other ranks write destination pages]** → Delay publication, aggregate completion, report invalid destination blocks, and never treat a partial transaction as computed cache.
- **[Torch reference operations may be too slow]** → Use them only as the correctness baseline; add an optimized Ascend path after numerical acceptance.
- **[AMORTIZE can retain deleted information]** → Document that it is not FORGET and exclude privacy/security deletion workloads.
- **[A single KV Connector prevents simultaneous PD/offload connectors]** → Keep those modes unsupported in v0 and reconsider a native scheduler protocol if composition becomes required.

## Migration Plan

1. Land the reference mapping/rotation logic and CPU tests with no runtime behavior enabled.
2. Add the external connector behind explicit KV-transfer configuration and an opt-in request directive.
3. Run patched-disabled and fallback tests against the pinned vLLM checkout.
4. Validate synthetic NPU cache transformations, then DeepSeek-V2-Lite single-request correctness on 4x910B.
5. Run semantic and performance matrices only after numerical and rollback gates pass.
6. Roll back by disabling the connector configuration; ordinary vLLM-Ascend APC and prefill remain unchanged.

## Open Questions

- Whether an existing torch-npu scatter/rotary combination is fast enough or a fused custom operator is required; this affects optimization tasks but not the public behavior or transaction design.
- The usable KV-cache budget and maximum concurrency on the shared 910B host after the currently resident TP4 services are identified or stopped.
