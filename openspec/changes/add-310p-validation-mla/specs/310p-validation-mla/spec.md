## Purpose

Provide a deliberately restricted Ascend 310P MLA execution path that can start DeepSeek-V2-Lite through vLLM and produce trustworthy service-level evidence for the existing Leyline validation arms.

## ADDED Requirements

### Requirement: Restricted DeepSeek-V2-Lite service startup
The system SHALL start DeepSeek-V2-Lite on Ascend 310P only within the qualified validation envelope: FP16, tensor parallel size four, eager execution, pipeline parallel size one, DCP/PCP size one, block size 128, prefix caching enabled, no quantization, and no speculative decoding.

#### Scenario: Supported validation configuration
- **WHEN** DeepSeek-V2-Lite is launched on Ascend 310P with every validation constraint satisfied
- **THEN** initialization completes with the validation MLA backend instead of rejecting MLA support

#### Scenario: Unsupported configuration
- **WHEN** a launch requests an unqualified model, dtype, topology, graph mode, quantization, or speculative mode
- **THEN** initialization fails before inference with an explanatory validation-envelope error

### Requirement: Logical MLA cache execution
The system SHALL maintain separate FP16 cKV and Kpe caches with logical shapes `[blocks, 128, 1, 512]` and `[blocks, 128, 1, 64]` and SHALL execute cache write, prefill attention, cache-hit tail attention, and decode without relying on MLA fused operators that reject Ascend 310P.

#### Scenario: Normal prefill and decode
- **WHEN** a supported request has no external cache directive
- **THEN** the service writes logical MLA cache rows and returns generated token IDs through the normal completions API

#### Scenario: Prefix-cache tail
- **WHEN** a supported request begins with complete cached blocks and contains an uncached tail
- **THEN** attention includes both the cached context and the newly written tail rows and returns a result without using an unsupported fused paged-cache operation

### Requirement: Correctness-first numerical behavior
The validation backend SHALL compute MLA attention using FP32 score and softmax accumulation where needed for stability, return FP16 model activations, and expose deterministic CPU-reference tests for cache mapping and attention math.

#### Scenario: Attention reference agreement
- **WHEN** the restricted attention path is evaluated with deterministic synthetic inputs
- **THEN** its cache rows and attention output agree with the independent reference within explicit validation tolerances

### Requirement: Leyline service-arm execution
The system SHALL allow the local Leyline connector to operate on the bound logical MLA caches and SHALL support the full, honest-edited, vanilla-APC, patched-disabled, and Leyline arms through the existing completions-based validation harness.

#### Scenario: Leyline transformation is applied
- **WHEN** a record request is followed by a structurally valid amortize request whose transformed prefix covers at least one complete block
- **THEN** the Leyline result reports `applied=true`, a positive transformed-token count, no fallback reason, and returns generated token IDs

#### Scenario: Honest baseline remains available
- **WHEN** the same edited prompt is sent without a Leyline amortize directive
- **THEN** the service computes it through normal prefill so its output can serve as the honest-edited comparison arm

### Requirement: Validation-only claim boundary
The system SHALL label the 310P MLA path experimental and SHALL NOT claim performance, concurrency, long-context, production reliability, or general MLA model support from this change.

#### Scenario: Qualification report
- **WHEN** validation results are produced
- **THEN** the report identifies the restricted runtime envelope and distinguishes base-model token-prefix agreement from instruction-model semantic-oracle validation
