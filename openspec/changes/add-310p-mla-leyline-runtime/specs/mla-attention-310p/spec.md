## Purpose

Provide a fail-closed FP16 DeepSeek MLA attention and cache implementation for Ascend 310P that can be validated independently of Leyline.

## ADDED Requirements

### Requirement: Explicit 310P MLA backend selection
The platform SHALL select `AscendMLABackend310` for non-sparse MLA on 310P and SHALL NOT silently select the ordinary GQA backend for any MLA request.

#### Scenario: Supported MLA selection
- **WHEN** a 310P request has `use_mla=true` and `use_sparse=false`
- **THEN** the selected backend is `AscendMLABackend310`

#### Scenario: Unsupported 310P attention combination
- **WHEN** a 310P request requests sparse MLA or another unimplemented attention combination
- **THEN** initialization fails with an explicit unsupported-capability error before cache allocation

### Requirement: Logical MLA cache contract
The 310P runner SHALL allocate and bind separate FP16 cKV and Kpe tensors using logical shapes `[blocks, block_size, 1, kv_lora_rank]` and `[blocks, block_size, 1, qk_rope_head_dim]`.

#### Scenario: DeepSeek-V2-Lite cache allocation
- **WHEN** DeepSeek-V2-Lite initializes with block size 128 and FP16 cache
- **THEN** every MLA layer exposes cKV dimension 512 and Kpe dimension 64 with consistent physical-slot semantics

### Requirement: FP16 MLA execution modes
The backend SHALL correctly execute no-cache prefill, decode, and chunked-prefill/cache-hit tails in FP16 on the qualified 310P environment.

#### Scenario: Normal inference without connector
- **WHEN** a deterministic DeepSeek-V2-Lite request executes without KV transfer
- **THEN** its greedy tokens and selected logits agree with the approved reference within the qualified numerical envelope

### Requirement: Hardware capability gate
The implementation SHALL remain disabled when any required 310P FP16 MLA operation or layout has not passed the baseline-matched hardware probe.

#### Scenario: Source support without device qualification
- **WHEN** backend source is present but the matching qualification artifact is absent or failed
- **THEN** the runtime reports the missing gate and does not advertise 310P MLA Leyline support
