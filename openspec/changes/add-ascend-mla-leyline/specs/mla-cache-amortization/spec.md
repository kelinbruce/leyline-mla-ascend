## Purpose

Provide transactional reuse of position-corrected DeepSeek MLA cache blocks after a semantically admissible deletion, while preserving normal vLLM behavior through explicit eligibility checks and safe fallback.

## ADDED Requirements

### Requirement: Eligible AMORTIZE deletion
The system SHALL accept an AMORTIZE cache transformation only for a single contiguous deletion whose edited token sequence matches the recorded source sequence outside the deleted span. The initial implementation SHALL require text-only DeepSeek-V2-Lite inference with BF16 KV cache, TP4, DCP1, eager execution, automatic prefix caching, and a 128-token block size.

#### Scenario: Structurally eligible deletion
- **WHEN** a request identifies a recorded source session, supplies one valid deletion span, and satisfies every supported runtime constraint
- **THEN** the system schedules MLA cache amortization for the reusable full-block prefix

#### Scenario: Unsupported edit or runtime
- **WHEN** a request contains a non-empty replacement, multiple edit spans, incompatible tokens, unsupported cache dtype, unsupported parallel mode, or another unsupported runtime feature
- **THEN** the system does not transform cache blocks and continues through honest prefill with an observable fallback reason

### Requirement: MLA cache transformation
For every reusable token and MLA layer, the system SHALL copy the 512-dimensional latent cKV value unchanged, SHALL transform only the 64-dimensional positional Kpe value from its old position to its edited position using a unit-magnitude delta rotation, and SHALL repack results into the destination request's physical block layout.

#### Scenario: Cross-block deletion shift
- **WHEN** a deletion causes target tokens to map across different source and destination 128-token block offsets
- **THEN** each destination row contains the cKV from its mapped source row and the corresponding delta-rotated Kpe

#### Scenario: DeepSeek YaRN scaling
- **WHEN** the source Kpe was produced with DeepSeek YaRN magnitude scaling
- **THEN** the delta transformation preserves the existing magnitude and does not apply YaRN mscale a second time

### Requirement: Full-block reuse boundary
The system SHALL publish only complete destination blocks for which every mapped source row is resident. Tokens outside that reusable region SHALL be processed by normal prefill.

#### Scenario: Partial source or destination tail
- **WHEN** the edited reusable region ends inside a source or destination block
- **THEN** the system truncates transformed reuse to the preceding complete eligible destination block and prefills the remaining tokens normally

### Requirement: Transactional block publication
The system SHALL reserve destination blocks without making their edited hashes visible, SHALL keep all source blocks referenced while transformation is in flight, and SHALL publish destination hashes only after successful completion on every tensor-parallel rank.

#### Scenario: Successful transformation
- **WHEN** every TP rank reports successful transformation of all planned layers and blocks
- **THEN** the destination blocks become prefix-cache-visible and the request resumes from the transformed token count

#### Scenario: Worker or transformation failure
- **WHEN** any rank reports an invalid block or transformation failure
- **THEN** no failed destination block is used as computed cache, all temporary references are released, and the request falls back to honest prefill

### Requirement: Cache identity isolation
The system SHALL reuse source blocks only when source and destination share the required model, tokenizer, RoPE configuration, KV layout, LoRA identity, and cache-salt or tenant identity.

#### Scenario: Incompatible source identity
- **WHEN** any cache identity component differs between the recorded source and edited request
- **THEN** the system rejects cache amortization and performs honest prefill without reading source cache data

### Requirement: AMORTIZE observability
The system SHALL report whether cache amortization was applied and SHALL expose the transformed token count, normal-prefill token count, transformation duration, and fallback reason without exposing prompt contents.

#### Scenario: Completed request metrics
- **WHEN** an AMORTIZE-marked request completes or falls back
- **THEN** its metrics distinguish transformed reuse from ordinary local APC hits and external cache hits
