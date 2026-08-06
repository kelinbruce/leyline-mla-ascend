## 1. Validation Tensor Primitives

- [x] 1.1 Add pure Torch helpers for FP32-accumulated RMSNorm, DeepSeek interleaved RoPE, and logical cKV/Kpe cache writes
- [x] 1.2 Add helpers for paged logical-row gathering, offset causal masks, dense prefill attention, and absorbed latent decode attention
- [x] 1.3 Add deterministic CPU unit tests for cache mapping, RoPE, causal prefill, cached-prefix tails, and decode reference agreement

## 2. Restricted 310P MLA Backend

- [x] 2.1 Replace the fail-closed MLA placeholder with a parent-backed validation implementation and enforce the single-sequence FP16 TP4 eager envelope
- [x] 2.2 Route prefill and decode cache writes through the validation tensor primitives without fused MLA operators
- [x] 2.3 Implement normal prefill and cached-prefix tail attention using dense validation math
- [x] 2.4 Implement paged decode by gathering logical cKV/Kpe rows and applying the existing value up-projection
- [x] 2.5 Add backend-focused unit tests for supported construction, rejected states, and method routing

## 3. Leyline Service Integration

- [x] 3.1 Verify and complete logical MLA cache registration with the local Leyline connector in the 310P runner lifecycle
- [x] 3.2 Make the validation harness fail a Leyline arm that silently falls back, transforms zero tokens, or omits generated token IDs
- [x] 3.3 Add unit tests for positive transform evidence and fallback rejection in the merged arm report

## 4. Launch and Verification

- [x] 4.1 Add a validation-only 310P runtime configuration and launch recipe for DeepSeek-V2-Lite TP4 FP16 eager service execution
- [x] 4.2 Run focused unit tests, OpenSpec strict validation, compilation, formatting, and diff checks
- [ ] 4.3 Run the no-connector and Leyline service arms on the target 310P host and record the environment and results
