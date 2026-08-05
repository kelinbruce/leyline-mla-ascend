## 1. Correct the validation contract

- [x] 1.1 Add base-model reference-prefix evaluation using server-returned token IDs
- [x] 1.2 Retain structured JSON oracle evaluation as an explicit separate mode
- [x] 1.3 Add exact-deletion chat-template rendering and disable duplicate special-token insertion
- [x] 1.4 Update report merging and schema versioning for both evaluation modes
- [x] 1.5 Add unit tests for raw/chat deletion and both evaluation contracts

## 2. Pin and inspect the 310P baseline

- [x] 2.1 Base the implementation on vLLM-Ascend `05e095a20`
- [x] 2.2 Record the vLLM v0.23.0 and CANN 9.1 beta pairing from the 310P container files
- [x] 2.3 Add a fail-closed preflight for the upstream MLA and KV-transfer guards
- [x] 2.4 Include 310P capability status in the environment manifest

## 3. Implement true 310P runtime support

- [ ] 3.1 Implement and unit-test the DeepSeek MLA attention/cache layout in the 310P runner
- [ ] 3.2 Integrate external KV Connector allocation, registration, completion, and rollback with `NPUModelRunner310`
- [ ] 3.3 Adapt the Leyline runtime matrix to supported 310P FP16 topology without weakening other safety checks
- [ ] 3.4 Establish FP16 cKV/Kpe numerical tolerances on 310P device captures
- [ ] 3.5 Run DeepSeek-V2-Lite end-to-end full/honest/Leyline correctness and failure rollback on 310P
- [ ] 3.6 Enable the runtime only after the preflight and all correctness gates pass

## 4. Verification

- [ ] 4.1 Run focused unit tests and formatting/static checks
- [ ] 4.2 Run base-checkpoint `reference_prefix` validation on the Ascend host
- [ ] 4.3 Optionally run Chat-checkpoint `structured_json` validation as a separate semantic experiment
