## Local verification

Date: 2026-08-05

- `ruff check` passed for every changed Python source and test file.
- Python bytecode compilation passed with `PYTHONPYCACHEPREFIX` redirected to a writable temporary directory.
- Pure Torch MLA numerical tests passed: 8 tests.
- Restricted runtime tests passed: 3 tests.
- 310P backend construction and routing tests passed against lightweight vLLM parent stubs: 6 tests.
- `pytest --confcutdir=tests/ut/distributed/kv_transfer/leyline tests/ut/distributed/kv_transfer/leyline/test_validation_harness.py` passed: 12 tests, including the single-command target runner contract.
- The standalone Leyline acceptance smoke check passed for positive evidence and rejected missing IDs, fallback, `applied=false`, and zero transformed tokens.
- `openspec validate add-310p-validation-mla --strict` passed.
- `git diff --check` passed.
- The 310P preflight recognizes the backend as implemented and reports only `310p_hardware_qualification_missing`.
- The 310P probe now has an `unfused_validation_mla_v1` profile. Only the ordinary-Torch operations used by the validation backend gate its status; unsupported fused MLA calls are retained as non-gating diagnostics.
- `run_310p_service_validation.py` packages probe, environment capture, cold no-connector service, Leyline service, arm execution, merge, and strict result acceptance into one target-host command.

## Pending verification

- CPU Torch was installed into a temporary test-only directory. The local macOS environment still does not contain vLLM or torch-npu, so full plugin-import and NPU execution remain target-host checks.
- This host has no Ascend 310P device or connection to the target host. A no-connector DeepSeek-V2-Lite service smoke test and the connector-backed Leyline arms remain to be run on the target environment.
- The checked-in historical 310P probe proves FP16 matrix multiply and logical cache gather/scatter execute on all four ranks. Its old schema does not satisfy the new unfused validation probe profile and cannot qualify this backend.
