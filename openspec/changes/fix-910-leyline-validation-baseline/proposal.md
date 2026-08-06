## Why

The first Ascend 910B run evaluated a local `DeepSeek-V2-Lite` checkpoint through raw completion token IDs while requiring instruction-following JSON output. Because the full and honest baselines did not establish a valid semantic oracle, the report cannot distinguish an invalid model/prompt baseline from Leyline cache-transformation errors.

## What Changes

- Keep the validation target on the current Ascend 910B BF16/TP4 Leyline branch; no Ascend 310 runtime or qualification work is included.
- Add a fail-closed checkpoint and prompt preflight that records local model/tokenizer identity and verifies instruction-following capability before running the long semantic workload.
- Split validation into a base-model `reference_prefix` mode and an instruction-model `structured_json` mode so reference agreement is never reported as semantic correctness.
- Apply tokenizer chat templates client-side for instruction checkpoints while preserving the declared deletion as an exact token interval, then submit explicit token IDs through `/v1/completions`.
- Compare server-returned generation token IDs for deterministic reference agreement and retain parsed JSON comparison only for structured semantic evaluation.
- Require the relevant full and honest baselines to pass before accepting a Leyline semantic result or starting performance measurements.
- **BREAKING**: Move Leyline correctness reports to schema version 2 with an explicit evaluation contract, checkpoint evidence, baseline-preflight result, and separate semantic/reference admission fields.

## Capabilities

### New Capabilities

- `leyline-validation-baseline`: Defines checkpoint-aware prompt construction, preflight, reference and semantic evaluation modes, and baseline gates for Ascend 910B Leyline validation.

### Modified Capabilities

None.

## Impact

- Affects `benchmarks/leyline/run_validation.py`, report merging, runner examples, environment evidence, validation documentation, and focused harness tests.
- Adds configuration for evaluation mode and prompt format; existing schema-v1 reports remain historical inputs and are not silently interpreted as schema-v2 evidence.
- Does not change the Leyline connector protocol, MLA cache transformation, BF16 runtime constraints, or any Ascend 310 code path.
