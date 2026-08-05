## Why

The first Ascend run used the `DeepSeek-V2-Lite` base checkpoint with raw token IDs but judged every arm against an instruction-following JSON oracle. The resulting `full_matches=false` does not establish a Leyline error: the checkpoint was continuing text rather than following the JSON instruction. Separately, the requested 310P baseline (`05e095a20`) explicitly rejects both MLA and KV transfer, so merely changing BF16 to FP16 would create an unsafe and non-functional configuration.

## What Changes

- Split validation into a base-model `reference_prefix` mode and an instruction-model `structured_json` mode.
- Preserve exact deletion indices when applying a tokenizer chat template, disable server-side special-token insertion, and use returned generation token IDs for comparison.
- Make the base-checkpoint example default to first-token agreement with the full arm, while clearly labeling this as behavioral/mechanism agreement rather than semantic oracle correctness.
- Add a 310P source/runtime preflight pinned to vLLM-Ascend `05e095a20` and its vLLM v0.23.0 container pairing.
- Define the missing 310P MLA, KV Connector, and FP16 hardware work as prerequisites. The existing 910B runtime remains the only enabled Leyline backend until those gates pass.

## Capabilities

### New Capabilities

- `leyline-310p-readiness`: Detect whether a checkout and runtime can safely launch DeepSeek MLA Leyline on 310P and identify unmet platform prerequisites.

### Modified Capabilities

- `amortize-validation`: Distinguish base-model reference agreement from instruction-model semantic-oracle validation.

## Impact

- Validation reports move to schema version 2 and include an explicit evaluation mode, match target, and `semantic_oracle_validated` flag.
- Existing `matches_oracle` is retained only in `structured_json` mode; `reference_prefix` emits `matches_reference`.
- 310P deployment remains fail-closed. Supporting it requires platform backend work beyond the current Leyline connector patch.
