## Purpose

Define reproducible, checkpoint-aware baseline evidence for Ascend 910B Leyline validation so model prompting failures cannot be mistaken for cache-transformation failures or semantic correctness.

## ADDED Requirements

### Requirement: 910-only validation scope
The validation change SHALL target the existing Ascend 910B BF16, TP4, DeepSeek MLA Leyline runtime and SHALL NOT claim or require Ascend 310 runtime support.

#### Scenario: Validation configuration is reported
- **WHEN** a correctness report is generated for this change
- **THEN** the report identifies the 910B runtime topology and contains no 310 capability or qualification claim

### Requirement: Explicit evaluation contract
Every validation run SHALL declare whether it evaluates base-model generation agreement or instruction-model semantic correctness, and the report SHALL NOT interpret reference agreement as structured-oracle correctness.

#### Scenario: Base checkpoint evaluation
- **WHEN** a base checkpoint is evaluated in `reference_prefix` mode
- **THEN** the full arm supplies the generated-token reference, other arms are compared over the configured prefix length, and semantic-oracle validation is reported as false

#### Scenario: Instruction checkpoint evaluation
- **WHEN** an instruction-tuned checkpoint is evaluated in `structured_json` mode
- **THEN** each semantic gate compares parsed output with the declared structured oracle and reference-only admission is reported as false

#### Scenario: Unsupported evaluation mode
- **WHEN** a run declares an unknown evaluation mode or an incompatible prompt/evaluation combination
- **THEN** validation fails before issuing the long correctness workload

### Requirement: Checkpoint and prompt preflight
The validation suite SHALL record reproducible model and tokenizer identity evidence and SHALL run a short deterministic prompt-format probe before treating a structured semantic workload as valid.

#### Scenario: Local checkpoint identity
- **WHEN** a model or tokenizer is loaded from a local directory
- **THEN** the report records the resolved paths, declared revisions, relevant configuration identity, chat-template availability, and stable hashes sufficient to distinguish the tested artifacts

#### Scenario: Structured probe succeeds
- **WHEN** `structured_json` mode is selected
- **THEN** a short request using the same checkpoint, tokenizer, endpoint, and effective prompt format must parse to its declared JSON result before semantic cases are admitted

#### Scenario: Structured probe fails
- **WHEN** the short structured probe does not produce the declared JSON result
- **THEN** the run reports the baseline preflight failure, skips semantic admission and Leyline semantic acceptance, and does not start performance validation, while still permitting explicitly requested behavioral and numerical diagnostic collection

### Requirement: Exact prompt construction
The validation suite SHALL preserve the declared edit as exactly one token deletion for both raw and chat-template prompts and SHALL prevent inference-time special-token insertion from changing the deletion indices.

#### Scenario: Raw prompt construction
- **WHEN** `prompt_format=raw` is selected
- **THEN** removing the declared token interval from the full prompt token IDs yields the edited prompt token IDs exactly

#### Scenario: Chat-template prompt construction
- **WHEN** `prompt_format=chat_template` is selected
- **THEN** the tokenizer renders a generation prompt with the removed span located inside the rendered user content, and removing that span yields the edited prompt token IDs exactly

#### Scenario: Missing or destructive chat template
- **WHEN** the tokenizer has no usable chat template or rendering cannot preserve the declared edit boundaries
- **THEN** validation fails before sending correctness arms

### Requirement: Server-returned token comparison
Reference evaluation SHALL compare generation token IDs returned by the inference server and SHALL NOT recreate them by tokenizing decoded response text.

#### Scenario: Reference prefix is available
- **WHEN** the full arm returns at least the configured number of generation token IDs
- **THEN** every reference-mode arm is evaluated against that exact token-ID prefix

#### Scenario: Generation token IDs are unavailable
- **WHEN** a required arm omits generation token IDs or returns fewer IDs than the configured reference length
- **THEN** the corresponding reference gate fails closed

### Requirement: First-token score diagnostics
The validation suite SHALL support pairwise first-token and divergence diagnostics across full, honest-edited re-prefill, and Leyline arms independently of structured semantic admission.

#### Scenario: Top-token scores are returned by the serving API
- **WHEN** diagnostic collection is enabled and an arm returns top-token log probabilities
- **THEN** the report records the selected token ID, top candidate IDs and log probabilities, top-1/top-2 score margin, and evidence type for that arm

#### Scenario: Raw first-token logits are captured internally
- **WHEN** internal raw-logit capture is enabled on the 910 runtime
- **THEN** the report records the raw-logit provenance and derives the same top-token, margin, and pairwise diagnostics without labeling API log probabilities as raw logits

#### Scenario: Arm outputs diverge
- **WHEN** two arms produce different token sequences
- **THEN** the report records their first-token agreement, actual common-prefix length, first divergent position, and available token-score evidence at the comparison point

### Requirement: Device cache capture diagnostics
The 910 validation suite SHALL support opt-in device capture of the cache rows used by each Leyline transformation and SHALL compare the captured destination rows with independent transformation expectations per layer and tensor-parallel rank.

#### Scenario: Leyline layer is transformed
- **WHEN** diagnostic capture is enabled for a Leyline request
- **THEN** the capture records request identity, layer, tensor-parallel rank, source and destination slots, source cKV and Kpe rows, actual destination cKV and Kpe rows, old and new positions, inverse frequencies, dtype, and synchronization state

#### Scenario: cKV is evaluated
- **WHEN** captured source and destination cKV rows are compared
- **THEN** the report checks bitwise equality and identifies every failing layer, rank, slot, and position delta

#### Scenario: Kpe is evaluated
- **WHEN** captured Kpe rows are compared
- **THEN** the report evaluates the actual destination against an independent FP32 delta-rotation reference and reports per-layer and aggregate absolute and relative error statistics

#### Scenario: Semantic baseline is unavailable
- **WHEN** structured preflight or structured baseline admission fails
- **THEN** cache capture and first-token diagnostics remain reportable as diagnostic evidence but cannot produce semantic or overall Leyline acceptance

### Requirement: Category-aware baseline admission
The validation suite SHALL establish the expected full, honest-edited, and counterfactual behavior for each workload category before evaluating Leyline acceptance.

#### Scenario: Semantically admissible workload
- **WHEN** an admissible or counterfactual-admissible case runs in `structured_json` mode
- **THEN** full, honest-edited, and every declared counterfactual must match the structured oracle before the case is semantically admitted

#### Scenario: Removed evidence is necessary
- **WHEN** a mechanism-diagnostic or negative-control case removes information required by the oracle
- **THEN** failure of the honest-edited arm is recorded as expected category behavior and the case is excluded from admissible success rates

#### Scenario: Reference-mode admission
- **WHEN** a case runs in `reference_prefix` mode
- **THEN** admission is labeled reference-only and requires the configured honest-edited and counterfactual token prefixes to match the full-arm reference

### Requirement: Leyline acceptance requires execution and baseline evidence
A Leyline result SHALL be accepted only when the applicable baseline is admitted and the response proves that cache transformation was applied without fallback.

#### Scenario: Leyline result is accepted
- **WHEN** the applicable baseline gates pass, the Leyline output matches the selected evaluation target, and execution metadata reports recording, application, positive transformed tokens, and no fallback
- **THEN** the report marks the Leyline result accepted under that evaluation contract

#### Scenario: Output matches after fallback
- **WHEN** the Leyline output matches the target but execution metadata indicates fallback, missing application, or zero transformed tokens
- **THEN** the report does not accept the result as Leyline evidence

### Requirement: Versioned report semantics
Correctness and merged reports SHALL use schema version 2 and SHALL expose evaluation mode, match target, prompt format, preflight evidence, baseline admission, semantic/reference admission, Leyline execution evidence, first-token score diagnostics, and cache-capture provenance separately.

#### Scenario: Schema-v2 reports are merged
- **WHEN** staged schema-v2 reports for different 910 execution arms are merged
- **THEN** the merged report recomputes gates using the declared evaluation contract and preserves checkpoint and preflight evidence

#### Scenario: Legacy report is supplied
- **WHEN** a schema-v1 report is supplied to schema-v2 merging
- **THEN** the tool rejects it or explicitly treats it as legacy evidence without silently assigning schema-v2 semantics

### Requirement: Performance follows correctness
The validation suite SHALL NOT start performance measurement until the applicable baseline and Leyline correctness gates have passed for the selected 910 configuration. Numerical and rollback evidence remain required by the existing overall Leyline validation workflow but are outside this baseline-fix report contract.

#### Scenario: Baseline remains invalid
- **WHEN** the checkpoint preflight or applicable full/honest baseline gate fails
- **THEN** performance validation is skipped with a recorded reason
