## Purpose

Defines a scientifically meaningful Ascend 910B qualification suite that separates base-model completion behavior, Chat semantic behavior, cache numerical correctness, and Leyline execution evidence.

## ADDED Requirements

### Requirement: Evaluation modes make distinct claims
The validation suite SHALL expose `completion_target`, `structured_json`, and `reference_prefix` as distinct evaluation modes. `completion_target` SHALL validate a declared completion-native target for a raw base checkpoint, `structured_json` SHALL validate a declared structured oracle for a Chat checkpoint, and `reference_prefix` SHALL remain diagnostic-only and SHALL NOT produce overall correctness acceptance.

#### Scenario: Raw base-model qualification
- **WHEN** a raw base checkpoint is evaluated for qualification
- **THEN** the suite uses `completion_target` and does not interpret instruction-following JSON behavior as the base-model oracle

#### Scenario: Reference-only diagnostic
- **WHEN** a run uses `reference_prefix`, including a one-token threshold
- **THEN** the report labels every match as diagnostic reference agreement and does not emit semantic or overall Leyline acceptance

### Requirement: Base completion targets are tokenization-stable and meaningful
Every `completion_target` case SHALL declare an expected non-whitespace completion. The planner SHALL prove that appending the target to the canonical prompt produces an exact, non-empty token suffix without replacing prompt-boundary tokens, and SHALL reject targets that resolve only to whitespace or unstable boundary tokenization.

#### Scenario: Stable single-token label
- **WHEN** a few-shot completion case ends at a label boundary and its declared label appends as an exact non-whitespace token suffix
- **THEN** the planner records the target token IDs and permits the case to run

#### Scenario: Trivial newline target
- **WHEN** the declared or observed target consists only of spaces, tabs, or line breaks
- **THEN** the case fails qualification instead of counting the whitespace token as agreement

### Requirement: Expanded base workload corpus
The canonical base corpus SHALL contain at least sixteen cases: at least six admissible completion-native cases across distinct task families, at least four token-position or block-boundary stress cases, at least two counterfactual-admissible cases with multiple removed-text variants, at least two deleted-only mechanism diagnostics, and at least two negative controls. The cases SHALL avoid JSON instructions and excessive repeated prose, and each admissible case SHALL use a short pattern-completion, code-completion, or label-completion form suitable for a base language model.

#### Scenario: Corpus composition validation
- **WHEN** the canonical base workload file is loaded
- **THEN** schema validation fails unless every minimum family count is met and every case has an explicit family, claim type, and expected completion contract

#### Scenario: Diverse admissible examples
- **WHEN** the canonical admissible corpus is inspected
- **THEN** it includes distinct examples for routing labels, inventory decisions, incident severity, feature state, file or build classification, and deterministic code or text completion rather than paraphrases of one repeated template

### Requirement: Position and block-boundary coverage is explicit
The workload planner SHALL be able to construct exact token spans for numerical stress cases and SHALL report the actual deletion length, deletion boundary offsets, old/new position deltas, transformed block boundaries, and normal-prefill tail. Qualification SHALL cover delta zero rows before a deletion, deletion shifts near a 128-token block boundary, a cross-block deletion, and at least one long shift representative of the target workload.

#### Scenario: Exact boundary stress generation
- **WHEN** a stress case requests a target token span such as 1, 127, 128, 129, or a configured long shift
- **THEN** the planner adjusts deterministic filler, verifies the final tokenized span exactly, and records the achieved positions rather than trusting character counts

#### Scenario: Required coverage is missing
- **WHEN** the combined numerical workload does not observe a required delta or boundary class
- **THEN** numerical qualification fails with the missing coverage listed

### Requirement: Baselines qualify before Leyline
An admissible `completion_target` case SHALL require full, honest-edited, and every declared counterfactual arm to produce the declared target token IDs before the Leyline result is evaluated. A structured case SHALL require a successful Chat preflight and the category-appropriate full, honest-edited, and counterfactual oracle results. Failed baselines SHALL make a case diagnostic-only.

#### Scenario: Full and honest differ
- **WHEN** full and honest-edited do not both produce the declared target
- **THEN** the case is excluded from admissible success rates and the Leyline result cannot repair the failed baseline

#### Scenario: Counterfactual changes target
- **WHEN** an allegedly irrelevant removed-text variant changes the full-arm target
- **THEN** the case fails counterfactual admission and reports the failing variant

### Requirement: Diagnostic cases remain separate from admissible claims
Deleted-only evidence cases and negative controls SHALL be reported in separate families and SHALL NOT contribute to admissible success rates. Retention of information that honest-edited prefill loses SHALL be described as retained-context evidence, not successful deletion semantics.

#### Scenario: Deleted-only answer is retained
- **WHEN** full and Leyline produce a target whose sole evidence was deleted while honest-edited cannot
- **THEN** the report records retained-context behavior and does not count the case as admissible success

### Requirement: Chat semantic corpus is independent
The canonical Chat corpus SHALL contain at least six structured cases across routing, inventory, access-policy or incident-response task families, including admissible, counterfactual-admissible, deleted-only diagnostic, and negative-control cases. It SHALL use a pinned instruction-tuned checkpoint, canonical client-side chat-template rendering, and a structured preflight through the same endpoint and decoding path.

#### Scenario: Chat corpus qualification
- **WHEN** the structured preflight and applicable full, honest-edited, and counterfactual baselines pass
- **THEN** the suite may evaluate Leyline against the declared structured oracle

#### Scenario: Base weights with a tokenizer template
- **WHEN** the tokenizer contains a chat template but the served weights fail structured preflight
- **THEN** the suite refuses semantic admission regardless of the template's presence

### Requirement: Token and distribution diagnostics are preserved
Every qualification run SHALL record generated token IDs, actual common-prefix length, first divergence, API score provenance, and top-1/top-2 margin for full, honest-edited, and Leyline. An opt-in internal capture SHALL record the complete raw-logit vector at the first declared completion token with checkpoint, request, rank, and sampling provenance.

#### Scenario: Raw logits are captured
- **WHEN** raw-logit capture is enabled for a qualification case
- **THEN** the report distinguishes raw logits from API log probabilities and reports full-to-honest and full-to-Leyline vector distance, top-k overlap, and selected-token margin

### Requirement: Cache numerical qualification is complete and independent
Overall qualification SHALL require device captures from every expected MLA layer and every TP rank. Source cKV and transformed destination cKV SHALL be bitwise equal, and destination Kpe SHALL pass configured error thresholds against an FP32 delta-rotation reference driven by native runtime RoPE frequencies or tables captured independently of connector-derived frequencies.

#### Scenario: Complete numerical pass
- **WHEN** all expected layer/rank captures are present, connector and native RoPE frequency provenance agree, cKV is bitwise equal, and every Kpe comparison passes
- **THEN** the numerical gate passes and links its manifest and aggregate errors into the correctness report

#### Scenario: Self-consistent but wrong connector frequency
- **WHEN** a connector-derived inverse-frequency vector disagrees with the native runtime RoPE state even if a comparison using that same connector vector would pass
- **THEN** numerical qualification fails and reports the frequency mismatch

### Requirement: Overall conclusions follow evidence gates
The suite SHALL distinguish execution success, numerical transform correctness, base reference behavior, Chat semantic correctness, and full-generation similarity. Overall qualification SHALL require compatible environment identity, successful record and transform execution, complete numerical and rollback gates, and the selected evaluation-mode baseline and Leyline target. Later autoregressive text divergence SHALL NOT by itself fail numerical correctness after those gates pass, and SHALL NOT be described as expected approximation before they pass.

#### Scenario: Numerical pass with later text divergence
- **WHEN** cache numerical evidence and the declared completion or semantic target pass but generated suffixes later diverge
- **THEN** the report may attribute the suffix difference to autoregressive amplification while preserving the successful target-level qualification

#### Scenario: No device capture
- **WHEN** a run has generated outputs but no complete device-cache evidence
- **THEN** it remains diagnostic and cannot conclude either implementation correctness or an inherent Leyline quality limit

### Requirement: Qualification environment is reproducible
The environment manifest SHALL record repository commits and cleanliness, installed distributions, imported module file paths, CANN version evidence, runtime topology, and content hashes for model and tokenizer artifacts. Repository and imported-module provenance SHALL agree before a run is qualified.

#### Scenario: Distribution and checkout identifiers differ
- **WHEN** distribution metadata names a different commit than the source checkout
- **THEN** the manifest resolves and records the actual imported module paths and blocks qualification if their provenance cannot be reconciled

### Requirement: Correctness results are repeatable
Qualification SHALL execute at least three isolated greedy repetitions of each required case with fresh request identifiers and cache salts, preserve every individual result, and report whether target decisions, execution metadata, and numerical summaries are stable across repetitions.

#### Scenario: One repetition changes its target
- **WHEN** otherwise identical isolated repetitions disagree on a required full, honest-edited, or Leyline target decision
- **THEN** the case is marked unstable and cannot contribute to overall qualification
