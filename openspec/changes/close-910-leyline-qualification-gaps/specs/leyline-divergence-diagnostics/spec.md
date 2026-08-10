## Purpose

Defines bounded full-vocabulary decode-step diagnostics and empirically discriminating retained-context workloads for explaining where and why Leyline generation diverges from its baselines.

## ADDED Requirements

### Requirement: Raw-logit capture supports bounded decode steps
Opt-in raw-logit capture SHALL support the first token and explicitly configured decode steps for selected validation requests. Each capture SHALL include run, case, arm, repetition, request, rank, decode-step, checkpoint, dtype, shape, and sampling provenance.

#### Scenario: Capture the logit that selects a divergent token
- **WHEN** decode step `d` is enabled for a selected request
- **THEN** the system stores the full-vocabulary vector used to select output token `d` and labels it as decode step `d`

#### Scenario: Request is outside the diagnostic selection
- **WHEN** raw-logit capture is enabled but a request does not match the configured validation run and case selection
- **THEN** no raw-logit artifact is written for that request

### Requirement: Diagnostic capture is storage bounded
Decode-step capture SHALL require an explicit maximum step count and request selection, SHALL reject unbounded capture configuration, and SHALL report estimated and actual artifact counts and bytes.

#### Scenario: Unbounded all-token capture is requested
- **WHEN** capture configuration omits a finite step limit or exceeds the configured safety maximum
- **THEN** the server refuses to enable the diagnostic capture

#### Scenario: Capture budget is reached
- **WHEN** the configured artifact-count or byte budget is exhausted
- **THEN** further captures stop and the resulting report is marked incomplete with the budget reason

### Requirement: First-divergence comparison uses aligned provenance
The comparator SHALL compute the first generated-token divergence from the immutable correctness evidence and compare the full-vocabulary vectors that selected that token only when case, repetition, arm, rank, run, and decode-step provenance align. A targeted rerun SHALL be correlated only when its generated prefix and divergence position reproduce the source behavior.

#### Scenario: Full and Leyline diverge at token four
- **WHEN** tokens zero through three agree, token four differs, and aligned step-four captures exist
- **THEN** the report compares full-to-Leyline step-four logits and retains first-token comparisons as separate evidence

#### Scenario: Targeted rerun diverges at another position
- **WHEN** a diagnostic rerun does not reproduce the source prefix and divergence index
- **THEN** its step logits are labeled non-correlatable and do not explain the source divergence

### Requirement: Retained-context diagnostics are empirically informative
A retained-context diagnostic SHALL count as informative only when, across all required repetitions, full produces the declared target, honest-edited does not, full and honest-edited select different first tokens, and both baseline decisions are stable. Informative diagnostics SHALL remain separate from admissible deletion success.

#### Scenario: Deleted codeword distinguishes baselines
- **WHEN** full stably selects the deleted codeword target and honest-edited stably selects another first token
- **THEN** the case is classified as an informative retained-context diagnostic

#### Scenario: Full and honest both produce the target
- **WHEN** a declared retained-context case produces the same target in full and honest-edited
- **THEN** it is classified as diagnostic-uninformative and cannot satisfy retained-context coverage

### Requirement: Retained-context coverage spans multiple families
The canonical diagnostic corpus SHALL contain at least four empirically informative cases across at least three distinct completion families, with deleted evidence preceding a transformable surviving cache region and the query evaluated through the normal-prefill tail.

#### Scenario: Only one diagnostic is informative
- **WHEN** corpus preflight finds fewer than four informative diagnostics or fewer than three represented families
- **THEN** retained-context coverage is incomplete even if the admissible target suite passes

### Requirement: Diagnostic conclusions compare Leyline with both baselines
For every informative diagnostic, the report SHALL state Leyline target behavior, first-token agreement, full-generation common prefix, and raw-logit distance relative to both full and honest-edited. The report SHALL NOT call retained-context behavior proven solely because Leyline differs from honest-edited.

#### Scenario: Leyline follows honest-edited
- **WHEN** Leyline selects the honest-edited first token and its raw-logit distribution is closer to honest-edited than full
- **THEN** the report records lack of retained-context evidence for that case without treating it as an admissible-task failure

#### Scenario: Leyline follows full
- **WHEN** Leyline selects the full target and its raw-logit distribution is closer to full than honest-edited
- **THEN** the report records positive retained-context evidence while keeping it outside admissible deletion success rates

