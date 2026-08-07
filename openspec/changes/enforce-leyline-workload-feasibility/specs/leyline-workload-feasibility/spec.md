## Purpose

Defines how Leyline validation proves that a workload can execute a real block-aligned cache transform before its semantic or numerical results are admitted as qualification evidence.

## ADDED Requirements

### Requirement: Exact connector-feasibility planning
Before sending qualification requests, the validator SHALL use the tokenized full and edited prompts, declared deletion, connector block size, locally computed prefix length, and complete resident source blocks to calculate the longest reusable target boundary using the same block-alignment and deletion mapping contract as the connector.

The feasibility result SHALL report full and edited token counts, complete source block count, maximum target boundary, reusable target range, predicted transformed token count, predicted post-deletion shifted token count, and a stable reason when no meaningful transform is possible. A qualification case SHALL include at least one transformed target position at or after the deletion start so that a delta-zero-only prefix cannot satisfy execution admission.

#### Scenario: Short source has no recordable block
- **WHEN** a full prompt contains fewer tokens than one connector block
- **THEN** the planner reports zero complete source blocks and reason `missing_source_blocks`

#### Scenario: Short edited prompt has no target block
- **WHEN** the source has a complete block but the edited prompt cannot expose one complete target block before generation
- **THEN** the planner reports zero predicted transformed tokens and reason `no_reusable_target_blocks`

#### Scenario: Mapped source block is not resident
- **WHEN** a complete target block maps through the deletion to a source block that is only partially populated
- **THEN** the planner excludes that target block and reports which required source block prevents reuse

#### Scenario: Transformable plan
- **WHEN** at least one complete target block maps only to complete resident source blocks beyond the locally computed prefix
- **THEN** the planner reports a positive block-aligned predicted transformed token count

#### Scenario: Only pre-deletion block is reusable
- **WHEN** the reusable target range ends at or before the deletion start
- **THEN** the planner reports zero shifted transformed tokens and the qualification case remains infeasible

### Requirement: Execution-aware workload admission
Each workload case SHALL declare whether transform execution is required or whether non-transformability is intentional diagnostic coverage. Qualification cases SHALL be rejected before requests are sent unless their predicted transformed token count meets the configured minimum.

#### Scenario: Required case is infeasible
- **WHEN** a qualification case marked as requiring execution predicts fewer transformed tokens than its minimum
- **THEN** validation stops with the case identifier, feasibility metrics, and stable rejection reason

#### Scenario: Intentional non-transformable diagnostic
- **WHEN** a diagnostic case explicitly allows zero transformed tokens
- **THEN** the validator retains the case as diagnostic evidence and does not count it as Leyline execution qualification

#### Scenario: Every scored Leyline case is feasible
- **WHEN** corpus admission completes successfully
- **THEN** every case eligible for Leyline semantic scoring has a positive predicted transform range

### Requirement: Tokenizer-aware surviving context
The workload planner SHALL support deterministic surviving filler that is present in both full and edited prompts and SHALL expand it using the active tokenizer until the requested transformable range is feasible without changing the exact deleted-token requirement or completion target boundary.

#### Scenario: Position-stress deletion remains exact
- **WHEN** surviving filler is expanded for a case requesting an exact deletion length
- **THEN** the resulting full and edited prompts still differ by exactly the requested contiguous token span

#### Scenario: Source coverage survives a long deletion
- **WHEN** a deletion shifts target positions into later source blocks
- **THEN** the planner adds enough surviving suffix for every source block needed by the minimum target range to be complete and resident

#### Scenario: Expansion cannot satisfy constraints
- **WHEN** bounded deterministic filler search cannot make a case feasible while preserving its token invariants
- **THEN** planning fails explicitly and does not silently accept an approximate workload

### Requirement: Transform smoke gate
The full qualification matrix SHALL run only after a designated smoke case demonstrates a recorded source, applied transform, positive block-aligned transformed token count, no fallback, complete expected layer and TP-rank coverage, and all evidence artifacts required by the active run mode.

#### Scenario: Smoke transform succeeds
- **WHEN** the designated smoke case satisfies every execution and evidence condition
- **THEN** the runner records the smoke result and proceeds to the full matrix

#### Scenario: Smoke transform falls back
- **WHEN** record or amortize falls back, reports zero transformed tokens, or reports incomplete layer or rank coverage
- **THEN** the runner stops before the full matrix and reports the failed smoke conditions

#### Scenario: Required capture is absent
- **WHEN** device capture is required but the smoke transform produces no joinable cache evidence
- **THEN** the smoke gate fails even if the response reports `applied=true`

### Requirement: Distinct execution and generation stability
Reports SHALL distinguish stability of connector decisions and transform metadata from exact generated-token stability across repetitions.

#### Scenario: Decisions stable but tokens vary
- **WHEN** repetitions have identical admission, fallback, application, and transform counts but different output token sequences
- **THEN** the report marks execution stability true and generation stability false

#### Scenario: Exact repeated result
- **WHEN** both connector evidence and output token sequences are identical across repetitions
- **THEN** the report marks both execution and generation stability true

### Requirement: Baseline requalification remains independent
Connector feasibility SHALL NOT make a case semantically admissible. Full, honest-edited, and applicable counterfactual targets SHALL still pass their baseline gates before a feasible Leyline result is interpreted.

#### Scenario: Feasible case has unstable baseline
- **WHEN** a case can execute a transform but its full or honest-edited arm misses the declared completion target
- **THEN** the report classifies it as an invalid baseline rather than a Leyline semantic failure
