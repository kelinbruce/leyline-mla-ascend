## Purpose

Defines reliable Leyline record and amortize transaction behavior so reported execution success proves one-shot source use, complete MLA transformation, and leak-free cleanup across all TP ranks.

## ADDED Requirements

### Requirement: Record and amortize have distinct lifecycle effects
Only a successful `record` directive SHALL create or replace a source session. An `amortize` directive SHALL consume the matching source session for its transform and SHALL NOT implicitly record the edited request under the same session identifier.

#### Scenario: Record completes successfully
- **WHEN** a record request finishes normally with eligible prompt blocks
- **THEN** the connector pins those source blocks, creates the named source session, and reports `recorded=true` for that record request

#### Scenario: Amortize completes successfully
- **WHEN** an amortize request consumes a source session and finishes generation
- **THEN** the connector reports transform application without setting `recorded=true` for the amortize request and without creating a replacement session

### Requirement: One-shot session references are released
After an amortize transform succeeds, fails, is cancelled, or falls back after acquiring a source session, all transaction-owned and session-owned block references SHALL be released exactly once and the consumed session SHALL no longer be addressable.

#### Scenario: Successful completion cleanup
- **WHEN** all TP workers complete a successful transform and the request later reaches its finish hook
- **THEN** no session, inflight plan, pending metadata, or extra source-block reference remains for that transaction

#### Scenario: Transform failure cleanup
- **WHEN** any TP worker fails the transform
- **THEN** the connector invalidates affected destination blocks, releases all source references, and schedules honest recomputation without leaving a reusable session

### Requirement: Execution evidence joins separate record and amortize requests
Validation SHALL establish record success from the record response and transform success from the amortize response. Leyline execution SHALL require record success, `applied=true`, positive transformed tokens, no fallback, and complete layer/rank evidence, but SHALL NOT require the amortize response itself to claim it recorded a session.

#### Scenario: Record response is missing
- **WHEN** an amortize response reports `applied=true` but the paired record response is absent or reports `recorded=false`
- **THEN** execution qualification fails with missing record evidence

### Requirement: Every expected MLA layer transforms on every TP rank
The worker SHALL determine the expected MLA cache layer set from the active model/runtime contract and SHALL require every expected layer to be registered and transformed on every TP rank. Transforming only a non-zero subset SHALL be a failure.

#### Scenario: Complete TP4 transform
- **WHEN** a 27-layer TP4 DeepSeek-V2-Lite request is transformed
- **THEN** the aggregated result proves all 27 expected MLA layers transformed successfully on each of the four ranks

#### Scenario: One layer is missing
- **WHEN** any expected layer is absent, incompatible, skipped, or fails on any rank
- **THEN** the transaction fails closed, invalidates transformed destination blocks, and uses honest recomputation

### Requirement: Completeness evidence is externally reportable
The Leyline response and validation report SHALL expose expected and transformed layer counts, expected and successful TP rank counts, and a completeness boolean without exposing cached prompt contents.

#### Scenario: Successful response metadata
- **WHEN** a transform qualifies as applied
- **THEN** its metadata shows matching expected/transformed layer counts, matching expected/successful rank counts, and `transform_complete=true`

### Requirement: Rollback preserves normal-path correctness
A partial or failed transform SHALL never expose mixed transformed and untransformed cache pages to generation. The connector SHALL invalidate all destination blocks touched by the transaction and account their tokens as normal prefill before generation proceeds.

#### Scenario: Failure after some layers transform
- **WHEN** transformation fails after one or more layers have already written destination rows
- **THEN** every destination block in the plan is invalidated and the request regenerates the complete affected token range through honest prefill
