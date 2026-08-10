## Purpose

Defines a guarded Ascend 910B fault-injection and rollback evidence contract proving that partial Leyline cache writes never reach generation and leave no transaction-owned state behind.

## ADDED Requirements

### Requirement: Device failure injection requires two explicit opt-ins
Runtime fault injection SHALL remain disabled by default and SHALL require both a server-start validation opt-in and a request-scoped injection directive. A client request alone SHALL NOT enable failure injection on a normal server.

#### Scenario: Request asks for injection on a normal server
- **WHEN** a request contains an injection directive but the server validation opt-in is disabled
- **THEN** the directive is rejected without altering cache transformation or normal serving behavior

#### Scenario: Validation server accepts a bounded failpoint
- **WHEN** the server validation opt-in is enabled and an authorized validation request names a supported layer/rank failpoint
- **THEN** the failpoint applies only to that transaction and is recorded in its diagnostic provenance

### Requirement: Injection exercises a partial TP-wide transform
The validation workflow SHALL support deterministic failure after at least one destination cache write on a selected layer or rank, so aggregation observes a partial transform rather than an early request-validation failure.

#### Scenario: One TP rank fails after a layer write
- **WHEN** the configured rank reaches the selected post-write failpoint
- **THEN** TP-wide aggregation reports transform failure, identifies the failed rank/layer, and does not report `applied=true`

### Requirement: Failed transformation falls back without mixed cache exposure
After injected partial failure, every destination block touched by the transaction SHALL be invalidated before generation, every affected token SHALL return to honest normal-prefill accounting, and generation SHALL not consume a mixture of transformed and untransformed destination cache rows.

#### Scenario: Partial write is rolled back
- **WHEN** any rank reports an injected transform failure after destination writes
- **THEN** the response reports failure metadata, touched-block invalidation, honest recomputation accounting, and a non-applied Leyline outcome

### Requirement: Rollback releases transaction state
Successful fallback after injected failure SHALL release the consumed source session, inflight plan, pending metadata, and transaction-owned source block references exactly once. The rollback report SHALL expose non-sensitive cleanup counters sufficient to verify zero remaining transaction state.

#### Scenario: Failed request completes through fallback
- **WHEN** generation finishes after an injected Leyline transform failure
- **THEN** the rollback evidence shows no remaining source session, inflight plan, pending metadata, or extra source-block reference for that transaction

### Requirement: Rollback qualification uses a signed evidence report
The rollback runner SHALL produce a schema-versioned report that records environment identity, run identity, injection point, record/amortize request IDs, transform and fallback metadata, target behavior, cleanup evidence, and source artifact hashes. Final qualification SHALL derive rollback success from this report rather than a configuration assertion.

#### Scenario: Rollback report is complete
- **WHEN** injection occurs after a partial write, fallback recomputes honestly, cleanup is complete, and the target agrees with the honest control
- **THEN** the rollback report passes and may satisfy the final rollback gate

#### Scenario: Boolean flag has no evidence
- **WHEN** configuration claims rollback success without a valid rollback report
- **THEN** final qualification remains rollback-unqualified
