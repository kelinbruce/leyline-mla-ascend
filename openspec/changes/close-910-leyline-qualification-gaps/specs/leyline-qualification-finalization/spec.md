## Purpose

Defines an immutable and provenance-checked finalization contract that combines staged Ascend 910B validation evidence without rerunning model requests or silently upgrading incomplete evidence.

## ADDED Requirements

### Requirement: Finalization is offline and preserves source evidence
The finalizer SHALL consume existing schema-v2 correctness and auxiliary evidence artifacts, SHALL NOT issue model requests, and SHALL preserve every source request identifier, repetition, cache salt, generated token sequence, arm result, and source artifact reference in the finalized report.

#### Scenario: Finalize a completed correctness run
- **WHEN** a correctness report and its auxiliary evidence are provided to the finalizer
- **THEN** the finalizer produces a derived report without contacting any inference endpoint and records cryptographic hashes of every source artifact

#### Scenario: Source report would be mutated
- **WHEN** finalization would require replacing a request identifier, repetition, generated output, or arm result from the source correctness report
- **THEN** finalization fails instead of rewriting the evidence

### Requirement: Evidence identity must agree
The finalizer SHALL verify corpus identity, evaluation contract, checkpoint and tokenizer hashes, imported module provenance, repository commits, runtime topology, block size, tensor-parallel rank set, and run identity wherever those fields apply. Missing or conflicting identity SHALL produce an explicit unqualified result.

#### Scenario: Cache-off checkpoint differs
- **WHEN** the connector-on and cache-off reports use different model hashes, tokenizer hashes, evaluation contracts, or runtime topology other than the explicitly staged cache configuration
- **THEN** the finalizer rejects the cache-off comparison and reports every conflicting field

#### Scenario: Auxiliary report references unknown requests
- **WHEN** a cache or logit report references a request that is absent from the source correctness report and is not an explicitly linked diagnostic rerun
- **THEN** finalization fails provenance validation

### Requirement: Staged cache-off evidence preserves repetitions
The finalizer SHALL align cache-off results by corpus, expanded case identifier, counterfactual variant, and repetition, and SHALL retain all individual cache-off outputs and target decisions. A representative aggregate arm SHALL NOT replace repetition-level evidence.

#### Scenario: Three stable cache-off repetitions
- **WHEN** every required case has three identity-compatible cache-off repetitions with stable target decisions
- **THEN** the final report records each repetition and marks the cache-off baseline complete

#### Scenario: Missing cache-off repetition
- **WHEN** any required case or variant has fewer than the configured repetitions
- **THEN** the cache-off baseline remains incomplete and cannot contribute to qualification

### Requirement: Qualification derives from validated reports rather than asserted booleans
Numerical and rollback qualification SHALL be derived from validated cache-comparison and rollback-report artifacts. Configuration booleans MAY remain for performance gating compatibility but SHALL NOT establish correctness qualification without linked evidence.

#### Scenario: Manual numerical flag without comparison
- **WHEN** a configuration sets numerical success but no complete cache-comparison artifact is supplied
- **THEN** the finalized result remains numerically unqualified

#### Scenario: Complete cache and rollback evidence
- **WHEN** the cache report proves every expected layer/rank pair, required delta, cKV equality, Kpe tolerance, and native RoPE agreement and the rollback report passes
- **THEN** the finalizer may advance to baseline and Leyline target classification

### Requirement: Final report explains the determining gate
Every case and top-level qualification result SHALL identify the first determining gate among environment, execution, numerical cache, rollback, baseline, Leyline target, and later autoregressive divergence. Missing evidence SHALL be distinguished from measured failure.

#### Scenario: Numerical evidence is absent
- **WHEN** semantic targets pass but no complete cache-comparison report is available
- **THEN** the result is classified as missing numerical evidence rather than implementation success or numerical failure

#### Scenario: Target passes after numerical qualification
- **WHEN** numerical, rollback, baseline, and Leyline target gates pass but later generated tokens diverge
- **THEN** the target is accepted and the suffix difference is reported separately as autoregressive divergence

