## Purpose

Define reproducible evidence that separates MLA positional transformation correctness from the semantic admissibility and performance value of an AMORTIZE workload.

## ADDED Requirements

### Requirement: Semantic admissibility gate
The validation suite SHALL classify a workload as valid AMORTIZE evidence only when the original full prompt and the honestly re-prefilled edited prompt both produce the declared correct task outcome.

#### Scenario: Surviving evidence is sufficient
- **WHEN** removed verbose or superseded content is followed by surviving structured evidence that independently determines the expected action
- **THEN** both the full and honestly edited baselines produce the same expected action and the workload is eligible for AMORTIZE evaluation

#### Scenario: Removed information is necessary
- **WHEN** honest deletion changes or prevents the correct task outcome
- **THEN** the workload is classified as a negative control or FORGET case and is excluded from valid AMORTIZE success rates

### Requirement: Counterfactual dependency check
The validation suite SHALL test that the declared outcome is invariant to approved counterfactual changes of the removed span while the surviving evidence remains fixed.

#### Scenario: Counterfactual invariance
- **WHEN** the removed span is replaced in the full baseline by semantically irrelevant variants consistent with the surviving evidence
- **THEN** the expected action remains unchanged across those variants

### Requirement: Numerical reference validation
The validation suite SHALL compare the device transformation against an independent FP32 reference over cKV copying, Kpe delta rotation, block mapping, and representative DeepSeek YaRN positions before accepting end-to-end results.

#### Scenario: Reference-equivalent cache
- **WHEN** synthetic or captured cache data is transformed over zero, small, block-boundary, and long-context position deltas
- **THEN** cKV is bitwise identical and Kpe error remains within the numerical envelope of one native BF16 rotation relative to the FP32 reference

### Requirement: End-to-end semantic validation
For every semantically admissible workload, the validation suite SHALL require the Leyline execution to preserve the expected structured action or task result. Logit distance to either full or honest-edited baselines SHALL be reported as analysis rather than used alone as semantic correctness.

#### Scenario: Valid Leyline outcome
- **WHEN** a valid workload is executed using transformed cache blocks
- **THEN** the declared action, tool selection, key arguments, or deterministic task result matches the oracle

### Requirement: Controlled performance comparison
The validation suite SHALL compare cache-off, vanilla APC, honest edited re-prefill, patched-disabled, and Leyline executions under matched prompts, cache residency, model configuration, and concurrency.

#### Scenario: Warm-cache performance run
- **WHEN** the source cache is resident and all correctness gates have passed
- **THEN** the report includes transformed tokens, actual prefill tokens, transformation time, TTFT percentiles, throughput, and NPU memory for concurrency levels 1, 4, 8, and 16

### Requirement: Reproducible environment record
Every hardware result SHALL record exact vLLM and vLLM-Ascend commits, model revision, tokenizer revision, CANN and driver versions, torch and torch-npu versions, NPU topology, runtime flags, and effective cache configuration.

#### Scenario: Result publication
- **WHEN** numerical, semantic, or performance results are emitted
- **THEN** the environment record is stored alongside the result and is sufficient to distinguish the tested software and hardware configuration
