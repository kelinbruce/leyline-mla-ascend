## Purpose

Define the numerical and end-to-end evidence required to qualify Leyline cache transformation with FP16 storage on Ascend 310P.

## ADDED Requirements

### Requirement: Dual numerical reference
The validation SHALL compare transformed Kpe rows against both an independent CPU FP32 delta-rotation reference and Kpe rows produced by honest native recomputation at the destination positions.

#### Scenario: Representative position shifts
- **WHEN** captured rows cover deltas 0, 1, 127, 128, 129, 1024 and near-maximum context positions
- **THEN** the report records errors against both references for every covered layer and rank

### Requirement: Exact cKV preservation
The transformation SHALL copy cKV without arithmetic and validation SHALL require bitwise equality between each source row and its mapped destination row.

#### Scenario: FP16 cKV copy
- **WHEN** a destination row is transformed from a resident source row
- **THEN** its cKV bytes are identical to the source cKV bytes

### Requirement: Calibrated FP16 Kpe envelope
FP16 acceptance thresholds SHALL be derived from baseline-matched device captures and SHALL include maximum absolute, maximum relative, and percentile error metrics. BF16 thresholds SHALL NOT be reused implicitly.

#### Scenario: Missing calibration evidence
- **WHEN** no passing FP16 calibration artifact matches the source, runtime, model, and topology
- **THEN** the runtime remains experimental and is not declared qualified

### Requirement: Layer/rank-complete capture
Qualification SHALL cover every transformed MLA layer on every participating TP rank and SHALL fail when any expected layer/rank is missing.

#### Scenario: Incomplete capture
- **WHEN** a report lacks one or more expected layer/rank results
- **THEN** numerical qualification fails even if the observed rows are within tolerance

### Requirement: End-to-end separation of concerns
Backend correctness SHALL compare identical prompts, Leyline correctness SHALL compare honest-edited and Leyline-edited execution, and semantic admissibility SHALL compare full and honest-edited task outcomes.

#### Scenario: Base checkpoint validation
- **WHEN** DeepSeek-V2-Lite base is tested
- **THEN** the backend and Leyline gates use returned token/reference agreement rather than an instruction-following JSON oracle
