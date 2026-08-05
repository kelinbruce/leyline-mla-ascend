## Purpose

Prevent unsupported DeepSeek MLA Leyline configurations from being launched on 310P while defining the evidence required to enable them safely.

## ADDED Requirements

### Requirement: Pinned 310P source baseline
The implementation SHALL identify vLLM-Ascend `05e095a20` as the requested 310P source baseline and SHALL record its vLLM v0.23.0 container pairing in environment evidence.

#### Scenario: Preflight on the implementation branch
- **WHEN** the 310P capability preflight runs
- **THEN** it reports the checkout HEAD and whether it contains the pinned baseline

### Requirement: Fail-closed platform capability detection
The system SHALL NOT advertise DeepSeek MLA Leyline as launchable while the active 310P runner rejects MLA or KV transfer.

#### Scenario: Upstream guards are present
- **WHEN** the 310P runner contains either unsupported-runtime guard
- **THEN** preflight reports `safe_to_launch_deepseek_mla_leyline=false` with a separate blocker for each missing capability

### Requirement: 310P FP16 runtime qualification
Before runtime enablement, the implementation SHALL support the DeepSeek MLA cache layout and KV Connector lifecycle in the 310P runner and SHALL validate cKV copying, Kpe rotation, transactional rollback, and end-to-end output agreement using FP16 on 310P hardware. Preflight SHALL require a passed hardware-qualification record matching the pinned baseline.

#### Scenario: Source support without hardware evidence
- **WHEN** source guards have been removed but FP16 numerical and end-to-end hardware gates have not passed
- **THEN** the implementation remains experimental and is not declared production-ready

#### Scenario: Qualified runtime
- **WHEN** source integration, numerical comparison, rollback, semantic/reference, and performance gates all pass on the pinned 310P environment
- **THEN** the preflight may report the configuration safe to launch
