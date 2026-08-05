## Purpose

Integrate the local Leyline cache transformation with the inherited vLLM KV Connector transaction on Ascend 310P without enabling unsupported connectors or layouts.

## ADDED Requirements

### Requirement: Connector allowlist
The 310P runner SHALL admit KV transfer only for the local Leyline connector under the qualified FP16 MLA runtime matrix and SHALL reject every other connector by default.

#### Scenario: Qualified local Leyline connector
- **WHEN** the configured connector is the local Leyline connector and all MLA/runtime gates pass
- **THEN** cache initialization, registration, and connector execution may proceed

#### Scenario: Other connector or unsupported mode
- **WHEN** another connector, graph mode, quantization, speculative decoding, sparse attention, DCP, or PCP is configured
- **THEN** initialization fails closed with an explicit reason

### Requirement: Connector-visible MLA cache
The worker SHALL register every bound MLA layer as compatible cKV and Kpe tensors whose slot mapping matches the scheduler transformation plan.

#### Scenario: Worker cache registration
- **WHEN** KV cache initialization completes for a qualified request
- **THEN** the Leyline worker receives the same cKV/Kpe tensors consumed by normal 310P MLA attention

### Requirement: TP-wide transactional completion
The system SHALL publish transformed destination blocks only after every participating rank completes every planned layer successfully.

#### Scenario: All ranks succeed
- **WHEN** all ranks transform all layers and synchronize successfully
- **THEN** the connector reports completion and the scheduler may publish the delayed destination hashes

#### Scenario: A rank or layer fails
- **WHEN** any rank or layer reports an error
- **THEN** all affected destination blocks are invalidated, source references are released, and the request resumes through honest prefill
