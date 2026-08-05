## MODIFIED Requirements

### Requirement: Explicit evaluation contract
The validation suite SHALL declare whether a run evaluates base-model generation agreement or instruction-model semantic correctness and SHALL NOT interpret reference agreement as oracle correctness.

#### Scenario: Base checkpoint validation
- **WHEN** `DeepSeek-V2-Lite` or another base checkpoint is evaluated in `reference_prefix` mode
- **THEN** the full arm supplies the generation-token reference, other arms are compared over the configured prefix length, and the report marks `semantic_oracle_validated` false

#### Scenario: Instruction checkpoint validation
- **WHEN** an instruction-tuned checkpoint is evaluated in `structured_json` mode
- **THEN** every semantic gate compares parsed output with the declared structured oracle and the report marks `semantic_oracle_validated` true

### Requirement: Exact prompt construction
The validation suite SHALL preserve the edit as exactly one token deletion for raw and chat-template prompts and SHALL prevent the inference server from inserting additional special tokens.

#### Scenario: Chat template rendering
- **WHEN** `prompt_format=chat_template` is selected
- **THEN** the generation prompt is rendered by the tokenizer, the removed span remains identifiable, and deleting the declared token interval from the full token sequence yields the edited token sequence exactly

### Requirement: Token-level reference comparison
The validation suite SHALL compare generation token IDs returned by the inference server rather than tokenizing decoded output text again.

#### Scenario: Reference prefix gate
- **WHEN** a reference-prefix result is produced
- **THEN** the report records the prefix length and matches each arm against the full arm's first N returned generation token IDs
