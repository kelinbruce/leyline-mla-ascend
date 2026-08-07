## 1. Feasibility Planning

- [x] 1.1 Add immutable feasibility metadata and a dependency-light planner that mirrors complete source-block and reusable target-block rules.
- [x] 1.2 Add tokenizer-aware surviving filler expansion while preserving exact deletion and completion-target tokenization.
- [x] 1.3 Validate canonical and counterfactual plans before network requests and emit stable per-case rejection diagnostics.

## 2. Workload and Configuration

- [x] 2.1 Extend the workload schema with execution expectation, minimum transform tokens, and bounded surviving filler fields.
- [x] 2.2 Rebuild the base workload cases so qualifying cases request at least one transformable block and unstable route cases use a closed completion pattern.
- [x] 2.3 Add smoke-case and fail-fast settings to the runner example configuration while retaining explicit legacy diagnostic opt-out.

## 3. Runtime Gates and Reporting

- [x] 3.1 Implement the designated one-repetition transform smoke gate and stop the full matrix when execution or required evidence is incomplete.
- [x] 3.2 Add feasibility and smoke evidence to the correctness report without changing historical result files.
- [x] 3.3 Report execution stability and exact generated-token stability separately while preserving the legacy `stable` alias.

## 4. Verification and Documentation

- [x] 4.1 Add unit tests for short source, short edited prompt, partially resident mapped source, long deletion, local APC, and positive transform planning.
- [x] 4.2 Add harness tests for filler expansion, required-versus-diagnostic admission, smoke pass/fail, and split stability reporting.
- [x] 4.3 Update the Leyline validation README with fail-fast behavior, workload authoring rules, and the 910B smoke-first command sequence.
- [x] 4.4 Run focused Leyline unit tests, JSON/schema checks, formatting checks, and strict OpenSpec validation.
