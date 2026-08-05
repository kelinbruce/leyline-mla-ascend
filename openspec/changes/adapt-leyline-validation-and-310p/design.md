## Context

The implementation branch is based on vLLM-Ascend `05e095a202bdcfef4da61168eae34bfd3b99da13`. Its 310P Dockerfiles pair the checkout with vLLM v0.23.0 and CANN `9.1.0-beta.1-310p`. In `vllm_ascend/_310p/model_runner_310p.py`, KV-cache allocation raises for any `kv_transfer_config` and for `model_config.use_mla`. The same limitations are still present on the inspected upstream main.

## Decisions

### 1. Make the evaluation claim explicit

`reference_prefix` takes the full arm's returned generation token IDs as the reference. It gates honest-edited/counterfactual admissibility and Leyline agreement over a configured prefix length. It does not consume the declared JSON oracle and cannot prove task-semantic correctness.

`structured_json` retains the declared-oracle gate. It is intended for an instruction-tuned checkpoint and may use `prompt_format=chat_template`.

### 2. Keep token deletion exact under both prompt formats

Raw prompts continue to tokenize prefix, removed span, and suffix independently. Chat-template prompts insert two reserved markers into one rendered user message, split the rendered text at those markers, remove them, then tokenize the three parts independently. The request sets `add_special_tokens=false` because the harness has already handled BOS/template formatting.

### 3. Compare actual generation token IDs

The completions request sets `return_token_ids=true`. Reference comparison uses those IDs directly rather than re-tokenizing decoded text, which may not be reversible for every tokenizer token.

### 4. Fail closed on 310P

The preflight reports the pinned baseline relationship, inspects the active 310P runner guards, evaluates the intended FP16 runtime configuration, and requires a baseline-matched hardware qualification record. It never edits or bypasses guards. `safe_to_launch_deepseek_mla_leyline` becomes true only after the source blockers, runtime prerequisites, and device gates are genuinely cleared.

### 5. Stage true 310P support

Enabling DeepSeek MLA Leyline on 310P requires, in order: a supported MLA cache/attention layout, KV Connector lifecycle integration in `NPUModelRunner310`, FP16 transformation tolerances established on hardware, rollback/E2E correctness, then performance tests. The existing 910B BF16 transform is a reference implementation, not proof that the 310P execution stack supports the same path.

## Risks / Trade-offs

- First-token agreement can hide later divergence. Reports record the configured prefix length, and stricter runs can increase it.
- Chat templates can transform user content. Marker preservation is checked and the harness fails if the rendered markers cannot be found.
- A source-text preflight detects known upstream guards but cannot replace hardware tests. Its output therefore lists the remaining hardware gates even after source support is added.
