# Leyline validation evidence

Each new on-device run belongs in its own immutable directory:

```text
results/leyline/<run-id>/
├── environment.*.json
├── runner.*.json
├── runtime.*.json
├── correctness-*.json
├── cache-comparison.json
├── logit-comparison*.json
├── rollback.json
├── qualification-final.json
├── cache-captures/
└── raw-logits/
```

Use the UTC `schema_v3_YYYYMMDD_HHMMSS` run ID from
`benchmarks/leyline/VALIDATION_910B.md`. The run directory is evidence, not a
working directory: preserve the copied configurations and reports that produced
the qualification conclusion.

`historical/` contains the pre-qualification schema-v2 captures gathered in
August 2026. Those reports are kept for reproducibility and debugging only;
they do not meet the current schema-v3 qualification contract.
