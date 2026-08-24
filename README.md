# Local NL-to-API JSON pipeline

This repository implements the assignment as a local, constrained parser: sparse
supervised models choose entity and structural candidates, while typed AST models
and the supplied field schema prevent invalid API JSON. It does not call hosted
services and it never evaluates target text as code.

The supplied labels have three `is_not_empty` filters with no `value` member,
despite the API grammar requiring one. During parsing only, those unary filters
are normalized to the documented API representation `value: ""`; any other
missing filter value remains a validation error.

Four supplied labels additionally use `sort` statements. Although omitted from
the written blueprint grammar, the AST supports validated `sort` nodes with a
schema-known field and an `asc` or `desc` direction so those labels remain
auditable and scoreable.

One legacy relation label places its child `statements` inside `parameters`; the
loader explicitly normalizes that placement to the canonical relation-node form.

## Run

```bash
python scripts/prepare_splits.py
python scripts/run_pipeline.py --mode train --run-id baseline-001
python scripts/run_pipeline.py --mode evaluate --run-id baseline-001
python scripts/run_pipeline.py --mode infer --run-id baseline-001 --question "Show suspicious phones"

# Validation-only hierarchical optimization, followed by the selected evaluation.
python scripts/optimize_hierarchy.py --run-id hierarchical-002
python scripts/run_pipeline.py --mode benchmark --run-id hierarchical-002
python scripts/run_pipeline.py --mode evaluate --run-id hierarchical-002
```

The split prephase is deterministic and refuses to replace an existing split;
pass `--force-resplit` only when deliberately recreating it. Training fits every
learned artifact on `train.csv` only. Validation is used for model configuration;
`test.csv` is used only by `evaluate`.

## Design and limitations

The model predicts the root entity and retrieves similar **training-only**
examples. The hierarchical version additionally predicts filter count, relation
presence, Boolean structure, and a root-conditioned field set. Those predictions
rerank only schema-compatible templates. Decoding then substitutes compatible
typed literals and validates the resulting AST. The supplied field descriptions
are static schema knowledge; they are not fitted from held-out queries.

`hierarchical-002` is the current accuracy champion. Its held-out canonical exact
match is 17.86%, root accuracy is 92.86%, schema validity is 100%, and measured
p95 latency is approximately 10 ms. Large Joblib model files are intentionally
ignored by Git and can be regenerated from the tracked training split.

`artifacts/<run-id>/errors.csv` identifies missed templates, fields, relations,
operators, and values for targeted future improvements. Evaluation reports and
predictions are tracked for auditability; generated model binaries are not.
