# Local entity extraction pipeline

This repository implements the assignment target locally: given a natural
language question, predict the order-independent set containing its root
`entityType` and every nested `relationTargetType`. It does not reconstruct an
API request, fields, values, operators, or AST at inference time.

## Run

Use the local `cognyte` Conda environment:

```bash
# The fixed, template-grouped splits are already present. Recreate only if needed.
conda run -n cognyte python scripts/prepare_splits.py

# Validation-only selection; test.csv is never read here.
conda run -n cognyte python scripts/optimize_entity_pipeline.py --run-id entity-final-001

# Refit the selected configuration on all non-test data.
conda run -n cognyte python scripts/run_entity_pipeline.py --mode train --run-id entity-final-001 \
  --family direct --threshold 0.6 --max-features 20000 --c 1 \
  --semantic-keyword-boost 0.3 --semantic-relation-boost 0.3 --include-validation

# Benchmark and evaluate the final model on the held-out test split.
conda run -n cognyte python scripts/run_entity_pipeline.py --mode benchmark --run-id entity-final-001
conda run -n cognyte python scripts/run_entity_pipeline.py --mode evaluate --run-id entity-final-001

# Minimal inference output.
conda run -n cognyte python scripts/run_entity_pipeline.py --mode infer --run-id entity-final-001 \
  --question "What SMS messages were sent from suspicious phones?"
```

The final command emits JSON such as `["CDR", "Phone"]`. Add `--diagnostics`
only to inspect probability diagnostics.

## Design and results

The selected model is sparse word/character TF-IDF with independent balanced
logistic entity heads. It runs fully locally on CPU and has no Torch,
transformers, network, or GPU dependency. The model, selection ablations,
metrics, predictions, errors, latency measurements, and the serialized model
are written under `artifacts/entity-final-001/`.

The final run scored 106/112 (94.64%) exact entity-set matches on the test
diagnostic. Its warmed local-CPU latency is **4.69 ms p50, 6.25 ms p95, and
6.79 ms p99** per query; cold model load is 74.64 ms. See the persisted
[`latency.json`](artifacts/entity-final-001/latency.json) and the complete
evaluation in the report. The test score is post-test-tuning because the
content-author rule was refined after reviewing a test error.

See [the entity extraction report](doc/entity_extraction_report.md) for the
accuracy, latency, optimization evidence, limitations, and representative test
examples.

See [the entity keyword guide](doc/entity_keyword_guide.md) for the static
semantic expansion lexicon and relation rules.
