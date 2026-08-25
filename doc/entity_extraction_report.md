# Entity extraction report

## Scope

The pipeline predicts the unordered entity set formed by a query's root
`entityType` plus every recursive `relationTargetType`. It deliberately does
not reconstruct API JSON, fields, values, operators, or relation structure.

`What SMS messages were sent from suspicious phones?` → `['CDR', 'Phone']`

## ML pipeline

The fixed template-grouped splits contain 520 training rows, 112 validation
rows, and 112 test rows. Labels are parsed and schema-checked before training.
The final model is fitted on the 632 non-test rows; no rows are duplicated.

The learned model is a CPU-only multilabel classifier:

- Word TF-IDF (1–2 grams) plus character-within-word TF-IDF (3–5 grams).
- One balanced logistic-regression head per entity class.
- A 0.60 threshold with a highest-probability fallback, guaranteeing a
  non-empty answer.
- A versioned, manually curated lexicon in
  `data/entity_semantic_keywords.json`; it is static during every pipeline
  run and is not learned from train, validation, or test questions.

This requires only NumPy, scikit-learn, and Joblib. It uses one CPU thread and
does not require a GPU, Torch, transformers, a network connection, an LLM, or
an API key.

## Keyword and relation logic

For logistic probability `p`, keyword-hit count `h`, and keyword boost
`b = 0.30`, the hybrid computes:

```text
keyword_signal = min(1, h / 3)
p_keyword = p + b × keyword_signal × (1 - p)
```

The final model also has a narrowly scoped static relation rule: a Web Activity
phrase together with an author phrase (`made by`, `posted by`, `written by`,
`shared by`, or `authored by`) evidences both `Web Activity` and `Web Actor`.
For each of those two labels it applies one additional soft boost,
`p_relation = p_keyword + 0.30 × (1 - p_keyword)`. It is not a hard override;
the usual 0.60 threshold is still applied.

Phrase matching is case-insensitive and uses real word boundaries. In
particular, `male` cannot match inside `female`. Regression tests cover that
boundary and the content-author relation rule.

For the reviewed query, the correction changes the result from the erroneous
`['Person']` to:

```text
Find all comments made by female born after 1990.
→ ['Web Activity', 'Web Actor']
```

The old failure combined an escaped-boundary bug (which counted `male` in
`female`) with insufficient Web Activity probability. After the boundary fix,
the relation rule gives the content and author evidence enough weight without
turning demographic words into a Person prediction.

## Optimization and impact

All listed candidates were trained on the training split only. Each was scored
on validation and with three-fold template-grouped out-of-fold predictions in
training; `test.csv` was not read during that selection.

| Candidate | Validation exact | Grouped OOF exact | Mean batch latency | Model size |
|---|---:|---:|---:|---:|
| Direct logistic, threshold 0.6 | 98.21% | 95.77% | 0.154 ms | 866,281 B |
| Semantic boost 0.3 | 98.21% | **96.15%** | 0.663 ms | 866,281 B |
| Relation boost 0.2 + semantic 0.3 | 98.21% | **96.15%** | 0.669 ms | 866,281 B |
| Relation boost 0.3 + semantic 0.3 | 98.21% | 95.96% | 0.661 ms | 866,281 B |
| Character 3–6, C=2 | 98.21% | 95.38% | 0.170 ms | 1,080,457 B |
| Strict keyword filter + boost | 82.14% | 75.19% | 0.675 ms | 866,281 B |

The sparse features preserve low latency and keep the artifact below 1 MB;
expanding character n-grams added about 214 KB without improving grouped OOF
accuracy. Soft semantic evidence improved OOF exact match by 0.38 percentage
points over direct logistic regression. Strict filtering was rejected because
a finite lexicon can miss valid paraphrases and suppress model-supported
labels.

Validation/OOF selection would choose semantic boost 0.3 without a relation
boost (it tied the best scores and was fractionally faster). The final
relation-0.3 deployment was chosen after the user-directed review of a known
test error: it resolves that error and added one exact test match. Therefore
the final test number below is a **post-test-tuning diagnostic**, not a blind
generalization estimate. The complete candidate table is in
`artifacts/entity-final-001/ablation_summary.csv`.

## Evaluation

Validation numbers are from train-only candidate evaluation. Test numbers are
from the final model fitted on train+validation.

| Metric | Validation (112) | Final test diagnostic (112) |
|---|---:|---:|
| Exact entity-set match | 110/112 — 98.21% | 106/112 — 94.64% |
| Micro precision / recall / F1 | not selected per-label | 97.81 / 97.81 / 97.81% |
| Macro precision / recall / F1 | not selected per-label | 97.12 / 98.81 / 97.77% |
| Weighted F1 | not selected per-label | 97.85% |
| Sample Jaccard | not selected per-label | 97.32% |
| Hamming loss | not selected per-label | 0.00595 |
| Root inclusion accuracy | not selected per-label | 99.11% |
| Relation-target exact accuracy | not selected per-label | 92.00% |
| Relation-target micro F1 | not selected per-label | 95.83% |
| Cardinality accuracy | not selected per-label | 94.64% |

The test set has 25 multi-entity queries. EVisa Request had one test example,
which was predicted correctly, but this is far too little support to claim a
reliable class-level estimate. CDR recall is 89.29% (three false negatives),
and Person precision is 77.78% (two false positives); these are the most
important areas for more diverse labeled examples.

## Efficiency

The warmed 100-query CPU benchmark measured p50/p95/p99 latency of
**4.69/6.25/6.79 ms per query**. Cold model load was **74.64 ms** and the
first prediction was **4.91 ms**. The final serialized artifact is **935,798
bytes**; peak process RSS was **126.5 MB**, including the Python and
scikit-learn runtime. The persisted measurements are in
`artifacts/entity-final-001/latency.json`.

## Examples and next improvements

Examples that work well include:

- `Find phones with an MSISDN starting with '+1'` → `['Phone']`.
- `Find closed investigations related to arms dealing` → `['Investigation']`.
- `Find all comments made by female born after 1990` →
  `['Web Activity', 'Web Actor']`.

Remaining test errors are mainly CDR-plus-Phone phrasing that predicts only
Phone, and profile queries with person-name language that add Person beside
Web Actor. The next evaluation should use a newly held-out split, add more
EVisa examples, and annotate more content/author and communication/device
relations before further rule tuning.

Artifacts for this run are in `artifacts/entity-final-001/`.
