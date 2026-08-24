# Entity Extraction Pipeline — Codex Agent Handoff

## Mission

Build, optimize, validate, select, and evaluate a local low-latency ML pipeline
that maps a natural-language query to the complete set of relevant entity names.

The required output is not an API AST and is not reconstructed JSON. It is an
order-independent collection containing:

1. the top-level JSON 'entityType'; and
2. every 'relationTargetType' found in relation statements, including nested
   relation statements if they occur.

Example:

~~~text
Input:
What SMS messages were sent from suspicious phones to 0549876543 containing
the word 'urgent'?

Output:
["CDR", "Phone"]
~~~

This interpretation is authoritative because doc/assignment.md says that the
entities occur in 'entityType' and, when present, 'relationTargetType', and its
only expected-output example is ['CDR', 'Phone'].

## Assignment compliance

The completed solution must directly demonstrate all of the following:

- extract relevant entities from questions using the supplied labeled data;
- run entirely locally on CPU or an optional local GPU;
- use optimization techniques appropriate for limited resources;
- deliver low single-query latency for real-time assistance;
- measure and document the effect of every accepted and rejected optimization
  on accuracy, latency, serialized model size, and preferably memory;
- evaluate entity extraction accuracy and efficiency;
- include representative correct and incorrect test examples;
- provide a Python implementation, usage instructions, approach explanation,
  limitations, and concrete future improvements.

Do not treat canonical JSON exact match, fields, values, operators, Boolean
groups, or relation types as prediction targets. They are outside the requested
output contract.

## Repository state and preservation rules

The repository is on branch 'main'. The worktree contains existing modified and
untracked work from earlier full-AST experiments. Those changes belong to the
user. Do not reset, discard, or overwrite them. Build the entity pipeline in new
files where practical and make only narrowly scoped changes to shared files.

The previous AST work is a negative architecture branch, not the new baseline.
'hierarchical-002' remains the old AST champion, but its 17.86% canonical JSON
score is irrelevant to the assignment's primary target. Its root classifier can
serve as a reference, but the entity pipeline should be trained and serialized
independently.

## Reuse from the current pipeline

Keep these components:

- src/nl_api/data.py:
  - safe parsing through ast.literal_eval and the typed AST;
  - question normalization;
  - recursive entity extraction through ApiRequest.entity_labels();
  - entity_labels, root_entity, and relation_targets columns;
  - deterministic row identifiers and template keys.
- src/nl_api/ast.py:
  - normalizes the supplied JSON variants;
  - ApiRequest.entity_labels() already returns the deduplicated root and all
    nested relation targets.
- src/nl_api/schema.py validates labels against supplied entities.
- scripts/prepare_splits.py provides a fixed seed and template-grouped splits.
- data/splits/split_manifest.json provides the split audit trail.
- tests/test_no_leakage.py checks split integrity.
- the artifact convention under artifacts/<run-id>/.
- local Conda environment 'cognyte' and the scikit-learn/joblib stack.
- warmed per-query latency measurement with I/O excluded.

Do not route the new inference path through CompositionalParser, field linkers,
operator models, value extraction, AST assembly, semantic retrieval, or
schema-constrained JSON decoding. They add latency and solve the wrong target.

## Current data profile

The existing template-grouped splits contain 744 rows:

| Split | Rows | One entity | Two entities |
|---|---:|---:|---:|
| Train | 520 | 443 | 77 |
| Validation | 112 | 102 | 10 |
| Test | 112 | 87 | 25 |

There are nine observed labels:

~~~text
CDR
EVisa Request
Insight
Investigation
Person
Phone
Report
Web Activity
Web Actor
~~~

All current rows contain one or two unique entity labels, and there are 11
observed label combinations. Treat those as dataset observations, not permanent
business rules. The implementation must continue to support an arbitrary
number of relation targets.

The multilabel rate differs across splits: 14.8% in train, 8.9% in validation,
and 22.3% in test. Report this distribution shift. Do not change thresholds or
architecture after inspecting test performance.

## Leakage policy

1. Use train.csv to fit every vectorizer, classifier, label encoder,
   calibration model, prior, and learned threshold.
2. Use template-grouped train out-of-fold predictions and validation.csv for
   hyperparameter and final threshold selection.
3. Do not use test.csv labels for feature design, parameter selection, routing,
   threshold adjustment, or error-driven implementation changes.
4. Freeze one candidate before running test evaluation.
5. Record training row IDs and a training-data fingerprint in every artifact.
6. Add tests proving that split row IDs/template keys are disjoint and that
   serialized models contain only train row identifiers.

Earlier AST sessions evaluated the current test split. The new entity pipeline
must nevertheless treat it as evaluation-only. Disclose this history as a
limitation. If a genuinely blind score is required, obtain a new external
holdout; do not silently repartition after seeing results.

## Exact target construction

For each parsed request:

~~~python
gold_entities = sorted(request.entity_labels())
~~~

Sorting is only for deterministic serialization. Scoring must use sets or a
fixed multilabel indicator vector, so these are equivalent:

~~~text
["CDR", "Phone"]
["Phone", "CDR"]
~~~

Do not derive labels from field-name prefixes. The assignment explicitly names
entityType and relationTargetType as the source of truth.

## New implementation surface

Prefer isolated, explainable files:

~~~text
src/nl_api/entity_pipeline.py
scripts/run_entity_pipeline.py
scripts/optimize_entity_pipeline.py
tests/test_entity_targets.py
tests/test_entity_pipeline.py
doc/entity_extraction_report.md
artifacts/entity-*/
~~~

entity_pipeline.py should expose a small serializable predictor with:

- train(rows, config);
- predict(question) returning an EntityPrediction;
- predict_batch(questions);
- dump(path) and load(path);
- deterministic label ordering;
- probabilities/confidences for diagnostics;
- no dependency on gold JSON at inference.

run_entity_pipeline.py should support at least:

~~~text
--mode train
--mode validate
--mode evaluate
--mode benchmark
--mode infer
~~~

Inference output should be minimal by default:

~~~json
["CDR", "Phone"]
~~~

Diagnostics may be available behind a flag but must not alter default output.

## Architecture experiments

Run inexpensive baselines first and change one variable family at a time.

### Baseline A — root-only reference

Use word-and-character TF-IDF plus balanced logistic regression to predict
entityType, then emit a singleton set. This cannot emit relation targets, but it
establishes root accuracy, singleton exact-set accuracy, latency, and size.

### Baseline B — direct multilabel sparse classifier

Train MultiLabelBinarizer over entity_labels and an independent one-vs-rest
logistic classifier using a word/character TF-IDF FeatureUnion.

Search a bounded grid:

- word n-grams (1,1) and (1,2);
- character char_wb n-grams (3,5) and (3,6);
- min_df 1 and 2;
- feature caps 10k, 20k, and 40k;
- logistic C in 0.25, 0.5, 1, 2, and 4;
- balanced versus unweighted heads;
- a global multilabel threshold selected without test data.

Handle constant labels inside an inner fold explicitly. Do not suppress the
training error.

### Candidate C — hierarchical root plus relation target

Compare the direct model with a decomposition aligned to the label source:

1. multiclass root model predicts entityType;
2. binary model predicts whether a relation target exists;
3. relation-target model predicts relationTargetType, conditioned on question
   features and optionally predicted root probabilities;
4. output the union of the predicted root and predicted targets.

The target head must receive the predicted root signal during normal inference.
Gold-root performance is allowed only as a clearly named oracle diagnostic.

Compare an explicit no-target class against a separate relation-presence gate.
Use shared learned fallback for rare combinations; never add root-specific
phrase rules or question-to-answer mappings.

### Candidate D — label-powerset reference

The training data has 11 observed entity combinations. Evaluate a multiclass
label-powerset TF-IDF model. It may capture root/target dependence cheaply but
cannot naturally emit unseen combinations. Retain it only if grouped validation
shows a real advantage and document the limitation.

### Optional semantic model

Only evaluate a compact local sentence encoder if sparse models leave a clear
accuracy gap. Include encoder load time, model size, RAM, and CPU p95. Do not
assume the installed Torch/transformers stack belongs in production. Prefer a
sparse model when accuracy is comparable.

## Threshold and calibration rules

Multilabel exact match is sensitive to thresholds:

1. generate template-grouped out-of-fold train probabilities;
2. search a small global threshold grid;
3. use per-label thresholds only with sufficient support and shrink them toward
   the global threshold;
4. confirm on validation;
5. freeze before test.

Always return a non-empty set. If nothing crosses the threshold, select the
highest-probability label and report the fallback rate. Never force a second
entity without model support.

Calibration may be compared when grouped OOF support is sufficient. Accept it
only if accuracy/confidence improves within the resource budget.

## Optimization protocol

Every ablation row must record:

- run ID and parent;
- model family and configuration;
- threshold policy;
- validation exact entity-set match;
- train grouped-OOF exact entity-set match;
- micro/macro F1 and relation-target recall;
- p50/p95 latency;
- complete serialized inference size;
- peak RSS when available;
- accepted/rejected status and reason.

Selection order:

1. validation exact entity-set match;
2. grouped-OOF exact entity-set match;
3. macro F1;
4. relation-target recall;
5. p95 latency;
6. serialized model size.

Set a CPU latency goal before test evaluation. A reasonable starting goal for
sparse inference is warmed p95 below 10 ms per query on this machine. Report all
regressions; do not bypass or hide them.

## Required accuracy metrics

Primary metric: exact entity-set match (multilabel subset accuracy).

Also report:

- exact count and rate;
- micro precision, recall, and F1;
- macro precision, recall, and F1;
- weighted F1;
- Hamming loss;
- sample-averaged Jaccard;
- per-entity precision, recall, F1, support, FP, and FN;
- root entityType accuracy as a secondary diagnostic;
- relation-presence precision, recall, F1, and accuracy;
- target exact accuracy and micro/macro F1 on relation rows;
- cardinality accuracy and predicted/gold cardinality distributions;
- exact match by gold root and singleton versus multilabel rows;
- calibration metrics if confidences are exposed.

Do not call root_entity_accuracy the assignment accuracy.

## Required efficiency metrics

Measure locally with network-independent inference:

- cold model-load time;
- cold first-prediction latency;
- warmed single-query p50, p95, p99 over at least 100 predictions;
- documented batch throughput;
- size of all required serialized artifacts;
- peak process RSS or a documented approximation;
- CPU model, thread count, and GPU use if any;
- whether Torch/transformers are required in the deployed path.

Exclude CSV I/O from model-only latency; optionally report end-to-end latency
separately. Do not confuse milliseconds with seconds.

## Required artifacts

Every frozen run should contain:

~~~text
artifacts/<run-id>/
  config.json
  metrics.json
  metrics.csv
  latency.json
  ablation_summary.csv
  predictions/
    validation.csv
    test.csv
  errors.csv
  models/
    entity_pipeline.joblib
~~~

Prediction CSV columns:

~~~text
row_id
question
gold_entities
predicted_entities
exact_match
missing_entities
extra_entities
gold_root
predicted_root
gold_has_relation
predicted_has_relation
confidence_or_probabilities
~~~

Serialize arrays as valid JSON. errors.csv must contain only non-exact rows and
use computed categories:

- wrong_root;
- missing_relation_target;
- extra_relation_target;
- wrong_relation_target;
- multiple_entity_errors.

## Final report and examples

Create doc/entity_extraction_report.md after evaluation. Include:

- selected approach and justification;
- full ablation results;
- accuracy, latency, size, and memory comparisons;
- at least three representative correct predictions, including a relation case;
- at least three representative errors covering missing, extra, and wrong labels
  when available;
- per-label weaknesses and rare-label behavior;
- the split multilabel-rate shift;
- open issues and practical improvements.

Choose examples only after freezing. Never tune from test examples.

## Tests and invariants

Add tests proving:

1. root-only JSON yields one entity;
2. relations add all relationTargetType values;
3. nested relations are traversed;
4. duplicates are deduplicated;
5. entity order does not affect exact scoring;
6. inference never reads target JSON;
7. blank input fails explicitly;
8. dump/load preserves predictions;
9. fixed-seed training is deterministic;
10. split row IDs/templates remain disjoint;
11. every prediction is non-empty and uses known labels;
12. errors.csv contains only errors;
13. generated CSV uses LF line endings.

Verification commands:

~~~bash
conda run -n cognyte pytest -q
conda run -n cognyte python -m py_compile \
  src/nl_api/entity_pipeline.py \
  scripts/run_entity_pipeline.py \
  scripts/optimize_entity_pipeline.py
git diff --check
~~~

Do not bypass exceptions or turn invalid predictions into successful metrics.
Diagnose and fix failures at their source.

## Execution sequence for the new agent

1. Read doc/assignment.md completely.
2. Read this handoff completely.
3. Inspect the dirty worktree and preserve unrelated changes.
4. Verify split integrity and target extraction.
5. Add target/evaluation tests.
6. Implement root-only and direct sparse multilabel baselines.
7. Run grouped OOF plus validation optimization.
8. Implement and compare hierarchical and label-powerset candidates.
9. Measure accuracy, latency, size, and memory for each candidate.
10. Select and freeze the development winner.
11. Run the full tests and validation benchmark.
12. Evaluate the frozen model on test once.
13. Generate metrics, predictions, filtered errors, and the report.
14. Update README with exact local commands.
15. Stop only after evaluation and artifact verification, unless genuinely
    blocked on authority or missing inputs.

## Definition of done

The work is complete only when:

- inference returns only entity lists;
- the label contains the root and every relation target;
- optimization uses train/validation/grouped OOF only;
- frozen test evaluation is complete;
- full multilabel accuracy and efficiency metrics are reported;
- optimization impacts and test examples are documented;
- artifacts are auditable and tests pass;
- there is no test leakage, hardcoded answer mapping, silent exception bypass,
  or hosted inference dependency.
