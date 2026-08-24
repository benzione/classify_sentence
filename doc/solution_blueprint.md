# Local NL-to-API JSON Pipeline: Implementation Blueprint

## 1. Purpose and scope

Build a fully local Python pipeline that converts a natural-language query into the API-request JSON represented in `data/user_queries.csv`.

The entity types (`entityType` and nested `relationTargetType`) are an important intermediate output, as emphasized by the assignment, but they are **not the final product**. The final product is a valid, schema-conformant request such as:

```json
{
  "entityType": "CDR",
  "statements": [
    {
      "type": "filter",
      "parameters": {
        "name": "ifc.ootb.CDR.technology",
        "operator": "equals",
        "value": "5G"
      }
    }
  ]
}
```

The implementation must run locally on CPU or a local GPU, with low online latency. Do not use hosted LLMs or hosted retrieval at inference time. The design must report measured accuracy, latency, model size, and the effect of each optimization.

## 2. Source-data facts and constraints

### 2.1 Files

- `data/user_queries.csv` has 744 `question` → `json` examples. The actual input column is named `question`, although the assignment text calls it `questions`.
- The target column is Python-literal-like text, rather than strict JSON: it uses single quotes and values such as `True`. Read it with `ast.literal_eval`, **never** `eval`.
- `data/fields_description.csv` provides 144 fields for the nine supported entity types: `CDR`, `EVisa Request`, `Insight`, `Investigation`, `Person`, `Phone`, `Report`, `Web Activity`, and `Web Actor`.
- The source JSON includes root entities, filters, nested relation blocks, AND-by-list semantics, and explicit `OR` blocks. It is therefore supervised data for structured parsing, not merely entity classification.

### 2.2 Target grammar

Model the API request as a typed abstract syntax tree (AST) before serializing it. The core grammar is:

```text
Request        := { entityType: Entity, statements: Statement[] }
Statement      := Filter | Relation | BooleanGroup
Filter         := { type: "filter", parameters: { name, operator, value } }
Relation       := { type: "relation", parameters: { relationType[], relationTargetType[] }, statements: Statement[] }
BooleanGroup   := { type: "operator", parameters: { operatorValue: "OR" }, statements: Statement[] }
```

The order of top-level AND statements is not semantically important, but the order and nesting inside an `OR` group is. Create a deterministic canonical ordering for evaluation and output. Do not alter semantics by flattening nested groups.

### 2.3 Explicit non-goals for version 1

- Do not infer API fields that are not in the supplied schema.
- Do not execute the API or claim answer/result accuracy; only generate the request.
- Do not train an unconstrained free-text JSON generator as the sole system. It can generate malformed JSON, wrong fields, and invalid nesting.

## 3. Required repository layout

The agent implementing this blueprint should create this structure:

```text
data/
  user_queries.csv                     # immutable input
  fields_description.csv               # immutable input
  splits/
    train.csv
    validation.csv
    test.csv
    split_manifest.json
    split_report.json
artifacts/
  <run_id>/
    config.json
    schema_index.joblib
    lexical_retriever.joblib
    models/
    calibration/
    predictions/
    metrics.json
    metrics.csv
    latency.json
    errors.csv
    ablation_summary.csv
scripts/
  prepare_splits.py
  run_pipeline.py
  evaluate.py                           # optional if not a mode of run_pipeline.py
tests/
  test_parsing.py
  test_schema_validation.py
  test_canonicalization.py
  test_no_leakage.py
```

`prepare_splits.py` is a one-time, deterministic **prephase**. `run_pipeline.py` is the actual train/validate/test/inference pipeline. It must refuse to overwrite a split unless an explicit `--force-resplit` flag is supplied.

## 4. Prephase: validate, normalize, and split the data

### 4.1 Input validation and normalized records

`scripts/prepare_splits.py` must:

1. Load both CSV files using UTF-8 and validate required headers.
2. Parse every target with `ast.literal_eval` into a Python object.
3. Recursively validate the AST:
   - one supported root `entityType`;
   - only known statement types;
   - every filter has `name`, `operator`, and `value`;
   - every field is allowed for the root entity or its relation target entity;
   - every relation has non-empty `relationType` and `relationTargetType` lists;
   - only supported operator values and value types are used.
4. Normalize without changing meaning:
   - trim text whitespace;
   - normalize field/operator aliases only if an explicit mapping is documented;
   - convert parsed targets to canonical strict JSON using `json.dumps`;
   - retain both `question_raw` and `question_normalized`;
   - add `row_id`, `root_entity`, `entity_labels`, `has_relation`, relation target labels, fields, operators, and AST complexity metadata.
5. Stop with a clear error report for invalid source rows. Do not silently discard rows.

`question_normalized` is only for duplicate/template grouping. It should lowercase and collapse whitespace, while preserving raw values separately. For an optional stronger template key, replace emails, phone numbers, dates, quoted spans, and standalone numbers with typed placeholders. Never feed a test example's labels or normalized AST into training.

### 4.2 Split policy

Create a fixed **70% train / 15% validation / 15% test** split with a documented seed (for example, `20260824`). Save the actual rows in `data/splits/`; no later script may dynamically re-split them.

Split by duplicate/template group, not individual row:

- Exact duplicate questions must be in exactly one split.
- Prefer grouping obvious near-template variants using the placeholder template key when doing so still preserves each rare class in train and test.
- Multi-label stratification should preserve root entities, relation targets, `has_relation`, and major structural features (`OR` presence, number of filters) across splits.
- With only nine `EVisa Request` examples, enforce at least 6 train examples and at least 1 validation and 1 test example; log the final count.

Use iterative multilabel stratification where possible. Because group-aware iterative stratification is not always available, implement a deterministic candidate search: generate multiple group-wise seeded splits, reject any that violate the constraints, and select the split minimizing a weighted distribution-distance score over the labels and complexity features. Record the algorithm, seed, source SHA-256 fingerprint, group IDs, row IDs, and class counts in `split_manifest.json` and `split_report.json`.

The test set is locked after this prephase. It may be read only once for final reporting. If exploratory experiments require repeated comparisons, use train/validation only or cross-validation **inside the training partition**.

### 4.3 Leakage rules

The following are mandatory:

- Fit TF-IDF vocabulary/IDF, BM25 corpus, nearest-neighbor examples, feature scaling, class weights, model parameters, thresholds, and fusion weights on train only.
- Tune hyperparameters and confidence thresholds on validation only.
- Never retrieve test questions as examples during training, validation, or test inference. At test inference, retrieval corpus is train only.
- Do not select a model, threshold, or augmentation strategy based on test metrics.
- Static schema descriptions may be used for all splits because they are supplied reference data, but they must not be augmented with held-out query/target information.
- Deduplicate before any optional augmentation. Augment training rows only; do not derive variants from validation or test questions.
- Include assertions in `test_no_leakage.py` that train/validation/test row IDs and duplicate groups are disjoint and that every fitted artifact declares its training split fingerprint.

## 5. Canonical target representation

Build `ApiRequest`, `Filter`, `Relation`, and `BooleanGroup` dataclasses/Pydantic models. All models must support:

- parsing source Python literals;
- strict schema validation;
- deterministic serialization to JSON;
- canonical comparison for metric calculation;
- conversion to and from a simplified internal representation.

Canonicalization rules must be published in the run configuration. A recommended safe rule is to sort sibling filters under an implicit AND by `(field, operator, normalized value)` but preserve statement order within explicit `OR` groups and preserve relation nesting. Use canonicalized ASTs for exact-match metrics, while retaining raw targets for auditability.

Normalize deterministic values before comparison:

- dates to `YYYY-MM-DD` where a date-only target is expected;
- phone, email, quoted phrase, Boolean, integer, float, and list values according to field type;
- relative time expressions to the API's structured `relative` object;
- known enum values only using schema/observed mappings, such as “SMS” → CDR `type: Text` and “call” → CDR `type: Voice` when validated by training examples.

Do not invent temporal reference dates. Relative terms must remain relative in the API representation.

## 6. Recommended end-to-end pipeline

Use a constrained hybrid parser. It has deterministic components for values and JSON construction, with ML/retrieval components to resolve semantic ambiguity.

```text
Question
  → normalize and extract typed literals
  → candidate entities and schemas
  → candidate fields/operators/relations
  → constrained AST search and scoring
  → schema validation and canonical JSON serialization
  → JSON response + confidence/diagnostics
```

### 6.1 Typed literal extraction (high precision)

Implement deterministic extractors before ML prediction for:

- email addresses;
- phone/MSISDN and IMEI/IMSI-looking identifiers;
- URLs, hashtags, IP addresses, quoted phrases;
- integers/floats and number words where unambiguous;
- Boolean cues (`suspicious`, `target`, `has attachments`);
- absolute dates, date ranges, years, quarters, and month expressions;
- relative time phrases (`last 12 hours`, `past two weeks`, `yesterday`).

The extractor must emit candidate spans and types, not final fields. For example, an email span might correspond to originator or destination email; a semantic/structural stage resolves that role.

### 6.2 Schema index and lexical retrieval

Create schema cards per entity and field from entity name, humanized field name, type, and description. Build:

- a BM25 index of **training questions**, retaining their parsed AST components;
- a BM25/TF-IDF index of schema cards;
- word and character TF-IDF features for supervised classifiers.

BM25 is useful in addition to TF-IDF classification: it retrieves highly similar labeled examples and handles rare template patterns. It is not an external service and remains local. At inference, retrieve only top-k training examples and schema fields. Save retrieval provenance (example row IDs and scores) in debug output; omit it from normal production responses if sensitive.

### 6.3 Semantic retrieval

Add a small local sentence embedding encoder only after the sparse baseline is measured. Candidates include `all-MiniLM-L6-v2` or `bge-small-en`; download/cache the model locally and record its exact revision/license. Use it to retrieve training examples and schema cards for paraphrase robustness.

For deployment optimization, export to ONNX and dynamically quantize to INT8. Verify that quantization has not changed canonical JSON metrics beyond a predefined allowable tolerance. The sparse system remains the safe fallback if the encoder is absent or fails to load.

### 6.4 Hierarchical supervised prediction

Prefer several small, interpretable models over one unconstrained sequence generator:

1. Root entity classifier: nine-way calibrated classifier.
2. Relation detector and relation target classifier, conditioned on the root entity.
3. Field candidate ranker, conditioned on predicted/beam-search entity context.
4. Operator classifier/ranker for each field candidate.
5. Value-role ranker, matching extracted values or spans to each selected field.
6. Boolean grouping classifier/rules, to identify explicit `OR` language.

The initial classifiers should be class-weighted linear models over concatenated word and character TF-IDF features. Use probabilities, not hard labels. Add semantic and BM25 scores as features in a learned fusion/reranker only after generating **out-of-fold** training predictions; otherwise the fusion model will overfit retrieval of its own examples.

The relation/field grammar must constrain predictions. For example, `CDR` can connect to `Phone` in patterns observed in the data, and Web Activity can relate to Web Actor; a request must not contain arbitrary unsupported label pairs.

### 6.5 Constrained AST decoding

Generate a small beam of valid AST candidates instead of independently accepting every field label. Score candidates using:

- entity, relation, field, operator, and value-role probabilities;
- BM25 support from retrieved training ASTs;
- semantic similarity support;
- schema-card similarity;
- deterministic value/type compatibility;
- penalties for unsupported field/entity combinations and invalid nesting.

Keep beam width modest (for example 5–20) and set a latency budget. Select only schema-valid candidates. If no complete candidate reaches the confidence threshold, return the highest-scoring valid partial request only if the product contract allows it; otherwise emit a structured low-confidence/error response. Never output invalid JSON or an unsupported field merely to fill a slot.

### 6.6 Output contract

Normal output must be strict JSON, serializable by `json.dumps`, and contain the API request under `request`:

```json
{
  "request": {
    "entityType": "CDR",
    "statements": []
  },
  "confidence": 0.91,
  "model_version": "run-20260824-001"
}
```

For assignment scoring, evaluate the `request` object alone. The wrapper supports production diagnostics and can be disabled with `--output-api-only`.

## 7. Preventing underfitting, overfitting, and minority-class failure

### 7.1 Underfitting controls

Symptoms: low train and validation component F1, generic predictions, missing relations/filters.

- Expand word and character n-gram ranges and vocabulary within memory limits.
- Add BM25 training-example retrieval and schema-card features.
- Add the small semantic encoder for paraphrases.
- Improve deterministic phrase/temporal parsers based only on training/validation error analysis.
- Increase AST beam width only if it improves validation exact match within latency budget.

### 7.2 Overfitting controls

Symptoms: high train exact match but much lower validation exact match, especially on template variants.

- Use regularized linear models; tune `C`, n-gram ranges, `min_df`, and vocabulary size on validation only.
- Keep model selection simple. Do not fine-tune a transformer initially on 744 examples.
- Group duplicate and near-template queries across split boundaries.
- Use out-of-fold predictions for every stacked/fusion feature.
- Use early stopping and a frozen encoder if a neural component is later trained.
- Maintain a small manually authored, unseen-phrasing robustness suite, never used for tuning.

### 7.3 Minority/majority class controls

`EVisa Request` is rare; CDR is frequent. Overall accuracy alone will hide failure on rare entities.

- Use class weights or per-task sample weights for all supervised classifiers.
- Use iterative multilabel stratification and minimum-support split constraints.
- Tune per-class thresholds on validation; do not let CDR's threshold dictate all labels.
- Use schema and retrieval evidence to support sparse classes.
- Consider carefully reviewed training-only paraphrase augmentation for minority examples, clearly marking synthetic rows and measuring its validation effect. Do not oversample by cloning examples without a reason; it can memorize templates.
- Report macro F1 and per-class precision/recall/F1 alongside micro metrics.

## 8. Hyperparameter and optimization procedure

All defaults belong in a versioned YAML/JSON config. `run_pipeline.py` must accept `--config`, `--run-id`, `--mode {train,evaluate,infer,benchmark}`, and `--output-api-only`.

Start with the following search ranges, then narrow them based on validation results and resource budget:

| Component | Initial values to evaluate | Selection criterion |
|---|---|---|
| Word TF-IDF | ngrams `(1,2)` / `(1,3)`, `min_df` 1/2, `max_features` 20k/50k | validation AST/component F1 and RAM |
| Char TF-IDF | ngrams `(3,5)` / `(3,6)`, `min_df` 1/2, 20k/50k features | robustness and latency |
| Linear model | logistic regression or LinearSVC + calibration; `C` 0.1, 0.5, 1, 2, 5 | macro F1 and calibration |
| Class weighting | none vs `balanced` vs capped inverse-frequency | minority recall without precision collapse |
| BM25 | `k1` 0.8–2.0, `b` 0.2–0.9, top-k 3/5/10 | component/exact-match gain |
| Semantic retrieval | top-k 3/5/10, cosine score threshold | paraphrase gains vs p95 latency |
| AST decoder | beam 5/10/20, max statements 3/5/8 | exact match under latency cap |
| Fusion | L2 regularization `C` 0.1–2 | validation improvement over best base model |
| Confidence thresholds | per-component grid or precision/recall target | calibrated F1 and safe fallback rate |

Run one change at a time after establishing a baseline. Do not optimize on test. Use a fixed validation objective, for example: maximize canonical JSON exact match; use macro component F1 as tie-breaker; reject configurations exceeding the p95 latency or RAM budget.

## 9. Experiment plan and KPI artifacts

Run and save an ablation table for these versions:

1. Deterministic value extraction + schema validation only.
2. Sparse supervised classifiers + constrained AST decoder.
3. Version 2 + BM25 training-query retrieval.
4. Version 3 + schema-card retrieval.
5. Version 4 + local semantic retrieval.
6. Version 5 with ONNX INT8 semantic encoder.
7. Final tuned configuration.

Every run writes `artifacts/<run_id>/metrics.json`, `metrics.csv`, `latency.json`, and `ablation_summary.csv`. At minimum record:

- dataset/split fingerprints, seed, source revision, package versions, hardware, and configuration;
- strict JSON validity rate and schema-validity rate;
- canonical full-JSON exact-match accuracy;
- AST tree/component F1;
- root entity accuracy and entity-label micro/macro precision, recall, F1;
- relation-target precision, recall, F1;
- field, operator, value, Boolean-group, and temporal-normalization accuracy/F1;
- per-class metrics and support counts;
- confusion matrices for root entity and relation target;
- p50/p95 warm inference latency, cold-start latency, throughput, peak RSS/RAM, and serialized artifact size;
- fallback/abstention rate and error categories.

Benchmark latency after model warm-up using at least 100 repeated local queries, with I/O excluded and included measurements reported separately. State hardware and whether CPU/GPU was used. Model-size reporting must include all artifacts needed at inference, including vectorizers, BM25 index, encoder, and schema index.

## 10. Error analysis and test cases

Write all validation and final test predictions to `predictions/` with question, gold AST, predicted AST, confidence, candidate scores, and a machine-readable error category. Review samples from each category:

- wrong root entity (`Report` vs `Insight`; `Web Actor` vs `Web Activity`);
- missed/extra relation;
- wrong field role (originator versus destination);
- wrong operator (`equals`, `contains`, `before`, `between`, `relative`);
- value extraction/normalization error;
- Boolean grouping/nesting error;
- schema validation rejection;
- low-confidence fallback.

Include several hand-authored end-to-end tests in the final report: clear CDR/Phone relations, Web Activity/Web Actor relations, report/insight text queries, date ranges and relative time, an eVisa query, explicit OR conditions, identifiers, and an intentionally ambiguous query. These tests illustrate behavior but must remain separate from quantitative test metrics unless frozen before tuning.

## 11. Completion checklist for the implementing agent

- [ ] `prepare_splits.py` parses and validates all source rows safely.
- [ ] Fixed train/validation/test CSVs and manifests are saved under `data/splits/`.
- [ ] Duplicate/template groups do not cross split boundaries.
- [ ] All fitting and tuning obey the leakage rules.
- [ ] The baseline produces strict, schema-valid API JSON locally.
- [ ] The final system uses constrained AST construction, not unconstrained JSON text generation.
- [ ] Ablations and optimization impact are measured and saved.
- [ ] Metrics cover JSON correctness, component accuracy, minority classes, and resource KPIs.
- [ ] Final report includes successful cases, failures, open issues, and future improvements.
- [ ] The README/report states that field descriptions are static schema knowledge and explains exactly how they are used.

## 12. Expected open issues to report honestly

- The dataset is small and likely template-heavy, so ordinary held-out results can overestimate production generalization.
- Some user language is inherently ambiguous without conversational clarification or API execution feedback.
- Relation-type directionality and API semantics should be verified with the API owner if a formal API schema exists.
- Full semantic equivalence is more meaningful than textual JSON equality; canonical AST comparison is the best available proxy without an executable backend.
- Additional real, naturally phrased examples and rare-class samples will likely improve the system more than increasing model size.
