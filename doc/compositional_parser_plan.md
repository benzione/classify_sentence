# Compositional Parser Implementation Plan

## Objective

Replace complete-template adaptation after root prediction with a parser that
constructs a schema-valid API AST from independently detected question
conditions. Keep the existing root classifier and retain the current decoder as
a validation-selected fallback while the compositional decoder matures.

No component may contain entity-specific question-to-answer mappings. Schema
fields and descriptions are supplied reference data; language rules must be
generic across entities.

## Evidence and baseline

- Held-out root accuracy: 92.86%.
- Held-out canonical exact match: 17.86% (20/112).
- Only 44/112 held-out value-free gold skeletons occur in training.
- Trying every compatible training template through the current assembler can
  produce only 24/112 held-out gold requests.
- With the gold AST supplied, current value adaptation preserves only 71/112
  held-out requests exactly.

These measurements make template retrieval and value adaptation the limiting
components. The new decoder must create unseen combinations and bind each value
to its own condition.

## Leakage and selection policy

1. Fit every learned component on `data/splits/train.csv` only.
2. Derive thresholds, routing, and model selection from validation only.
3. Do not use test labels during implementation or optimization.
4. Use the test split once after the complete validation winner is fixed.
5. Record training row IDs, configuration, component metrics, latency, and
   predictions for every promoted candidate.

## Implementation stages

### Stage 1 — Condition-level supervision

Status: completed.

Decompose every training AST into atomic condition records containing:

- root entity;
- scope entity/path;
- field;
- operator;
- typed value;
- relation context;
- Boolean-group context;
- aligned question span and alignment confidence.

Alignment confidence levels:

- `exact`: target literal occurs verbatim, case-insensitively;
- `normalized`: a typed value such as a date or relative time matches after
  deterministic normalization;
- `lexical`: the condition clause contains schema field-name evidence;
- `unaligned`: no defensible span was found; retained for request-level
  structure training but excluded from span-supervised field training.

Acceptance gate: deterministic export, no dropped filters, stable row IDs, and
an alignment coverage report broken down by root and confidence.

Result: 1,277/1,277 training filters retained. Alignments comprise 777 exact,
230 normalized, 202 lexical, and 68 unaligned conditions.

### Stage 2 — Candidate condition spans

Status: completed for the deterministic candidate generator.

Detect candidate spans without target access using typed literals, relative-time
expressions, quoted phrases, and clause boundaries. Preserve `between X and Y`
as one condition and preserve explicit `OR` evidence.

Acceptance gate: candidate recall against exact/normalized training alignments,
reported overall and by root. Precision is secondary at this stage because the
field linker can reject candidates.

Result: 99.90% recall on train and 99.05% on validation for exact/normalized
condition spans. Learned span detection remains a later enhancement if candidate
precision becomes the bottleneck.

### Stage 3 — Shared span-to-schema field linker

Status: in progress.

Train one shared pairwise ranker over:

```text
condition span + root + scope candidate
field name + description + field type
```

Use positive aligned span/field pairs and hard negative fields from the same
root or relation scope. At inference, score only schema-valid candidates.

Acceptance gate: validation top-1 and top-3 field-and-scope accuracy must beat
the existing document-level field head, especially for CDR and Web Activity.

Initial result: the best shared MiniLM multiclass linker (`compositional-linker-005`)
reaches 45.02% top-1 and 70.62% top-3 validation field accuracy with gold scope
and aligned spans. It is promising for Phone, Person, Report, Insight, and
Investigation, but CDR and Web Activity remain below the acceptance gate. The
linker is therefore experimental and is not connected to production decoding.

### Stage 4 — Structure decoder

Status: pending.

Predict condition count, relation target/type, root versus relation scope,
explicit OR groups, and sort nodes. Decode a tree/set rather than an arbitrary
classifier-chain field order.

Acceptance gate: component count, relation, Boolean, and scope metrics improve
without reducing schema validity below 100%.

### Stage 5 — Operator and value binding

Status: pending.

Predict an operator after selecting a field. Copy or normalize the value from
that condition's span. Use generic typed normalization for dates, relative time,
numbers, emails, URLs, phones, IP addresses, strings, and Booleans.

Acceptance gate: when supplied the gold field and scope, operator/value exact
match must materially exceed the current adapter's 71/112 held-out diagnostic;
selection is performed on validation equivalents only.

### Stage 6 — Constrained AST search

Status: pending.

Assemble predicted nodes under schema constraints. Use a small beam when field,
scope, or structure decisions are ambiguous. Invalid fields, relation targets,
operators, and AST transitions must be rejected during construction rather than
repaired afterward.

Acceptance gate: 100% strict JSON and schema validity.

### Stage 7 — Validation-selected hybrid routing

Status: pending.

Compare the compositional and incumbent decoders per root on validation. Retain
the incumbent for roots where the new decoder is worse. Routing decisions are
configuration learned from validation results, not hardcoded semantic mappings.

Acceptance gate: higher overall validation canonical exact match with measured
latency and no root-specific regression hidden by aggregate metrics.

### Stage 8 — Optimization and final evaluation

Status: pending.

Optimize confidence thresholds, candidate counts, beam width, and optional
encoder compression on validation. Select exact match first, then latency and
model size. Run the selected candidate once on the locked test split and produce
`metrics.json`, `latency.json`, predictions, and `errors.csv`.

## Component metrics

Every validation run must report:

- root accuracy;
- candidate-span recall;
- condition-count accuracy;
- field top-1/top-3 accuracy;
- exact field-set accuracy;
- field-and-scope accuracy;
- operator accuracy with gold and predicted fields;
- value exact match with gold and predicted fields;
- relation and Boolean accuracy;
- canonical exact match;
- strict JSON/schema validity;
- p50/p95 latency and serialized model size.

## Rollback rule

The incumbent `hierarchical-002` remains the production candidate until a new
model wins validation and then improves or safely matches the locked test result
at an acceptable latency and model size. Negative experiments remain documented
and disabled by default.
