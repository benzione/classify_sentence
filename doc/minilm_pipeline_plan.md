# MiniLM pipeline plan

## Goal

Add MiniLM as a second, independent entity-set extraction pipeline. It must not
replace, import, mutate, or serve as a fallback for the existing sparse
TF-IDF/logistic pipeline. Both pipelines consume the same fixed split CSVs and
produce the same entity-set output and evaluation artifacts, allowing a final
deployment choice based on accuracy, latency, memory, and model size.

## Independent architecture

```text
fixed split CSVs
     ├── TF-IDF + logistic scripts/artifacts (existing, unchanged)
     └── MiniLM + linear multilabel heads scripts/artifacts/minilm-*/
```

The MiniLM path will use the pretrained `sentence-transformers/all-MiniLM-L6-v2`
sentence encoder only to create dense query embeddings. A separate balanced
one-vs-rest logistic-regression head is trained for every entity label. This
keeps the output contract identical while avoiding a fragile end-to-end
Transformer fine-tune on the small 520-row training split.

The encoder is loaded from a pinned local Hugging Face snapshot. Training and
inference run locally after that explicit model download; no user query is sent
to a hosted inference API.

## Training and selection

1. Load `train.csv`, normalize only the training queries, and generate MiniLM
   embeddings in batches on CPU or a local CUDA device when available.
2. Fit the independent multilabel heads on training embeddings only.
3. Compare a small validation-only grid: balanced/unweighted heads, `C`, and
   global threshold. Compute three-fold template-grouped out-of-fold training
   exact-set accuracy as a guard against split-specific selection.
4. Select by validation exact entity-set match, grouped OOF exact match,
   warmed CPU p95 latency, then serialized size. `test.csv` is not read during
   selection.
5. Refit the selected MiniLM configuration on train+validation, then evaluate
   once on test. The existing sparse pipeline is not retrained or modified.

## Optimization

- Use `all-MiniLM-L6-v2` rather than a larger Transformer.
- Use batched embedding generation during training/evaluation.
- Persist precomputed embeddings only as local temporary data; do not include
  them in the deployment artifact.
- Export the encoder to ONNX when supported, apply dynamic INT8 quantization,
  and benchmark the PyTorch and ONNX variants separately. Select ONNX only if
  its validation predictions are equivalent and it improves latency or size.
- Limit CPU inference to one provider thread for comparable, repeatable p95
  measurements.

## KPIs and decision table

Both pipelines will report:

| Area | Required measurements |
|---|---|
| Accuracy | exact entity-set match, micro/macro/weighted F1, sample Jaccard, Hamming loss |
| Relations | relation-presence F1, relation-target exact accuracy and micro F1 |
| Efficiency | warmed p50/p95/p99, cold load, cold first prediction, artifact bytes, peak RSS, CPU/GPU use |
| Reliability | per-entity precision/recall/F1, errors CSV, fallback rate |

The final report will show the sparse and MiniLM numbers side-by-side. The
winner is the pipeline meeting the agreed latency target with the highest
validation and newly held-out evaluation accuracy. The current test split has
already informed rule changes in the sparse pipeline, so it is a diagnostic
comparison rather than a blind final selection benchmark.

## Deliverables

- `src/nl_api/minilm_pipeline.py`: MiniLM encoder and independent multilabel
  classifier implementation.
- `scripts/run_minilm_pipeline.py`: train, validate, evaluate, benchmark, and
  infer commands.
- `scripts/optimize_minilm_pipeline.py`: validation-only selection.
- MiniLM-specific tests, artifacts, documentation, and a requirements extra.

## Completed baseline run (CPU)

The independent implementation was run locally with a cached MiniLM snapshot
and native PyTorch inference. Validation-only selection chose unweighted heads
with `C=1` and threshold `0.50`:

| KPI | Sparse TF-IDF pipeline | MiniLM pipeline |
|---|---:|---:|
| Validation exact entity-set match | 98.21% | 91.96% |
| Train grouped OOF exact match | 96.15% | 87.50% |
| Test diagnostic exact entity-set match | 94.64% (106/112) | 83.93% (94/112) |
| Test micro F1 | 97.81% | 91.73% |
| Warmed CPU p50 / p95 / p99 | 4.69 / 6.25 / 6.79 ms | 15.34 / 30.17 / 32.34 ms |
| Peak process RSS | 126.5 MB | 416.7 MB |
| Serialized trained heads | 935,798 B total sparse artifact | 19,599 B plus 87 MB MiniLM encoder cache |

MiniLM is not selected for deployment in this run: it is approximately 4.8x
slower at p95 and 10.7 percentage points lower in test exact-set match. ONNX
Runtime was unavailable, so INT8 ONNX conversion was not performed; it remains
the next MiniLM-specific optimization experiment, not a reason to change the
current sparse deployment.

## Risks and mitigations

The pretrained model may not be cached locally or the environment may not have
the required CPU runtime. The implementation will fail clearly with setup
instructions rather than silently falling back to TF-IDF. Small labelled data
can make MiniLM worse than sparse features; the independent KPI comparison is
therefore the selection mechanism, not an assumption that a Transformer wins.
