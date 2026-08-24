#!/usr/bin/env python3
"""Train, validate, evaluate, benchmark, and infer with entity-only models."""
from __future__ import annotations

import argparse, csv, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             jaccard_score, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from nl_api.data import read_split
from nl_api.entity_pipeline import EntityConfig, EntityPipeline


def model_path(run_id: str) -> Path: return ROOT / "artifacts" / run_id / "models" / "entity_pipeline.joblib"

def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p, method="nearest")) if values else 0.0

def averaged_prf(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

def evaluate(model: EntityPipeline, rows: list[dict], output: Path) -> dict:
    predictions, gold_sets, predicted_sets, latencies, fallback_count = [], [], [], [], 0
    for row in rows:
        start = time.perf_counter(); prediction = model.predict(row["question_raw"]); latencies.append((time.perf_counter()-start)*1000)
        gold, predicted = set(row["entity_labels"]), set(prediction.entities)
        fallback_count += prediction.fallback_used
        gold_sets.append(gold); predicted_sets.append(predicted)
        predictions.append({"row_id": row["row_id"], "question": row["question_raw"], "gold_entities": json.dumps(sorted(gold)), "predicted_entities": json.dumps(sorted(predicted)), "exact": gold == predicted, "confidence": prediction.confidence, "fallback_used": prediction.fallback_used, "probabilities": json.dumps(prediction.probabilities, sort_keys=True)})
    labels = model.labels
    y_true = np.asarray([[int(label in values) for label in labels] for values in gold_sets]); y_pred = np.asarray([[int(label in values) for label in labels] for values in predicted_sets])
    exact = int(sum(a == b for a, b in zip(gold_sets, predicted_sets)))
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    by_entity = {label: {"precision": float(per_precision[i]), "recall": float(per_recall[i]), "f1": float(per_f1[i]), "support": int(support[i]), "fp": int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum()), "fn": int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())} for i, label in enumerate(labels)}
    relation_rows = [i for i, row in enumerate(rows) if row["has_relation"]]
    target_labels = sorted({label for row in rows for label in row["relation_targets"]})
    target_metrics = {"rows": len(relation_rows)}
    if relation_rows and target_labels:
        truth = np.asarray([[int(label in rows[i]["relation_targets"]) for label in target_labels] for i in relation_rows])
        pred = np.asarray([[int(label in predicted_sets[i] - {rows[i]["root_entity"]}) for label in target_labels] for i in relation_rows])
        target_metrics.update({"exact_accuracy": float(accuracy_score(truth, pred)), "micro_f1": float(f1_score(truth, pred, average="micro", zero_division=0)), "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0))})
    root_correct = sum(row["root_entity"] in predicted_sets[i] for i, row in enumerate(rows))
    rel_true = np.asarray([row["has_relation"] for row in rows]); rel_pred = np.asarray([len(predicted_sets[i]) > 1 for i in range(len(rows))])
    rp, rr, rf, _ = precision_recall_fscore_support(rel_true, rel_pred, average="binary", zero_division=0)
    by_root = {}
    for root in sorted({row["root_entity"] for row in rows}):
        idx = [i for i, row in enumerate(rows) if row["root_entity"] == root]
        by_root[root] = {"samples": len(idx), "exact_count": sum(gold_sets[i] == predicted_sets[i] for i in idx), "exact_rate": sum(gold_sets[i] == predicted_sets[i] for i in idx)/len(idx)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=predictions[0].keys() if predictions else [], lineterminator="\n"); writer.writeheader(); writer.writerows(predictions)
    return {"samples": len(rows), "exact_entity_set_match": exact/len(rows) if rows else 0, "exact_count": exact, "micro": averaged_prf(y_true, y_pred, "micro"), "macro": averaged_prf(y_true, y_pred, "macro"), "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), "hamming_loss": float(hamming_loss(y_true, y_pred)), "sample_jaccard": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)), "root_entity_accuracy": root_correct/len(rows) if rows else 0, "relation_presence": {"precision": float(rp), "recall": float(rr), "f1": float(rf), "accuracy": float((rel_true == rel_pred).mean())}, "relation_target": target_metrics, "cardinality_accuracy": sum(len(a)==len(b) for a,b in zip(gold_sets,predicted_sets))/len(rows), "gold_cardinality": dict(Counter(map(len, gold_sets))), "predicted_cardinality": dict(Counter(map(len, predicted_sets))), "fallback_rate": fallback_count/len(rows), "by_entity": by_entity, "by_gold_root": by_root, "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95), "p99": percentile(latencies, 99)}}

def write_csv(path: Path, report: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n"); writer.writerow(("metric", "value"))
        for key, value in report.items():
            if not isinstance(value, dict): writer.writerow((key, value))

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("train", "validate", "evaluate", "benchmark", "infer"), required=True); parser.add_argument("--run-id", required=True); parser.add_argument("--question"); parser.add_argument("--diagnostics", action="store_true"); parser.add_argument("--family", choices=("root", "direct", "hierarchical", "powerset"), default="direct"); parser.add_argument("--threshold", type=float, default=0.5); parser.add_argument("--semantic-keyword-boost", type=float, default=0.0); parser.add_argument("--semantic-relation-boost", type=float, default=0.0); parser.add_argument("--semantic-keyword-filter", action="store_true"); parser.add_argument("--semantic-keyword-min-hits", type=int, default=1); parser.add_argument("--word-ngrams", choices=("1,1", "1,2"), default="1,2"); parser.add_argument("--char-ngrams", choices=("3,5", "3,6"), default="3,5"); parser.add_argument("--min-df", type=int, choices=(1,2), default=1); parser.add_argument("--max-features", type=int, choices=(10000,20000,40000), default=20000); parser.add_argument("--c", type=float, default=1); parser.add_argument("--unweighted", action="store_true"); parser.add_argument("--include-validation", action="store_true", help="fit final model on train.csv plus validation.csv; do not run validation scoring afterward"); parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits"); args = parser.parse_args()
    path = model_path(args.run_id); artifact = path.parents[1]
    if args.mode == "train":
        rows = read_split(args.split_dir / "train.csv")
        if args.include_validation: rows += read_split(args.split_dir / "validation.csv")
        config = EntityConfig(args.family, tuple(map(int,args.word_ngrams.split(","))), tuple(map(int,args.char_ngrams.split(","))), args.min_df, args.max_features, args.c, not args.unweighted, args.threshold, 20260824, args.semantic_keyword_boost, args.semantic_keyword_filter, args.semantic_keyword_min_hits, args.semantic_relation_boost)
        model = EntityPipeline.train(rows, config); model.dump(path); artifact.mkdir(parents=True, exist_ok=True)
        metadata = model.metadata() | {"training_splits": ["train", "validation"] if args.include_validation else ["train"]}
        (artifact/"config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.include_validation:
            print(json.dumps({"training_rows": len(rows), "training_splits": metadata["training_splits"], "validation_scoring": "not run: validation rows were included in fitting"}, indent=2)); return
        report = evaluate(model, read_split(args.split_dir/"validation.csv"), artifact/"predictions"/"validation.csv"); (artifact/"metrics.json").write_text(json.dumps({"validation":report},indent=2),encoding="utf-8"); write_csv(artifact/"metrics.csv",report); print(json.dumps(report,indent=2)); return
    if not path.exists(): raise SystemExit(f"model does not exist: {path}")
    model = EntityPipeline.load(path)
    if args.mode == "infer":
        if not args.question: raise SystemExit("--question is required for infer")
        prediction = model.predict(args.question); print(json.dumps(prediction.__dict__ if args.diagnostics else prediction.to_list(), ensure_ascii=False)); return
    if args.mode == "validate": rows, name = read_split(args.split_dir/"validation.csv"), "validation.csv"
    elif args.mode == "evaluate": rows, name = read_split(args.split_dir/"test.csv"), "test.csv"
    else:
        validation = read_split(args.split_dir/"validation.csv"); rows, name = (validation*((100+len(validation)-1)//len(validation)))[:100], "benchmark.csv"
        # Warm the sparse vectorizer and estimators before timing.
        model.predict_batch([r["question_raw"] for r in rows[:10]])
    report = evaluate(model, rows, artifact/"predictions"/name)
    if args.mode == "benchmark":
        # Measure deployment cold start separately from warmed model-only calls.
        import resource
        load_start = time.perf_counter(); cold_model = EntityPipeline.load(path); load_ms = (time.perf_counter() - load_start) * 1000
        first_start = time.perf_counter(); cold_model.predict(rows[0]["question_raw"]); first_ms = (time.perf_counter() - first_start) * 1000
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux reports KiB.
        peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
        report["deployment"] = {"cold_model_load_ms": load_ms, "cold_first_prediction_ms": first_ms,
                                "serialized_size_bytes": path.stat().st_size,
                                "peak_rss_bytes": peak_rss_bytes,
                                "threads": 1, "gpu_used": False, "torch_or_transformers_required": False}
    target = artifact/("latency.json" if args.mode=="benchmark" else "metrics.json"); target.write_text(json.dumps(({args.mode:report} if args.mode != "benchmark" else report),indent=2),encoding="utf-8"); write_csv(artifact/"metrics.csv",report)
    if args.mode == "evaluate":
        with (artifact/"predictions"/name).open(encoding="utf-8",newline="") as source: bad=[r for r in csv.DictReader(source) if r["exact"] != "True"]
        with (artifact/"errors.csv").open("w",encoding="utf-8",newline="") as dest: writer=csv.DictWriter(dest,fieldnames=bad[0].keys() if bad else ["row_id"],lineterminator="\n"); writer.writeheader(); writer.writerows(bad)
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
