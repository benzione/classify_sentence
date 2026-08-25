#!/usr/bin/env python3
"""Train, evaluate, benchmark, and infer with the independent MiniLM pipeline."""
from __future__ import annotations

import argparse, csv, json, resource, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from nl_api.data import read_split
from nl_api.minilm_pipeline import MiniLMConfig, MiniLMPipeline


def model_path(run_id: str) -> Path: return ROOT / "artifacts" / run_id / "models" / "minilm_pipeline.joblib"
def percentile(values: list[float], value: float) -> float: return float(np.percentile(values, value, method="nearest")) if values else 0.0

def evaluate(model: MiniLMPipeline, rows: list[dict], output: Path) -> dict:
    predictions, gold_sets, predicted_sets, latency = [], [], [], []
    for row in rows:
        started = time.perf_counter(); prediction = model.predict(row["question_raw"]); latency.append((time.perf_counter()-started)*1000)
        gold, predicted = set(row["entity_labels"]), set(prediction.entities); gold_sets.append(gold); predicted_sets.append(predicted)
        predictions.append({"row_id": row["row_id"], "question": row["question_raw"], "gold_entities": json.dumps(sorted(gold)), "predicted_entities": json.dumps(sorted(predicted)), "exact": gold == predicted, "confidence": prediction.confidence, "fallback_used": prediction.fallback_used, "probabilities": json.dumps(prediction.probabilities, sort_keys=True)})
    labels = model.labels
    truth = np.asarray([[int(label in values) for label in labels] for values in gold_sets])
    predicted = np.asarray([[int(label in values) for label in labels] for values in predicted_sets])
    per_p, per_r, per_f, support = precision_recall_fscore_support(truth, predicted, average=None, zero_division=0)
    by_entity = {label: {"precision": float(per_p[i]), "recall": float(per_r[i]), "f1": float(per_f[i]), "support": int(support[i]), "fp": int(((predicted[:, i] == 1) & (truth[:, i] == 0)).sum()), "fn": int(((predicted[:, i] == 0) & (truth[:, i] == 1)).sum())} for i, label in enumerate(labels)}
    relation_rows = [i for i, row in enumerate(rows) if row["has_relation"]]
    target_labels = sorted({label for row in rows for label in row["relation_targets"]})
    target = {"rows": len(relation_rows)}
    if relation_rows and target_labels:
        relation_truth = np.asarray([[int(label in rows[i]["relation_targets"]) for label in target_labels] for i in relation_rows])
        relation_pred = np.asarray([[int(label in predicted_sets[i] - {rows[i]["root_entity"]}) for label in target_labels] for i in relation_rows])
        target.update({"exact_accuracy": float(accuracy_score(relation_truth, relation_pred)), "micro_f1": float(f1_score(relation_truth, relation_pred, average="micro", zero_division=0)), "macro_f1": float(f1_score(relation_truth, relation_pred, average="macro", zero_division=0))})
    rel_truth = np.asarray([row["has_relation"] for row in rows]); rel_pred = np.asarray([len(values) > 1 for values in predicted_sets])
    rp, rr, rf, _ = precision_recall_fscore_support(rel_truth, rel_pred, average="binary", zero_division=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=predictions[0].keys()); writer.writeheader(); writer.writerows(predictions)
    return {"samples": len(rows), "exact_entity_set_match": sum(a == b for a, b in zip(gold_sets, predicted_sets))/len(rows), "exact_count": sum(a == b for a, b in zip(gold_sets, predicted_sets)), "micro": _prf(truth, predicted, "micro"), "macro": _prf(truth, predicted, "macro"), "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)), "hamming_loss": float(hamming_loss(truth, predicted)), "sample_jaccard": float(jaccard_score(truth, predicted, average="samples", zero_division=0)), "root_entity_accuracy": sum(row["root_entity"] in predicted_sets[i] for i, row in enumerate(rows))/len(rows), "relation_presence": {"precision": float(rp), "recall": float(rr), "f1": float(rf), "accuracy": float((rel_truth == rel_pred).mean())}, "relation_target": target, "cardinality_accuracy": sum(len(a) == len(b) for a, b in zip(gold_sets, predicted_sets))/len(rows), "gold_cardinality": dict(Counter(map(len, gold_sets))), "predicted_cardinality": dict(Counter(map(len, predicted_sets))), "fallback_rate": sum(row["fallback_used"] for row in predictions)/len(rows), "by_entity": by_entity, "latency_ms": {"p50": percentile(latency, 50), "p95": percentile(latency, 95), "p99": percentile(latency, 99)}}

def _prf(truth: np.ndarray, predicted: np.ndarray, average: str) -> dict:
    p, r, f, _ = precision_recall_fscore_support(truth, predicted, average=average, zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("train", "validate", "evaluate", "benchmark", "infer"), required=True); parser.add_argument("--run-id", required=True); parser.add_argument("--question"); parser.add_argument("--diagnostics", action="store_true"); parser.add_argument("--threshold", type=float, default=.5); parser.add_argument("--c", type=float, default=1); parser.add_argument("--unweighted", action="store_true"); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--device", default="cpu"); parser.add_argument("--include-validation", action="store_true"); args = parser.parse_args()
    path, artifact = model_path(args.run_id), model_path(args.run_id).parents[1]
    if args.mode == "train":
        rows = read_split(ROOT / "data/splits/train.csv")
        if args.include_validation: rows += read_split(ROOT / "data/splits/validation.csv")
        config = MiniLMConfig(device=args.device, batch_size=args.batch_size, c=args.c, balanced=not args.unweighted, threshold=args.threshold)
        model = MiniLMPipeline.train(rows, config); model.dump(path); artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "config.json").write_text(json.dumps(model.metadata() | {"training_splits": ["train", "validation"] if args.include_validation else ["train"]}, indent=2), encoding="utf-8")
        if args.include_validation: print(json.dumps({"training_rows": len(rows), "validation_scoring": "not run: validation rows were included in fitting"}, indent=2)); return
        report = evaluate(model, read_split(ROOT / "data/splits/validation.csv"), artifact / "predictions/validation.csv"); (artifact / "metrics.json").write_text(json.dumps({"validation": report}, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2)); return
    if not path.exists(): raise SystemExit(f"model does not exist: {path}")
    model = MiniLMPipeline.load(path)
    if args.mode == "infer":
        if not args.question: raise SystemExit("--question is required for infer")
        prediction = model.predict(args.question); print(json.dumps(prediction.__dict__ if args.diagnostics else prediction.to_list(), ensure_ascii=False)); return
    if args.mode == "validate": rows, filename = read_split(ROOT / "data/splits/validation.csv"), "validation.csv"
    elif args.mode == "evaluate": rows, filename = read_split(ROOT / "data/splits/test.csv"), "test.csv"
    else:
        validation = read_split(ROOT / "data/splits/validation.csv"); rows, filename = (validation * 1)[:100], "benchmark.csv"; model.predict_batch([row["question_raw"] for row in rows[:10]])
    report = evaluate(model, rows, artifact / "predictions" / filename)
    if args.mode == "benchmark":
        started = time.perf_counter(); cold = MiniLMPipeline.load(path); load_ms = (time.perf_counter()-started)*1000
        started = time.perf_counter(); cold.predict(rows[0]["question_raw"]); first_ms = (time.perf_counter()-started)*1000
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss); rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        report["deployment"] = {"cold_model_load_ms": load_ms, "cold_first_prediction_ms": first_ms, "serialized_head_size_bytes": path.stat().st_size, "peak_rss_bytes": rss_bytes, "threads": 1, "gpu_used": model.config.device != "cpu", "onnx_runtime_used": False, "encoder": model.config.model_name}
        (artifact / "latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    else: (artifact / "metrics.json").write_text(json.dumps({args.mode: report}, indent=2), encoding="utf-8")
    if args.mode == "evaluate":
        with (artifact / "predictions" / filename).open(encoding="utf-8", newline="") as source: errors = [row for row in csv.DictReader(source) if row["exact"] != "True"]
        with (artifact / "errors.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=errors[0].keys() if errors else ["row_id"]); writer.writeheader(); writer.writerows(errors)
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
