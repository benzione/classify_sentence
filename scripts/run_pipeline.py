#!/usr/bin/env python3
"""Train, evaluate, benchmark, or locally infer with the constrained parser."""
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl_api.ast import ApiRequest
from nl_api.data import read_split
from nl_api.pipeline import LocalParser
from nl_api.schema import Schema


def locations(run_id: str) -> tuple[Path, Path]:
    artifact = ROOT / "artifacts" / run_id
    return artifact, artifact / "models" / "parser.joblib"


def evaluate(parser: LocalParser, rows: list[dict], output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True); predictions = []
    exact = root = valid = 0; latencies = []
    for row in rows:
        start = time.perf_counter(); prediction = parser.predict(row["question_raw"]); latencies.append((time.perf_counter() - start) * 1000)
        gold = ApiRequest.from_dict(json.loads(row["target_json"])); predicted = prediction.request
        is_exact = gold.canonical_json() == predicted.canonical_json(); exact += is_exact; root += gold.entity_type == predicted.entity_type
        # predict() already validates and propagates any schema failure.  Metrics must
        # never turn a bad request into a successful-looking run.
        parser.schema.validate(predicted); valid += 1
        category = "correct" if is_exact else "wrong_root" if gold.entity_type != predicted.entity_type else "wrong_structure_or_value"
        predictions.append({"row_id": row["row_id"], "question": row["question_raw"], "gold_json": gold.canonical_json(), "predicted_json": predicted.canonical_json(), "confidence": prediction.confidence, "error_category": category, "diagnostics": json.dumps(prediction.diagnostics, sort_keys=True)})
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=predictions[0].keys() if predictions else []); writer.writeheader(); writer.writerows(predictions)
    ordered = sorted(latencies)
    return {"samples": len(rows), "canonical_exact_match": exact / len(rows) if rows else 0, "root_entity_accuracy": root / len(rows) if rows else 0, "strict_json_validity": 1.0, "schema_validity": valid / len(rows) if rows else 0, "latency_ms": {"p50": ordered[len(ordered)//2] if ordered else 0, "p95": ordered[min(len(ordered)-1, int(len(ordered)*.95))] if ordered else 0}}


def write_metric_csv(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("metric", "value"))
        for key, value in metrics.items():
            if isinstance(value, dict):
                for nested, nested_value in value.items(): writer.writerow((f"{key}.{nested}", nested_value))
            else: writer.writerow((key, value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "evaluate", "infer", "benchmark"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--question")
    parser.add_argument("--output-api-only", action="store_true")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--slot-threshold", type=float, default=1.0)
    parser.add_argument("--slot-mode", choices=("off", "augment", "replace"), default="off")
    parser.add_argument("--hierarchy-structure-weight", type=float, default=0.0)
    parser.add_argument("--hierarchy-field-weight", type=float, default=0.0)
    parser.add_argument("--hierarchy-field-mode", choices=("mean_probability", "topk_f1"), default="mean_probability")
    args = parser.parse_args()
    artifact, model_path = locations(args.run_id)
    if args.mode == "train":
        train = read_split(args.split_dir / "train.csv"); validation = read_split(args.split_dir / "validation.csv")
        schema = Schema.load(ROOT / "data" / "fields_description.csv")
        model = LocalParser.train(
            schema, train, args.slot_threshold, args.slot_mode,
            args.hierarchy_structure_weight, args.hierarchy_field_weight,
            args.hierarchy_field_mode,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True); model.dump(str(model_path))
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "config.json").write_text(json.dumps({"model": "word+char TF-IDF / balanced LogisticRegression / hierarchical schema-constrained retrieval", "canonicalization": "sort implicit root AND filters only", "slot_mode": args.slot_mode, "slot_threshold": args.slot_threshold, "hierarchy_structure_weight": args.hierarchy_structure_weight, "hierarchy_field_weight": args.hierarchy_field_weight, "hierarchy_field_mode": args.hierarchy_field_mode, "training_row_count": len(train), "training_row_ids": [row["row_id"] for row in train]}, indent=2), encoding="utf-8")
        metrics = evaluate(model, validation, artifact / "predictions" / "validation.csv")
        (artifact / "metrics.json").write_text(json.dumps({"validation": metrics}, indent=2), encoding="utf-8")
        write_metric_csv(artifact / "metrics.csv", metrics)
        with (artifact / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(("version", "components", "validation_exact_match", "validation_p95_ms")); writer.writerow(("sparse-baseline", "word+char TF-IDF, balanced logistic root classifier, train-only cosine retrieval", metrics["canonical_exact_match"], metrics["latency_ms"]["p95"]))
        print(json.dumps(metrics, indent=2)); return
    if not model_path.exists(): raise SystemExit(f"model does not exist: run train first ({model_path})")
    model = LocalParser.load(str(model_path))
    if args.mode == "infer":
        if not args.question: raise SystemExit("--question is required for infer")
        prediction = model.predict(args.question)
        result = prediction.request.to_dict() if args.output_api_only else {"request": prediction.request.to_dict(), "confidence": prediction.confidence, "model_version": args.run_id}
        print(json.dumps(result, ensure_ascii=False)); return
    if args.mode == "evaluate":
        rows = read_split(args.split_dir / "test.csv")
    else:
        # At least 100 warmed local predictions, without accidentally multiplying
        # an entire split and turning a latency benchmark into an I/O benchmark.
        validation_rows = read_split(args.split_dir / "validation.csv")
        rows = (validation_rows * ((100 + len(validation_rows) - 1) // len(validation_rows)))[:100]
    report = evaluate(model, rows, artifact / "predictions" / ("test.csv" if args.mode == "evaluate" else "benchmark.csv"))
    (artifact / ("metrics.json" if args.mode == "evaluate" else "latency.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.mode == "evaluate":
        # Keep the complete audit trail under predictions/ and provide a genuinely
        # filtered error report for human review.
        prediction_path = artifact / "predictions" / "test.csv"
        with prediction_path.open("r", encoding="utf-8", newline="") as stream:
            all_predictions = list(csv.DictReader(stream))
        with (artifact / "errors.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=all_predictions[0].keys() if all_predictions else [])
            writer.writeheader()
            writer.writerows(row for row in all_predictions if row["error_category"] != "correct")
        write_metric_csv(artifact / "metrics.csv", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
