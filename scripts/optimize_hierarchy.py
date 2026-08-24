#!/usr/bin/env python3
"""Validation-only search for hierarchical retrieval weights."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nl_api.data import read_split
from nl_api.pipeline import LocalParser
from nl_api.schema import Schema
from run_pipeline import evaluate, write_metric_csv


def weights(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",")]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("weights must be a comma-separated list of non-negative numbers")
    return values


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--run-id", required=True)
    argument_parser.add_argument("--structure-weights", type=weights, default=weights("0,0.1,0.25,0.5,0.75,1"))
    argument_parser.add_argument("--field-weights", type=weights, default=weights("0,0.1,0.25,0.5,0.75,1"))
    argument_parser.add_argument("--field-mode", choices=("mean_probability", "topk_f1"), default="mean_probability")
    argument_parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    args = argument_parser.parse_args()

    artifact = ROOT / "artifacts" / args.run_id
    train_rows = read_split(args.split_dir / "train.csv")
    validation_rows = read_split(args.split_dir / "validation.csv")
    model = LocalParser.train(Schema.load(ROOT / "data" / "fields_description.csv"), train_rows)
    model.hierarchy_field_mode = args.field_mode
    results: list[dict] = []
    best: tuple[float, float, float, float, dict] | None = None

    for structure_weight in args.structure_weights:
        for field_weight in args.field_weights:
            model.hierarchy_structure_weight = structure_weight
            model.hierarchy_field_weight = field_weight
            name = f"structure-{structure_weight:g}_field-{field_weight:g}"
            metrics = evaluate(model, validation_rows, artifact / "candidates" / name / "validation.csv")
            result = {
                "structure_weight": structure_weight,
                "field_weight": field_weight,
                "canonical_exact_match": metrics["canonical_exact_match"],
                "root_entity_accuracy": metrics["root_entity_accuracy"],
                "p95_ms": metrics["latency_ms"]["p95"],
            }
            results.append(result)
            rank = (result["canonical_exact_match"], -result["p95_ms"], -structure_weight, -field_weight)
            if best is None or rank > best[:4]:
                best = (*rank, metrics)

    assert best is not None
    winning = max(
        results,
        key=lambda result: (
            result["canonical_exact_match"], -result["p95_ms"],
            -result["structure_weight"], -result["field_weight"],
        ),
    )
    model.hierarchy_structure_weight = winning["structure_weight"]
    model.hierarchy_field_weight = winning["field_weight"]
    model_path = artifact / "models" / "parser.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.dump(str(model_path))
    validation_metrics = evaluate(model, validation_rows, artifact / "predictions" / "validation.csv")
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "optimization.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (artifact / "config.json").write_text(json.dumps({
        "model": "hierarchical structure and field conditioned retrieval",
        "hierarchy_structure_weight": winning["structure_weight"],
        "hierarchy_field_weight": winning["field_weight"],
        "hierarchy_field_mode": args.field_mode,
        "selection_split": "validation",
        "training_row_count": len(train_rows),
        "training_row_ids": [row["row_id"] for row in train_rows],
    }, indent=2), encoding="utf-8")
    (artifact / "metrics.json").write_text(json.dumps({"validation": validation_metrics}, indent=2), encoding="utf-8")
    write_metric_csv(artifact / "metrics.csv", validation_metrics)
    with (artifact / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"winner": winning, "validation": validation_metrics}, indent=2))


if __name__ == "__main__":
    main()
