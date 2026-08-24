#!/usr/bin/env python3
"""Validation-only optimization for root-routed component experts."""
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


def integers(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",")]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def rank(result: dict) -> tuple[float, float, int, int]:
    return (
        result["canonical_exact_match"], -result["p95_ms"],
        -result["expert_min_rows"], -result["expert_min_field_support"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--min-rows", type=integers, default=integers("30,40,50,60,80,164"))
    parser.add_argument("--field-support", type=integers, default=integers("1,2,3,5"))
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    args = parser.parse_args()

    artifact = ROOT / "artifacts" / args.run_id
    train_rows = read_split(args.split_dir / "train.csv")
    validation_rows = read_split(args.split_dir / "validation.csv")
    model = LocalParser.train(
        Schema.load(ROOT / "data" / "fields_description.csv"), train_rows,
        train_root_experts=True,
    )
    results: list[dict] = []

    def trial(stage: str) -> dict:
        name = (
            f"{stage}_rows-{model.expert_min_rows}_support-{model.expert_min_field_support}"
            f"_fallback-{model.expert_fallback}_structure-{model.hierarchy_structure_weight:g}"
            f"_field-{model.hierarchy_field_weight:g}_mode-{model.hierarchy_field_mode}"
            f"_expert-model-{model.expert_field_model}"
        )
        metrics = evaluate(model, validation_rows, artifact / "candidates" / name / "validation.csv")
        result = {
            "stage": stage,
            "expert_min_rows": model.expert_min_rows,
            "expert_min_field_support": model.expert_min_field_support,
            "expert_fallback": model.expert_fallback,
            "structure_weight": model.hierarchy_structure_weight,
            "field_weight": model.hierarchy_field_weight,
            "field_mode": model.hierarchy_field_mode,
            "expert_field_model": model.expert_field_model,
            "canonical_exact_match": metrics["canonical_exact_match"],
            "root_entity_accuracy": metrics["root_entity_accuracy"],
            "p95_ms": metrics["latency_ms"]["p95"],
        }
        results.append(result)
        return result

    # First select which roots and fields have enough support. The previous
    # hierarchy winner supplies fixed weights so this stage changes only routing.
    model.hierarchy_structure_weight = 0.3
    model.hierarchy_field_weight = 2.0
    model.hierarchy_field_mode = "mean_probability"
    model.expert_field_model = "classifier_chain"
    routing_results: list[dict] = []
    for minimum_rows in args.min_rows:
        for minimum_support in args.field_support:
            for fallback in ("rules", "shared"):
                model.expert_min_rows = minimum_rows
                model.expert_min_field_support = minimum_support
                model.expert_fallback = fallback
                routing_results.append(trial("routing"))
    routing_winner = max((result for result in routing_results if result["expert_min_rows"] < 164), key=rank)

    # With routing fixed, optimize only a compact set of component score mixtures.
    model.expert_min_rows = routing_winner["expert_min_rows"]
    model.expert_min_field_support = routing_winner["expert_min_field_support"]
    model.expert_fallback = routing_winner["expert_fallback"]
    scoring_results: list[dict] = []
    for expert_field_model in ("classifier_chain", "independent"):
        for structure_weight, field_weight, field_mode in (
            (0.0, 1.0, "mean_probability"), (0.3, 1.0, "mean_probability"),
            (0.3, 2.0, "mean_probability"), (0.5, 1.0, "topk_f1"),
        ):
            model.expert_field_model = expert_field_model
            model.hierarchy_structure_weight = structure_weight
            model.hierarchy_field_weight = field_weight
            model.hierarchy_field_mode = field_mode
            scoring_results.append(trial("scoring"))
    winner = max(results, key=rank)

    model.expert_min_rows = winner["expert_min_rows"]
    model.expert_min_field_support = winner["expert_min_field_support"]
    model.expert_fallback = winner["expert_fallback"]
    model.hierarchy_structure_weight = winner["structure_weight"]
    model.hierarchy_field_weight = winner["field_weight"]
    model.hierarchy_field_mode = winner["field_mode"]
    model.expert_field_model = winner["expert_field_model"]
    model_path = artifact / "models" / "parser.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.dump(str(model_path))
    validation_metrics = evaluate(model, validation_rows, artifact / "predictions" / "validation.csv")

    active_experts = {
        entity: expert.row_count for entity, expert in model.root_experts.items()
        if expert.row_count >= model.expert_min_rows
    }
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "optimization.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (artifact / "config.json").write_text(json.dumps({
        "model": "root-routed component experts with confidence-by-support backoff",
        "expert_min_rows": model.expert_min_rows,
        "expert_min_field_support": model.expert_min_field_support,
        "expert_fallback": model.expert_fallback,
        "active_experts": active_experts,
        "hierarchy_structure_weight": model.hierarchy_structure_weight,
        "hierarchy_field_weight": model.hierarchy_field_weight,
        "hierarchy_field_mode": model.hierarchy_field_mode,
        "expert_field_model": model.expert_field_model,
        "selection_split": "validation",
        "training_row_count": len(train_rows),
        "training_row_ids": [row["row_id"] for row in train_rows],
    }, indent=2), encoding="utf-8")
    (artifact / "validation_metrics.json").write_text(json.dumps({"validation": validation_metrics}, indent=2), encoding="utf-8")
    write_metric_csv(artifact / "validation_metrics.csv", validation_metrics)
    with (artifact / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"winner": winner, "active_experts": active_experts, "validation": validation_metrics}, indent=2))


if __name__ == "__main__":
    main()
