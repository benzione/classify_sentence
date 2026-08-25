#!/usr/bin/env python3
"""Validation-only selection for the independent MiniLM pipeline."""
from __future__ import annotations

import argparse, csv, json, sys, tempfile, time
from dataclasses import asdict, replace
from pathlib import Path

from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from nl_api.data import read_split
from nl_api.minilm_pipeline import MiniLMConfig, MiniLMPipeline


def score(model: MiniLMPipeline, rows: list[dict]) -> tuple[float, float]:
    started = time.perf_counter(); predictions = model.predict_batch([row["question_raw"] for row in rows]); elapsed = (time.perf_counter()-started)*1000/len(rows)
    return sum(set(prediction.entities) == set(row["entity_labels"]) for prediction, row in zip(predictions, rows))/len(rows), elapsed

def grouped_oof(rows: list[dict], config: MiniLMConfig) -> float:
    predicted: dict[int, set[str]] = {}
    folds = GroupKFold(n_splits=3)
    for train_index, test_index in folds.split(rows, groups=[row["template_key"] for row in rows]):
        model = MiniLMPipeline.train([rows[i] for i in train_index], config)
        for index, result in zip(test_index, model.predict_batch([rows[i]["question_raw"] for i in test_index])): predicted[index] = set(result.entities)
    return sum(predicted[i] == set(row["entity_labels"]) for i, row in enumerate(rows))/len(rows)

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", default="minilm-001"); parser.add_argument("--device", default="cpu"); args = parser.parse_args()
    train, validation = read_split(ROOT / "data/splits/train.csv"), read_split(ROOT / "data/splits/validation.csv")
    base = MiniLMConfig(device=args.device, batch_size=32)
    candidates = [("balanced-C0.5-t0.45", replace(base, c=.5, threshold=.45)), ("balanced-C1-t0.5", replace(base, c=1, threshold=.5)), ("balanced-C1-t0.6", replace(base, c=1, threshold=.6)), ("balanced-C2-t0.5", replace(base, c=2, threshold=.5)), ("unweighted-C1-t0.5", replace(base, c=1, threshold=.5, balanced=False))]
    results = []
    for name, config in candidates:
        model = MiniLMPipeline.train(train, config); validation_exact, ms = score(model, validation); oof = grouped_oof(train, config)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "minilm_pipeline.joblib"; model.dump(temporary); size = temporary.stat().st_size
        results.append({"candidate": name, "config": json.dumps(asdict(config), sort_keys=True), "validation_exact_entity_set_match": validation_exact, "train_grouped_oof_exact_entity_set_match": oof, "per_query_ms": ms, "serialized_head_size_bytes": size})
    results.sort(key=lambda value: (-value["validation_exact_entity_set_match"], -value["train_grouped_oof_exact_entity_set_match"], value["per_query_ms"]))
    selected = results[0]
    for row in results: row["status"] = "accepted: validation/OOF winner" if row is selected else "rejected: lower validation/OOF ranking"
    artifact = ROOT / "artifacts" / args.run_id; artifact.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    with (artifact / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(results)
    config = MiniLMConfig(**json.loads(selected["config"])); model = MiniLMPipeline.train(train, config); model.dump(artifact / "models/minilm_pipeline.joblib")
    (artifact / "config.json").write_text(json.dumps(model.metadata() | {"selection": selected["candidate"], "test_data_used": False}, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "all_candidates": results}, indent=2))

if __name__ == "__main__": main()
