#!/usr/bin/env python3
"""Validation-only selection for the local entity extractor.

The compact candidate list changes one resource/threshold family at a time;
the held-out test split is intentionally never read here.
"""
from __future__ import annotations

import csv, json, sys, tempfile, time
from dataclasses import asdict, replace
from pathlib import Path

from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from nl_api.data import read_split
from nl_api.entity_pipeline import EntityConfig, EntityPipeline


def score(model: EntityPipeline, rows: list[dict]) -> tuple[float, float]:
    start = time.perf_counter(); predicted = model.predict_batch([r["question_raw"] for r in rows]); elapsed = (time.perf_counter()-start)*1000/len(rows)
    exact = sum(set(p.entities) == set(r["entity_labels"]) for p, r in zip(predicted, rows))/len(rows)
    return exact, elapsed

def oof_score(rows: list[dict], config: EntityConfig) -> float:
    groups = [r["template_key"] for r in rows]; folds = GroupKFold(n_splits=3)
    predictions: dict[int, set[str]] = {}
    for train_i, test_i in folds.split(rows, groups=groups):
        model = EntityPipeline.train([rows[i] for i in train_i], config)
        for i, prediction in zip(test_i, model.predict_batch([rows[i]["question_raw"] for i in test_i])): predictions[i] = set(prediction.entities)
    return sum(predictions[i] == set(row["entity_labels"]) for i, row in enumerate(rows))/len(rows)

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", default="entity-001"); args = parser.parse_args()
    train, validation = read_split(ROOT/"data/splits/train.csv"), read_split(ROOT/"data/splits/validation.csv")
    base = EntityConfig(max_features=20_000, c=1, threshold=.5)
    candidates = [
        ("A-root-reference", replace(base, family="root")),
        ("B-direct-10k", replace(base, family="direct", max_features=10_000)),
        ("B-direct-threshold-0.6", replace(base, family="direct", threshold=.6)),
        ("B-direct-char-3-6-C2", replace(base, family="direct", char_ngrams=(3,6), c=2)),
        ("C-hierarchical", replace(base, family="hierarchical")),
        ("D-label-powerset", replace(base, family="powerset")),
        ("H-semantic-boost-0.1", replace(base, family="direct", threshold=.6, semantic_keyword_boost=.1)),
        ("H-semantic-boost-0.2", replace(base, family="direct", threshold=.6, semantic_keyword_boost=.2)),
        ("H-semantic-boost-0.3", replace(base, family="direct", threshold=.6, semantic_keyword_boost=.3)),
        ("H-semantic-relation-0.1", replace(base, family="direct", threshold=.6, semantic_keyword_boost=.3, semantic_relation_boost=.1)),
        ("H-semantic-relation-0.2", replace(base, family="direct", threshold=.6, semantic_keyword_boost=.3, semantic_relation_boost=.2)),
        ("H-semantic-relation-0.3", replace(base, family="direct", threshold=.6, semantic_keyword_boost=.3, semantic_relation_boost=.3)),
        ("H-semantic-filter-boost-0.2", replace(base, family="direct", threshold=.6, semantic_keyword_boost=.2, semantic_keyword_filter=True)),
        ("H-semantic-filter-min2-boost-0.2", replace(base, family="direct", threshold=.6, semantic_keyword_boost=.2, semantic_keyword_filter=True, semantic_keyword_min_hits=2)),
    ]
    results = []
    for name, config in candidates:
        model = EntityPipeline.train(train, config); validation_exact, ms = score(model, validation); oof = oof_score(train, config)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "entity_pipeline.joblib"; model.dump(temporary); size = temporary.stat().st_size
        results.append({"candidate": name, "family": config.family, "config": json.dumps(asdict(config), sort_keys=True), "validation_exact_entity_set_match": validation_exact, "train_grouped_oof_exact_entity_set_match": oof, "per_query_ms": ms, "serialized_size_bytes": size})
    # Tie-break order is validation exact, OOF exact, then latency.  Persist only
    # the frozen winner; it is the sole model allowed to see the test set later.
    results.sort(key=lambda r: (-r["validation_exact_entity_set_match"], -r["train_grouped_oof_exact_entity_set_match"], r["per_query_ms"]))
    selected = results[0]
    for row in results:
        row["status"] = "accepted: validation/OOF winner" if row is selected else "rejected: lower validation/OOF ranking"
    artifact = ROOT/"artifacts"/args.run_id; artifact.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    config_dict = json.loads(selected["config"])
    config_dict["word_ngrams"] = tuple(config_dict["word_ngrams"])
    config_dict["char_ngrams"] = tuple(config_dict["char_ngrams"])
    winner_config = EntityConfig(**config_dict); winner = EntityPipeline.train(train, winner_config); path=artifact/"models"/"entity_pipeline.joblib"; winner.dump(path)
    with (artifact/"ablation_summary.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(results)
    (artifact/"config.json").write_text(json.dumps(winner.metadata() | {"selection": selected["candidate"], "test_data_used": False},indent=2),encoding="utf-8")
    print(json.dumps({"selected": selected, "all_candidates": results}, indent=2))

if __name__ == "__main__": main()
