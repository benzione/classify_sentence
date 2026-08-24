#!/usr/bin/env python3
"""Train and validate a shared semantic condition-to-schema field linker."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl_api.field_linker import FieldLinker, MulticlassFieldLinker, condition_text, field_card, lexical_overlap, pair_features
from nl_api.schema import Schema


def read_jsonl(path: Path, alignments: set[str]) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return [record for record in records if record["alignment"] in alignments and record["span_text"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--alignments", default="exact,normalized")
    parser.add_argument("--negative-count", type=int, default=5)
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--ranker", choices=("pairwise", "multiclass"), default="pairwise")
    args = parser.parse_args()
    if args.negative_count < 1: raise SystemExit("--negative-count must be positive")
    alignments = set(args.alignments.split(","))
    if not alignments <= {"exact", "normalized", "lexical"}: raise SystemExit("unsupported alignment confidence")

    schema = Schema.load(ROOT / "data" / "fields_description.csv")
    component_dir = ROOT / "data" / "components"
    train = read_jsonl(component_dir / "train_conditions.jsonl", alignments)
    validation = read_jsonl(component_dir / "validation_conditions.jsonl", {"exact", "normalized"})
    encoder = SentenceTransformer(args.encoder, device="cpu")
    cards = {field.name: field_card(field) for field in schema.fields}
    card_names = sorted(cards)
    card_matrix = encoder.encode([cards[name] for name in card_names], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    field_embeddings = {name: embedding for name, embedding in zip(card_names, card_matrix)}

    train_texts = [condition_text(record) for record in train]
    train_embeddings = encoder.encode(train_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    if args.ranker == "pairwise":
        features: list[np.ndarray] = []; labels: list[int] = []
        for record, query_embedding, query_text in zip(train, train_embeddings, train_texts):
            field = record["field"]
            features.append(pair_features(query_embedding, field_embeddings[field], lexical_overlap(query_text, cards[field]))); labels.append(1)
            allowed = schema.by_entity.get(record["scope"][0], {})
            negatives = sorted(
                (candidate for candidate in allowed if candidate != field),
                key=lambda candidate: (-float(np.dot(query_embedding, field_embeddings[candidate])), candidate),
            )[:args.negative_count]
            for candidate in negatives:
                features.append(pair_features(query_embedding, field_embeddings[candidate], lexical_overlap(query_text, cards[candidate]))); labels.append(0)
        classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20260824)
        classifier.fit(np.vstack(features), labels)
        linker = FieldLinker(schema, args.encoder, classifier, field_embeddings)
    else:
        classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=20260824)
        classifier.fit(train_embeddings, [record["field"] for record in train])
        linker = MulticlassFieldLinker(schema, args.encoder, classifier)

    start = time.perf_counter()
    validation_texts = [condition_text(record) for record in validation]
    validation_embeddings = encoder.encode(validation_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    encoding_ms = (time.perf_counter() - start) * 1000
    counts: Counter[str] = Counter(); by_root: dict[str, Counter[str]] = defaultdict(Counter); predictions = []
    for record, embedding in zip(validation, validation_embeddings):
        ranking = linker.rank(record, embedding); fields = [field for field, _ in ranking]
        top1 = bool(fields and fields[0] == record["field"]); top3 = record["field"] in fields[:3]
        counts["samples"] += 1; counts["top1"] += top1; counts["top3"] += top3
        by_root[record["root_entity"]]["samples"] += 1; by_root[record["root_entity"]]["top1"] += top1; by_root[record["root_entity"]]["top3"] += top3
        predictions.append({
            "row_id": record["row_id"], "condition_index": record["condition_index"],
            "root_entity": record["root_entity"], "scope": json.dumps(record["scope"]),
            "span_text": record["span_text"], "gold_field": record["field"],
            "predicted_field": fields[0] if fields else "", "top3": json.dumps(fields[:3]),
            "correct": top1,
        })
    metrics = {
        "validation_samples": counts["samples"],
        "field_top1_accuracy": counts["top1"] / counts["samples"] if counts["samples"] else 0.0,
        "field_top3_accuracy": counts["top3"] / counts["samples"] if counts["samples"] else 0.0,
        "encoding_ms_total": encoding_ms,
        "encoding_ms_per_condition": encoding_ms / len(validation) if validation else 0.0,
        "by_root": {
            root: {
                "samples": values["samples"],
                "top1_accuracy": values["top1"] / values["samples"],
                "top3_accuracy": values["top3"] / values["samples"],
            } for root, values in sorted(by_root.items())
        },
    }
    artifact = ROOT / "artifacts" / args.run_id; (artifact / "models").mkdir(parents=True, exist_ok=True); (artifact / "predictions").mkdir(parents=True, exist_ok=True)
    joblib.dump(linker, artifact / "models" / "field_linker.joblib")
    (artifact / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (artifact / "config.json").write_text(json.dumps({
        "model": f"shared MiniLM condition embeddings plus {args.ranker} logistic ranker",
        "encoder": args.encoder, "alignments": sorted(alignments), "negative_count": args.negative_count,
        "ranker": args.ranker,
        "training_condition_count": len(train), "training_row_ids": sorted({record["row_id"] for record in train}),
        "selection_split": "validation",
    }, indent=2), encoding="utf-8")
    with (artifact / "predictions" / "validation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=predictions[0].keys()); writer.writeheader(); writer.writerows(predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
