#!/usr/bin/env python3
"""Export auditable condition-level supervision from a labeled split."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl_api.ast import ApiRequest
from nl_api.data import read_split
from nl_api.schema import Schema
from nl_api.supervision import candidate_spans, condition_targets, json_line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "components")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output_dir / f"{args.split}_conditions.jsonl"
    report_path = args.output_dir / f"{args.split}_alignment_report.json"
    if not args.force and (output.exists() or report_path.exists()):
        raise SystemExit("component outputs already exist; pass --force to replace them")

    rows = read_split(args.split_dir / f"{args.split}.csv")
    schema = Schema.load(ROOT / "data" / "fields_description.csv")
    counts: Counter[str] = Counter(); by_root: dict[str, Counter[str]] = defaultdict(Counter)
    lines: list[str] = []
    aligned_recalled = aligned_total = 0
    for row in rows:
        request = ApiRequest.from_dict(json.loads(row["target_json"]))
        targets = condition_targets(row["question_raw"], request, schema)
        if len(targets) != row["filter_count"]:
            raise RuntimeError(f"row {row['row_id']}: decomposed {len(targets)} filters, expected {row['filter_count']}")
        candidates = candidate_spans(row["question_raw"])
        for index, target in enumerate(targets):
            lines.append(json_line(target, row["row_id"], row["question_raw"], index))
            counts[target.alignment] += 1; by_root[target.root_entity][target.alignment] += 1
            if target.alignment in {"exact", "normalized"}:
                aligned_total += 1
                if any(candidate.start <= target.span_start and candidate.end >= target.span_end for candidate in candidates): aligned_recalled += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "split": args.split,
        "request_count": len(rows),
        "condition_count": len(lines),
        "alignment": dict(sorted(counts.items())),
        "alignment_by_root": {root: dict(sorted(values.items())) for root, values in sorted(by_root.items())},
        "candidate_recall_on_exact_or_normalized": aligned_recalled / aligned_total if aligned_total else 0.0,
        "candidate_recalled": aligned_recalled,
        "candidate_target_count": aligned_total,
        "training_row_ids": [row["row_id"] for row in rows] if args.split == "train" else None,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
