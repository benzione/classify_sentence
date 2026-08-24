#!/usr/bin/env python3
"""Validate immutable inputs and create one deterministic, group-safe split."""
from __future__ import annotations

import argparse, json, random, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl_api.data import load_records, source_fingerprint, write_records
from nl_api.schema import Schema

SEED = 20260824
RATIOS = (0.70, 0.15, 0.15)


def labels(row: dict) -> set[str]:
    return {f"root={row['root_entity']}", f"relation={row['has_relation']}", f"or={row['has_or']}", f"filters={min(row['filter_count'], 4)}", *(f"target={x}" for x in row["relation_targets"])}


def make_candidate(groups: list[list[dict]], seed: int) -> list[list[dict]]:
    """Greedily assign whole template groups, prioritising rare-label balance."""
    rng = random.Random(seed); shuffled = list(groups); rng.shuffle(shuffled)
    shuffled.sort(key=lambda group: (max(sum(label in labels(row) for row in group) for label in set().union(*(labels(r) for r in group))), len(group)), reverse=True)
    total = sum(map(len, groups)); targets = [total * ratio for ratio in RATIOS]
    all_labels = Counter(label for group in groups for row in group for label in labels(row))
    result = [[], [], []]; counts = [Counter(), Counter(), Counter()]; sizes = [0, 0, 0]
    # The assignment's rare-class constraint is imposed before generic balancing.
    # This operates on whole groups, never individual examples.
    rare_groups = [group for group in shuffled if any(row["root_entity"] == "EVisa Request" for row in group)]
    if sum(sum(row["root_entity"] == "EVisa Request" for row in group) for group in rare_groups) >= 8:
        rng.shuffle(rare_groups)
        required = (6, 1, 1); assigned = [0, 0, 0]
        for group in rare_groups:
            index = next((i for i in range(3) if assigned[i] < required[i]), min(range(3), key=lambda i: assigned[i]))
            result[index].extend(group); counts[index].update(label for row in group for label in labels(row)); sizes[index] += len(group)
            assigned[index] += sum(row["root_entity"] == "EVisa Request" for row in group)
        shuffled = [group for group in shuffled if group not in rare_groups]
    for group in shuffled:
        group_counts = Counter(label for row in group for label in labels(row))
        def cost(index: int) -> float:
            size_cost = ((sizes[index] + len(group) - targets[index]) / max(targets[index], 1)) ** 2
            label_cost = sum(((counts[index][label] + value - all_labels[label] * RATIOS[index]) / max(all_labels[label], 1)) ** 2 for label, value in group_counts.items())
            # Split-size compliance is a hard operational requirement; label
            # distribution is the secondary objective once capacity is respected.
            return 10_000 * size_cost + label_cost
        # Capacity is the primary constraint. The previous squared-distance
        # objective could get stuck after rare-class seeding because it compared
        # one-step overshoots instead of each split's remaining capacity.
        index = min(range(3), key=lambda i: (sizes[i] / max(targets[i], 1), cost(i)))
        result[index].extend(group); counts[index].update(group_counts); sizes[index] += len(group)
    return result


def valid(parts: list[list[dict]]) -> bool:
    # EVisa is specifically constrained by the blueprint if it exists in input.
    visa = [sum(row["root_entity"] == "EVisa Request" for row in part) for part in parts]
    return visa[0] >= 6 and visa[1] >= 1 and visa[2] >= 1


def score(parts: list[list[dict]], records: list[dict]) -> float:
    all_counts = Counter(label for row in records for label in labels(row)); total = len(records)
    value = 0.0
    for index, part in enumerate(parts):
        observed = Counter(label for row in part for label in labels(row))
        for label, overall in all_counts.items():
            value += ((observed[label] / max(len(part), 1)) - (overall / total)) ** 2
        value += 10_000 * ((len(part) / total) - RATIOS[index]) ** 2
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force-resplit", action="store_true")
    parser.add_argument("--candidates", type=int, default=400)
    args = parser.parse_args()
    split_dir = args.data_dir / "splits"
    if split_dir.exists() and any(split_dir.iterdir()):
        if not args.force_resplit: raise SystemExit(f"{split_dir} already exists; use --force-resplit to replace it")
        shutil.rmtree(split_dir)
    schema_path, queries_path = args.data_dir / "fields_description.csv", args.data_dir / "user_queries.csv"
    schema = Schema.load(schema_path); records = load_records(queries_path, schema)
    grouped = defaultdict(list)
    for record in records: grouped[record["template_key"]].append(record)
    groups = list(grouped.values())
    candidates = [make_candidate(groups, args.seed + i) for i in range(args.candidates)]
    viable = [parts for parts in candidates if valid(parts)]
    if not viable: raise SystemExit("no group-wise split satisfies the EVisa minimums; increase --candidates or inspect groups")
    best = min(viable, key=lambda parts: score(parts, records))
    names = ("train", "validation", "test")
    for name, rows in zip(names, best): write_records(split_dir / f"{name}.csv", sorted(rows, key=lambda x: x["row_id"]))
    manifest = {"algorithm": "seeded group-wise greedy candidate search", "seed": args.seed, "candidates": args.candidates, "source_sha256": source_fingerprint([queries_path, schema_path]), "group_key": "placeholder template_key", "groups": {name: sorted({row["template_key"] for row in rows}) for name, rows in zip(names, best)}, "row_ids": {name: [row["row_id"] for row in rows] for name, rows in zip(names, best)}}
    report = {"counts": {name: len(rows) for name, rows in zip(names, best)}, "root_entities": {name: dict(sorted(Counter(row["root_entity"] for row in rows).items())) for name, rows in zip(names, best)}, "objective": score(best, records)}
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (split_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
