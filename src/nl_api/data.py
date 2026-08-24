from __future__ import annotations

import ast, csv, hashlib, json, re
from collections import Counter
from pathlib import Path
from typing import Any

from .ast import ApiRequest, BooleanGroup, Filter, Relation, Sort, Statement
from .schema import Schema


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def template_key(question: str) -> str:
    text = normalize_question(question)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", "<email>", text)
    text = re.sub(r"\b(?:\+?\d[\d .()-]{5,}\d)\b", "<phone>", text)
    text = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", "<date>", text)
    text = re.sub(r"(['\"]).*?\1", "<quoted>", text)
    return re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", text)


def source_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def structural_features(request: ApiRequest) -> dict[str, Any]:
    fields: list[str] = []; operators: list[str] = []; targets: list[str] = []; count = 0; has_or = False
    def visit(items: tuple[Statement, ...]) -> None:
        nonlocal count, has_or
        for item in items:
            if isinstance(item, Filter): fields.append(item.name); operators.append(item.operator); count += 1
            elif isinstance(item, Sort): fields.append(item.name)
            elif isinstance(item, Relation): targets.extend(item.target_type); visit(item.statements)
            elif isinstance(item, BooleanGroup): has_or = True; visit(item.statements)
    visit(request.statements)
    return {"root_entity": request.entity_type, "entity_labels": sorted(request.entity_labels()), "relation_targets": sorted(set(targets)), "has_relation": bool(targets), "has_or": has_or, "filter_count": count, "fields": sorted(set(fields)), "operators": sorted(set(operators))}


def load_records(query_csv: str | Path, schema: Schema) -> list[dict[str, Any]]:
    path = Path(query_csv)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or {"question", "json"} - set(reader.fieldnames):
            raise ValueError("user_queries.csv must have question and json columns")
        records = []
        for number, row in enumerate(reader, start=2):
            raw = row["question"]
            if not raw or not raw.strip(): raise ValueError(f"row {number}: blank question")
            try: target = ast.literal_eval(row["json"])
            except (SyntaxError, ValueError) as exc: raise ValueError(f"row {number}: invalid Python literal target: {exc}") from exc
            request = ApiRequest.from_dict(target); schema.validate(request)
            record = {"row_id": number - 2, "question_raw": raw, "question_normalized": normalize_question(raw), "template_key": template_key(raw), "target_json": request.canonical_json()}
            record.update(structural_features(request)); records.append(record)
    if not records: raise ValueError("query CSV has no rows")
    return records


def write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["row_id", "question_raw", "question_normalized", "template_key", "target_json", "root_entity", "entity_labels", "relation_targets", "has_relation", "has_or", "filter_count", "fields", "operators"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (list, dict)) else row[key] for key in keys})


def read_split(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["row_id"] = int(row["row_id"]); row["target_json"] = row["target_json"]
        for key in ("entity_labels", "relation_targets", "fields", "operators"): row[key] = json.loads(row[key])
        for key in ("has_relation", "has_or"): row[key] = row[key].lower() == "true"
        row["filter_count"] = int(row["filter_count"])
    return rows
