"""Condition-level supervision and target-free span candidates."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from .ast import ApiRequest, BooleanGroup, Filter, Relation, Sort, Statement
from .schema import Schema


@dataclass(frozen=True)
class SpanCandidate:
    start: int
    end: int
    text: str
    kind: str


@dataclass(frozen=True)
class ConditionTarget:
    root_entity: str
    scope: tuple[str, ...]
    relation_path: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    boolean_path: tuple[str, ...]
    field: str
    operator: str
    value: Any
    span_start: int | None
    span_end: int | None
    span_text: str | None
    alignment: str

    def to_dict(self, row_id: int, question: str, condition_index: int) -> dict[str, Any]:
        return {
            "row_id": row_id,
            "condition_index": condition_index,
            "question": question,
            "root_entity": self.root_entity,
            "scope": list(self.scope),
            "relation_path": [
                {"relation_type": list(relation_type), "target_type": list(target_type)}
                for relation_type, target_type in self.relation_path
            ],
            "boolean_path": list(self.boolean_path),
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "span_text": self.span_text,
            "alignment": self.alignment,
        }


def _date_value(text: str) -> str | None:
    cleaned = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text.strip(), flags=re.IGNORECASE).replace(",", "")
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _typed_candidates(question: str) -> list[SpanCandidate]:
    patterns = (
        ("quoted", r"['\"][^'\"]+['\"]"),
        ("email", r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        ("url", r"https?://[^\s,'\"]+"),
        ("relative", r"\b(?:last|past|previous)\s+(?:(?:\d+)\s+)?(?:hour|day|week|month|year)s?\b|\b(?:yesterday|today)\b"),
        ("date", r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\s+\d{2,4}\b|\b[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
        ("number", r"(?<!\w)\+?\d[\d,.:/+()-]*(?!\w)"),
        ("boolean", r"\b(?:true|false)\b"),
    )
    output: list[SpanCandidate] = []
    seen: set[tuple[int, int, str]] = set()
    for kind, pattern in patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            key = (match.start(), match.end(), kind)
            if key not in seen:
                output.append(SpanCandidate(match.start(), match.end(), match.group(0), kind)); seen.add(key)
    return output


def _clause_candidates(question: str) -> list[SpanCandidate]:
    boundaries: list[tuple[int, int]] = []
    start = 0
    for separator in re.finditer(r"\s*(?:,|;|\band\b|\bor\b)\s*", question, re.IGNORECASE):
        prefix = question[start:separator.start()].casefold()
        # The conjunction inside "between X and Y" belongs to one condition.
        if separator.group(0).strip().casefold() == "and" and re.search(r"\bbetween\b", prefix):
            continue
        boundaries.append((start, separator.start())); start = separator.end()
    boundaries.append((start, len(question)))
    output = []
    for left, right in boundaries:
        while left < right and question[left].isspace(): left += 1
        while right > left and question[right - 1].isspace(): right -= 1
        if left < right: output.append(SpanCandidate(left, right, question[left:right], "clause"))
    return output


def candidate_spans(question: str) -> list[SpanCandidate]:
    """Return deterministic inference candidates without reading target labels."""
    candidates = _typed_candidates(question) + _clause_candidates(question)
    return sorted(candidates, key=lambda item: (item.start, item.end, item.kind))


def _exact_span(question: str, value: Any) -> SpanCandidate | None:
    if not isinstance(value, (str, int, float, bool)) or value == "":
        return None
    raw = str(value)
    match = re.search(r"(?<!\w)" + re.escape(raw) + r"(?!\w)", question, re.IGNORECASE)
    return SpanCandidate(match.start(), match.end(), match.group(0), "exact") if match else None


def _normalized_span(question: str, value: Any, candidates: list[SpanCandidate]) -> SpanCandidate | None:
    if isinstance(value, str) and _date_value(value):
        target = _date_value(value)
        for candidate in candidates:
            if candidate.kind == "date" and _date_value(candidate.text) == target:
                return SpanCandidate(candidate.start, candidate.end, candidate.text, "normalized")
    if isinstance(value, list) and value and all(isinstance(item, str) and _date_value(item) for item in value):
        dates = [candidate for candidate in candidates if candidate.kind == "date" and _date_value(candidate.text) in value]
        if len({_date_value(candidate.text) for candidate in dates}) == len(set(value)):
            return SpanCandidate(min(item.start for item in dates), max(item.end for item in dates), question[min(item.start for item in dates):max(item.end for item in dates)], "normalized")
    if isinstance(value, dict) and value.get("mode") == "previous":
        target_count = int(value.get("count", 1)); target_unit = str(value.get("time_res", "")).casefold().rstrip("s")
        for candidate in candidates:
            if candidate.kind != "relative": continue
            text = candidate.text.casefold()
            if text == "yesterday" and target_count == 1 and target_unit == "day": return SpanCandidate(candidate.start, candidate.end, candidate.text, "normalized")
            match = re.search(r"\b(?:last|past|previous)\s+(?:(\d+)\s+)?(hour|day|week|month|year)s?\b", text)
            if match and int(match.group(1) or 1) == target_count and match.group(2) == target_unit:
                return SpanCandidate(candidate.start, candidate.end, candidate.text, "normalized")
    return None


def _lexical_span(question: str, field: str, schema: Schema, candidates: list[SpanCandidate]) -> SpanCandidate | None:
    definition = schema.by_field.get(field)
    if definition is None: return None
    field_tokens = set(re.findall(r"[a-z0-9]+", f"{field.rsplit('.', 1)[-1]} {definition.description}".casefold()))
    field_tokens = {token for token in field_tokens if len(token) >= 3}
    ranked: list[tuple[int, int, SpanCandidate]] = []
    for candidate in candidates:
        if candidate.kind != "clause": continue
        tokens = set(re.findall(r"[a-z0-9]+", candidate.text.casefold()))
        overlap = len(tokens & field_tokens)
        if overlap: ranked.append((overlap, -len(candidate.text), candidate))
    if not ranked: return None
    selected = max(ranked, key=lambda item: (item[0], item[1], -item[2].start))[2]
    return SpanCandidate(selected.start, selected.end, selected.text, "lexical")


def condition_targets(question: str, request: ApiRequest, schema: Schema) -> tuple[ConditionTarget, ...]:
    """Decompose a gold AST and align every filter without dropping failures."""
    candidates = candidate_spans(question)
    output: list[ConditionTarget] = []

    def visit(
        statements: Iterable[Statement],
        scope: tuple[str, ...],
        relation_path: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
        boolean_path: tuple[str, ...],
    ) -> None:
        for statement in statements:
            if isinstance(statement, Filter):
                span = _exact_span(question, statement.value)
                alignment = "exact"
                if span is None:
                    span = _normalized_span(question, statement.value, candidates); alignment = "normalized"
                if span is None:
                    span = _lexical_span(question, statement.name, schema, candidates); alignment = "lexical"
                if span is None: alignment = "unaligned"
                output.append(ConditionTarget(
                    request.entity_type, scope, relation_path, boolean_path,
                    statement.name, statement.operator, statement.value,
                    span.start if span else None, span.end if span else None,
                    span.text if span else None, alignment,
                ))
            elif isinstance(statement, Relation):
                relation = (statement.relation_type, statement.target_type)
                visit(statement.statements, statement.target_type, relation_path + (relation,), boolean_path)
            elif isinstance(statement, BooleanGroup):
                visit(statement.statements, scope, relation_path, boolean_path + (statement.operator,))
            elif isinstance(statement, Sort):
                continue
    visit(request.statements, (request.entity_type,), (), ())
    return tuple(output)


def json_line(target: ConditionTarget, row_id: int, question: str, condition_index: int) -> str:
    return json.dumps(target.to_dict(row_id, question, condition_index), ensure_ascii=False, sort_keys=True)
