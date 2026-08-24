"""Train-only decomposition of API ASTs into supervised parser components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from .ast import ApiRequest, BooleanGroup, Filter, Relation, Sort, Statement


@dataclass(frozen=True)
class FilterComponent:
    scope: tuple[str, ...]
    field: str
    operator: str
    value: Any


@dataclass(frozen=True)
class RequestComponents:
    root_entity: str
    filters: tuple[FilterComponent, ...]
    relations: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    has_or: bool
    sorts: tuple[tuple[str, str], ...]


def decompose(request: ApiRequest) -> RequestComponents:
    filters: list[FilterComponent] = []; relations: list[tuple[tuple[str, ...], tuple[str, ...]]] = []; sorts: list[tuple[str, str]] = []; has_or = False
    def visit(items: tuple[Statement, ...], scope: tuple[str, ...]) -> None:
        nonlocal has_or
        for item in items:
            if isinstance(item, Filter): filters.append(FilterComponent(scope, item.name, item.operator, item.value))
            elif isinstance(item, Sort): sorts.append((item.name, item.direction))
            elif isinstance(item, Relation):
                relations.append((item.relation_type, item.target_type)); visit(item.statements, item.target_type)
            elif isinstance(item, BooleanGroup): has_or = True; visit(item.statements, scope)
    visit(request.statements, (request.entity_type,))
    return RequestComponents(request.entity_type, tuple(filters), tuple(relations), has_or, tuple(sorts))


def skeleton(component: RequestComponents) -> tuple:
    """Value-free, deterministic AST target for structural prediction."""
    return (component.root_entity, tuple(sorted((x.scope, x.field, x.operator) for x in component.filters)), tuple(sorted(component.relations)), component.has_or, tuple(sorted(component.sorts)))


def _window(question: str, start: int, end: int, radius: int = 36) -> str:
    """Return a compact condition context without crossing strong punctuation."""
    left = max(question.rfind(";", 0, start), question.rfind("?", 0, start), question.rfind(".", 0, start)) + 1
    right_candidates = [position for delimiter in (";", "?", ".") if (position := question.find(delimiter, end)) >= 0]
    right = min(right_candidates) if right_candidates else len(question)
    return question[max(left, start - radius):min(right, end + radius)].strip()


def aligned_filter_examples(question: str, request: ApiRequest) -> list[tuple[str, FilterComponent]]:
    """Create auditable training examples only when a literal AST value occurs."""
    output: list[tuple[str, FilterComponent]] = []
    lowered = question.casefold()
    for component in decompose(request).filters:
        if not isinstance(component.value, (str, int, float)):
            continue
        value = str(component.value).casefold()
        if not value:
            continue
        match = re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", lowered)
        if match:
            output.append((_window(question, match.start(), match.end()), component))
    return output


def literal_windows(question: str) -> list[str]:
    """Produce deterministic inference contexts around typed literal candidates."""
    patterns = (
        r"['\"][^'\"]+['\"]",
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        r"https?://[^\s,'\"]+",
        r"(?<!\w)\+?\d[\d,.:/+()-]*(?!\w)",
        r"\b(?:true|false|suspicious|target)\b",
    )
    windows: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            window = _window(question, match.start(), match.end())
            if window and window not in windows: windows.append(window)
    return windows
