from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import json


class SchemaError(ValueError):
    """Raised when an API request does not obey the supplied schema."""


@dataclass(frozen=True)
class Filter:
    name: str
    operator: str
    value: Any

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Filter":
        params = value.get("parameters")
        if not isinstance(params, dict) or {"name", "operator"} - set(params):
            raise SchemaError("filter requires parameters.name, parameters.operator and parameters.value")
        # Three supplied labels omit the semantically empty operand for this unary
        # predicate. Normalize that documented source defect to the API's explicit
        # representation; every other missing operand remains invalid.
        if "value" not in params:
            if params.get("operator") == "is_not_empty":
                params = {**params, "value": ""}
            else:
                raise SchemaError("filter requires parameters.name, parameters.operator and parameters.value")
        if not isinstance(params["name"], str) or not isinstance(params["operator"], str):
            raise SchemaError("filter name and operator must be strings")
        return cls(params["name"].strip(), params["operator"].strip(), params["value"])

    def to_dict(self) -> dict[str, Any]:
        return {"type": "filter", "parameters": {"name": self.name, "operator": self.operator, "value": self.value}}


@dataclass(frozen=True)
class Sort:
    name: str
    direction: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Sort":
        params = value.get("parameters")
        if not isinstance(params, dict) or not isinstance(params.get("name"), str) or params.get("direction") not in {"asc", "desc"}:
            raise SchemaError("sort requires parameters.name and direction asc or desc")
        return cls(params["name"].strip(), params["direction"])

    def to_dict(self) -> dict[str, Any]:
        return {"type": "sort", "parameters": {"name": self.name, "direction": self.direction}}


@dataclass(frozen=True)
class Relation:
    relation_type: tuple[str, ...]
    target_type: tuple[str, ...]
    statements: tuple[Statement, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Relation":
        params = value.get("parameters")
        if not isinstance(params, dict):
            raise SchemaError("relation requires parameters")
        rel, target = params.get("relationType"), params.get("relationTargetType")
        if not isinstance(rel, list) or not rel or not all(isinstance(x, str) and x for x in rel):
            raise SchemaError("relationType must be a non-empty string list")
        if not isinstance(target, list) or not target or not all(isinstance(x, str) and x for x in target):
            raise SchemaError("relationTargetType must be a non-empty string list")
        # One supplied legacy relation places its children inside parameters.
        # Accept only that known alternate placement and serialize the canonical
        # grammar with statements at the relation-node level.
        children_owner = value if "statements" in value else params
        return cls(tuple(rel), tuple(target), tuple(parse_statement(x) for x in _statements(children_owner)))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "relation", "parameters": {"relationType": list(self.relation_type), "relationTargetType": list(self.target_type)}, "statements": [x.to_dict() for x in self.statements]}


@dataclass(frozen=True)
class BooleanGroup:
    operator: str
    statements: tuple[Statement, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BooleanGroup":
        params = value.get("parameters")
        operator = params.get("operatorValue") if isinstance(params, dict) else None
        if operator != "OR":
            raise SchemaError("only explicit OR operator groups are supported")
        return cls(operator, tuple(parse_statement(x) for x in _statements(value)))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "operator", "parameters": {"operatorValue": self.operator}, "statements": [x.to_dict() for x in self.statements]}


Statement = Filter | Sort | Relation | BooleanGroup


def _statements(value: dict[str, Any]) -> list[dict[str, Any]]:
    children = value.get("statements")
    if not isinstance(children, list) or not children:
        raise SchemaError("compound statement requires non-empty statements")
    if not all(isinstance(x, dict) for x in children):
        raise SchemaError("statements must be objects")
    return children


def parse_statement(value: dict[str, Any]) -> Statement:
    if not isinstance(value, dict):
        raise SchemaError("statement must be an object")
    statement_type = value.get("type")
    if statement_type == "filter": return Filter.from_dict(value)
    if statement_type == "sort": return Sort.from_dict(value)
    if statement_type == "relation": return Relation.from_dict(value)
    if statement_type == "operator": return BooleanGroup.from_dict(value)
    raise SchemaError(f"unsupported statement type: {statement_type!r}")


@dataclass(frozen=True)
class ApiRequest:
    entity_type: str
    statements: tuple[Statement, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ApiRequest":
        if not isinstance(value, dict) or not isinstance(value.get("entityType"), str):
            raise SchemaError("request requires string entityType")
        statements = value.get("statements")
        if not isinstance(statements, list):
            raise SchemaError("request requires statements list")
        return cls(value["entityType"].strip(), tuple(parse_statement(x) for x in statements))

    def to_dict(self) -> dict[str, Any]:
        return {"entityType": self.entity_type, "statements": [x.to_dict() for x in self.statements]}

    def canonical_dict(self) -> dict[str, Any]:
        # Only implicit root AND children are reorderable. Explicit groups retain order.
        def key(item: Statement) -> tuple[str, str, str]:
            if isinstance(item, Filter): return ("filter", item.name, item.operator + "|" + json.dumps(item.value, sort_keys=True, default=str))
            if isinstance(item, Sort): return ("sort", item.name, item.direction)
            if isinstance(item, Relation): return ("relation", "|".join(item.target_type), "|".join(item.relation_type))
            return ("operator", item.operator, "")
        return {"entityType": self.entity_type, "statements": [x.to_dict() for x in sorted(self.statements, key=key)]}

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def entity_labels(self) -> set[str]:
        output = {self.entity_type}
        def visit(items: Iterable[Statement]) -> None:
            for item in items:
                if isinstance(item, Relation): output.update(item.target_type); visit(item.statements)
                elif isinstance(item, BooleanGroup): visit(item.statements)
        visit(self.statements)
        return output
