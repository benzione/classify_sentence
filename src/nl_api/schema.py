from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

from .ast import ApiRequest, BooleanGroup, Filter, Relation, SchemaError, Sort, Statement


@dataclass(frozen=True)
class FieldDefinition:
    entity: str
    name: str
    field_type: str
    description: str


class Schema:
    def __init__(self, fields: list[FieldDefinition]):
        self.fields = fields
        self.entities = frozenset(x.entity for x in fields)
        self.by_entity = {entity: {x.name: x for x in fields if x.entity == entity} for entity in self.entities}
        self.by_field = {x.name: x for x in fields}

    @classmethod
    def load(cls, path: str | Path) -> "Schema":
        with Path(path).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"entity_name", "field_name", "field_type", "description"}
            if reader.fieldnames is None or required - set(reader.fieldnames):
                raise SchemaError(f"schema CSV must contain {sorted(required)}")
            fields = [FieldDefinition(row["entity_name"].strip(), row["field_name"].strip(), row["field_type"].strip(), row["description"].strip()) for row in reader]
        if not fields or any(not x.entity or not x.name for x in fields):
            raise SchemaError("schema has blank entity or field names")
        return cls(fields)

    def validate(self, request: ApiRequest) -> None:
        if request.entity_type not in self.entities:
            raise SchemaError(f"unknown root entity {request.entity_type!r}")
        def visit(items: tuple[Statement, ...], entity: str) -> None:
            for item in items:
                if isinstance(item, Filter):
                    definition = self.by_field.get(item.name)
                    if definition is None:
                        raise SchemaError(f"unknown field {item.name!r}")
                    # Field names can be shared between relation entities; their definition is authoritative.
                    if definition.entity != entity and item.name not in self.by_entity.get(entity, {}):
                        raise SchemaError(f"field {item.name!r} is not valid for entity {entity!r}")
                    if not item.operator:
                        raise SchemaError("filter operator cannot be blank")
                elif isinstance(item, Sort):
                    if item.name not in self.by_entity.get(entity, {}):
                        raise SchemaError(f"sort field {item.name!r} is not valid for entity {entity!r}")
                elif isinstance(item, Relation):
                    for target in item.target_type:
                        if target not in self.entities: raise SchemaError(f"unknown relation target {target!r}")
                    visit(item.statements, item.target_type[0])
                elif isinstance(item, BooleanGroup): visit(item.statements, entity)
        visit(request.statements, request.entity_type)
