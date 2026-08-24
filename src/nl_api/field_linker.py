"""Shared semantic span-to-schema ranking utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from .schema import FieldDefinition, Schema


def field_card(definition: FieldDefinition) -> str:
    leaf = definition.name.rsplit(".", 1)[-1]
    return f"entity: {definition.entity}; field: {leaf}; type: {definition.field_type}; description: {definition.description}"


def condition_text(record: dict) -> str:
    scope = " / ".join(record["scope"])
    return f"condition: {record['span_text']}; question: {record['question']}; root: {record['root_entity']}; scope: {scope}"


def lexical_overlap(left: str, right: str) -> float:
    first = set(re.findall(r"[a-z0-9]+", left.casefold()))
    second = set(re.findall(r"[a-z0-9]+", right.casefold()))
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def pair_features(query: np.ndarray, field: np.ndarray, overlap: float) -> np.ndarray:
    """Interaction features let a linear ranker model query/schema agreement."""
    cosine = float(np.dot(query, field))
    return np.concatenate((np.abs(query - field), query * field, np.asarray([cosine, overlap], dtype=np.float32)))


@dataclass
class FieldLinker:
    schema: Schema
    encoder_name: str
    classifier: LogisticRegression
    field_embeddings: dict[str, np.ndarray]

    def rank(self, record: dict, query_embedding: np.ndarray) -> list[tuple[str, float]]:
        entity = record["scope"][0]
        candidates = self.schema.by_entity.get(entity, {})
        query_text = condition_text(record)
        rows = []
        for field, definition in candidates.items():
            feature = pair_features(query_embedding, self.field_embeddings[field], lexical_overlap(query_text, field_card(definition)))
            rows.append((field, feature))
        if not rows: return []
        probabilities = self.classifier.predict_proba(np.vstack([feature for _, feature in rows]))[:, 1]
        return sorted(((field, float(score)) for (field, _), score in zip(rows, probabilities)), key=lambda item: (-item[1], item[0]))


@dataclass
class MulticlassFieldLinker:
    """Shared semantic field classifier with schema-constrained ranking."""
    schema: Schema
    encoder_name: str
    classifier: LogisticRegression

    def rank(self, record: dict, query_embedding: np.ndarray) -> list[tuple[str, float]]:
        allowed = self.schema.by_entity.get(record["scope"][0], {})
        probabilities = self.classifier.predict_proba(query_embedding.reshape(1, -1))[0]
        return sorted(
            ((field, float(score)) for field, score in zip(self.classifier.classes_, probabilities) if field in allowed),
            key=lambda item: (-item[1], item[0]),
        )
