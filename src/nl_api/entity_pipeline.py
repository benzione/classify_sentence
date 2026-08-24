"""Small, local, entity-set predictors.

This module deliberately predicts only ``entityType`` and recursive
``relationTargetType`` labels.  It does not inspect JSON, fields, or values at
inference time.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from .data import normalize_question


@dataclass(frozen=True)
class EntityConfig:
    family: str = "direct"  # root, direct, hierarchical, powerset
    word_ngrams: tuple[int, int] = (1, 2)
    char_ngrams: tuple[int, int] = (3, 5)
    min_df: int = 1
    max_features: int = 20_000
    c: float = 1.0
    balanced: bool = True
    threshold: float = 0.5
    random_state: int = 20260824
    semantic_keyword_boost: float = 0.0
    semantic_keyword_filter: bool = False
    semantic_keyword_min_hits: int = 1
    semantic_relation_boost: float = 0.0


@dataclass(frozen=True)
class EntityPrediction:
    entities: tuple[str, ...]
    probabilities: dict[str, float]
    confidence: float
    fallback_used: bool = False

    def to_list(self) -> list[str]:
        return list(self.entities)


class _BinaryHeads:
    """Independent binary heads with an explicit constant-label path."""
    def __init__(self, labels: list[str], estimators: list[Any]):
        self.labels, self.estimators = labels, estimators

    @classmethod
    def fit(cls, matrix: Any, target: np.ndarray, labels: list[str], config: EntityConfig) -> "_BinaryHeads":
        estimators = []
        for index in range(target.shape[1]):
            values = target[:, index]
            if len(np.unique(values)) == 1:
                model: Any = DummyClassifier(strategy="constant", constant=int(values[0]))
            else:
                model = LogisticRegression(
                    C=config.c, class_weight="balanced" if config.balanced else None,
                    max_iter=1000, random_state=config.random_state,
                )
            model.fit(matrix, values)
            estimators.append(model)
        return cls(labels, estimators)

    def probabilities(self, matrix: Any) -> np.ndarray:
        return np.column_stack([model.predict_proba(matrix)[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(matrix.shape[0]) for model in self.estimators])


class EntityPipeline:
    """A train-only sparse entity extractor suitable for CPU inference."""
    def __init__(self, config: EntityConfig, vectorizer: FeatureUnion, labels: list[str], *,
                 direct_heads: _BinaryHeads | None = None, root_model: Any | None = None,
                 presence_model: Any | None = None, target_heads: _BinaryHeads | None = None,
                 powerset_model: Any | None = None, powerset_classes: list[tuple[str, ...]] | None = None,
                 training_row_ids: list[int] | None = None, training_fingerprint: str = "",
                 training_example_count: int = 0, semantic_keywords: dict[str, list[str]] | None = None,
                 semantic_relation_rules: list[dict[str, Any]] | None = None):
        self.config = config; self.vectorizer = vectorizer; self.labels = labels
        self.direct_heads = direct_heads; self.root_model = root_model
        self.presence_model = presence_model; self.target_heads = target_heads
        self.powerset_model = powerset_model; self.powerset_classes = powerset_classes or []
        self.training_row_ids = training_row_ids or []; self.training_fingerprint = training_fingerprint
        self.training_example_count = training_example_count
        self.semantic_keywords = semantic_keywords or {}
        self._semantic_patterns = _compile_semantic_patterns(self.semantic_keywords)
        self.semantic_relation_rules = semantic_relation_rules or []

    @staticmethod
    def build_vectorizer(config: EntityConfig) -> FeatureUnion:
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=config.word_ngrams, min_df=config.min_df,
                                      max_features=config.max_features, sublinear_tf=True, dtype=np.float32)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=config.char_ngrams,
                                      min_df=config.min_df, max_features=config.max_features,
                                      sublinear_tf=True, dtype=np.float32)),
        ])

    @classmethod
    def train(cls, rows: list[dict[str, Any]], config: EntityConfig | None = None) -> "EntityPipeline":
        config = config or EntityConfig()
        if config.family not in {"root", "direct", "hierarchical", "powerset"}:
            raise ValueError(f"unsupported entity model family: {config.family}")
        if not rows: raise ValueError("cannot train entity pipeline without rows")
        labels = sorted({label for row in rows for label in row["entity_labels"]})
        fit_rows = list(rows)
        vectorizer = cls.build_vectorizer(config)
        matrix = vectorizer.fit_transform([normalize_question(row["question_raw"]) for row in fit_rows])
        ids = [int(row["row_id"]) for row in rows]
        fingerprint = hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()
        targets = np.asarray([[int(label in row["entity_labels"]) for label in labels] for row in fit_rows], dtype=int)
        kwargs = dict(training_row_ids=ids, training_fingerprint=fingerprint, training_example_count=len(fit_rows))
        root_y = [row["root_entity"] for row in fit_rows]
        semantic_keywords, semantic_relation_rules = _load_semantic_resources(labels)
        root_model = LogisticRegression(C=config.c, class_weight="balanced" if config.balanced else None,
                                        max_iter=1000, random_state=config.random_state).fit(matrix, root_y)
        if config.family == "root":
            return cls(config, vectorizer, labels, root_model=root_model, semantic_keywords=semantic_keywords, semantic_relation_rules=semantic_relation_rules, **kwargs)
        if config.family == "direct":
            return cls(config, vectorizer, labels, direct_heads=_BinaryHeads.fit(matrix, targets, labels, config), semantic_keywords=semantic_keywords, semantic_relation_rules=semantic_relation_rules, **kwargs)
        if config.family == "powerset":
            combinations = sorted({tuple(sorted(row["entity_labels"])) for row in rows})
            encoded = ["\x1f".join(sorted(row["entity_labels"])) for row in fit_rows]
            model = LogisticRegression(C=config.c, class_weight="balanced" if config.balanced else None,
                                       max_iter=1000, random_state=config.random_state).fit(matrix, encoded)
            return cls(config, vectorizer, labels, powerset_model=model, powerset_classes=combinations, semantic_keywords=semantic_keywords, semantic_relation_rules=semantic_relation_rules, **kwargs)
        # Target labels are learned only from recursive relationTargetType columns.
        target_y = np.asarray([[int(label in row["relation_targets"]) for label in labels] for row in fit_rows], dtype=int)
        has_relation = np.asarray([int(row["has_relation"]) for row in fit_rows])
        presence = _fit_binary(matrix, has_relation, config)
        return cls(config, vectorizer, labels, root_model=root_model, presence_model=presence,
                   target_heads=_BinaryHeads.fit(matrix, target_y, labels, config), semantic_keywords=semantic_keywords, semantic_relation_rules=semantic_relation_rules, **kwargs)

    def predict(self, question: str) -> EntityPrediction:
        return self.predict_batch([question])[0]

    def predict_batch(self, questions: Iterable[str]) -> list[EntityPrediction]:
        normalized = [normalize_question(q) for q in questions]
        matrix = self.vectorizer.transform(normalized)
        if self.config.family == "root":
            probabilities = self.root_model.predict_proba(matrix)
            return [EntityPrediction((str(self.root_model.classes_[int(np.argmax(probabilities[row]))]),), {str(label): float(probabilities[row, i]) for i, label in enumerate(self.root_model.classes_)}, float(probabilities[row].max())) for row in range(matrix.shape[0])]
        if self.config.family == "direct":
            probability_matrix = self.direct_heads.probabilities(matrix)
            return self._semantic_predictions(probability_matrix, normalized)
        if self.config.family == "powerset":
            probabilities = self.powerset_model.predict_proba(matrix)
            result = []
            for row in range(matrix.shape[0]):
                chosen = int(np.argmax(probabilities[row])); entities = tuple(sorted(self.powerset_model.classes_[chosen].split("\x1f")))
                diagnostics = {" / ".join(c.split("\x1f")): float(probabilities[row, i]) for i, c in enumerate(self.powerset_model.classes_)}
                result.append(EntityPrediction(entities, diagnostics, float(probabilities[row, chosen])))
            return result
        roots = self.root_model.predict_proba(matrix); presence = _positive_probability(self.presence_model, matrix)
        target_probability = self.target_heads.probabilities(matrix)
        result = []
        for row in range(matrix.shape[0]):
            root = str(self.root_model.classes_[int(np.argmax(roots[row]))])
            chosen = {root}
            if presence[row] >= self.config.threshold:
                chosen.update(label for col, label in enumerate(self.labels) if target_probability[row, col] >= self.config.threshold)
            probs = {label: float(target_probability[row, col]) for col, label in enumerate(self.labels)}
            probs["__relation_presence__"] = float(presence[row])
            result.append(EntityPrediction(tuple(sorted(chosen)), probs, float(max(roots[row].max(), presence[row]))))
        return result

    def _threshold_predictions(self, probabilities: np.ndarray) -> list[EntityPrediction]:
        result = []
        for row in range(probabilities.shape[0]):
            selected = [label for index, label in enumerate(self.labels) if probabilities[row, index] >= self.config.threshold]
            fallback = not selected
            if fallback: selected = [self.labels[int(np.argmax(probabilities[row]))]]
            probs = {label: float(probabilities[row, index]) for index, label in enumerate(self.labels)}
            result.append(EntityPrediction(tuple(sorted(selected)), probs, float(np.max(probabilities[row])), fallback))
        return result

    def _semantic_predictions(self, probabilities: np.ndarray, questions: list[str]) -> list[EntityPrediction]:
        if self.config.semantic_keyword_boost <= 0 and not self.config.semantic_keyword_filter:
            return self._threshold_predictions(probabilities)
        result = []
        patterns = getattr(self, "_semantic_patterns", None)
        if patterns is None:
            patterns = _compile_semantic_patterns(self.semantic_keywords)
            self._semantic_patterns = patterns
        for row, question in enumerate(questions):
            hits = np.asarray([_keyword_hits(question, patterns.get(label, ())) for label in self.labels])
            active = hits >= self.config.semantic_keyword_min_hits
            adjusted = probabilities[row].copy()
            if self.config.semantic_keyword_boost > 0:
                signal = np.minimum(1.0, hits / 3.0)
                adjusted += self.config.semantic_keyword_boost * signal * (1.0 - adjusted)
            if self.config.semantic_relation_boost > 0:
                for entity in _relation_entities(question, hits, self.labels, self.semantic_relation_rules):
                    col = self.labels.index(entity)
                    adjusted[col] += self.config.semantic_relation_boost * (1.0 - adjusted[col])
            selected = [label for col, label in enumerate(self.labels) if adjusted[col] >= self.config.threshold and (not self.config.semantic_keyword_filter or active[col])]
            fallback = not selected
            if fallback:
                candidates = np.flatnonzero(active) if active.any() else np.arange(len(self.labels))
                selected = [self.labels[candidates[int(np.argmax(adjusted[candidates]))]]]
            probs = {label: float(adjusted[col]) for col, label in enumerate(self.labels)}
            result.append(EntityPrediction(tuple(sorted(selected)), probs, float(np.max(adjusted)), fallback))
        return result

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "EntityPipeline":
        model = joblib.load(path)
        if not isinstance(model, cls): raise TypeError("artifact is not an EntityPipeline")
        return model

    def metadata(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "labels": self.labels, "training_row_ids": self.training_row_ids,
                "training_fingerprint": self.training_fingerprint,
                "training_example_count": self.training_example_count,
                "semantic_keyword_source": "data/entity_semantic_keywords.json",
                "semantic_keyword_counts": {label: len(words) for label, words in self.semantic_keywords.items()},
                "semantic_relation_rule_count": len(self.semantic_relation_rules)}


def _fit_binary(matrix: Any, labels: np.ndarray, config: EntityConfig) -> Any:
    if len(np.unique(labels)) == 1: model: Any = DummyClassifier(strategy="constant", constant=int(labels[0]))
    else: model = LogisticRegression(C=config.c, class_weight="balanced" if config.balanced else None, max_iter=1000, random_state=config.random_state)
    return model.fit(matrix, labels)


def _positive_probability(model: Any, matrix: Any) -> np.ndarray:
    probs = model.predict_proba(matrix); return probs[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(matrix.shape[0])





def _load_semantic_resources(labels: list[str]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    path = Path(__file__).resolve().parents[2] / "data" / "entity_semantic_keywords.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    entities = payload["entities"]
    return ({label: list(entities[label]["discriminators"]) + list(entities[label].get("semantic_expansions", ())) for label in labels}, payload.get("relation_patterns", []))


def _compile_semantic_patterns(keywords: dict[str, list[str]]) -> dict[str, tuple[re.Pattern[str], ...]]:
    return {label: tuple(re.compile(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)") for phrase in phrases) for label, phrases in keywords.items()}


def _keyword_hits(question: str, patterns: Iterable[re.Pattern[str]]) -> int:
    return sum(bool(pattern.search(question)) for pattern in patterns)


def _relation_entities(question: str, hits: np.ndarray, labels: list[str], rules: Iterable[dict[str, Any]]) -> set[str]:
    entities: set[str] = set()
    for rule in rules:
        source, target = rule["source_entity"], rule["target_entity"]
        if source not in labels or target not in labels: continue
        source_hits = hits[labels.index(source)]
        if source_hits >= int(rule.get("minimum_source_hits", 1)) and any(_phrase_match(question, phrase) for phrase in rule["phrases"]):
            entities.update((source, target))
    return entities


def _phrase_match(question: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", question))
