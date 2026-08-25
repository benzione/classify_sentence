"""Independent MiniLM entity-set extraction pipeline.

This module intentionally has no dependency on the sparse entity pipeline.
MiniLM supplies fixed local sentence embeddings; separate logistic heads learn
the multilabel entity decision on those embeddings.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from .data import normalize_question


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_ENCODERS: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class MiniLMConfig:
    model_name: str = MODEL_NAME
    device: str = "cpu"
    batch_size: int = 32
    c: float = 1.0
    balanced: bool = True
    threshold: float = 0.5
    random_state: int = 20260825
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class MiniLMPrediction:
    entities: tuple[str, ...]
    probabilities: dict[str, float]
    confidence: float
    fallback_used: bool = False

    def to_list(self) -> list[str]:
        return list(self.entities)


def _encoder(model_name: str, device: str) -> Any:
    """Load only from the local Hugging Face cache; never invoke hosted inference."""
    key = (model_name, device)
    if key not in _ENCODERS:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("MiniLM requires sentence-transformers and torch; install requirements-minilm.txt") from exc
        try:
            _ENCODERS[key] = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise RuntimeError(f"MiniLM model {model_name!r} is unavailable in the local Hugging Face cache") from exc
    return _ENCODERS[key]


def encode_questions(questions: Iterable[str], config: MiniLMConfig) -> np.ndarray:
    normalized = [normalize_question(question) for question in questions]
    if not normalized:
        return np.empty((0, 384), dtype=np.float32)
    encoder = _encoder(config.model_name, config.device)
    vectors = encoder.encode(normalized, batch_size=config.batch_size,
                             normalize_embeddings=config.normalize_embeddings,
                             show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32)


class _Heads:
    def __init__(self, labels: list[str], estimators: list[Any]):
        self.labels, self.estimators = labels, estimators

    @classmethod
    def fit(cls, matrix: np.ndarray, target: np.ndarray, labels: list[str], config: MiniLMConfig) -> "_Heads":
        estimators: list[Any] = []
        for column in range(target.shape[1]):
            values = target[:, column]
            if len(np.unique(values)) == 1:
                estimator: Any = DummyClassifier(strategy="constant", constant=int(values[0]))
            else:
                estimator = LogisticRegression(C=config.c, class_weight="balanced" if config.balanced else None,
                                               max_iter=1000, random_state=config.random_state)
            estimator.fit(matrix, values); estimators.append(estimator)
        return cls(labels, estimators)

    def probabilities(self, matrix: np.ndarray) -> np.ndarray:
        return np.column_stack([
            estimator.predict_proba(matrix)[:, list(estimator.classes_).index(1)] if 1 in estimator.classes_
            else np.zeros(matrix.shape[0])
            for estimator in self.estimators
        ])


class MiniLMPipeline:
    """A local MiniLM encoder plus independently trained multilabel heads."""
    def __init__(self, config: MiniLMConfig, labels: list[str], heads: _Heads, *,
                 training_row_ids: list[int], training_fingerprint: str, training_example_count: int):
        self.config = config; self.labels = labels; self.heads = heads
        self.training_row_ids = training_row_ids; self.training_fingerprint = training_fingerprint
        self.training_example_count = training_example_count

    @classmethod
    def train(cls, rows: list[dict[str, Any]], config: MiniLMConfig | None = None) -> "MiniLMPipeline":
        if not rows: raise ValueError("cannot train MiniLM pipeline without rows")
        config = config or MiniLMConfig()
        labels = sorted({label for row in rows for label in row["entity_labels"]})
        vectors = encode_questions((row["question_raw"] for row in rows), config)
        target = np.asarray([[int(label in row["entity_labels"]) for label in labels] for row in rows], dtype=int)
        ids = [int(row["row_id"]) for row in rows]
        fingerprint = hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()
        return cls(config, labels, _Heads.fit(vectors, target, labels, config),
                   training_row_ids=ids, training_fingerprint=fingerprint, training_example_count=len(rows))

    def predict(self, question: str) -> MiniLMPrediction:
        return self.predict_batch([question])[0]

    def predict_batch(self, questions: Iterable[str]) -> list[MiniLMPrediction]:
        probabilities = self.heads.probabilities(encode_questions(questions, self.config))
        predictions = []
        for row in probabilities:
            selected = [label for index, label in enumerate(self.labels) if row[index] >= self.config.threshold]
            fallback = not selected
            if fallback: selected = [self.labels[int(np.argmax(row))]]
            values = {label: float(row[index]) for index, label in enumerate(self.labels)}
            predictions.append(MiniLMPrediction(tuple(sorted(selected)), values, float(np.max(row)), fallback))
        return predictions

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "MiniLMPipeline":
        model = joblib.load(path)
        if not isinstance(model, cls): raise TypeError("artifact is not a MiniLMPipeline")
        return model

    def metadata(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "labels": self.labels,
                "training_row_ids": self.training_row_ids, "training_fingerprint": self.training_fingerprint,
                "training_example_count": self.training_example_count, "encoder": "local MiniLM sentence embeddings",
                "hosted_inference_used": False, "onnx_runtime_used": False}
