from __future__ import annotations

import json, re
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from .ast import ApiRequest, BooleanGroup, Filter, Relation, Sort, Statement
from .components import aligned_filter_examples, decompose, literal_windows
from .schema import Schema


@dataclass
class Prediction:
    request: ApiRequest
    confidence: float
    diagnostics: dict[str, Any]


@dataclass
class ComponentExpert:
    """A component model fitted only on one root entity's training rows."""
    entity: str
    row_count: int
    vectorizer: FeatureUnion
    field_binarizer: MultiLabelBinarizer
    field_classifier: OneVsRestClassifier
    count_classifier: Any
    relation_classifier: Any
    boolean_classifier: Any
    chain_classifier: ClassifierChain | None
    chain_fields: tuple[str, ...]
    constant_fields: tuple[str, ...]
    field_support: dict[str, int]


class LocalParser:
    """Train-only component models followed by schema-constrained AST assembly."""
    def __init__(
        self,
        schema: Schema,
        vectorizer: FeatureUnion,
        classifier: LogisticRegression,
        field_binarizer: MultiLabelBinarizer,
        field_classifier: OneVsRestClassifier,
        field_operators: dict[str, list[str]],
        train_rows: list[dict],
        slot_vectorizer: FeatureUnion | None = None,
        slot_classifier: LogisticRegression | None = None,
        slot_threshold: float = 1.0,
        slot_mode: str = "off",
        count_classifier: Any | None = None,
        relation_classifier: Any | None = None,
        boolean_classifier: Any | None = None,
        hierarchy_structure_weight: float = 0.0,
        hierarchy_field_weight: float = 0.0,
        hierarchy_field_mode: str = "mean_probability",
        root_experts: dict[str, ComponentExpert] | None = None,
        expert_min_rows: int = 10**9,
        expert_min_field_support: int = 1,
        expert_fallback: str = "rules",
        expert_field_model: str = "classifier_chain",
    ):
        self.schema, self.vectorizer, self.classifier, self.train_rows = schema, vectorizer, classifier, train_rows
        self.field_binarizer, self.field_classifier, self.field_operators = field_binarizer, field_classifier, field_operators
        self.slot_vectorizer, self.slot_classifier = slot_vectorizer, slot_classifier
        self.slot_threshold, self.slot_mode = slot_threshold, slot_mode
        self.count_classifier = count_classifier
        self.relation_classifier = relation_classifier
        self.boolean_classifier = boolean_classifier
        self.hierarchy_structure_weight = hierarchy_structure_weight
        self.hierarchy_field_weight = hierarchy_field_weight
        self.hierarchy_field_mode = hierarchy_field_mode
        self.root_experts = root_experts or {}
        self.expert_min_rows = expert_min_rows
        self.expert_min_field_support = expert_min_field_support
        self.expert_fallback = expert_fallback
        self.expert_field_model = expert_field_model
        self.train_matrix = vectorizer.transform([r["question_normalized"] for r in train_rows])
        self.train_requests = [ApiRequest.from_dict(json.loads(r["target_json"])) for r in train_rows]
        self.train_components = [decompose(request) for request in self.train_requests]
        estimators = getattr(field_classifier, "estimators_", ())
        self._field_coefficients = np.vstack([model.coef_[0] for model in estimators]) if estimators else None
        self._field_intercepts = np.asarray([model.intercept_[0] for model in estimators]) if estimators else None
        self.observed_values: dict[str, Counter[str]] = defaultdict(Counter)
        def collect(items: tuple[Statement, ...]) -> None:
            for item in items:
                if isinstance(item, Filter) and isinstance(item.value, str) and item.value: self.observed_values[item.name][item.value] += 1
                elif isinstance(item, (Relation, BooleanGroup)): collect(item.statements)
        for request in self.train_requests: collect(request.statements)

    @staticmethod
    def build_vectorizer() -> FeatureUnion:
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000, sublinear_tf=True)),
        ])

    @classmethod
    def train(
        cls,
        schema: Schema,
        rows: list[dict],
        slot_threshold: float = 1.0,
        slot_mode: str = "off",
        hierarchy_structure_weight: float = 0.0,
        hierarchy_field_weight: float = 0.0,
        hierarchy_field_mode: str = "mean_probability",
        expert_min_rows: int = 10**9,
        expert_min_field_support: int = 1,
        expert_fallback: str = "rules",
        expert_field_model: str = "classifier_chain",
        train_root_experts: bool = False,
    ) -> "LocalParser":
        if len({r["root_entity"] for r in rows}) < 2: raise ValueError("training split needs at least two root entities")
        vectorizer = cls.build_vectorizer(); matrix = vectorizer.fit_transform([r["question_normalized"] for r in rows])
        classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20260824)
        classifier.fit(matrix, [r["root_entity"] for r in rows])
        label_sets: list[list[str]] = []
        field_operators: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            request = ApiRequest.from_dict(json.loads(row["target_json"])); found: list[str] = []
            def visit(items: tuple[Statement, ...]) -> None:
                for item in items:
                    if isinstance(item, Filter): found.append(item.name); field_operators[item.name][item.operator] += 1
                    elif isinstance(item, (Relation, BooleanGroup)): visit(item.statements)
            visit(request.statements); label_sets.append(sorted(set(found)))
        field_binarizer = MultiLabelBinarizer(); targets = field_binarizer.fit_transform(label_sets)
        field_classifier = OneVsRestClassifier(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20260824))
        field_classifier.fit(matrix, targets)
        count_classifier = cls._fit_head(matrix, [row["filter_count"] for row in rows])
        relation_classifier = cls._fit_head(matrix, [row["has_relation"] for row in rows])
        boolean_classifier = cls._fit_head(matrix, [row["has_or"] for row in rows])
        root_experts = cls._fit_root_experts(rows) if train_root_experts else {}
        operators = {field: [operator for operator, _ in counts.most_common()] for field, counts in field_operators.items()}
        slot_texts: list[str] = []; slot_fields: list[str] = []
        for row in rows:
            request = ApiRequest.from_dict(json.loads(row["target_json"]))
            for context, component in aligned_filter_examples(row["question_raw"], request):
                slot_texts.append(context); slot_fields.append(component.field)
        if len(set(slot_fields)) < 2: raise ValueError("aligned slot training needs at least two fields")
        slot_vectorizer = cls.build_vectorizer(); slot_matrix = slot_vectorizer.fit_transform(slot_texts)
        slot_classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20260824)
        slot_classifier.fit(slot_matrix, slot_fields)
        return cls(
            schema, vectorizer, classifier, field_binarizer, field_classifier,
            operators, rows, slot_vectorizer, slot_classifier, slot_threshold,
            slot_mode, count_classifier, relation_classifier, boolean_classifier,
            hierarchy_structure_weight, hierarchy_field_weight,
            hierarchy_field_mode, root_experts, expert_min_rows,
            expert_min_field_support, expert_fallback, expert_field_model,
        )

    @classmethod
    def _fit_root_experts(cls, rows: list[dict]) -> dict[str, ComponentExpert]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["root_entity"]].append(row)
        experts: dict[str, ComponentExpert] = {}
        for entity, entity_rows in grouped.items():
            if len(entity_rows) < 2:
                continue
            vectorizer = cls.build_vectorizer()
            matrix = vectorizer.fit_transform([row["question_normalized"] for row in entity_rows])
            label_sets: list[list[str]] = []
            support: Counter[str] = Counter()
            for row in entity_rows:
                component = decompose(ApiRequest.from_dict(json.loads(row["target_json"])))
                fields = sorted({item.field for item in component.filters})
                label_sets.append(fields)
                support.update(fields)
            binarizer = MultiLabelBinarizer()
            targets = binarizer.fit_transform(label_sets)
            field_classifier = OneVsRestClassifier(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20260824))
            field_classifier.fit(matrix, targets)
            chain_fields = tuple(sorted(
                (field for field, count in support.items() if count < len(entity_rows)),
                key=lambda field: (-support[field], field),
            ))
            chain_classifier = None
            if chain_fields:
                chain_targets = np.asarray([[field in fields for field in chain_fields] for fields in label_sets], dtype=int)
                chain_classifier = ClassifierChain(
                    LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20260824),
                    order=list(range(len(chain_fields))),
                    random_state=20260824,
                )
                chain_classifier.fit(matrix, chain_targets)
            experts[entity] = ComponentExpert(
                entity=entity,
                row_count=len(entity_rows),
                vectorizer=vectorizer,
                field_binarizer=binarizer,
                field_classifier=field_classifier,
                count_classifier=cls._fit_head(matrix, [row["filter_count"] for row in entity_rows]),
                relation_classifier=cls._fit_head(matrix, [row["has_relation"] for row in entity_rows]),
                boolean_classifier=cls._fit_head(matrix, [row["has_or"] for row in entity_rows]),
                chain_classifier=chain_classifier,
                chain_fields=chain_fields,
                constant_fields=tuple(sorted(field for field, count in support.items() if count == len(entity_rows))),
                field_support=dict(support),
            )
        return experts

    @staticmethod
    def _fit_head(matrix: Any, labels: list[Any]) -> Any:
        if len(set(labels)) == 1:
            model: Any = DummyClassifier(strategy="most_frequent")
        else:
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20260824)
        model.fit(matrix, labels)
        return model

    def dump(self, path: str) -> None:
        joblib.dump({"schema": self.schema, "vectorizer": self.vectorizer, "classifier": self.classifier, "field_binarizer": self.field_binarizer, "field_classifier": self.field_classifier, "field_operators": self.field_operators, "train_rows": self.train_rows, "slot_vectorizer": self.slot_vectorizer, "slot_classifier": self.slot_classifier, "slot_threshold": self.slot_threshold, "slot_mode": self.slot_mode, "count_classifier": self.count_classifier, "relation_classifier": self.relation_classifier, "boolean_classifier": self.boolean_classifier, "hierarchy_structure_weight": self.hierarchy_structure_weight, "hierarchy_field_weight": self.hierarchy_field_weight, "hierarchy_field_mode": self.hierarchy_field_mode, "root_experts": self.root_experts, "expert_min_rows": self.expert_min_rows, "expert_min_field_support": self.expert_min_field_support, "expert_fallback": self.expert_fallback, "expert_field_model": self.expert_field_model, "training_fingerprint": self._train_fingerprint()}, path)

    @classmethod
    def load(cls, path: str) -> "LocalParser":
        saved = joblib.load(path)
        return cls(
            saved["schema"], saved["vectorizer"], saved["classifier"],
            saved["field_binarizer"], saved["field_classifier"],
            saved["field_operators"], saved["train_rows"],
            saved.get("slot_vectorizer"), saved.get("slot_classifier"),
            saved.get("slot_threshold", 1.0), saved.get("slot_mode", "off"),
            saved.get("count_classifier"), saved.get("relation_classifier"),
            saved.get("boolean_classifier"),
            saved.get("hierarchy_structure_weight", 0.0),
            saved.get("hierarchy_field_weight", 0.0),
            saved.get("hierarchy_field_mode", "mean_probability"),
            saved.get("root_experts"), saved.get("expert_min_rows", 10**9),
            saved.get("expert_min_field_support", 1),
            # Models saved before root experts existed used the shared hierarchy.
            saved.get("expert_fallback", "shared"),
            saved.get("expert_field_model", "classifier_chain"),
        )

    def _train_fingerprint(self) -> str:
        import hashlib
        return hashlib.sha256(",".join(str(row["row_id"]) for row in self.train_rows).encode()).hexdigest()

    def predict(self, question: str) -> Prediction:
        normalized = re.sub(r"\s+", " ", question.strip().lower())
        if not normalized: raise ValueError("question cannot be blank")
        query = self.vectorizer.transform([normalized]); probabilities = self.classifier.predict_proba(query)[0]
        class_prob = dict(zip(self.classifier.classes_, probabilities)); root = max(class_prob, key=class_prob.get)
        similarities = cosine_similarity(query, self.train_matrix).ravel()
        # Only templates whose root matches the classifier may decode. This makes root and AST constraints explicit.
        compatible = [i for i, request in enumerate(self.train_requests) if request.entity_type == root]
        hierarchy_scores, decoder = self._routed_hierarchy_scores(normalized, query, root, compatible)
        best = max(compatible, key=lambda index: similarities[index] + hierarchy_scores.get(index, 0.0)) if compatible else int(similarities.argmax())
        request = self._assemble(question, root, self.train_requests[best])
        self.schema.validate(request)
        confidence = float(0.65 * class_prob.get(root, 0.0) + 0.35 * max(similarities[best], 0.0))
        return Prediction(request, confidence, {"root_probabilities": class_prob, "retrieved_row_id": self.train_rows[best]["row_id"], "retrieval_score": float(similarities[best]), "hierarchy_score": round(hierarchy_scores.get(best, 0.0), 12), "component_decoder": decoder})

    def _routed_hierarchy_scores(self, normalized: str, query: Any, root: str, candidates: list[int]) -> tuple[dict[int, float], str]:
        expert = self.root_experts.get(root)
        if expert is not None and expert.row_count >= self.expert_min_rows:
            expert_query = expert.vectorizer.transform([normalized])
            if self.expert_field_model == "classifier_chain" and expert.chain_classifier is not None:
                chain_probabilities = np.asarray(expert.chain_classifier.predict_proba(expert_query))[0]
                field_prob = {field: 1.0 for field in expert.constant_fields}
                field_prob.update(zip(expert.chain_fields, (float(value) for value in chain_probabilities)))
            else:
                field_prob = dict(zip(expert.field_binarizer.classes_, expert.field_classifier.predict_proba(expert_query)[0]))
            scores = self._component_scores(
                candidates,
                self._class_probability(expert.count_classifier, expert_query),
                self._class_probability(expert.relation_classifier, expert_query),
                self._class_probability(expert.boolean_classifier, expert_query),
                field_prob,
                expert.field_support,
            )
            return scores, f"root_expert:{root}"
        if self.expert_fallback == "shared":
            return self._hierarchy_scores(query, candidates), "shared_model"
        return {}, "schema_rules"

    @staticmethod
    def _class_probability(model: Any, matrix: Any) -> dict[Any, float]:
        if model is None:
            return {}
        return {label: float(probability) for label, probability in zip(model.classes_, model.predict_proba(matrix)[0])}

    def _hierarchy_scores(self, query: Any, candidates: list[int]) -> dict[int, float]:
        if not candidates or (self.hierarchy_structure_weight <= 0 and self.hierarchy_field_weight <= 0):
            return {}
        count_prob = self._class_probability(self.count_classifier, query)
        relation_prob = self._class_probability(self.relation_classifier, query)
        boolean_prob = self._class_probability(self.boolean_classifier, query)
        if self._field_coefficients is None or self._field_intercepts is None:
            probabilities = self.field_classifier.predict_proba(query)[0]
        else:
            decision = np.asarray(query @ self._field_coefficients.T).ravel() + self._field_intercepts
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(decision, -40.0, 40.0)))
        field_prob = dict(zip(self.field_binarizer.classes_, probabilities))
        return self._component_scores(candidates, count_prob, relation_prob, boolean_prob, field_prob)

    def _component_scores(
        self,
        candidates: list[int],
        count_prob: dict[Any, float],
        relation_prob: dict[Any, float],
        boolean_prob: dict[Any, float],
        field_prob: dict[str, float],
        field_support: dict[str, int] | None = None,
    ) -> dict[int, float]:
        predicted_count = max(count_prob, key=count_prob.get) if count_prob else 0
        candidate_vocabulary = {item.field for index in candidates for item in self.train_components[index].filters}
        if field_support is not None:
            candidate_vocabulary = {field for field in candidate_vocabulary if field_support.get(field, 0) >= self.expert_min_field_support}
        predicted_fields = {
            field for field, _ in sorted(
                ((field, field_prob.get(field, 0.0)) for field in candidate_vocabulary),
                key=lambda item: item[1], reverse=True,
            )[:predicted_count]
        }
        result: dict[int, float] = {}
        for index in candidates:
            component = self.train_components[index]
            shape_score = (
                count_prob.get(len(component.filters), 0.0)
                + relation_prob.get(bool(component.relations), 0.0)
                + boolean_prob.get(component.has_or, 0.0)
            ) / 3.0
            fields = {item.field for item in component.filters}
            scored_fields = fields if field_support is None else {field for field in fields if field in candidate_vocabulary}
            if self.hierarchy_field_mode == "topk_f1":
                denominator = len(scored_fields) + len(predicted_fields)
                field_score = 2.0 * len(scored_fields & predicted_fields) / denominator if denominator else 0.0
            else:
                field_score = sum(float(field_prob.get(field, 0.0)) for field in scored_fields) / max(len(scored_fields), 1)
            result[index] = self.hierarchy_structure_weight * shape_score + self.hierarchy_field_weight * field_score
        return result

    def _assemble(self, question: str, root: str, fallback: ApiRequest) -> ApiRequest:
        """Augment a retrieved AST only with explicit schema-backed conditions."""
        request = self._adapt(fallback, question)
        clauses = self._clauses(question)
        statements: list[Statement] = list(request.statements)
        seen = {(item.name, item.operator, json.dumps(item.value, sort_keys=True, default=str)) for item in request.statements if isinstance(item, Filter)}
        for clause in clauses:
            for field, score in self._explicit_field_candidates(clause, root):
                operator = self._operator(clause, field)
                value = self._value(clause, field, operator)
                if value is None:
                    continue
                key = (field, operator, json.dumps(value, sort_keys=True, default=str))
                if key not in seen: statements.append(Filter(field, operator, value)); seen.add(key)
        slot_filters = self._slot_filters(question, root)
        for item, score in slot_filters:
            key = (item.name, item.operator, json.dumps(item.value, sort_keys=True, default=str))
            if key not in seen: statements.append(item); seen.add(key)
        if self.slot_mode == "replace" and slot_filters and not any(isinstance(item, (Relation, BooleanGroup, Sort)) for item in statements):
            # Replacement is intentionally limited to flat requests. Complex
            # structure stays under the safe retrieved decoder.
            statements = [item for item, _ in slot_filters]
        return ApiRequest(root, tuple(statements))

    def _slot_filters(self, question: str, entity: str) -> list[tuple[Filter, float]]:
        if self.slot_mode == "off" or self.slot_vectorizer is None or self.slot_classifier is None: return []
        output: list[tuple[Filter, float]] = []
        allowed = self.schema.by_entity.get(entity, {})
        contexts = literal_windows(question)
        if not contexts: return []
        probability_rows = self.slot_classifier.predict_proba(self.slot_vectorizer.transform(contexts))
        for context, probabilities in zip(contexts, probability_rows):
            ranked = sorted(zip(self.slot_classifier.classes_, probabilities), key=lambda item: item[1], reverse=True)
            field, probability = next(((field, float(probability)) for field, probability in ranked if field in allowed), (None, 0.0))
            if field is None or probability < self.slot_threshold: continue
            operator = self._operator(context, field); value = self._value(context, field, operator)
            if value is not None: output.append((Filter(field, operator, value), probability))
        return output

    def _explicit_field_candidates(self, clause: str, entity: str) -> list[tuple[str, float]]:
        """Return fields named explicitly in the text; no guessed fields are added."""
        tokens = set(re.findall(r"[a-z0-9]+", clause.lower()))
        result = []
        for field in self.schema.by_entity.get(entity, {}):
            leaf = set(re.findall(r"[a-z0-9]+", field.rsplit(".", 1)[-1].lower()))
            if leaf & tokens: result.append((field, 1.0))
        return result

    @staticmethod
    def _clauses(question: str) -> list[str]:
        # Do not split an OR expression: its grouping requires the relation/Boolean
        # component decoder rather than unsafe implicit-AND construction.
        parts = re.split(r"\band\b|,", question, flags=re.IGNORECASE)
        return [part.strip() for part in parts if part.strip()]

    def _field_candidates(self, clause: str, entity: str) -> list[tuple[str, float]]:
        scores = self.field_classifier.predict_proba(self.vectorizer.transform([clause]))[0]
        allowed = self.schema.by_entity.get(entity, {})
        candidates = [(field, float(score)) for field, score in zip(self.field_binarizer.classes_, scores) if field in allowed]
        if not candidates: return []
        candidates.sort(key=lambda item: item[1], reverse=True)
        tokens = set(re.findall(r"[a-z0-9]+", clause.lower()))
        exact_name_matches = [(field, score) for field, score in candidates if set(re.findall(r"[a-z0-9]+", field.rsplit(".", 1)[-1].lower())) & tokens]
        if exact_name_matches:
            # A concrete schema field mentioned by name is more specific than a
            # classifier's broad entity-level prior.
            return exact_name_matches[:3]
        # Relative threshold permits several independent conditions without a
        # fixed field list; a high floor avoids blindly recreating every template field.
        best = candidates[0][1]
        selected = [(field, score) for field, score in candidates if score >= max(0.30, best * 0.72)][:3]
        # Schema-card lexical evidence is static reference knowledge. A field name
        # mentioned verbatim in the question is a strong candidate even when it is
        # rare in the supervised split (for example, identifier fields).
        for field, score in candidates:
            definition = allowed[field]
            card_tokens = set(re.findall(r"[a-z0-9]+", f"{definition.name} {definition.description}".lower()))
            if tokens & {token for token in card_tokens if len(token) >= 4} and (field, score) not in selected:
                selected.append((field, score))
        return selected[:3]

    def _operator(self, clause: str, field: str) -> str:
        text = clause.lower()
        rules = ((r"\b(?:start(?:s|ing)?|begin(?:s|ning)?)\s+with\b", "begins_with"), (r"\b(?:end(?:s|ing)?)\s+with\b", "ends_with"), (r"\bmore than\b|\bgreater than\b|\bover\b", "greater"), (r"\bless than\b|\bfewer than\b|\bunder\b", "less"), (r"\bbetween\b", "between"), (r"\bbefore\b|\bearlier than\b", "before"), (r"\bafter\b|\blater than\b", "after"), (r"\bcontains?\b|\bmention(?:ing)?\b|\bincluding\b", "contains"), (r"\b(?:last|past|previous)\b", "relative"), (r"\bnot empty\b|\bhas\b", "is_not_empty"))
        for pattern, operator in rules:
            if re.search(pattern, text): return operator
        return self.field_operators.get(field, ["equals"])[0]

    def _value(self, clause: str, field: str, operator: str) -> Any | None:
        if operator in {"is_not_empty", "is_empty"}: return ""
        definition = self.schema.by_field.get(field)
        if definition is None: return None
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", clause)
        email = re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", clause)
        url = re.findall(r"https?://[^\s,'\"]+", clause)
        numbers = re.findall(r"(?<![\w.])\+?\d[\d,.-]*(?![\w.])", clause)
        if operator == "relative":
            match = re.search(r"\b(?:last|past|previous)\s+(?:(\d+)\s+)?(hour|day|week|month|year)s?\b", clause, re.IGNORECASE)
            if match: return {"mode": "previous", "time_res": match.group(2).lower(), "count": int(match.group(1) or 1)}
        if operator == "between":
            dates = self._dates(clause)
            if len(dates) >= 2: return dates[:2]
        if definition.field_type == "date":
            dates = self._dates(clause)
            if dates: return dates[0]
        if email: return email[0]
        if url: return url[0]
        if quoted: return quoted[-1]
        if definition.field_type in {"integer", "float"} and numbers:
            raw = numbers[-1].replace(",", "")
            try: return int(raw) if definition.field_type == "integer" else float(raw)
            except ValueError: return None
        if numbers and any(token in field.lower() for token in ("msisdn", "imei", "imsi", "phone", "number")):
            return numbers[-1]
        # Enum-like terms are learned from the words immediately following a field
        # cue; no API values or entity-specific mappings are embedded here.
        match = re.search(r"\b(?:type|status|platform|technology|gender|country)\s+(?:is|of|=)?\s*([A-Za-z][A-Za-z _-]{1,40})", clause, re.IGNORECASE)
        return match.group(1).strip().rstrip("?.") if match else None

    @staticmethod
    def _dates(text: str) -> list[str]:
        """Normalize unambiguous absolute date spans with stdlib only."""
        patterns = (r"\b\d{4}-\d{1,2}-\d{1,2}\b", r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b", r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}\b", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
        output: list[str] = []
        for raw in (match.group(0) for pattern in patterns for match in re.finditer(pattern, text)):
            cleaned = raw.replace(",", "")
            for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    parsed = datetime.strptime(cleaned, fmt)
                    value = parsed.strftime("%Y-%m-%d")
                    if value not in output: output.append(value)
                    break
                except ValueError: pass
        return output

    def _adapt(self, request: ApiRequest, question: str) -> ApiRequest:
        # Retrieval supplies structure; typed, training-derived value correction
        # updates compatible filters without copying a previous request verbatim.
        emails = re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", question)
        urls = re.findall(r"https?://[^\s,'\"]+", question)
        phones = re.findall(r"(?<!\d)(?:\+?\d[\d .()-]{5,}\d)(?!\d)", question)
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
        def substitute(items: tuple[Statement, ...]) -> tuple[Statement, ...]:
            output: list[Statement] = []
            for item in items:
                if isinstance(item, Relation): output.append(Relation(item.relation_type, item.target_type, substitute(item.statements)))
                elif isinstance(item, BooleanGroup): output.append(BooleanGroup(item.operator, substitute(item.statements)))
                elif isinstance(item, Filter):
                    value = item.value; lower = item.name.lower(); definition = self.schema.by_field.get(item.name)
                    candidates = emails if "email" in lower else urls if "url" in lower else phones if any(x in lower for x in ("msisdn", "phone", "imei", "imsi")) else quoted if item.operator == "contains" else []
                    if len(candidates) == 1 and isinstance(value, str): value = candidates[0].strip()
                    operator = item.operator
                    if definition and definition.field_type in {"integer", "float"}:
                        operator = self._operator(question, item.name)
                        extracted = self._value(question, item.name, operator)
                        if extracted is not None: value = extracted
                    elif definition and definition.field_type == "date":
                        if re.search(r"\b(?:last|past|previous|before|after|between|earlier|later)\b", question, re.I): operator = self._operator(question, item.name)
                        extracted = self._value(question, item.name, operator)
                        if extracted is not None: value = extracted
                    elif isinstance(value, str):
                        observed = self._observed_value(question, item.name)
                        if observed is not None: value = observed
                    output.append(Filter(item.name, operator, value))
                elif isinstance(item, Sort): output.append(item)
            return tuple(output)
        return ApiRequest(request.entity_type, substitute(request.statements))

    def _observed_value(self, question: str, field: str) -> Any | None:
        """Use only values seen in training, matched as whole phrases in input."""
        text = question.casefold(); matches: list[tuple[str, int]] = []
        for value, frequency in self.observed_values.get(field, {}).items():
            pattern = r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)"
            if re.search(pattern, text): matches.append((value, frequency))
        # Prefer the training-majority spelling, then apply a total ordering so
        # process-level hash randomization can never change a prediction.
        ranked = sorted(matches, key=lambda item: (item[1], len(item[0]), item[0].casefold(), item[0]))
        return ranked[-1][0] if ranked else None
