import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl_api.entity_pipeline import (EntityConfig, EntityPipeline,
                                    _compile_semantic_patterns,
                                    _keyword_hits, _relation_entities)


ROWS = [
    {"row_id": 1, "question_raw": "find phones", "entity_labels": ["Phone"], "relation_targets": [], "has_relation": False, "root_entity": "Phone"},
    {"row_id": 2, "question_raw": "find people", "entity_labels": ["Person"], "relation_targets": [], "has_relation": False, "root_entity": "Person"},
    {"row_id": 3, "question_raw": "calls from phones", "entity_labels": ["CDR", "Phone"], "relation_targets": ["Phone"], "has_relation": True, "root_entity": "CDR"},
    {"row_id": 4, "question_raw": "call records", "entity_labels": ["CDR"], "relation_targets": [], "has_relation": False, "root_entity": "CDR"},
]


def test_direct_predict_is_sorted_nonempty_and_serializable(tmp_path):
    model = EntityPipeline.train(ROWS, EntityConfig(max_features=100, c=1.0))
    prediction = model.predict("find phones")
    assert prediction.entities and prediction.entities == tuple(sorted(prediction.entities))
    path = tmp_path / "entity.joblib"; model.dump(path)
    assert EntityPipeline.load(path).training_row_ids == [1, 2, 3, 4]


def test_hierarchical_handles_constant_relation_target_labels():
    model = EntityPipeline.train(ROWS, EntityConfig(family="hierarchical", max_features=100))
    assert model.predict("calls from phones").entities


def test_semantic_keyword_hybrid_loads_static_lexicon_and_predicts():
    model = EntityPipeline.train(ROWS, EntityConfig(max_features=100, semantic_keyword_boost=.2, semantic_keyword_filter=True))
    assert "phone" in model.semantic_keywords["Phone"]
    assert model.predict("find phones").entities


def test_semantic_word_boundary_does_not_match_male_inside_female():
    patterns = _compile_semantic_patterns({"Person": ["male", "female"]})
    assert _keyword_hits("female", patterns["Person"]) == 1


def test_content_author_relation_rule_promotes_both_required_entities():
    rules = [{"source_entity": "Web Activity", "target_entity": "Web Actor",
              "phrases": ["made by"], "minimum_source_hits": 1}]
    entities = _relation_entities("comments made by a female", np.array([1, 0, 2]),
                                  ["Web Activity", "Web Actor", "Person"], rules)
    assert entities == {"Web Activity", "Web Actor"}
