import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl_api.ast import ApiRequest
from nl_api.data import structural_features


def test_entity_labels_include_nested_relation_targets_once():
    request = ApiRequest.from_dict({"entityType": "CDR", "statements": [{"type": "relation", "parameters": {"relationType": ["a"], "relationTargetType": ["Phone"]}, "statements": [{"type": "relation", "parameters": {"relationType": ["b"], "relationTargetType": ["Person", "Phone"]}, "statements": [{"type": "filter", "parameters": {"name": "x", "operator": "equals", "value": "y"}}]}]}]})
    assert sorted(request.entity_labels()) == ["CDR", "Person", "Phone"]
    assert structural_features(request)["relation_targets"] == ["Person", "Phone"]
