import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nl_api.ast import ApiRequest
from nl_api.components import aligned_filter_examples, decompose, literal_windows, skeleton

def test_decomposition_retains_relation_scope_and_or():
    request = ApiRequest.from_dict({"entityType":"CDR","statements":[{"type":"operator","parameters":{"operatorValue":"OR"},"statements":[{"type":"filter","parameters":{"name":"a","operator":"equals","value":"x"}}]},{"type":"relation","parameters":{"relationType":["r"],"relationTargetType":["Phone"]},"statements":[{"type":"filter","parameters":{"name":"b","operator":"equals","value":"y"}}]}]})
    parts = decompose(request)
    assert parts.has_or and parts.filters[1].scope == ("Phone",) and skeleton(parts)[0] == "CDR"

def test_literal_alignment_is_scoped_to_the_filter_value():
    request = ApiRequest.from_dict({"entityType":"Phone","statements":[{"type":"filter","parameters":{"name":"kind","operator":"equals","value":"Foreign"}}]})
    examples = aligned_filter_examples("Find Foreign phones", request)
    assert len(examples) == 1 and examples[0][1].field == "kind"

def test_literal_windows_are_deterministic_and_local():
    windows = literal_windows("Find phones with IMSI 12345 and type 'Foreign'. Ignore this.")
    assert windows and all("Ignore" not in window for window in windows)
