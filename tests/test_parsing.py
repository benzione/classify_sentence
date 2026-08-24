import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nl_api.ast import ApiRequest, SchemaError

def test_parses_literal_shape():
    request = ApiRequest.from_dict({"entityType": "Phone", "statements": [{"type": "filter", "parameters": {"name": "x", "operator": "equals", "value": True}}]})
    assert request.entity_type == "Phone"

def test_rejects_unknown_statement():
    try: ApiRequest.from_dict({"entityType": "Phone", "statements": [{"type": "oops"}]})
    except SchemaError: return
    assert False, "unknown statements must not be accepted"
