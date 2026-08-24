import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nl_api.ast import ApiRequest

def test_root_filters_are_canonicalized():
    def request(names): return ApiRequest.from_dict({"entityType": "Phone", "statements": [{"type":"filter", "parameters":{"name":name,"operator":"equals","value":"x"}} for name in names]})
    assert request(["b", "a"]).canonical_json() == request(["a", "b"]).canonical_json()
