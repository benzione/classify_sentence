import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nl_api.ast import ApiRequest, SchemaError
from nl_api.schema import Schema

def test_unknown_field_is_rejected():
    schema = Schema.load(Path(__file__).resolve().parents[1] / "data" / "fields_description.csv")
    request = ApiRequest.from_dict({"entityType":"Phone", "statements":[{"type":"filter","parameters":{"name":"not.real","operator":"equals","value":"x"}}]})
    with pytest.raises(SchemaError): schema.validate(request)
