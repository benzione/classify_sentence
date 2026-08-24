import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl_api.ast import ApiRequest
from nl_api.schema import Schema
from nl_api.supervision import candidate_spans, condition_targets


SCHEMA = Schema.load(Path(__file__).resolve().parents[1] / "data" / "fields_description.csv")


def request(statements):
    return ApiRequest.from_dict({"entityType": "Phone", "statements": statements})


def filter_statement(field, operator, value):
    return {"type": "filter", "parameters": {"name": field, "operator": operator, "value": value}}


def test_exact_and_normalized_condition_alignment():
    target = request([
        filter_statement("ifc.Phone.type", "equals", "Foreign"),
        filter_statement("ifc.ootb.Participant.lastActiveDate", "before", "2024-01-01"),
    ])
    aligned = condition_targets("Find Foreign phones active before 1st January 2024", target, SCHEMA)
    assert [item.alignment for item in aligned] == ["exact", "normalized"]
    assert aligned[1].span_text == "1st January 2024"


def test_relative_time_and_relation_boolean_context_are_retained():
    target = ApiRequest.from_dict({"entityType": "CDR", "statements": [{
        "type": "relation", "parameters": {"relationType": ["r"], "relationTargetType": ["Phone"]},
        "statements": [{"type": "operator", "parameters": {"operatorValue": "OR"}, "statements": [
            filter_statement("ifc.ootb.Participant.lastActiveDate", "relative", {"mode": "previous", "time_res": "day", "count": 1})
        ]}]
    }]})
    aligned = condition_targets("Calls related to phones active yesterday", target, SCHEMA)
    assert aligned[0].alignment == "normalized"
    assert aligned[0].scope == ("Phone",)
    assert aligned[0].relation_path and aligned[0].boolean_path == ("OR",)


def test_between_dates_remains_one_candidate_clause():
    question = "Reports created between January 1st 2024 and February 28th 2024"
    clauses = [item.text for item in candidate_spans(question) if item.kind == "clause"]
    assert clauses == [question]


def test_unaligned_filter_is_retained():
    target = request([filter_statement("ifc.ootb.Phone.IMSI", "equals", "not-present")])
    aligned = condition_targets("Find a device", target, SCHEMA)
    assert len(aligned) == 1
    assert aligned[0].alignment == "unaligned"
    assert aligned[0].span_start is None
