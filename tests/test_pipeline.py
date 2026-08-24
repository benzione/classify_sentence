import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl_api.pipeline import LocalParser


def test_observed_value_uses_training_frequency_and_total_tie_break():
    parser = LocalParser.__new__(LocalParser)
    parser.observed_values = {
        "platform": Counter({"Twitter": 1, "twitter": 2}),
        "status": Counter({"Open": 1, "open": 1}),
    }
    assert parser._observed_value("Find Twitter accounts", "platform") == "twitter"
    assert parser._observed_value("Find open cases", "status") == "open"
