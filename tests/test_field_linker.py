import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl_api.field_linker import lexical_overlap, pair_features


def test_pair_features_include_symmetric_and_multiplicative_interactions():
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    field = np.asarray([0.5, 0.5], dtype=np.float32)
    features = pair_features(query, field, 0.25)
    assert features.tolist() == [0.5, 0.5, 0.5, 0.0, 0.5, 0.25]


def test_lexical_overlap_is_case_insensitive():
    assert lexical_overlap("MSISDN begins", "msisdn identifier") == 1 / 3
