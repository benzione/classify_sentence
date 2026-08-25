import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nl_api.minilm_pipeline import MiniLMConfig, _Heads


def test_minilm_heads_are_independent_multilabel_classifiers():
    vectors = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.1, 0.0], [0.9, 1.0]], dtype=np.float32)
    heads = _Heads.fit(vectors, np.asarray([[0, 0], [1, 1], [0, 1], [1, 1]]), ["A", "B"], MiniLMConfig())
    assert heads.probabilities(vectors).shape == (4, 2)


def test_minilm_configuration_defaults_to_local_cpu_encoder():
    config = MiniLMConfig()
    assert config.device == "cpu"
    assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
