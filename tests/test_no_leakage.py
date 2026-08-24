import json
from pathlib import Path
import pytest

def test_split_ids_and_template_groups_disjoint():
    base = Path(__file__).resolve().parents[1] / "data" / "splits"
    if not base.exists(): pytest.skip("splits have not been prepared")
    manifest = json.loads((base / "split_manifest.json").read_text())
    ids = [set(manifest["row_ids"][name]) for name in ("train", "validation", "test")]
    groups = [set(manifest["groups"][name]) for name in ("train", "validation", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
