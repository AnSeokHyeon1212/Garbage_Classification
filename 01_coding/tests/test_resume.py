"""다른 계정에서 사다리를 이어 돌릴 때: 앞 실험 결과가 모두 있어야 시작한다"""
import pytest

from nbcells import load


def test_check_resume(tmp_path):
    ns = load(["data", "cache", "model", "train", "ladder"])
    ladder = [{"id": "M1"}, {"id": "M2"}, {"id": "M3"}]
    (tmp_path / "M1").mkdir()
    (tmp_path / "M1" / "metrics.json").write_text("{}")
    assert ns["check_resume"](ladder, tmp_path, "M2") == ["M1"]
    with pytest.raises(RuntimeError, match="M2"):
        ns["check_resume"](ladder, tmp_path, "M3")       # M2 결과가 없으면 멈춤
    with pytest.raises(AssertionError):
        ns["check_resume"](ladder, tmp_path, "M99")
