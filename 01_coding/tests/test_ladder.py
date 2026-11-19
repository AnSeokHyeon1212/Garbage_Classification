"""Task 5: 실험 사다리 (채택 규칙, 단계 목록, 작은 실행)"""
from nbcells import load


def _m(v, pc_c=0.9, pc_s=0.9):
    return {"val_avg": v, "per_class_val": {"clothes": pc_c, "shoes": pc_s}}


def test_adopt_rule():
    adopt = load(["data", "cache", "model", "train", "ladder"])["adopt"]
    assert adopt(_m(0.711), _m(0.70)) is True
    assert adopt(_m(0.710), _m(0.70)) is True          # 정확히 +0.01도 채택
    assert adopt(_m(0.709), _m(0.70)) is False


def test_ladder_ids_unique_and_one_change():
    L = load(["data", "cache", "model", "train", "ladder"])["LADDER"]
    assert [s["id"] for s in L] == [f"R{i}" for i in range(1, 13)]
    assert L[0]["change"] == {}
    assert all(1 <= len(s["change"]) <= 2 for s in L[1:])   # epochs+patience만 2개


def test_run_ladder_tiny(tmp_path):
    ns = load(["data", "cache", "model", "train", "ladder"], CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=24)
    tiny = [{**s, "change": {**s["change"], "epochs": 1, "batch": 8, "img_size": 64}} for s in ns["LADDER"][:2]]
    df = ns["run_ladder"](tiny, tmp_path / "exp")
    assert list(df.id) == ["R1", "R2"] and (tmp_path / "exp/ladder.csv").exists()
    assert bool(df.adopted.iloc[0]) is True                  # 기준 실험은 항상 채택
    assert df.best_after.iloc[-1] in {"R1", "R2"}


def test_run_ladder_stop_before(tmp_path):
    ns = load(["data", "cache", "model", "train", "ladder"], CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=24)
    tiny = [{**s, "change": {**s["change"], "epochs": 1, "batch": 8, "img_size": 64}} for s in ns["LADDER"][:3]]
    df = ns["run_ladder"](tiny, tmp_path / "exp", stop_before="R2")
    assert list(df.id) == ["R1"]
