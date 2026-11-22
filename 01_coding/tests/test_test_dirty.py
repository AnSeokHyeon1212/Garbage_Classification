"""test_dirty(더러운 사진 추가 평가셋) 데이터 검사 — CSV가 없으면 건너뜀"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "01_coding" / "RESULT" / "data_split"
CSV = SPLIT / "test_dirty.csv"

pytestmark = pytest.mark.skipif(not CSV.exists(), reason="test_dirty.csv 없음")


def load():
    return pd.read_csv(CSV)


def test_size_and_balance():
    df = load()
    assert len(df) == 1200
    assert (df.groupby("label").size() == 100).all()
    for (label, grp), g in df.groupby(["label", "group"]):
        clean = (g.corruption == "none").sum()
        assert abs(clean - len(g) / 2) <= 1, (label, grp, clean, len(g))


def test_no_overlap_with_used():
    df = load()
    used = set()
    for name in ["train_mix_clean", "val_ext_clean", "test_ext_clean", "val_hard_clean", "extra_external"]:
        used |= set(pd.read_csv(SPLIT / f"{name}.csv", encoding="utf-8-sig")["path"])
    sp = pd.read_csv(SPLIT / "split.csv")
    used |= set(sp.path)   # split.csv 전체
    kag = df[df.source == "kaggle"]
    ext = df[df.source != "kaggle"]
    # Kaggle 행은 원본 경로(origin)가 split == "test"인 것만 허용
    test_paths = set(sp[sp.split == "test"].path)
    assert set(kag.origin) <= test_paths
    # 외부 행은 path·origin 모두 사용된 경로와 겹치지 않음
    assert not (set(ext.path) & used)
    assert not (set(ext.origin) & used)


def test_only_trained_kinds():
    df = load()
    ext = df[df.source != "kaggle"]
    allowed = {"aihub", "realwaste", "tricascade", "household_real", "household_default", "hf_shoes", "tb_trash"}
    assert set(ext.source) <= allowed, set(ext.source) - allowed
    for bad in ["Vegetation/", "coffee_grounds", "/clothing/", "office_paper"]:
        assert not ext.origin.str.contains(bad, regex=False).any(), bad
    tb = ext[ext.origin.str.contains("TrashBusters/", regex=False)]
    assert (tb.label == "trash").all()
    assert (ext[ext.source == "realwaste"].label == "plastic").all()
