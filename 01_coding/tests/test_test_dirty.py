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


def used_paths():
    used = set()
    for name in ["train_mix_clean", "val_ext_clean", "test_ext_clean", "val_hard_clean", "extra_external"]:
        used |= set(pd.read_csv(SPLIT / f"{name}.csv", encoding="utf-8-sig")["path"])
    used |= set(pd.read_csv(SPLIT / "split.csv").path)   # split.csv 전체
    return used


def source_path(source, origin):
    """외부 행의 origin을 사용된 경로와 비교할 수 있는 원래 경로로 바꿈. 대응이 없으면 None"""
    if source == "realwaste":   # Plastic/x.jpg
        return "realwaste/realwaste-main/RealWaste/" + origin
    if origin.startswith("hf_extra/"):   # tricascade·hf_shoes
        return "external_data/" + origin
    return None   # AI Hub는 파일명으로, Household·TrashBusters는 사용 목록에 원본 경로가 없어 사진 해시로 비교


def test_no_overlap_with_used():
    df = load()
    used = used_paths()
    sp = pd.read_csv(SPLIT / "split.csv")
    kag = df[df.source == "kaggle"]
    ext = df[df.source != "kaggle"]
    # Kaggle 행은 원본 경로(origin)가 split == "test"인 것만 허용
    assert set(kag.origin) <= set(sp[sp.split == "test"].path)
    # 외부 행은 평가셋 안의 path·origin이 사용된 경로와 겹치지 않음
    assert not (set(ext.path) & used)
    assert not (set(ext.origin) & used)
    # 외부 행의 원래 출처 파일이 사용된 파일과 같지 않음
    src = {r.origin: source_path(r.source, r.origin) for r in ext.itertuples()}
    mapped = {o: p for o, p in src.items() if p}
    assert mapped, "대응되는 원래 경로가 없음"
    assert not (set(mapped.values()) & used), sorted(set(mapped.values()) & used)[:5]
    # AI Hub는 사용된 AI Hub 파일명과 비교 (origin은 "<zip>//<파일명>" 형태)
    used_aihub = {Path(p).name for p in used if "/aihub" in p}
    names = {Path(o).name for o in ext[ext.source == "aihub"].origin}
    assert names and not (names & used_aihub), sorted(names & used_aihub)[:5]


def test_no_near_duplicate_with_used():
    """외부 원본 사진과 사용된 사진의 사진 해시 거리가 4 초과 (해시 캐시가 없으면 건너뜀)"""
    import json
    import numpy as np
    cache_path = SPLIT / "phash_cache.json"
    if not cache_path.exists():
        pytest.skip("phash_cache.json 없음")
    cache = json.loads(cache_path.read_text())
    used = [p for p in used_paths() if p in cache]
    df = load()
    ext = df[(df.source != "kaggle") & (df.corruption == "none") & df.path.isin(cache)]
    if len(used) < 1000 or len(ext) == 0:
        pytest.skip("해시 캐시에 필요한 사진이 없음")
    hu = np.array([int(cache[p], 16) for p in used], dtype=np.uint64)
    he = np.array([int(cache[p], 16) for p in ext.path], dtype=np.uint64)
    mn = min(int(np.bitwise_count(hu ^ h).min()) for h in he)
    assert mn > 4, mn


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
    # 출처마다 학습한 종류에 해당하는 클래스만 허용
    by_source = {
        "aihub": {"battery", "brown-glass", "green-glass", "white-glass", "metal"},
        "realwaste": {"plastic"},
        "tricascade": {"cardboard", "biological"},
        "household_real": {"plastic", "paper", "white-glass", "trash"},
        "household_default": {"trash"},
        "hf_shoes": {"shoes"},
        "tb_trash": {"trash"},
    }
    for src, g in ext.groupby("source"):
        assert set(g.label) <= by_source[src], (src, set(g.label) - by_source[src])
