"""손상 검증셋 val_dirty, 선택 지표 val_avg4, 배경 의존도 bg_drop, 학습 없는 다시 채점, 3차 사다리"""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch

from nbcells import load

TAGS = ["data", "cache", "model", "train", "ladder"]
ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "01_coding/RESULT/data_split"
EXP = ROOT / "01_coding/RESULT/experiments"


@pytest.fixture(scope="module")
def ns():
    return load(TAGS)


def test_val_dirty_sources(ns):
    vd, vh = ns["val_dirty"], ns["val_hard"]
    assert len(vd) == 1198 and (vd.corruption != "none").all()
    assert vd.corruption.value_counts().min() >= 140
    assert set(vd.orig_path) == set(vh.orig_path)
    split = pd.read_csv(SPLIT / "split.csv")
    banned = set(split[split.split == "test"].path) | set(pd.read_csv(SPLIT / "test_ext_clean.csv").path)
    td = pd.read_csv(SPLIT / "test_dirty.csv")
    banned |= set(td.path) | set(td.origin)
    assert not (set(vd.orig_path) | set(vd.path)) & banned


def test_adopt_key(ns):
    assert ns["adopt"]({"val_avg4": .80}, {"val_avg4": .79}, key="val_avg4") is True
    assert ns["adopt"]({"val_avg4": .795}, {"val_avg4": .79}, key="val_avg4") is False


def test_ladder_m_unchanged(ns, tmp_path):
    for i in range(1, 10):
        d = tmp_path / f"M{i}"
        d.mkdir()
        for f in ["cfg.json", "metrics.json"]:
            shutil.copy(EXP / f"M{i}" / f, d / f)
    assert "val_avg4" in json.load(open(tmp_path / "M8" / "metrics.json"))   # M8은 다시 채점돼 3차 지표가 있는 상태
    out = ns["run_ladder"](ns["LADDER_M"], tmp_path, None, ns["M_BASE"], "ladder_m.csv", key="val_avg")
    saved = pd.read_csv(EXP / "ladder_m.csv")
    assert out.adopted.tolist() == saved.adopted.tolist()
    assert out.best_after.tolist() == saved.best_after.tolist()
    assert list(out.columns) == list(saved.columns)          # 1·2차 결과 파일 형식은 그대로 (3차 지표 열 없음)


def test_rescore_does_not_train(tmp_path):
    small = load(TAGS, CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    cfg = {**json.load(open(EXP / "M8" / "cfg.json", encoding="utf-8")), "width": 8}
    d = tmp_path / "exp" / "X1"
    d.mkdir(parents=True)
    json.dump(cfg, open(d / "cfg.json", "w"))
    json.dump({"val": 0.5, "val_ext": 0.5, "val_hard": 0.5}, open(d / "metrics.json", "w"))
    torch.save(small["build_model"](cfg).state_dict(), d / "best.pth")

    def boom(*a, **k):
        raise AssertionError("학습하면 안 됨")
    small["train_one_epoch"] = boom
    m = small["rescore_experiment"]("X1", tmp_path / "exp")
    assert {"val_dirty", "val_avg4", "bg_drop"} <= set(m)
    assert abs(m["val_avg4"] - (1.5 + m["val_dirty"]) / 4) < 1e-9
    assert set(json.load(open(d / "metrics.json"))) >= {"val_dirty", "val_avg4", "bg_drop"}


def test_ladder_n_definition(ns):
    assert [s["id"] for s in ns["LADDER_N"]] == ["N1", "N2"]
    assert ns["LADDER_N"][0]["change"] == {"aug_add": ["robust"]}
    assert ns["LADDER_N"][1]["change"] == {"aug_add": ["robust", "bg_swap"]}   # N1이 기각돼도 robust + bg_swap


def test_metrics_saved_even_if_extra_fails(tmp_path):
    """3차 지표 계산이 실패해도 학습 결과(metrics.json)는 남아야 다시 학습하지 않는다"""
    small = load(TAGS, CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    cfg = {**json.load(open(EXP / "M8" / "cfg.json", encoding="utf-8")), "width": 8, "epochs": 1}

    def fail(*a, **k):
        raise RuntimeError("연결 끊김 흉내")
    small["extra_metrics"] = fail
    small["run_experiment"]("X2", cfg, tmp_path / "exp")
    m = json.load(open(tmp_path / "exp" / "X2" / "metrics.json", encoding="utf-8"))
    assert {"val", "val_ext", "val_hard", "val_avg"} <= set(m)
