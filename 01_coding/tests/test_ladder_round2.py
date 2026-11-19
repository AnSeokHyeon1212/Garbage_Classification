"""Task 4: 채택 규칙(클래스 조건 없음), 2차 사다리 M1~M9, aug_add, ladder_m.csv"""
import json

import pandas as pd

from nbcells import load

TAGS = ["data", "cache", "model", "train", "ladder"]


def _m(v, pc_c=0.9, pc_s=0.9):
    return {"val_avg": v, "per_class_val": {"clothes": pc_c, "shoes": pc_s}}


def test_adopt_no_class_guard():
    adopt = load(TAGS)["adopt"]
    assert adopt(_m(0.80, pc_c=0.50), _m(0.70)) is True


def test_ladder_m_ids_and_aug_add(tmp_path):
    ns = load(TAGS, CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    assert [s["id"] for s in ns["LADDER_M"]] == [f"M{i}" for i in range(1, 10)]
    tiny = [{**s, "change": {**s["change"], "epochs": 1, "batch": 8, "img_size": 64}} for s in ns["LADDER_M"][:3]]
    df = ns["run_ladder"](tiny, tmp_path / "exp", base_cfg=ns["M_BASE"], csv_name="ladder_m.csv")
    cfg3 = json.load(open(tmp_path / "exp/M3/cfg.json"))
    assert set(cfg3["aug"]) == {"dark_blur", "hflip", "vflip", "rot"} and "aug_add" not in cfg3
    assert (tmp_path / "exp/ladder_m.csv").exists() and "per_class_delta" in df


def test_round2_does_not_touch_round1(tmp_path):
    ns = load(TAGS, CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    (tmp_path / "exp/R1").mkdir(parents=True)
    (tmp_path / "exp/R1/cfg.json").write_text('{"old": 1}'); (tmp_path / "exp/R1/metrics.json").write_text("{}")
    tiny = [{**ns["LADDER_M"][0], "change": {"train_data": "mix2", "epochs": 1, "batch": 8, "img_size": 64}}]
    ns["run_ladder"](tiny, tmp_path / "exp", base_cfg=ns["M_BASE"], csv_name="ladder_m.csv")
    assert json.load(open(tmp_path / "exp/R1/cfg.json")) == {"old": 1}
    assert "R1" not in ns["pd"].read_csv(tmp_path / "exp/ladder_m.csv").id.tolist()
    assert json.load(open(tmp_path / "exp/R1/cfg.json")) == {"old": 1}
    assert "R1" not in pd.read_csv(tmp_path / "exp/ladder_m.csv").id.tolist()
