"""Task 6: 최종 TEST 1회 보호, 혼동행렬·클래스 보고 함수, 결과 없을 때 보고 셀 동작"""
import numpy as np
import pandas as pd
import pytest

from nbcells import load

TAGS = ["data", "cache", "model", "train", "ladder", "final"]


def test_final_test_runs_once(tmp_path):
    ns = load(TAGS, CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=8)
    m = ns["build_model"]({"width": 32, "hidden": ()}).eval()
    sets = {"test": ns["test_df"].head(8), "test_ext": ns["test_ext"].head(8)}
    r = ns["final_test_once"](m, sets, tmp_path, img_size=64)
    assert set(r) == {"test", "test_ext"} and (tmp_path / "final_test.json").exists()
    assert (tmp_path / "final_test_confusion.csv").exists() and (tmp_path / "final_test_ext_confusion.csv").exists()
    with pytest.raises(AssertionError, match="이미 1회"):
        ns["final_test_once"](m, sets, tmp_path, img_size=64)


def test_class_report_values():
    ns = load(TAGS)
    cm = pd.DataFrame([[8, 2], [1, 9]], index=["metal", "paper"], columns=["metal", "paper"])
    r = ns["class_report"](cm)
    assert r.loc["metal", "재현율"] == 0.8 and r.loc["metal", "정밀도"] == round(8 / 9, 3)
    assert r.loc["paper", "오탐"] == 2 and r.loc["paper", "미탐"] == 1


def test_confusion_shape():
    ns = load(TAGS)
    cm = ns["confusion"](np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert list(cm.columns) == ns["CLASSES"] and list(cm.index) == ["battery", "biological"]


def test_report_and_eval_round2_tiny(tmp_path):
    """작은 사다리(2단계) 결과로 10~12절 셀이 끝까지 돌고, TEST가 1회 저장되는지"""
    small = dict(CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    ns = load(TAGS, **small)
    tiny = [{**s, "change": {**s["change"], "epochs": 1, "batch": 8, "img_size": 64}} for s in ns["LADDER_M"][:2]]
    ns["run_ladder"](tiny, tmp_path / "exp", base_cfg=ns["M_BASE"], csv_name="ladder_m.csv")
    heads = {k: ns[k].head(16) for k in ["val_df", "val_ext", "val_hard", "test_df", "test_ext"]}
    ns2 = load(TAGS + ["report", "eval"], EXP_ROOT=tmp_path / "exp", FINAL_DIR=tmp_path / "ft",
               RUN_FINAL_TEST=True, **small, **heads)
    assert ns2["FINAL_ID"] in {"M1", "M2"}
    assert (tmp_path / "ft/final_test.json").exists()
    assert "val_avg (세 지표 평균)" in ns2["scores"].index
    assert "aihub_metal_as_glass_battery" in ns2 and isinstance(ns2["source_acc"], pd.DataFrame)
    assert isinstance(ns2["metal_pred"], pd.DataFrame)     # AI Hub metal이 무엇으로 예측됐는지 (없으면 빈 표)


def test_report_cells_survive_without_results(tmp_path):
    ns = load(TAGS + ["report"], EXP_ROOT=tmp_path)     # 실험 결과가 아직 없는 상태
    assert ns["FINAL_ID"] is None


R1_CSV = (
    "id,change,why,val,val_ext,val_hard,val_avg,best_epoch,epochs_run,minutes,adopted,best_after\n"
    "R1,{},기준,0.8,0.5,0.6,0.6333,10,17,5.0,True,R1\n"
)


def test_report_round1_only(tmp_path):
    (tmp_path / "ladder.csv").write_text(R1_CSV, encoding="utf-8")
    shown = []
    ns = load(TAGS + ["report"], EXP_ROOT=tmp_path, display=lambda *a, **k: shown.extend(a))
    assert ns["FINAL_ID"] is None and len(ns["ladder_df_r1"]) == 1
    assert any(isinstance(o, pd.DataFrame) and "R1" in o.get("id", pd.Series(dtype=str)).tolist() for o in shown)


def test_eval_raises_when_best_pth_missing(tmp_path):
    """최종 실험의 best.pth가 없으면 다른 가중치로 조용히 대체하지 말고 FileNotFoundError를 낸다"""
    small = dict(CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    ns = load(TAGS, **small)
    tiny = [{**s, "change": {**s["change"], "epochs": 1, "batch": 8, "img_size": 64}} for s in ns["LADDER_M"][:1]]
    ns["run_ladder"](tiny, tmp_path / "exp", base_cfg=ns["M_BASE"], csv_name="ladder_m.csv")
    for p in (tmp_path / "exp").glob("*/best.pth"):
        p.unlink()
    heads = {k: ns[k].head(16) for k in ["val_df", "val_ext", "val_hard", "test_df", "test_ext"]}
    with pytest.raises(FileNotFoundError, match="best.pth"):
        load(TAGS + ["report", "eval"], EXP_ROOT=tmp_path / "exp", FINAL_DIR=tmp_path / "ft", **small, **heads)
