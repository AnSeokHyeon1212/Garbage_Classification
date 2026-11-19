"""Task 4: 학습 함수와 실험 실행기 (아주 작은 설정, CPU)"""
import pandas as pd

from nbcells import load

TINY = dict(epochs=2, patience=5, batch=8, img_size=64, max_per_class=3)


def _run(tmp_path, **kw):
    ns = load(["data", "cache", "model", "train"], CACHE_ROOT=tmp_path / "cache", SUBSET_FOR_TEST=24)
    cfg = {**ns["BASE_CFG"], **TINY, **kw}
    return ns, ns["run_experiment"]("T1", cfg, tmp_path / "exp")


def test_run_writes_outputs(tmp_path):
    ns, m = _run(tmp_path)
    d = tmp_path / "exp/T1"
    assert {"history.csv", "best.pth", "cfg.json", "metrics.json"} <= {p.name for p in d.iterdir()}
    assert abs(m["val_avg"] - (m["val"] + m["val_ext"] + m["val_hard"]) / 3) < 1e-9
    assert set(m["per_class_val"]) == set(ns["CLASSES"])


def test_resume_skips_finished(tmp_path):
    ns, m = _run(tmp_path)
    mt = (tmp_path / "exp/T1/best.pth").stat().st_mtime
    _, m2 = _run(tmp_path)
    assert m2 == m and (tmp_path / "exp/T1/best.pth").stat().st_mtime == mt


def test_best_checkpoint_matches_best_epoch(tmp_path):
    ns, m = _run(tmp_path, epochs=3)
    h = pd.read_csv(tmp_path / "exp/T1/history.csv")
    assert m["best_epoch"] == int(h.loc[h.score.idxmax(), "epoch"])


def test_cosine_reaches_zero(tmp_path):
    ns, m = _run(tmp_path, scheduler="cosine", epochs=3)
    h = pd.read_csv(tmp_path / "exp/T1/history.csv")
    assert h.lr.iloc[0] == 1e-3 and h.lr.iloc[-1] < 1e-3


def test_experiment_root_local():
    ns = load(["data", "cache", "model", "train"])
    assert ns["experiment_root"](False) == ns["ROOT"] / "01_coding" / "RESULT" / "experiments"
