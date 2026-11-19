"""Task 3: 정리 데이터, 비율 유지, 클래스 가중치, 배경 점검 지표"""
from nbcells import load

TAGS = ["data", "cache", "model", "train"]


def test_load_train_mix2():
    ns = load(TAGS)
    assert len(ns["load_train_df"]({**ns["BASE_CFG"], "train_data": "mix2"})) == 27050


def test_class_weights_mean_one_and_order():
    ns = load(TAGS)
    w = ns["class_weights"](ns["train_mix"])
    assert len(w) == 12 and abs(float(w.mean()) - 1) < 1e-6
    i = ns["CLASSES"].index
    assert w[i("white-glass")] > w[i("shoes")]


def test_round2_metrics_keys(tmp_path):
    ns = load(TAGS, CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    base = {**ns["BASE_CFG"], "epochs": 1, "batch": 8, "img_size": 64, "train_data": "mix2", "resize": "letterbox"}
    a = ns["run_experiment"]("A", base, tmp_path / "e")
    b = ns["run_experiment"]("B", {**base, "class_weight": "sqrt_inv"}, tmp_path / "e")
    for m in (a, b):
        assert {"val_ext_by_source", "aihub_metal_as_glass_battery", "per_class_val_ext"} <= set(m)
        v = m["aihub_metal_as_glass_battery"]
        assert v is None or 0 <= v <= 1


def test_eval_criterion_unweighted():
    ns = load(TAGS)
    tc, ec = ns["make_criteria"]({**ns["BASE_CFG"], "class_weight": "sqrt_inv"}, ns["train_mix"])
    assert tc.weight is not None and ec.weight is None
    tc0, _ = ns["make_criteria"](ns["BASE_CFG"], ns["train_mix"])
    assert tc0.weight is None
