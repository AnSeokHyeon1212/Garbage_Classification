"""E9 단일 / E9+E8 확률 평균 / E9+E8 스태킹을 검증셋(val, val_ext, val_hard)에서 오답 기준으로 비교: python ensemble_compare.py
- TEST·test_ext는 사용하지 않는다 (TEST는 E9로 1회 사용 완료, D29)
- RESULT/ensemble/ensemble_preds.csv 가 있으면 다시 추론하지 않고 그 예측을 읽는다 (없을 때만 체크포인트로 추론)
- 산출물: RESULT/ensemble/ensemble_preds.csv (이미지별 예측), RESULT/ensemble/ensemble_cm/cm_<set>_<method>.csv (9개),
          RESULT/ensemble/ensemble_compare.json (오답 수·McNemar), RESULT/figures/ensemble_confusion.png
- 결정은 D28·A안: 최종 모델 = E9 단일. 이 비교는 보고서용 분석이다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
import gc_lib as G

SETS = ["val", "val_ext", "val_hard"]
METHODS = ["single", "avg", "stack"]
NAMES = {"single": "E9 단일 (최종)", "avg": "E9+E8 확률 평균", "stack": "E9+E8 스태킹"}
PRED_CSV = G.ENS_DIR / "ensemble_preds.csv"
CM_DIR = G.ENS_DIR / "ensemble_cm"


def predict_all() -> pd.DataFrame:
    import torch
    from stacking import probs
    torch.set_num_threads(12)
    split = pd.read_csv(G.SPLIT_CSV)
    sets = {"val": split[split.split == "val"], "val_ext": pd.read_csv(G.VAL_EXT_CSV),
            "val_hard": pd.read_csv(G.DATA_DIR / "val_hard.csv")}
    ck = torch.load(G.CKPT / "STACK.pth")
    assert (ck["best"], ck["partner"]) == ("E9", "E8"), ck["best"]
    head = torch.nn.Linear(24, 12)
    head.load_state_dict(ck["state"])
    out = []
    for k, df in sets.items():
        p9, p8 = probs("E9", df), probs("E8", df)
        with torch.no_grad():
            st = head(torch.cat([p9.clamp_min(1e-6).log(), p8.clamp_min(1e-6).log()], 1))
        d = df[["path", "label", "source"]].copy()
        d["set"] = k
        d["corruption"] = df["corruption"].values if "corruption" in df else ""
        for m, p in {"single": p9, "avg": (p9 + p8) / 2, "stack": st}.items():
            d[m] = [G.CLASSES[i] for i in p.argmax(1).numpy()]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def mcnemar(d: pd.DataFrame, m: str) -> dict:
    a, b = d.single == d.label, d[m] == d.label
    only_single, only_m = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_single, only_single + only_m).pvalue if only_single + only_m else 1.0
    return dict(only_single=only_single, only_other=only_m, p=round(float(p), 4))


def main():
    if PRED_CSV.exists():
        d = pd.read_csv(PRED_CSV, keep_default_na=False)
    else:
        d = predict_all()
        d.to_csv(PRED_CSV, index=False, encoding="utf-8")
    CM_DIR.mkdir(exist_ok=True)
    res = {"note": "검증셋만 사용. val_hard는 val·val_ext를 손상시킨 셋이라 독립 아님, 160px라 E8에 유리", "sets": {}}
    for k in SETS:
        s = d[d.set == k]
        res["sets"][k] = {"n": len(s), "errors": {m: int((s[m] != s.label).sum()) for m in METHODS},
                          "mcnemar_vs_single": {m: mcnemar(s, m) for m in ["avg", "stack"]}}
        for m in METHODS:
            cm = pd.crosstab(s.label, s[m]).reindex(index=G.CLASSES, columns=G.CLASSES, fill_value=0)
            cm.index.name, cm.columns.name = "true", "pred"
            cm.to_csv(CM_DIR / f"cm_{k}_{m}.csv", encoding="utf-8-sig")
    res["total_errors"] = {m: int((d[m] != d.label).sum()) for m in METHODS}
    json.dump(res, open(G.ENS_DIR / "ensemble_compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    plot(d)
    print(json.dumps(res["total_errors"], ensure_ascii=False))


def plot(d: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(21, 8))
    for a, m in zip(ax, METHODS):
        cm = pd.crosstab(d.label, d[m]).reindex(index=G.CLASSES, columns=G.CLASSES, fill_value=0).values
        rate = cm / cm.sum(1, keepdims=True)
        a.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(12):
            for j in range(12):
                if cm[i, j]:
                    a.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                           color="white" if rate[i, j] > 0.5 else ("crimson" if i != j and cm[i, j] >= 10 else "black"))
        a.set_xticks(range(12), G.CLASSES, rotation=60, ha="right", fontsize=8)
        a.set_yticks(range(12), G.CLASSES, fontsize=8)
        a.set(title=f"{NAMES[m]} — 오답 {int((d[m] != d.label).sum())} / {len(d)}", xlabel="예측", ylabel="정답")
    fig.suptitle("검증셋(val + val_ext + val_hard) 혼동행렬 — 색=행 비율, 숫자=장수, 빨강=10장 이상 오답", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(G.FIGURES / "ensemble_confusion.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
