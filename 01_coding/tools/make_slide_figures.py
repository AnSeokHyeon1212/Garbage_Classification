"""슬라이드용 그래프 7개를 저장된 CSV/JSON에서 만든다 (재학습·재평가 없음)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent / "RESULT"
OUT = ROOT / "figures_slides"

# 디자인 토큰
BG, TEXT, SUB, TERT, LINE, GRID = "#FFFFFF", "#171717", "#707070", "#9A9A9A", "#DFDFDF", "#EDEDED"
MAIN, DEEP, GRAY, EXT, WARN = "#3ECF8E", "#24B47E", "#C7C7C7", "#644FC1", "#FF2201"
FS = 14
HALF, FULL = (6.4, 4.0), (12.8, 5.6)

plt.rcParams.update({
    "font.family": "Malgun Gothic", "font.size": FS, "axes.unicode_minus": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
})


def style_axes(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(labelsize=FS, colors=TEXT, length=0)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=1)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def class_counts():
    d = pd.read_csv(ROOT / "data_split/train_mix_clean.csv")
    c = d["label"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=HALF)
    ax.barh(c.index, c.values, color=MAIN, height=0.7)
    for i, v in enumerate(c.values):
        ax.text(v + 40, i, f"{v:,}", va="center", fontsize=FS, color=TEXT)
    ax.set_xlim(0, c.max() * 1.18)
    style_axes(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("학습 장수")
    save(fig, "class_counts.png")
    return int(c.sum())


def ladder_val_avg():
    d = pd.read_csv(ROOT / "experiments/ladder_m.csv")
    short = {"M1": "정리한\n혼합 데이터", "M2": "비율 유지\n(letterbox)", "M3": "반전·회전", "M4": "밝기·색\n변화",
             "M5": "Trivial\nAugment", "M6": "클래스\n가중치", "M7": "라벨\n스무딩", "M8": "epoch\n30→60",
             "M9": "입력\n160px"}
    # M1은 출발점(진회색), 반영된 단계 M3·M8만 초록
    cols = [SUB if i == "M1" else DEEP if i == "M8" else MAIN if i == "M3" else GRAY for i in d["id"]]
    fig, ax = plt.subplots(figsize=FULL)
    x = np.arange(len(d))
    ax.bar(x, d["val_avg"], color=cols, width=0.65)
    for xi, v in zip(x, d["val_avg"]):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", fontsize=FS)
    ax.set_xticks(x, [f"{i}\n{short[i]}" for i in d["id"]])
    ax.set_ylim(0.70, 0.90)
    ax.set_ylabel("val_avg")
    style_axes(ax)
    save(fig, "ladder_val_avg.png")
    return dict(zip(d["id"], d["val_avg"]))


def curves_m1_m3_m8():
    fig, axes = plt.subplots(1, 3, figsize=FULL, sharey=True)
    for ax, m in zip(axes, ["M1", "M3", "M8"]):
        h = pd.read_csv(ROOT / f"experiments/{m}/history.csv")
        ax.plot(h["epoch"], h["train_acc"], color=TEXT, lw=2, label="train")
        ax.plot(h["epoch"], h["val_acc"], color=DEEP, lw=2, label="val")
        ax.plot(h["epoch"], h["val_ext_acc"], color=EXT, lw=2, label="val_ext")
        if m == "M8":
            ax.axvline(50, color=TERT, ls=":", lw=1.8)
            ax.text(48, 0.43, "epoch 50", ha="right", fontsize=FS, color=SUB)
        ax.set_title(m, fontsize=FS, fontweight="bold", color=TEXT)
        ax.set_xlabel("epoch")
        ax.set_ylim(0.4, 1.0)
        style_axes(ax)
    axes[0].set_ylabel("정확도")
    axes[0].legend(frameon=False, fontsize=FS, loc="lower right")
    fig.tight_layout()
    save(fig, "curves_m1_m3_m8.png")


def final_curve_gap():
    h = pd.read_csv(ROOT / "experiments/M8/history.csv")
    r = h[h["epoch"] == 50].iloc[0]
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(h["epoch"], h["train_acc"], color=TEXT, lw=2, label="train")
    ax.plot(h["epoch"], h["val_acc"], color=DEEP, lw=2, label="val")
    ax.annotate("", xy=(50, r["val_acc"]), xytext=(50, r["train_acc"]),
                arrowprops=dict(arrowstyle="<->", color=WARN, lw=1.8))
    ax.text(47, 0.62, f"epoch 50\n{r['train_acc']:.3f} vs {r['val_acc']:.3f}",
            ha="right", va="center", fontsize=FS, color=TEXT)
    ax.set_xlabel("epoch")
    ax.set_ylabel("정확도")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, fontsize=FS, loc="lower right")
    style_axes(ax)
    save(fig, "final_curve_gap.png")
    return r["train_acc"], r["val_acc"]


def confusion_test():
    c = pd.read_csv(ROOT / "final_test/final_test_confusion.csv", encoding="utf-8-sig", index_col=0)
    m = c.values
    cmap = LinearSegmentedColormap.from_list("w2g", ["#FFFFFF", DEEP])
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.imshow(m, cmap=cmap, vmin=0, vmax=m.max())
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=11,
                        color="#FFFFFF" if m[i, j] > m.max() * 0.55 else TEXT)
    ax.set_xticks(range(len(c.columns)), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c.index)), c.index)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답", labelpad=10)
    ax.tick_params(labelsize=FS, length=0)
    for s in ax.spines.values():
        s.set_color(LINE)
    save(fig, "confusion_test.png")
    return int(np.trace(m)), int(m.sum())


def test_scores():
    mj = json.loads((ROOT / "experiments/M8/metrics.json").read_text(encoding="utf-8"))
    tj = json.loads((ROOT / "final_test/final_test.json").read_text(encoding="utf-8"))
    groups = [("일반", mj["val"], tj["test"]["acc"]), ("외부", mj["val_ext"], tj["test_ext"]["acc"])]
    fig, ax = plt.subplots(figsize=HALF)
    w = 0.36
    for k, (_, v, t) in enumerate(groups):
        ax.bar(k - w / 2 - 0.01, v, w, color=GRAY, label="검증" if k == 0 else None)
        ax.bar(k + w / 2 + 0.01, t, w, color=MAIN, edgecolor=DEEP, linewidth=1.5,
               label="TEST" if k == 0 else None)
        ax.text(k - w / 2 - 0.01, v + 0.008, f"{v:.3f}", ha="center", fontsize=FS)
        ax.text(k + w / 2 + 0.01, t + 0.008, f"{t:.3f}", ha="center", fontsize=FS)
    ax.set_xticks([0, 1], ["val / test", "val_ext / test_ext"])
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("정확도")
    ax.legend(frameon=False, fontsize=FS, loc="upper left", ncol=2)
    style_axes(ax)
    save(fig, "test_scores.png")
    return mj["val"], mj["val_ext"], tj["test"]["acc"], tj["test_ext"]["acc"]


KO = {"dark": "어둡게", "blur": "흐림", "motion_blur": "흔들림", "jpeg": "JPEG 압축",
      "glare": "빛번짐", "low_light_noise": "저조도+잡음", "stain": "얼룩", "noise": "잡음"}


def dirty_by_corruption():
    from matplotlib.patches import Patch
    d = pd.read_csv(ROOT / "final_test/dirty_test_predictions.csv")
    clean = d[d.corruption == "none"].ok.mean()
    p = d[d.corruption != "none"].groupby(["corruption", "severity"])["ok"].mean().unstack()
    p = p.sort_values("strong", ascending=False)  # 위에서 아래로: 강한 손상에 강한 순
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    h = 0.38
    ys, labels = [], []
    ax.barh(0, clean, color=DEEP, height=0.6)
    ax.text(clean + 0.01, 0, f"{clean:.3f}", va="center", fontsize=FS)
    ys.append(0)
    labels.append("원본")
    for i, (k, r) in enumerate(p.iterrows(), start=1):
        y = i * 1.2 + 0.2
        sc = WARN if round(r["strong"], 3) <= 0.60 else SUB
        ax.barh(y - h / 2, r["mild"], color=GRAY, height=h)
        ax.barh(y + h / 2, r["strong"], color=sc, height=h)
        ax.text(r["mild"] + 0.01, y - h / 2, f"{r['mild']:.3f}", va="center", fontsize=12)
        ax.text(r["strong"] + 0.01, y + h / 2, f"{r['strong']:.3f}", va="center", fontsize=12)
        ys.append(y)
        labels.append(KO.get(k, k))
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("정확도")
    style_axes(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(color=GRAY, label="약"), Patch(color=SUB, label="강"),
                       Patch(color=WARN, label="강 (0.60 이하)")],
              frameon=False, fontsize=FS, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3,
              handlelength=1.2, columnspacing=1.2)
    save(fig, "dirty_by_corruption.png")
    print(p.round(3).to_string())
    return d.ok.mean(), clean, d[d.corruption != "none"].ok.mean()


def loss_curve():
    """M8 train·val 손실: val loss 최저 epoch와 저장한 epoch 50을 표시 (과적합 점검)"""
    h = pd.read_csv(ROOT / "experiments/M8/history.csv")
    i = h["val_loss"].idxmin()
    lo_ep, lo = int(h.loc[i, "epoch"]), h.loc[i, "val_loss"]
    s = h[h["epoch"] == 50].iloc[0]
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(h["epoch"], h["train_loss"], color=TEXT, lw=2, label="train loss")
    ax.plot(h["epoch"], h["val_loss"], color=DEEP, lw=2, label="val loss")
    ax.scatter([lo_ep], [lo], color=DEEP, s=60, zorder=3)
    ax.annotate(f"최저 epoch {lo_ep}\n{lo:.3f}", (lo_ep, lo), xytext=(lo_ep - 4, lo + 0.45),
                ha="center", fontsize=FS, color=TEXT, arrowprops=dict(arrowstyle="-", color=TERT))
    ax.axvline(50, color=TERT, ls=":", lw=1.8)
    ax.annotate(f"저장 epoch 50\nval {s.val_loss:.3f}", (50, s.val_loss), xytext=(48.5, s.val_loss + 0.55),
                ha="right", fontsize=FS, color=SUB, arrowprops=dict(arrowstyle="-", color=TERT))
    ax.set_xlabel("epoch")
    ax.set_ylabel("손실 (loss)")
    ax.set_ylim(0, max(h["train_loss"].max(), h["val_loss"].max()) * 1.05)
    ax.legend(frameon=False, fontsize=FS, loc="upper center")
    style_axes(ax)
    save(fig, "loss_curve.png")
    return lo_ep, lo, s.val_loss


if __name__ == "__main__":
    print("M8 val loss min epoch/value, ep50", [round(float(x), 3) for x in loss_curve()])
    print("class total", class_counts())
    print("val_avg", {k: round(v, 3) for k, v in ladder_val_avg().items()})
    curves_m1_m3_m8()
    print("M8 ep50 train/val", [round(x, 3) for x in final_curve_gap()])
    tr, tot = confusion_test()
    print("test diag/total", tr, tot, round(tr / tot, 4))
    print("val,val_ext,test,test_ext", [round(x, 4) for x in test_scores()])
    print("dirty all/clean/corrupt", [round(x, 3) for x in dirty_by_corruption()])
