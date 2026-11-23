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
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
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
    short = {"M1": "정리 데이터", "M2": "비율 유지", "M3": "반전·회전", "M4": "밝기·색",
             "M5": "trivial", "M6": "클래스\n가중치", "M7": "라벨\n스무딩", "M8": "epoch 60", "M9": "160px"}
    cols = [DEEP if i == "M8" else MAIN if a else GRAY for i, a in zip(d["id"], d["adopted"])]
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
    ax.set_ylabel("정답")
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


def dirty_by_corruption():
    d = pd.read_csv(ROOT / "final_test/dirty_test_predictions.csv")
    g = d.groupby("corruption")["ok"].mean().sort_values()
    cols = [DEEP if k == "none" else WARN if v < 0.6 else GRAY for k, v in g.items()]
    fig, ax = plt.subplots(figsize=HALF)
    ax.barh(["원본" if k == "none" else k for k in g.index], g.values, color=cols, height=0.7)
    for i, v in enumerate(g.values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=FS)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("정확도")
    style_axes(ax, "x")
    ax.spines["left"].set_visible(False)
    save(fig, "dirty_by_corruption.png")
    clean = d[d.corruption == "none"].ok.mean()
    dirty = d[d.corruption != "none"].ok.mean()
    return d.ok.mean(), clean, dirty


if __name__ == "__main__":
    print("class total", class_counts())
    print("val_avg", {k: round(v, 3) for k, v in ladder_val_avg().items()})
    curves_m1_m3_m8()
    print("M8 ep50 train/val", [round(x, 3) for x in final_curve_gap()])
    tr, tot = confusion_test()
    print("test diag/total", tr, tot, round(tr / tot, 4))
    print("val,val_ext,test,test_ext", [round(x, 4) for x in test_scores()])
    print("dirty all/clean/corrupt", [round(x, 3) for x in dirty_by_corruption()])
