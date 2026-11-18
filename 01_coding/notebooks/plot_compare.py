"""야간 실험 비교 그림 1장: python plot_compare.py E3 E4 E5 ...
figures/overnight_compare.png — 왼쪽 train·val loss, 오른쪽 val·val_ext accuracy (실험별 색)
개별 곡선 PNG는 채택된 모델만 G.plot_curve(<id>)로 따로 저장한다.
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gc_lib as G


def main(ids):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, e in enumerate(ids):
        h = pd.read_csv(G.HISTORY / f"{e}.csv")
        c = f"C{k}"
        ax[0].plot(h.epoch, h.train_loss, "--", color=c, alpha=0.6)
        ax[0].plot(h.epoch, h.val_loss, "-o", color=c, ms=3, label=f"{e} val (점선=train)")
        ax[1].plot(h.epoch, h.val_acc, "-o", color=c, ms=3, label=f"{e} val")
        ax[1].plot(h.epoch, h.val_ext_acc, ":^", color=c, ms=3, label=f"{e} val_ext")
    ax[0].set(title="Loss (실선 val, 점선 train)", xlabel="epoch", ylabel="loss")
    ax[1].set(title="Accuracy (실선 val, 점선 val_ext)", xlabel="epoch", ylabel="accuracy")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = G.FIGURES / "overnight_compare.png"
    fig.savefig(p, dpi=200)
    print(p)


if __name__ == "__main__":
    main(sys.argv[1:])
