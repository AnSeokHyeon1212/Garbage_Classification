"""AI Hub 유리 사진 점검: 바닥(배경)과 다른 픽셀로 병 영역을 찾아 (1) 병이 사진에서 차지하는 비율, (2) 병의 색(갈색·녹색·투명)을 계산.
결과는 검토용 CSV와 병 영역을 확대한 검토 시트로 저장한다 (자동 판정은 후보 표시용이고, 최종 제외는 시트를 보고 결정)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
S = 256


def analyze(path):
    im = Image.open(ROOT / path).convert("RGB")
    im.thumbnail((S, S))
    a = np.asarray(im.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32)
    h, w, _ = a.shape
    border = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3), a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    bg = np.median(border, 0)
    spread = np.median(np.abs(border - bg), 0).mean()
    dist = np.abs(a - bg).mean(-1)
    mask = dist > max(25, 4 * spread)
    mask = ndimage.binary_opening(mask, iterations=1)
    lab, n = ndimage.label(mask)
    if n == 0:
        return dict(area=0.0, box=(0, 0, w, h), hue=np.nan, sat=0.0, val=0.0)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lab == k)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    obj = np.asarray(im, dtype=np.float32)[lab == k] / 255
    hsv = np.asarray(Image.fromarray((obj[None] * 255).astype(np.uint8)).convert("HSV"), dtype=np.float32)[0]
    colored = hsv[:, 1] > 60
    hue = float(np.median(hsv[colored, 0]) * 360 / 255) if colored.mean() > 0.15 else np.nan
    area = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
    return dict(area=area, box=box, hue=hue, sat=float(colored.mean()), val=float(hsv[:, 2].mean()))


def color_of(r):
    if np.isnan(r["hue"]):
        return "clear"
    if 15 <= r["hue"] <= 50:
        return "brown"
    if 60 <= r["hue"] <= 170:
        return "green"
    return "other"


def sheet(df, name, T=96, C=14, R=10):
    N = C * R
    for p in range(0, len(df), N):
        W = Image.new("RGB", (C * T, R * T), "white")
        d = ImageDraw.Draw(W)
        for k, (_, x) in enumerate(df.iloc[p:p + N].iterrows()):
            im = Image.open(ROOT / x.path).convert("RGB")
            im.thumbnail((S, S))
            b = x.box
            m = int(0.15 * max(b[2] - b[0], b[3] - b[1])) + 4
            im = im.crop((max(0, b[0] - m), max(0, b[1] - m), min(im.width, b[2] + m), min(im.height, b[3] + m)))
            im.thumbnail((T - 2, T - 2))
            X, Y = (k % C) * T, (k // C) * T
            W.paste(im, (X, Y))
            d.rectangle([X, Y, X + 54, Y + 10], fill="yellow")
            d.text((X + 1, Y - 1), f"{x.rid} {x.color[:2]} {x.area:.2f}", fill="black")
        W.save(OUT / f"{name}_{p // N:02d}.jpg", quality=85)


if __name__ == "__main__":
    OUT = Path(sys.argv[1])   # 검토 결과 폴더
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[2]   # extra | existing
    if which == "extra":
        df = pd.read_csv(ROOT / "01_coding/RESULT/data_split/extra_external.csv")
        df = df[(df.source == "aihub") & df.label.str.endswith("glass")]
    else:
        S_ = ROOT / "01_coding/RESULT/data_split"
        df = pd.concat([pd.read_csv(S_ / f).assign(set=f[:-4]) for f in ["train_mix.csv", "val_ext.csv", "test_ext.csv"]])
        df = df[(df.source == "aihub") & df.label.str.endswith("glass")]
    df = df.reset_index(drop=True)
    res = pd.DataFrame([analyze(p) for p in df.path])
    df = pd.concat([df, res], axis=1)
    df["color"] = df.apply(color_of, axis=1)
    df = df.sort_values(["label", "color", "area"]).reset_index(drop=True)
    df["rid"] = df.groupby("label").cumcount()
    df.to_csv(OUT / f"glass_{which}.csv", index=False, encoding="utf-8-sig")
    print(pd.crosstab(df.label, df.color))
    print(df.groupby("label").area.describe(percentiles=[.1, .25, .5]).round(3))
    for lab, g in df.groupby("label"):
        sheet(g, f"{which}_{lab}")
