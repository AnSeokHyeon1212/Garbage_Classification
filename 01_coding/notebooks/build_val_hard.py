"""val_hard 생성 (한 번만 실행, 이후 고정).

- 구성: Kaggle val에서 클래스당 50장(seed=42) + val_ext 500장(외부, 10클래스).
  clothes·shoes는 외부 데이터가 없어 Kaggle val 100장 → 클래스당 100장, 총 1,200장 (Kaggle 700 / 외부 500)
- 각 사진을 160x160으로 줄인 뒤 seed 고정으로 손상 1종 적용: blur / dark / glare (1/3씩)
  * blur : GaussianBlur 반경 2.0~3.0 (학습 degrade의 σ0.5~2, kernel 5보다 강함)
  * dark : 밝기 0.15~0.35배 (학습 degrade의 0.3~1.0보다 어두움)
  * glare: 무작위 위치에 흰 원형 빛번짐(반경 25~45%, 강도 0.6~0.9) — 학습 증강에 없음
- 출력: external_data/val_hard/<label>/<i>.png, 01_coding/results/val_hard.csv
- 원본 split.csv, val_ext.csv는 읽기만 함. Kaggle test는 사용하지 않음.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
import gc_lib as G

SEED, SIZE = 42, 160
OUT_DIR = G.ROOT / "external_data" / "val_hard"
OUT_CSV = G.RESULTS / "val_hard.csv"


def glare(img: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    cy, cx = rng.uniform(0.15, 0.85) * h, rng.uniform(0.15, 0.85) * w
    r = rng.uniform(0.25, 0.45) * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.clip(1 - np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r, 0, 1) ** 1.5 * rng.uniform(0.6, 0.9)
    a = 1 - (1 - a) * (1 - m[..., None])  # screen 합성: 밝은 빛이 겹친 효과
    return Image.fromarray((a * 255).clip(0, 255).astype(np.uint8))


def corrupt(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(rng.uniform(2.0, 3.0)))
    if kind == "dark":
        return ImageEnhance.Brightness(img).enhance(rng.uniform(0.15, 0.35))
    return glare(img, rng)


def main():
    if OUT_CSV.exists():
        print(f"이미 있음(고정): {OUT_CSV} — 재생성하지 않음")
        return
    split = pd.read_csv(G.SPLIT_CSV)
    val = split[split["split"] == "val"]
    parts = []
    for c, g in val.groupby("label"):
        n = 100 if c in ("clothes", "shoes") else 50
        parts.append(g.sample(n=n, random_state=SEED))
    src = pd.concat(parts + [pd.read_csv(G.VAL_EXT_CSV)], ignore_index=True)
    src = src.sort_values(["label", "source", "path"]).reset_index(drop=True)

    rng = random.Random(SEED)
    kinds = ["blur", "dark", "glare"]
    rows = []
    for i, r in src.iterrows():
        kind = rng.choice(kinds)
        img = Image.open(G.ROOT / r.path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        out = corrupt(img, kind, rng)
        p = OUT_DIR / r.label / f"{i:04d}_{kind}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        out.save(p)
        rows.append(dict(path=p.relative_to(G.ROOT).as_posix(), label=r.label, split="val_hard",
                         source=r.source, orig_path=r.path, corruption=kind))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(df.groupby(["label", "source"]).size().unstack(fill_value=0))
    print(df.corruption.value_counts())
    print(f"val_hard {len(df)}장 → {OUT_CSV}")


if __name__ == "__main__":
    main()
