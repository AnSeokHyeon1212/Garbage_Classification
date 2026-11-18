"""T-03 외부 데이터 통합 → results/train_mix.csv, results/val_ext.csv (split.csv는 읽기만)

1) 후보 수집: V2 original, RealWaste, AI Hub 직접촬영(zip에서 필요한 만큼만 꺼내 external_data/aihub/에 짧은 변 384px로 저장)
2) phash: Kaggle 전체(train/val/test) 대비 + 외부끼리. hamming<=4
   - 같은 클래스 근접 → 중복으로 제거
   - 다른 클래스 근접 → 라벨 충돌로 제외
3) val_ext 먼저 고정(RealWaste·AI Hub, 클래스당 50) → 나머지에서 D19 표대로 train 샘플(seed=42), 부족분은 Kaggle로 채움
"""
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import gc_lib as G

SEED = 42
HAM = 4
ROOT = G.ROOT
EXT_DIR = ROOT / "external_data" / "aihub"
AIHUB = ROOT / "232.재활용품 분류 및 선별 데이터" / "01-1.정식개방데이터"
V2 = ROOT / "archive(1)" / "original"
RW = ROOT / "realwaste" / "realwaste-main" / "RealWaste"

V2_MAP = {c: c for c in ["biological", "cardboard", "metal", "paper", "plastic", "trash"]}  # glass/battery/clothes/shoes 미사용
RW_MAP = {"Food Organics": "biological", "Miscellaneous Trash": "trash", "Cardboard": "cardboard",
          "Metal": "metal", "Paper": "paper", "Plastic": "plastic"}  # Glass/Vegetation/Textile 제외
# AI Hub zip → (클래스, 꺼낼 후보 수). 후보 수 = 필요량(train+val_ext) × 약 1.35 (중복 제거 여유분)
AI_ZIPS = [
    ("Validation/01.원천데이터/VS_2.직접촬영_07.유리병_001.갈색.zip", "brown-glass", 850),
    ("Validation/01.원천데이터/VS_2.직접촬영_07.유리병_002.녹색.zip", "green-glass", 830),
    ("Validation/01.원천데이터/VS_2.직접촬영_07.유리병_003.투명.zip", "white-glass", 810),
    ("Training/01.원천데이터/TS_2.직접촬영_08.건전지_001.건전지.zip", "battery", 650),
    ("Validation/01.원천데이터/VS_2.직접촬영_08.건전지_001.건전지.zip", "battery", 150),
    ("Training/01.원천데이터/TS_2.직접촬영_01.금속캔_001.철캔_3.zip", "metal", 200),
]
# D19 표: 클래스 → {source: train 장수}
D19 = {
    "battery": {"kaggle": 500, "aihub": 500},
    "biological": {"kaggle": 579, "v2": 60, "realwaste": 361},
    "brown-glass": {"kaggle": 425, "aihub": 575},
    "green-glass": {"kaggle": 440, "aihub": 560},
    "white-glass": {"kaggle": 450, "aihub": 550},
    "cardboard": {"kaggle": 400, "v2": 300, "realwaste": 300},
    "clothes": {"kaggle": 1000},
    "metal": {"kaggle": 400, "v2": 200, "realwaste": 300, "aihub": 100},
    "paper": {"kaggle": 400, "v2": 300, "realwaste": 300},
    "plastic": {"kaggle": 400, "v2": 300, "realwaste": 300},
    "shoes": {"kaggle": 1000},
    "trash": {"kaggle": 400, "v2": 200, "realwaste": 400},
}
# val_ext: 클래스 → {source: 장수}. clothes/shoes는 RealWaste·AI Hub에 없음
VAL_EXT = {
    "battery": {"aihub": 50}, "brown-glass": {"aihub": 50}, "green-glass": {"aihub": 50},
    "white-glass": {"aihub": 50}, "biological": {"realwaste": 50}, "trash": {"realwaste": 50},
    "cardboard": {"realwaste": 50}, "paper": {"realwaste": 50}, "plastic": {"realwaste": 50},
    "metal": {"realwaste": 25, "aihub": 25},
}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def extract_aihub() -> pd.DataFrame:
    rows = []
    for zrel, label, n in AI_ZIPS:
        out_dir = EXT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(AIHUB / zrel) as zf:
            names = sorted(i.filename for i in zf.infolist() if not i.is_dir() and i.filename.lower().endswith(".jpg"))
            pick = random.Random(SEED).sample(names, min(n, len(names)))
            broken = 0
            for name in pick:
                dst = out_dir / Path(name).name
                if not dst.exists():
                    try:
                        img = Image.open(io.BytesIO(zf.read(name)))
                        img.draft("RGB", (768, 768))  # 큰 JPEG를 빠르게 축소 디코딩
                        img = ImageOps.exif_transpose(img).convert("RGB")
                    except Exception:  # zip 안의 깨진/비이미지 파일은 건너뜀
                        broken += 1
                        continue
                    s = 384 / min(img.size)
                    if s < 1:
                        img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
                    img.save(dst, quality=92)
                rows.append(dict(path=rel(dst), label=label, source="aihub", origin=f"{Path(zrel).name}/{name}"))
        print(f"aihub {label}: {len(pick) - broken}장 추출, 깨진 파일 {broken}장 제외 ({Path(zrel).name})", flush=True)
    return pd.DataFrame(rows)


def list_folder(base: Path, mapping: dict, source: str) -> pd.DataFrame:
    rows = []
    for src_cls, label in mapping.items():
        for f in sorted((base / src_cls).iterdir()):
            if f.suffix.lower() in IMG_EXT:
                rows.append(dict(path=rel(f), label=label, source=source, origin=f"{src_cls}/{f.name}"))
    return pd.DataFrame(rows)


def phash_all(paths, cache: dict) -> np.ndarray:
    out = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(paths):
        if p not in cache:
            with Image.open(ROOT / p) as im:
                im.draft("L", (128, 128))
                cache[p] = str(imagehash.phash(im.convert("L")))
        out[i] = np.uint64(int(cache[p], 16))
        if (i + 1) % 5000 == 0:
            print(f"  phash {i + 1}/{len(paths)}", flush=True)
    return out


def min_ham(a: np.ndarray, b: np.ndarray, chunk=400):
    """a의 각 해시에 대해 b 중 가장 가까운 거리와 그 인덱스."""
    dmin = np.full(len(a), 99, dtype=np.int64)
    arg = np.zeros(len(a), dtype=np.int64)
    for s in range(0, len(a), chunk):
        d = np.bitwise_count(a[s:s + chunk, None] ^ b[None, :]).astype(np.int64)
        arg[s:s + chunk] = d.argmin(1)
        dmin[s:s + chunk] = d.min(1)
    return dmin, arg


def main():
    random.seed(SEED)
    split = pd.read_csv(G.SPLIT_CSV)
    ext = pd.concat([list_folder(V2, V2_MAP, "v2"), list_folder(RW, RW_MAP, "realwaste"), extract_aihub()],
                    ignore_index=True)
    print("외부 후보:", ext.groupby("source").size().to_dict(), flush=True)

    cache_path = G.RESULTS / "phash_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    hk = phash_all(split["path"].tolist(), cache)
    he = phash_all(ext["path"].tolist(), cache)
    cache_path.write_text(json.dumps(cache))

    # ① Kaggle 전체 대비
    d, j = min_ham(he, hk)
    near = d <= HAM
    same = near & (split["label"].values[j] == ext["label"].values)
    ext["status"] = "keep"
    ext.loc[same, "status"] = "dup_kaggle"
    ext.loc[near & ~same, "status"] = "conflict_kaggle"
    ext["kaggle_match"] = np.where(near, split["path"].values[j], "")

    # ② 외부끼리 (남은 것만). 같은 클래스 근접 → 먼저 나온 1장만 유지, 다른 클래스 근접 → 양쪽 제외
    idx = np.where(ext["status"] == "keep")[0]
    h = he[idx]
    labels = ext["label"].values[idx]
    order = {"aihub": 0, "realwaste": 1, "v2": 2}  # val_ext 후보(AI Hub·RealWaste)를 우선 유지
    pri = np.array([order[s] for s in ext["source"].values[idx]])
    drop_dup, drop_conf = set(), set()
    for s in range(0, len(idx), 400):
        dd = np.bitwise_count(h[s:s + 400, None] ^ h[None, :])
        for r, c in zip(*np.where(dd <= HAM)):
            a, b = s + r, c
            if a >= b:
                continue
            if labels[a] != labels[b]:
                drop_conf.update([a, b])
            else:  # 우선순위 낮은 쪽(동률이면 뒤쪽) 제거
                drop_dup.add(b if (pri[a], a) <= (pri[b], b) else a)
    ext.loc[idx[sorted(drop_dup - drop_conf)], "status"] = "dup_external"
    ext.loc[idx[sorted(drop_conf)], "status"] = "conflict_external"

    report = ext.pivot_table(index=["source", "label"], columns="status", values="path", aggfunc="count",
                             fill_value=0)
    report.to_csv(G.RESULTS / "dedup_report.csv", encoding="utf-8-sig")
    print(report, flush=True)

    pool = ext[ext["status"] == "keep"].copy()

    # ③ val_ext 먼저 고정
    ve = []
    for label, srcs in VAL_EXT.items():
        for src, n in srcs.items():
            cand = pool[(pool.label == label) & (pool.source == src)]
            ve.append(cand.sample(n=min(n, len(cand)), random_state=SEED))
    val_ext = pd.concat(ve)
    pool = pool.drop(val_ext.index)

    # ④ train: D19 표대로. 외부 부족분 → Kaggle로 채움
    k_train = split[split["split"] == "train"]
    tr, short = [], []
    for label, srcs in D19.items():
        got = 0
        for src, n in srcs.items():
            if src == "kaggle":
                continue
            cand = pool[(pool.label == label) & (pool.source == src)]
            take = cand.sample(n=min(n, len(cand)), random_state=SEED)
            if len(take) < n:
                short.append(f"{label}/{src}: {len(take)}/{n}")
            tr.append(take)
            got += len(take)
        kc = k_train[k_train.label == label]
        need_k = 1000 - got
        take_k = kc.sample(n=min(need_k, len(kc)), random_state=SEED)
        if len(take_k) < need_k:
            short.append(f"{label}/kaggle: {len(take_k)}/{need_k} (Kaggle train 부족)")
        tr.append(take_k.assign(origin="split.csv"))
    train_mix = pd.concat(tr)

    cols = ["path", "label", "split", "source"]
    train_mix = train_mix.assign(split="train")[cols].sort_values(["label", "source", "path"])
    val_ext = val_ext.assign(split="val_ext")[cols].sort_values(["label", "source", "path"])
    assert not set(train_mix.path) & set(val_ext.path)
    assert not set(val_ext.path) & set(split.path)
    assert set(train_mix[train_mix.source == "kaggle"].path) <= set(k_train.path)  # Kaggle val/test 미포함
    train_mix.to_csv(G.RESULTS / "train_mix.csv", index=False, encoding="utf-8")
    val_ext.to_csv(G.VAL_EXT_CSV, index=False, encoding="utf-8")

    counts = pd.concat([train_mix, val_ext]).pivot_table(index="label", columns=["split", "source"],
                                                         values="path", aggfunc="count", fill_value=0)
    counts["train_total"] = counts["train"].sum(axis=1)
    counts["val_ext_total"] = counts["val_ext"].sum(axis=1)
    counts.loc["합계"] = counts.sum()
    counts.to_csv(G.RESULTS / "mix_counts.csv", encoding="utf-8-sig")
    print(counts, flush=True)
    print("부족분:", short or "없음", flush=True)
    print(f"train_mix {len(train_mix)} / val_ext {len(val_ext)}", flush=True)


if __name__ == "__main__":
    main()
