"""남은 TRAIN 풀에서 두 개의 고정 표본을 만든다 (한 번만, 이후 고정). 순서 = test_ext 먼저 → meta_train.

남은 풀 = (Kaggle split=train 중 train_mix 미사용) + (외부 후보 중 train_mix·val_ext 미사용)
  - 외부 후보: V2/RealWaste 원본 폴더 + external_data/aihub (build_mix.py와 같은 클래스 매핑)
  - 안전 재검사(phash hamming<=4): Kaggle 전체(train/val/test)·train_mix·val_ext와 근접한 외부 사진 제외, 풀 내부 근접쌍은 둘 다 제외
1) test_ext.csv : 최종 TEST 보조용 외부 사진 9클래스 × 40 = 360 (val_ext와 같은 출처 구성).
                  biological·clothes·shoes는 남은 외부 사진이 없어 제외. 튜닝·선택에 쓰지 않고 최종 1회만 평가.
2) meta_train.csv: 스태킹 메타 분류기 학습용. 남은 풀에서 클래스당 100장 층화 추출(seed=42).
                   어떤 CNN도 학습에 쓰지 않은 TRAIN 내부 사진. VAL/VAL_EXT/TEST 미포함.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gc_lib as G
from build_mix import HAM, RW, RW_MAP, SEED, V2, V2_MAP, list_folder, min_ham, phash_all

TEST_EXT = {
    "battery": {"aihub": 40}, "brown-glass": {"aihub": 40}, "green-glass": {"aihub": 40},
    "white-glass": {"aihub": 40}, "metal": {"aihub": 20, "realwaste": 20},
    "cardboard": {"realwaste": 40}, "paper": {"realwaste": 40}, "plastic": {"realwaste": 40},
    "trash": {"realwaste": 40},
}
META_PER_CLASS = 100
COLS = ["path", "label", "split", "source"]


def aihub_files() -> pd.DataFrame:
    base = G.ROOT / "external_data" / "aihub"
    rows = [dict(path=f.relative_to(G.ROOT).as_posix(), label=d.name, source="aihub")
            for d in sorted(base.iterdir()) if d.is_dir() for f in sorted(d.iterdir())]
    return pd.DataFrame(rows)


def main():
    out_t, out_m = G.DATA_DIR / "test_ext.csv", G.DATA_DIR / "meta_train.csv"
    if out_t.exists() or out_m.exists():
        print("이미 있음(고정) — 재생성하지 않음")
        return
    split = pd.read_csv(G.SPLIT_CSV)
    mix = pd.read_csv(G.DATA_DIR / "train_mix.csv")
    vext = pd.read_csv(G.VAL_EXT_CSV)
    used = set(mix.path) | set(vext.path)

    ext = pd.concat([list_folder(V2, V2_MAP, "v2"), list_folder(RW, RW_MAP, "realwaste"), aihub_files()],
                    ignore_index=True)[["path", "label", "source"]]
    ext = ext[~ext.path.isin(used)].reset_index(drop=True)

    cache_path = G.DATA_DIR / "phash_cache.json"
    cache = json.loads(cache_path.read_text())
    n0 = len(cache)
    he = phash_all(ext.path.tolist(), cache)
    ref = pd.concat([split[["path"]], mix[["path"]], vext[["path"]]]).drop_duplicates()
    hr = phash_all(ref.path.tolist(), cache)
    if len(cache) > n0:
        cache_path.write_text(json.dumps(cache))
    d, _ = min_ham(he, hr)
    keep = d > HAM  # 이미 쓰인 사진(또는 Kaggle 전체)과 근접 → 제외
    dd = np.bitwise_count(he[:, None] ^ he[None, :]) if len(he) < 6000 else None
    if dd is not None:
        np.fill_diagonal(dd, 99)
        keep &= dd.min(1) > HAM  # 풀 내부 근접쌍은 양쪽 제외 (같은 물체 반복 촬영 방지)
    ext = ext[keep].reset_index(drop=True)
    print("외부 남은 풀(재검사 후):", ext.groupby(["label", "source"]).size().to_dict())

    # 1) test_ext 먼저
    te = []
    for label, srcs in TEST_EXT.items():
        for src, n in srcs.items():
            cand = ext[(ext.label == label) & (ext.source == src)]
            assert len(cand) >= n, f"{label}/{src}: {len(cand)} < {n}"
            te.append(cand.sample(n=n, random_state=SEED))
    test_ext = pd.concat(te)
    ext = ext.drop(test_ext.index)

    # 2) meta_train: Kaggle train 미사용분 + 남은 외부 → 클래스당 100
    k_left = split[(split.split == "train") & ~split.path.isin(used)][["path", "label", "source"]]
    pool = pd.concat([k_left, ext], ignore_index=True)
    meta = []
    for label, g in pool.groupby("label"):
        assert len(g) >= META_PER_CLASS, f"{label}: {len(g)}"
        meta.append(g.sample(n=META_PER_CLASS, random_state=SEED))
    meta = pd.concat(meta)

    test_ext = test_ext.assign(split="test_ext")[COLS].sort_values(["label", "source", "path"])
    meta = meta.assign(split="meta_train")[COLS].sort_values(["label", "source", "path"])
    others = set(split[split.split != "train"].path) | used
    assert not set(test_ext.path) & set(meta.path)
    assert not (set(test_ext.path) | set(meta.path)) & others
    test_ext.to_csv(out_t, index=False, encoding="utf-8")
    meta.to_csv(out_m, index=False, encoding="utf-8")
    print("test_ext:", len(test_ext), test_ext.groupby(["label", "source"]).size().to_dict())
    print("meta_train:", len(meta), meta.groupby("source").size().to_dict())


if __name__ == "__main__":
    main()
