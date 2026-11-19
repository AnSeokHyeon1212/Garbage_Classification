"""추가 외부 데이터 구축: 클래스당 외부 학습 사진을 최대 2,000장까지 채운다.

선정 기준 (사용자 결정, 2026-11-19)
- 피사체가 잘 보이는 한 장 = 한 물체 사진만 사용 (잘라내야 하는 다물체 사진 제외)
  → AI Hub는 '직접촬영'만 사용, '영상추출'(컨베이어 프레임, 여러 물체)은 제외
- trash 보강은 스티로폼 컵·용기만
- 필요한 장수만 골라 저장 (짧은 변 최대 384px, JPEG)

후보 → phash 중복 제거(기존 Kaggle·외부·val_ext·test_ext 전부 + 후보끼리, hamming ≤ 4) → 클래스별 필요량만 선택
출력: RESULT/data_split/extra_external.csv, drive_upload/extra_public.zip, drive_upload/aihub_extra.zip
"""
import io
import json
import random
import zipfile
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "01_coding" / "RESULT" / "data_split"
AIHUB = ROOT / "232.재활용품 분류 및 선별 데이터" / "01-1.정식개방데이터"
V2 = ROOT / "archive(1)" / "original"
RW = ROOT / "realwaste" / "realwaste-main" / "RealWaste"
CLOTH = ROOT / "external_data" / "clothing_small"
HH = ROOT / "external_data" / "household_zip"   # Recyclable and Household Waste.zip에서 꺼낸 사진 (짧은 변 384px)
HF_TB = ROOT / "download" / "hf_trashbusters"
HF_SHOE = ROOT / "download" / "hf_shoes"
OUT_AIHUB = ROOT / "external_data" / "aihub_extra"
OUT_HF = ROOT / "external_data" / "hf_extra"
UPLOAD = ROOT / "drive_upload"
SEED, HAM, TARGET, SHORT = 42, 4, 2000, 384

AI_ZIPS = [  # 직접촬영만
    ("Training/01.원천데이터/TS_2.직접촬영_08.건전지_001.건전지.zip", "battery"),
    ("Validation/01.원천데이터/VS_2.직접촬영_08.건전지_001.건전지.zip", "battery"),
    ("Validation/01.원천데이터/VS_2.직접촬영_07.유리병_001.갈색.zip", "brown-glass"),
    ("Validation/01.원천데이터/VS_2.직접촬영_07.유리병_002.녹색.zip", "green-glass"),
    ("Validation/01.원천데이터/VS_2.직접촬영_07.유리병_003.투명.zip", "white-glass"),
    ("Training/01.원천데이터/TS_2.직접촬영_01.금속캔_001.철캔_3.zip", "metal"),
    ("Training/01.원천데이터/TS_2.직접촬영_01.금속캔_001.철캔_1.zip", "metal"),
]
TB_MAP = {  # TrashBusters original_label → 우리 클래스 (household_garbage·tricascade 출처만)
    "food_waste": "biological", "eggshells": "biological", "A_Foods": "biological",
    "cardboard_boxes": "cardboard", "cardboard_packaging": "cardboard", "C_Cardboard": "cardboard",
    "styrofoam_cups": "trash", "styrofoam_food_containers": "trash",
    "D_Newspaper": "paper",
    "plastic_water_bottles": "plastic", "plastic_soda_bottles": "plastic", "plastic_detergent_bottles": "plastic",
    "plastic_food_containers": "plastic", "O_Plastic Bottle": "plastic",
    "G_Brown Glass": "brown-glass", "I_Green Glass": "green-glass", "J_White Glass": "white-glass",
}
# 육안 점검 후 제외: tea_bags(상자·광고), office_paper·newspaper·magazines(문구 아이콘·인물·표지),
# F_Papers(깃발·조명 등), Q_Plastic Containers(유리병·금속통 섞임)
TB_SOURCES = {"household_garbage", "tricascade"}   # trashnet·garbage_dataset은 Kaggle과 겹칠 가능성이 커서 제외
# 예외: 캔 이외의 다양한 금속(냄비·호일·통·스프레이 캔)은 출처와 무관하게 사용 (Kaggle과 겹치는 것은 phash로 제거)
TB_METAL = {("household_garbage", "aerosol_cans"), ("tricascade", "M_Metal Containers"),
            ("garbage_dataset", "metal"), ("trashnet", "metal")}
V2_MAP = {c: c for c in ["biological", "cardboard", "metal", "paper", "plastic", "trash", "clothes", "shoes"]}
RW_MAP = {"Food Organics": "biological", "Miscellaneous Trash": "trash", "Cardboard": "cardboard",
          "Metal": "metal", "Paper": "paper", "Plastic": "plastic"}
CLOTH_MAP = {"shoes": "shoes", "hat": None}         # 나머지 옷 카테고리는 clothes, 모자는 제외
# 같은 클래스 안에서 실제 환경 사진을 먼저 고른다 (숫자가 작을수록 먼저)
PRIORITY = {"metal_diverse": -1, "aihub": 0, "realwaste": 0, "household_real": 0, "clothing": 1, "v2": 1,
            "tricascade": 2, "household_default": 2, "hf_shoes": 3}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def save_small(img: Image.Image, dst: Path):
    img = ImageOps.exif_transpose(img).convert("RGB")
    s = SHORT / min(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=92)


def used_paths() -> set:
    dfs = [pd.read_csv(SPLIT / f) for f in ["split.csv", "train_mix.csv", "val_ext.csv", "test_ext.csv"]]
    return set(pd.concat(dfs)["path"])


def local_candidates(used: set) -> list:
    rows = []
    for base, mapping, src in [(V2, V2_MAP, "v2"), (RW, RW_MAP, "realwaste")]:
        for folder, label in mapping.items():
            for f in sorted((base / folder).iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and rel(f) not in used:
                    rows.append(dict(path=rel(f), label=label, source=src, origin=f"{folder}/{f.name}"))
    # Recyclable and Household Waste 원본 zip: 검토 시트를 보고 고른 번호만 (household_selected.json)
    sel = json.loads((Path(__file__).parent / "household_selected.json").read_text())
    hh_map = {"clothing": "clothes", "shoes": "shoes", "office_paper": "paper", "newspaper": "paper"}
    for key, ids in sel.items():
        cat, ctx = key.split("/")
        for i in ids:
            f = HH / cat / ctx / f"{i}.jpg"
            rows.append(dict(path=rel(f), label=hh_map[cat], source=f"household_{'real' if ctx == 'real_world' else 'default'}",
                             origin=f"RecyclableHousehold/{cat}/{ctx}/Image_{i}"))
    for f in sorted(CLOTH.rglob("*.jpg")):
        label = CLOTH_MAP.get(f.parent.name, "clothes")
        if label:
            rows.append(dict(path=rel(f), label=label, source="clothing", origin=f"{f.parent.name}/{f.name}"))
    return rows


def aihub_candidates(per_class: dict) -> list:
    """직접촬영 zip에서 클래스별 필요량 × 1.4장만 무작위로 꺼낸다 (이미 쓴 파일명 제외)"""
    done = {p.name for p in (ROOT / "external_data" / "aihub").rglob("*.jpg")}
    names_by_cls = {}
    for zrel, label in AI_ZIPS:
        with zipfile.ZipFile(AIHUB / zrel) as zf:
            names_by_cls.setdefault(label, []).extend(
                (zrel, n) for n in zf.namelist() if n.lower().endswith(".jpg") and Path(n).name not in done)
    rows = []
    for label, names in names_by_cls.items():
        k = min(len(names), int(per_class.get(label, 0) * 1.4))
        pick = random.Random(SEED).sample(names, k)
        by_zip = {}
        for zrel, n in pick:
            by_zip.setdefault(zrel, []).append(n)
        for zrel, ns in by_zip.items():
            with zipfile.ZipFile(AIHUB / zrel) as zf:
                for n in ns:
                    dst = OUT_AIHUB / label / Path(n).name
                    if not dst.exists():
                        try:
                            im = Image.open(io.BytesIO(zf.read(n)))
                            im.draft("RGB", (768, 768))
                            save_small(im, dst)
                        except Exception:
                            continue
                    rows.append(dict(path=rel(dst), label=label, source="aihub", origin=f"{Path(zrel).name}/{n}"))
        print(f"aihub {label}: 후보 {k}장", flush=True)
    return rows


def hf_candidates() -> list:
    rows = []
    for f in sorted(HF_TB.glob("*.parquet")):
        t = pq.read_table(f, columns=["image", "source_dataset", "original_label", "image_context"]).to_pandas()
        is_metal = pd.Series(list(zip(t.source_dataset, t.original_label)), index=t.index).isin(TB_METAL)
        t = t[(t.source_dataset.isin(TB_SOURCES) & t.original_label.isin(TB_MAP)) | is_metal]
        for i, r in t.iterrows():
            if (r.source_dataset, r.original_label) in TB_METAL:
                label, src = "metal", "metal_diverse"
            else:
                label = TB_MAP[r.original_label]
                src = "tricascade" if r.source_dataset == "tricascade" else f"household_{'real' if r.image_context == 'real_world' else 'default'}"
            dst = OUT_HF / label / f"{src}_{r.original_label.replace(' ', '_')}_{f.stem[:5]}{i}.jpg"
            if not dst.exists():
                try:
                    save_small(Image.open(io.BytesIO(r.image["bytes"])), dst)
                except Exception:
                    continue
            rows.append(dict(path=rel(dst), label=label, source=src, origin=f"TrashBusters/{r.source_dataset}/{r.original_label}"))
    for f in sorted(HF_SHOE.glob("*.parquet")):
        t = pq.read_table(f).to_pandas()
        col = [c for c in t.columns if c.startswith("image")][0]
        for i, r in t.iterrows():
            dst = OUT_HF / "shoes" / f"hf_shoes_{f.stem[:5]}{i}.jpg"
            if not dst.exists():
                save_small(Image.open(io.BytesIO(r[col]["bytes"])), dst)
            rows.append(dict(path=rel(dst), label="shoes", source="hf_shoes", origin=f"ShoeSandalBoot/{i}"))
    print("HF 후보:", pd.DataFrame(rows).groupby(["label", "source"]).size().to_dict(), flush=True)
    return rows


def phash(paths, cache):
    out = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(paths):
        if p not in cache:
            with Image.open(ROOT / p) as im:
                im.draft("L", (128, 128))
                cache[p] = str(imagehash.phash(im.convert("L")))
        out[i] = np.uint64(int(cache[p], 16))
    return out


def min_ham(a, b, chunk=400):
    d = np.full(len(a), 99, dtype=np.int64)
    for s in range(0, len(a), chunk):
        d[s:s + chunk] = np.bitwise_count(a[s:s + chunk, None] ^ b[None, :]).min(1)
    return d


def main():
    mix = pd.read_csv(SPLIT / "train_mix.csv")
    have = mix[mix.source != "kaggle"].groupby("label").size().reindex(sorted(set(mix.label)), fill_value=0)
    need = (TARGET - have).clip(lower=0).to_dict()
    print("현재 외부 학습 장수:", have.to_dict(), "\n필요:", need, flush=True)

    used = used_paths()
    cand = pd.DataFrame(local_candidates(used) + aihub_candidates(need) + hf_candidates())
    cand = cand[cand.label.map(need) > 0].reset_index(drop=True)

    # --- 학습 방해 데이터 제거 (육안 검토 결과, 2026-11-19) ---
    # ① 육안으로 고른 제외 목록 (v2 paper의 크라프트지·종이컵·봉투·상자 등)
    rej = set(pd.read_csv(Path(__file__).parent / "extra_rejects.csv").path)
    # ② v2 cardboard: 우유팩·플라스틱 병 등이 많이 섞여 있어 추가분에서는 쓰지 않음 (TriCascade 등으로 대체)
    v2_cb = (cand.source == "v2") & (cand.label == "cardboard")
    # ③ AI Hub 유리: 병이 사진의 2% 미만(너무 작음)이거나 병 색이 라벨과 다르면 제외 (glass_check.analyze)
    from glass_check import analyze, color_of
    g = (cand.source == "aihub") & cand.label.str.endswith("glass")
    exp = {"brown-glass": "brown", "green-glass": "green", "white-glass": "clear"}
    bad_glass = pd.Series(False, index=cand.index)
    for i in cand.index[g]:
        r = analyze(cand.path[i])
        bad_glass[i] = r["area"] < 0.02 or color_of(r) != exp[cand.label[i]]
    drop = cand.path.isin(rej) | v2_cb | bad_glass
    print("제외:", {"육안 목록": int(cand.path.isin(rej).sum()), "v2 cardboard": int(v2_cb.sum()),
                  "AI Hub 유리(작음·색 불일치)": int(bad_glass.sum())}, flush=True)
    cand = cand[~drop].reset_index(drop=True)

    cache_path = SPLIT / "phash_cache.json"
    cache = json.loads(cache_path.read_text())
    existing = sorted(used | set(pd.read_csv(SPLIT / "val_hard.csv")["path"]))
    he = phash(existing, cache)
    hc = phash(cand.path.tolist(), cache)
    cache_path.write_text(json.dumps(cache))

    cand["dup_existing"] = min_ham(hc, he) <= HAM
    cand["prio"] = cand.source.map(PRIORITY)
    cand = cand[~cand.dup_existing].sample(frac=1, random_state=SEED).sort_values("prio", kind="stable")
    hc2 = np.array([np.uint64(int(cache[p], 16)) for p in cand.path])
    keep = np.ones(len(cand), bool)   # 후보끼리 중복: 우선순위가 높은(앞) 것을 남김
    for i in range(1, len(cand)):
        if keep[i] and (np.bitwise_count(hc2[:i][keep[:i]] ^ hc2[i]) <= HAM).any():
            keep[i] = False
    cand = cand[keep]

    def round_robin(g: pd.DataFrame, n: int) -> pd.DataFrame:
        """출처 편향을 줄이기 위해 출처별로 한 장씩 돌아가며 고른다 (출처 순서는 우선순위)"""
        g = g.assign(_k=g.groupby("source").cumcount()).sort_values(["_k", "prio"], kind="stable")
        return g.head(n).drop(columns="_k")

    picked = pd.concat([round_robin(g, need[c]) for c, g in cand.groupby("label")])
    picked = picked[["path", "label", "source", "origin"]].sort_values(["label", "source", "path"])
    picked.to_csv(SPLIT / "extra_external.csv", index=False, encoding="utf-8-sig")
    summary = picked.groupby(["label", "source"]).size().unstack(fill_value=0)
    summary["추가 합계"] = summary.sum(1)
    summary["기존 외부"] = have.reindex(summary.index)
    summary["외부 총계"] = summary["추가 합계"] + summary["기존 외부"]
    summary.to_csv(SPLIT / "extra_counts.csv", encoding="utf-8-sig")
    print(summary.to_string(), flush=True)

    for name, mask in [("aihub_extra.zip", picked.source == "aihub"), ("extra_public.zip", picked.source != "aihub")]:
        with zipfile.ZipFile(UPLOAD / name, "w", zipfile.ZIP_STORED) as z:
            for p in picked[mask].path:
                z.write(ROOT / p, p)
            if name == "extra_public.zip":
                z.write(SPLIT / "extra_external.csv", "01_coding/RESULT/data_split/extra_external.csv")
                z.write(SPLIT / "extra_counts.csv", "01_coding/RESULT/data_split/extra_counts.csv")
        print(name, int(mask.sum()), "장", round((UPLOAD / name).stat().st_size / 1e6, 1), "MB", flush=True)


if __name__ == "__main__":
    main()
