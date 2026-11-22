"""더러운 사진 추가 평가셋(test_dirty) 만들기 — 기록용 (원본 zip·AI Hub 원본이 있어야 실행됨)

구성: 12클래스 × 100장 = 1,200장. 클래스마다 Kaggle test 40장 + 학습·검증·테스트에 쓰지 않은 외부 실사 60장.
외부 사진은 학습한 종류의 물체만 쓰고(사람이 시트를 보고 거름), 외부가 60장에 못 미치면 모자란 만큼 Kaggle test로 채운다.
각 묶음의 절반은 원본 그대로, 절반은 손상(학습 증강에 없던 종류 포함, 약·강)을 고정 seed로 적용해 저장한다.

단계
  python build_test_dirty.py candidates   # 외부 후보를 꺼내 중복 제거 후 검토 시트 생성
  (시트를 사람이 보고 tools/test_dirty_review.json에 제외 번호·유리 색을 적는다)
  python build_test_dirty.py finalize     # 클래스별 60장 확정 → 손상 적용 → CSV·zip
"""
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from build_extra_external import AIHUB, AI_ZIPS, HAM, ROOT, SEED, SPLIT, min_ham, phash  # noqa: E402

HH_ZIP = ROOT / "Recyclable and Household Waste.zip"
RW = ROOT / "realwaste" / "realwaste-main" / "RealWaste"
OUT = ROOT / "external_data" / "test_dirty"
CAND = OUT / "candidates"
SHEETS = OUT / "sheets"
REVIEW = Path(__file__).parent / "test_dirty_review.json"
SHORT = 384
N_KAGGLE, N_EXT = 40, 60

# Household 실제 환경(real_world) 카테고리 → 우리 클래스. 유리는 색을 사람이 보고 정한다
# 아래 food_waste·eggshells·cardboard_*·shoes는 계획(PLAN)에서 쓰지 않는다. 지우면 후보끼리 중복 제거 결과가
# 달라질 수 있어(이미 검토한 시트와 번호가 어긋날 위험) 후보 목록 재현을 위해 그대로 둔다.
# (v2: 학습에 없던 종류인 커피 찌꺼기·옷 더미·사무용지·금속 캔 웹 사진은 뺀다)
HH_MAP = {
    "food_waste": "biological", "eggshells": "biological",
    "cardboard_boxes": "cardboard", "cardboard_packaging": "cardboard",
    "shoes": "shoes",
    "newspaper": "paper",
    "plastic_water_bottles": "plastic", "plastic_soda_bottles": "plastic",
    "plastic_detergent_bottles": "plastic", "plastic_food_containers": "plastic",
    "styrofoam_cups": "trash", "styrofoam_food_containers": "trash",
    "glass_beverage_bottles": "glass", "glass_food_jars": "glass", "glass_cosmetic_containers": "glass",
}
# 클래스별 외부 60장의 출처 목표 (갈색·녹색 유리는 AI Hub만, white-glass는 AI Hub 30 + Household 투명 단지 30)
# 학습한 종류의 물체만 쓴다. 옷은 외부 후보가 없어 Kaggle test로 채운다(finalize가 처리)
PLAN = {
    "battery": {"aihub": 60},
    "brown-glass": {"aihub": 60}, "green-glass": {"aihub": 60},
    "white-glass": {"aihub": 30, "household": 30},
    "metal": {"aihub": 60},
    "plastic": {"realwaste": 40, "household": 20},
    "cardboard": {"tricascade": 60},
    "trash": {"household": 50, "tb_trash": 10},
    "shoes": {"hf_shoes": 60},
    "biological": {"tricascade": 60},
    "paper": {"household": 60}, "clothes": {"household": 60},
}
CORRUPTIONS = ["noise", "jpeg", "motion_blur", "low_light_noise", "glare", "stain", "dark", "blur"]


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
    used = set()
    for f in SPLIT.glob("*.csv"):
        if f.name == "test_dirty.csv":   # v1 결과는 폐기 대상이라 v2 후보를 가리지 않는다
            continue
        df = pd.read_csv(f)
        if "path" in df.columns:
            used |= set(df["path"])
    return used


def candidates():
    rng = random.Random(SEED)
    used = used_paths()
    used_names = {Path(p).name for p in used}
    rows = []
    # Household 실제 환경 사진 (카테고리당 최대 250장 전부 후보)
    with zipfile.ZipFile(HH_ZIP) as z:
        for n in z.namelist():
            parts = n.split("/")
            if len(parts) < 5 or parts[-2] != "real_world" or parts[-3] not in HH_MAP:
                continue
            cat, i = parts[-3], Path(n).stem.split("_")[-1]
            dst = CAND / "household" / cat / f"{i}.jpg"
            if not dst.exists():
                save_small(Image.open(io.BytesIO(z.read(n))), dst)
            rows.append(dict(path=rel(dst), label=HH_MAP[cat], source="household_real", origin=f"RecyclableHousehold/{cat}/real_world/Image_{i}"))
    # AI Hub 직접 촬영 (학습·검증·테스트에 쓴 파일명 제외, 클래스당 120장 무작위)
    by_cls = {}
    for zrel, label in AI_ZIPS:
        with zipfile.ZipFile(AIHUB / zrel) as zf:
            by_cls.setdefault(label, []).extend((zrel, n) for n in zf.namelist()
                                                if n.lower().endswith(".jpg") and Path(n).name not in used_names)
    for label, names in by_cls.items():
        for zrel, n in sorted(rng.sample(names, min(len(names), 120))):
            dst = CAND / "aihub" / label / Path(n).name
            if not dst.exists():
                with zipfile.ZipFile(AIHUB / zrel) as zf:
                    im = Image.open(io.BytesIO(zf.read(n)))
                    im.draft("RGB", (768, 768))
                    save_small(im, dst)
            rows.append(dict(path=rel(dst), label=label, source="aihub", origin=f"{Path(zrel).name}/{n}"))
    # RealWaste plastic 남은 사진
    for f in sorted((RW / "Plastic").iterdir()):
        if rel(f) not in used:
            rows.append(dict(path=rel(f), label="plastic", source="realwaste", origin=f"Plastic/{f.name}"))
    # Household 실제 환경 cardboard·스티로폼은 학습에 거의 다 써서, 2차 수집 때 꺼내 두고 쓰지 않은 사진으로 보충
    # (TrashBusters TriCascade 골판지, Household 스티로폼·골판지, ShoeSandalBoot 신발)
    # "household" 접두어(cardboard)는 계획에서 쓰지 않지만, 위 HH_MAP과 같은 이유로 후보 목록 재현을 위해 둔다.
    for label, prefixes in {"cardboard": ["tricascade", "household"], "trash": ["household"], "shoes": ["hf_shoes"],
                             "biological": ["tricascade"]}.items():
        files = sorted((ROOT / "external_data" / "hf_extra" / label).glob("*.jpg"))
        if label == "shoes":   # 신발 사진은 1만 장이 넘어 300장만 무작위 후보로
            files = rng.sample([f for f in files if rel(f) not in used], 300)
        for f in files:
            pre = next((p for p in prefixes if f.stem.startswith(p)), None)
            if pre and rel(f) not in used:
                src = "tricascade" if pre == "tricascade" else ("hf_shoes" if pre == "hf_shoes" else
                                                               "household_" + f.stem.split("_")[1])
                rows.append(dict(path=rel(f), label=label, source=src, origin=f"hf_extra/{label}/{f.name}"))
    # trash 보충: Household 기본 배경 스티로폼, TrashBusters의 trash 라벨(garbage_dataset·trashnet; 마스크·칫솔만 쓰도록 사람이 시트에서 고름)
    with zipfile.ZipFile(HH_ZIP) as z:
        for n in z.namelist():
            p = n.split("/")
            if len(p) >= 5 and p[-2] == "default" and p[-3] in ("styrofoam_cups", "styrofoam_food_containers"):
                dst = CAND / "household_default" / p[-3] / f"{Path(n).stem.split('_')[-1]}.jpg"
                if not dst.exists():
                    save_small(Image.open(io.BytesIO(z.read(n))), dst)
                rows.append(dict(path=rel(dst), label="trash", source="household_default",
                                 origin=f"RecyclableHousehold/{p[-3]}/default/{Path(n).name}"))
    import pyarrow.parquet as pq
    for f in sorted((ROOT / "download" / "hf_trashbusters").glob("*.parquet")):
        t = pq.read_table(f, columns=["image", "source_dataset", "original_label"]).to_pandas()
        t = t[t.original_label.eq("trash") & t.source_dataset.isin(["garbage_dataset", "trashnet"])]
        for i, r in t.iterrows():
            dst = CAND / "tb_trash" / f"{r.source_dataset}_{f.stem[:5]}{i}.jpg"
            if not dst.exists():
                save_small(Image.open(io.BytesIO(r.image["bytes"])), dst)
            rows.append(dict(path=rel(dst), label="trash", source="tb_trash", origin=f"TrashBusters/{r.source_dataset}/trash/{i}"))
    cand = pd.DataFrame(rows)

    # 학습·검증·테스트 사진과 같은 사진(phash hamming ≤ 4) 제외, 후보끼리 중복도 제외
    cache_path = SPLIT / "phash_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    he = phash(sorted(used), cache)
    hc = phash(cand.path.tolist(), cache)
    cache_path.write_text(json.dumps(cache))
    cand["dup"] = min_ham(hc, he) <= HAM
    keep = np.ones(len(cand), bool)
    for i in range(1, len(cand)):
        if keep[i] and (np.bitwise_count(hc[:i][keep[:i]] ^ hc[i]) <= HAM).any():
            keep[i] = False
    print("기존 데이터와 중복:", int(cand.dup.sum()), "/ 후보끼리 중복:", int((~keep).sum()))
    cand = cand[~cand.dup & keep].drop(columns="dup").reset_index(drop=True)

    # 검토 시트: (출처, 클래스)마다 번호를 붙여 60장씩
    cand["sheet_id"] = cand.groupby(["source", "label"]).cumcount()
    cand.to_csv(OUT / "candidates.csv", index=False, encoding="utf-8")
    SHEETS.mkdir(parents=True, exist_ok=True)
    index = []
    for (src, lab), g in cand.groupby(["source", "label"]):
        g = g.sample(frac=1, random_state=SEED)
        for s in range(0, min(len(g), 180), 60):
            name = f"{src}_{lab}_{s // 60}"
            make_sheet(g.iloc[s:s + 60], SHEETS / f"{name}.jpg")
            index += [dict(path=p_, sheet=name) for p_ in g.path.iloc[s:s + 60]]
    pd.DataFrame(index).to_csv(OUT / "sheet_index.csv", index=False, encoding="utf-8")
    print(cand.groupby(["label", "source"]).size().to_string())


def make_sheet(g: pd.DataFrame, dst: Path, cell=150, cols=10):
    rows = (len(g) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(g.itertuples()):
        im = Image.open(ROOT / r.path).convert("RGB")
        im.thumbnail((cell - 4, cell - 18))
        x, y = (k % cols) * cell, (k // cols) * cell
        sheet.paste(im, (x + 2, y + 16))
        d.text((x + 3, y + 1), str(r.sheet_id), fill="red")
    sheet.save(dst, quality=85)


# ---------- 손상 ----------
def corrupt(img: Image.Image, kind: str, strong: bool, rng: np.random.Generator) -> Image.Image:
    img = img.convert("RGB")
    a = np.asarray(img, dtype=np.float32)
    if kind == "noise":
        a = a + rng.normal(0, 25 if strong else 12, a.shape)
    elif kind == "low_light_noise":
        a = a * (0.25 if strong else 0.45) + rng.normal(0, 12 if strong else 6, a.shape)
    elif kind == "dark":
        a = a * (0.3 if strong else 0.5)
    elif kind == "glare":
        h, w = a.shape[:2]
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        yy, xx = np.mgrid[:h, :w]
        r = (0.45 if strong else 0.3) * max(h, w)
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))[..., None]
        a = a + g * (200 if strong else 130)
    elif kind == "stain":
        out = img.copy()
        d = ImageDraw.Draw(out, "RGBA")
        h, w = a.shape[:2]
        for _ in range(4 if strong else 2):
            rx, ry = rng.uniform(0.08, 0.18) * w, rng.uniform(0.08, 0.18) * h
            cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
            col = tuple(int(v) for v in rng.integers(40, 110, 3)) + (170 if strong else 120,)
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col)
        return out
    else:
        if kind == "jpeg":
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=8 if strong else 20)
            return Image.open(buf).convert("RGB")
        if kind == "blur":
            return img.filter(ImageFilter.GaussianBlur(4 if strong else 2))
        if kind == "motion_blur":
            k = 15 if strong else 9
            kern = np.zeros((k, k), np.float32)
            kern[k // 2] = 1
            ang = rng.uniform(0, 180)
            kim = Image.fromarray((kern * 255).astype(np.uint8)).rotate(ang)
            kern = np.asarray(kim, np.float32)
            kern /= kern.sum()
            from scipy.signal import fftconvolve
            a = np.stack([fftconvolve(a[..., c], kern, mode="same") for c in range(3)], -1)
        else:
            raise ValueError(kind)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def finalize():
    rev = json.loads(REVIEW.read_text(encoding="utf-8"))
    cand = pd.read_csv(OUT / "candidates.csv")
    # 사람이 본 시트의 사진만 사용. 시트마다 "reject"(빼는 번호) 또는 "accept"(쓸 번호만) 중 하나
    idx = pd.read_csv(OUT / "sheet_index.csv").merge(cand[["path", "sheet_id"]], on="path")
    ok = set()
    for sheet, rule in rev["sheets"].items():
        g = idx[idx.sheet == sheet]
        assert len(g), sheet
        if "accept" in rule:
            g = g[g.sheet_id.isin(rule["accept"])]
        else:
            g = g[~g.sheet_id.isin(rule.get("reject", []))]
        ok |= set(g.path)
    cand = cand[cand.path.isin(ok)].copy()
    # Household 유리는 사람이 정한 색만 사용
    glass_color = {int(k): v for k, v in rev.get("household_glass_color", {}).items()}
    hg = cand.label == "glass"
    cand.loc[hg, "label"] = cand.loc[hg, "sheet_id"].map(glass_color)
    cand = cand.dropna(subset=["label"])

    picks = []
    for label, plan in PLAN.items():
        short = 0
        for src, n in plan.items():
            g = cand[(cand.label == label) & (cand.source.str.startswith(src))].sample(frac=1, random_state=SEED)
            picks.append(g.head(n + short))
            short = max(0, n + short - len(g))
        if short:   # 계획한 출처가 모자라면 같은 클래스의 다른 출처에서 채움
            rest = cand[(cand.label == label) & ~cand.path.isin(pd.concat(picks).path)].sample(frac=1, random_state=SEED)
            picks.append(rest.head(short))
    ext = pd.concat(picks)
    n_ext = ext.groupby("label").size().reindex(list(PLAN), fill_value=0)
    print("외부 사진 장수:", n_ext.to_dict())

    # Kaggle: 클래스당 40장 + 외부가 60장에 못 미친 만큼 더 (클래스당 합계 100장 유지)
    split = pd.read_csv(SPLIT / "split.csv")
    test = split[split.split == "test"]
    kag = pd.concat([g.sample(N_KAGGLE + (N_EXT - int(n_ext[c])), random_state=SEED)
                     for c, g in test.groupby("label")]).assign(source="kaggle")
    kag["origin"] = kag.path
    df = pd.concat([kag[["path", "label", "source", "origin"]], ext[["path", "label", "source", "origin"]]])
    df["group"] = np.where(df.source == "kaggle", "kaggle", "external")

    # 묶음(클래스 × kaggle/external)마다 절반 원본, 절반 손상 (종류는 돌아가며, 약·강 번갈아)
    rng = np.random.default_rng(SEED)
    rows = []
    for (label, grp), g in df.groupby(["label", "group"]):
        g = g.sample(frac=1, random_state=SEED).reset_index(drop=True)
        for k, r in g.iterrows():
            kind, strong = ("none", False) if k < len(g) // 2 else (CORRUPTIONS[k % len(CORRUPTIONS)], bool((k // len(CORRUPTIONS)) % 2))
            dst = OUT / "images" / label / f"{grp}_{k:03d}_{kind}{'_strong' if strong else ''}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(ROOT / r.path)
            im = ImageOps.exif_transpose(im).convert("RGB")
            s = SHORT / min(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
            if kind != "none":
                im = corrupt(im, kind, strong, rng)
            im.save(dst, quality=92)
            rows.append(dict(path=rel(dst), label=label, source=r.source, group=grp, corruption=kind,
                             severity=("strong" if strong else "mild") if kind != "none" else "none", origin=r.origin))
    out = pd.DataFrame(rows)
    out.to_csv(SPLIT / "test_dirty.csv", index=False, encoding="utf-8")
    print(out.groupby(["label", "group"]).size().unstack().to_string())
    print(out.groupby(["corruption", "severity"]).size().to_string())
    with zipfile.ZipFile(ROOT / "drive_upload" / "test_dirty.zip", "w", zipfile.ZIP_STORED) as z:
        for p in out.path:
            z.write(ROOT / p, p)
        z.write(SPLIT / "test_dirty.csv", "01_coding/RESULT/data_split/test_dirty.csv")
    print("zip:", ROOT / "drive_upload" / "test_dirty.zip")


if __name__ == "__main__":
    {"candidates": candidates, "finalize": finalize}[sys.argv[1]]()
