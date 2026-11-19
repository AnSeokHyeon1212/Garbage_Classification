"""정리된 데이터 분할 만들기 (2026-11-19, 사용자 요청: 학습·val_ext·test_ext·val_hard 점검)

제외 규칙 (모든 세트에 같은 기준)
  ① AI Hub 유리: 병이 사진의 2% 미만이거나 병 색이 라벨과 다름 (glass_check.analyze)
  ② 육안 검토 제외 목록: split_rejects.csv (기존 세트), extra_rejects.csv (추가 데이터, build 단계에서 이미 적용)
  ③ v2 cardboard: 우유팩·플라스틱 병이 많이 섞여 있어 전부 제외
보충
  test_ext → val_ext 순서로, 빠진 장수만큼 같은 출처·같은 클래스의 깨끗한 추가 데이터(아직 어떤 학습에도 안 쓴 것)에서 옮겨 채움
학습 데이터
  Kaggle train 클래스당 1,000장(clothes는 외부가 적어 2,835장, seed 42) + 외부(기존 train_mix 외부 + 추가 데이터) 클래스당 최대 2,000장
val_hard
  Kaggle val 700장은 그대로, 외부 500장은 정리된 val_ext로 다시 만든다 (유지된 사진은 기존 손상 이미지를 그대로 사용)
출력: RESULT/data_split/{train_mix,val_ext,test_ext,val_hard}_clean.csv, clean_report.csv
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from glass_check import analyze, color_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "01_coding" / "RESULT" / "data_split"
VH_DIR = ROOT / "external_data" / "val_hard"
SEED, TARGET_EXT, KAGGLE_N, CLOTHES_KAGGLE = 42, 2000, 1000, 2835
EXP = {"brown-glass": "brown", "green-glass": "green", "white-glass": "clear"}


def bad_glass(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for i in df.index[(df.source == "aihub") & df.label.isin(EXP)]:
        r = analyze(df.path[i])
        out[i] = r["area"] < 0.02 or color_of(r) != EXP[df.label[i]]
    return out


def clean(df: pd.DataFrame, rej: set, name: str, report: list) -> pd.DataFrame:
    g = bad_glass(df)
    m = df.path.isin(rej)
    v = (df.source == "v2") & (df.label == "cardboard")
    for k, mask in [("유리 작음·색 불일치", g), ("육안 제외", m), ("v2 cardboard", v)]:
        for lab, n in df[mask].groupby("label").size().items():
            report.append(dict(set=name, label=lab, reason=k, n=int(n)))
    return df[~(g | m | v)]


def refill(cur: pd.DataFrame, orig: pd.DataFrame, pool: pd.DataFrame, name: str, report: list):
    """orig의 (클래스, 출처)별 장수가 되도록 pool에서 같은 출처 사진을 옮겨 온다"""
    need = orig.groupby(["label", "source"]).size() - cur.groupby(["label", "source"]).size().reindex(
        orig.groupby(["label", "source"]).size().index, fill_value=0)
    add = []
    for (lab, src), n in need[need > 0].items():
        cand = pool[(pool.label == lab) & (pool.source == src)].sample(frac=1, random_state=SEED).head(int(n))
        add.append(cand)
        report.append(dict(set=name, label=lab, reason=f"보충({src})", n=len(cand)))
    add = pd.concat(add) if add else pool.head(0)
    out = pd.concat([cur, add.assign(split=name)[cur.columns]], ignore_index=True)
    return out, pool[~pool.path.isin(add.path)]


def glare(img, rng):
    a = np.asarray(img, dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    cy, cx = rng.uniform(0.15, 0.85) * h, rng.uniform(0.15, 0.85) * w
    r = rng.uniform(0.25, 0.45) * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.clip(1 - np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r, 0, 1) ** 1.5 * rng.uniform(0.6, 0.9)
    return Image.fromarray(((1 - (1 - a) * (1 - m[..., None])) * 255).clip(0, 255).astype(np.uint8))


def corrupt(img, kind, rng):   # 기존 val_hard 생성과 같은 손상 (160px, blur 2~3 / 밝기 0.15~0.35 / 빛번짐)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(rng.uniform(2.0, 3.0)))
    if kind == "dark":
        return ImageEnhance.Brightness(img).enhance(rng.uniform(0.15, 0.35))
    return glare(img, rng)


def main():
    report = []
    rej = set(pd.read_csv(Path(__file__).parent / "split_rejects.csv").path)
    old = {k: pd.read_csv(S / f"{k}.csv") for k in ["train_mix", "val_ext", "test_ext", "val_hard"]}
    extra = pd.read_csv(S / "extra_external.csv").assign(split="train")

    ve, te = clean(old["val_ext"], rej, "val_ext", report), clean(old["test_ext"], rej, "test_ext", report)
    tm_ext = clean(old["train_mix"][old["train_mix"].source != "kaggle"], rej, "train_mix", report)
    pool = clean(extra, rej, "extra", report)          # 추가 데이터 (육안·v2 규칙은 build에서 이미 적용)

    te, pool = refill(te, old["test_ext"], pool, "test_ext", report)
    ve, pool = refill(ve, old["val_ext"], pool, "val_ext", report)

    # 학습: Kaggle + 외부(클래스당 최대 2,000, 출처별 번갈아)
    split = pd.read_csv(S / "split.csv")
    ktr = split[split.split == "train"]
    kag = pd.concat([g.sample(min(len(g), CLOTHES_KAGGLE if c == "clothes" else KAGGLE_N), random_state=SEED)
                     for c, g in ktr.groupby("label")]).assign(source="kaggle")
    ext = pd.concat([tm_ext, pool], ignore_index=True).drop_duplicates("path")
    ext = ext.sample(frac=1, random_state=SEED)
    ext = ext.assign(_k=ext.groupby(["label", "source"]).cumcount()).sort_values("_k", kind="stable")
    ext = ext.groupby("label").head(TARGET_EXT).drop(columns="_k")
    train = pd.concat([kag[["path", "label", "split", "source"]], ext[["path", "label", "split", "source"]]],
                      ignore_index=True).assign(split="train")
    assert not set(train.path) & (set(ve.path) | set(te.path)), "학습과 검증/테스트가 겹침"
    assert not set(ve.path) & set(te.path)

    # val_hard: Kaggle 부분 유지 + 외부 부분 재구성
    vh = old["val_hard"]
    keep = vh[(vh.source == "kaggle") | vh.orig_path.isin(ve.path)]
    new_src = ve[~ve.path.isin(vh.orig_path)].reset_index(drop=True)
    rng, kinds, rows = random.Random(SEED + 1), ["blur", "dark", "glare"], []
    for i, r in new_src.iterrows():
        kind = kinds[i % 3]
        img = Image.open(ROOT / r.path).convert("RGB").resize((160, 160), Image.BILINEAR)
        p = VH_DIR / r.label / f"c{i:04d}_{kind}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        corrupt(img, kind, rng).save(p)
        rows.append(dict(path=p.relative_to(ROOT).as_posix(), label=r.label, split="val_hard",
                         source=r.source, orig_path=r.path, corruption=kind))
    vh_new = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    report.append(dict(set="val_hard", label="(외부)", reason="재생성", n=len(rows)))

    for k, d in [("train_mix", train), ("val_ext", ve), ("test_ext", te), ("val_hard", vh_new)]:
        d.to_csv(S / f"{k}_clean.csv", index=False, encoding="utf-8")
    rep = pd.DataFrame(report)
    rep.to_csv(S / "clean_report.csv", index=False, encoding="utf-8-sig")
    print(rep.pivot_table(index=["set", "label"], columns="reason", values="n", aggfunc="sum", fill_value=0).to_string())
    for k, d in [("train", train), ("val_ext", ve), ("test_ext", te), ("val_hard", vh_new)]:
        print(k, len(d), d.groupby("label").size().to_dict())
    print("학습 외부 출처:\n", pd.crosstab(train.label, train.source).to_string())


if __name__ == "__main__":
    main()
