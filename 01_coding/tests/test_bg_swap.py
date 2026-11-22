"""배경 교란 bg_swap: 사진마다 추정한 물체 상자 바깥만 바꾸고, 클래스·출처와 무관하게 같은 확률로 건다"""
import inspect
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

from nbcells import load

NS = load(["data", "cache"])


def _scene(size=192, box=(0.4, 0.4, 0.6, 0.6), bg=(128, 128, 128), obj=(220, 30, 30)):
    a = np.full((size, size, 3), bg, dtype=np.uint8)
    x0, y0, x1, y1 = (int(v * size) for v in box)
    a[y0:y1, x0:x1] = obj
    return Image.fromarray(a)


def test_object_box_finds_object():
    x0, y0, x1, y1 = NS["object_box"](_scene())
    assert x0 <= 0.4 and y0 <= 0.4 and x1 >= 0.6 and y1 >= 0.6
    assert (x1 - x0) * (y1 - y0) < 0.25


def test_object_box_on_textured_ground():
    """아스팔트처럼 거친 바닥 위 작은 물체 (AI Hub 사진): 바닥 질감을 물체로 보면 안 된다"""
    rng = np.random.default_rng(3)
    a = np.clip(rng.normal(120, 30, (192, 192, 1)).repeat(3, -1), 0, 255).astype(np.uint8)
    a[86:106, 70:122] = (30, 160, 60)                       # 가운데 초록 건전지
    x0, y0, x1, y1 = NS["object_box"](Image.fromarray(a))
    assert x0 <= 70 / 192 and x1 >= 122 / 192 and y0 <= 86 / 192 and y1 >= 106 / 192
    assert (x1 - x0) * (y1 - y0) < 0.3


def test_object_box_on_real_aihub_photos():
    """실제 AI Hub 건전지 사진 40장: 상자가 사진 전체로 잡히면 배경 교란이 효과가 없다"""
    tm = NS["train_mix"]
    g = tm[(tm.label == "battery") & (tm.source == "aihub")].sample(40, random_state=0)
    files = NS["cache_resized"](g, NS["CACHE_ROOT"], name="테스트").file
    area = np.array([(lambda b: (b[2] - b[0]) * (b[3] - b[1]))(NS["object_box"](Image.open(f))) for f in files])
    assert np.median(area) < 0.7 and (area > 0.7).mean() < 0.35, (np.median(area), (area > 0.7).mean())


def test_object_box_fallback():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (192, 192, 3), dtype=np.uint8))
    assert np.allclose(NS["object_box"](img), (0.1, 0.1, 0.9, 0.9))


def test_bg_replace_keeps_object():
    for box in [(0.4, 0.4, 0.6, 0.6), (0.0, 0.3, 0.25, 0.7)]:   # 가운데 물체, 왼쪽 가장자리에 걸친 물체
        img = _scene(bg=(255, 255, 255), box=box)
        b = NS["object_box"](img)
        donor = _scene(bg=(20, 90, 20), obj=(20, 90, 20))
        a = np.asarray(img)
        x0, y0, x1, y1 = (int(v * 192) for v in box)
        for mode in ["swap", "noise", "blur", "solid"]:
            out = np.asarray(NS["bg_replace"](img, b, donor if mode == "swap" else None, mode))
            assert out.shape == a.shape
            kept = (out[y0:y1, x0:x1] == a[y0:y1, x0:x1]).all(-1).mean()
            assert kept >= 0.95, (box, mode, kept)
            outside = np.ones(a.shape[:2], bool)
            bx0, by0, bx1, by1 = (int(v * 192) for v in b)
            outside[max(0, by0 - 10):by1 + 10, max(0, bx0 - 10):bx1 + 10] = False
            if outside.any() and mode != "blur":   # 단색 배경은 흐려도 그대로라 blur는 바깥 변화 검사에서 뺌
                assert np.abs(out[outside].astype(float) - a[outside]).mean() > 10, (box, mode)


def _fake_df(tmp_path, n=2000):
    rng = np.random.default_rng(1)
    files, labels = [], []
    classes = NS["CLASSES"]
    for i in range(24):   # 사진 파일 24장을 돌려 씀
        f = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (192, 192, 3), dtype=np.uint8)).save(f)
        files.append(str(f))
    return pd.DataFrame({"file": [files[i % 24] for i in range(n)], "label": [classes[i % 12] for i in range(n)],
                         "box": ["0.2,0.2,0.8,0.8"] * n})


def test_bg_swap_is_label_free(tmp_path):
    params = set(inspect.signature(NS["bg_replace"]).parameters)
    assert not params & {"label", "source", "cls"}
    calls = []
    orig = NS["bg_replace"]
    NS["bg_replace"] = lambda img, box, donor, mode, strength=None: calls.append(1) or img
    try:
        df = _fake_df(tmp_path, 12000)
        random.seed(0)
        ds = NS["GarbageDataset"](df, None, bg_swap=True)
        hit = []
        for i in range(len(ds)):
            before = len(calls)
            ds[i]
            hit.append(len(calls) > before)
    finally:
        NS["bg_replace"] = orig
    rate = pd.Series(hit).groupby(df.label).mean()
    # 클래스마다 약 0.5 (클래스당 1,000장이면 우연에 의한 표준편차 약 0.016)
    assert rate.max() - rate.min() < 0.08 and abs(rate.mean() - 0.5) < 0.03


def test_bg_swap_speed(tmp_path):
    """배경 교란이 학습 변환 시간을 40% 넘게 늘리면 안 된다 (Colab은 이미지 처리 CPU가 병목)"""
    df = _fake_df(tmp_path, 128)
    tf = NS["build_transforms"](128, ["hflip", "dark_blur", "robust"])[0]

    def per_image(bg):
        random.seed(0)
        ds = NS["GarbageDataset"](df, tf, bg_swap=bg)
        t = time.perf_counter()
        for i in range(len(ds)):
            ds[i]
        return (time.perf_counter() - t) / len(ds)
    per_image(False)                                   # 파일 캐시 데우기
    base, with_bg = per_image(False), per_image(True)
    assert with_bg <= base * 1.4, (base, with_bg)
