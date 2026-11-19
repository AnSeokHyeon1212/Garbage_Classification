"""Task 3: 이미지 캐시(192px), 증강·전처리"""
import os

import pandas as pd
from PIL import Image

from nbcells import load


def test_cache_size(tmp_path):
    ns = load(["data", "cache"])
    out = ns["cache_resized"](ns["val_hard"].head(5), tmp_path)
    assert len(out) == 5 and all(Image.open(f).size == (192, 192) for f in out.file)


def test_cache_idempotent(tmp_path):
    ns = load(["data", "cache"])
    a = ns["cache_resized"](ns["val_hard"].head(3), tmp_path)
    t = [os.path.getmtime(f) for f in a.file]
    b = ns["cache_resized"](ns["val_hard"].head(3), tmp_path)
    assert t == [os.path.getmtime(f) for f in b.file]


def test_cache_skips_missing_files(tmp_path):
    ns = load(["data", "cache"])
    df = pd.concat([ns["val_hard"].head(2),
                    pd.DataFrame([{"path": "no/such.jpg", "label": "metal", "source": "aihub"}])])
    assert len(ns["cache_resized"](df, tmp_path)) == 2


def test_transforms_output():
    ns = load(["data", "cache"])
    tr, ev = ns["build_transforms"](128, ["hflip", "rrc", "dark_blur"])
    img = Image.new("RGB", (192, 192))
    assert tr(img).shape == (3, 128, 128) and ev(img).shape == (3, 128, 128)


def test_split_frames_in_data_cells():
    ns = load(["data"])
    assert len(ns["val_df"]) == 2327 and len(ns["test_df"]) == 2328
