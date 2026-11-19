"""Task 2: 비율 유지(letterbox) 캐시, 정리된 정규화 통계, 새 증강"""
from PIL import Image

from nbcells import load


def test_letterbox_keeps_aspect():
    ns = load(["data", "cache"])
    img = Image.new("RGB", (300, 100), (255, 0, 0))
    out = ns["letterbox"](img, 192, (10, 20, 30))
    assert out.size == (192, 192)
    assert out.getpixel((96, 5)) == (10, 20, 30) and out.getpixel((96, 96)) == (255, 0, 0)


def test_letterbox_cache_separate_dir(tmp_path):
    ns = load(["data", "cache"])
    a = ns["cache_resized"](ns["val_ext"].head(2), tmp_path, mode="letterbox")
    assert all("192_lb" in f for f in a.file)


def test_new_aug_flags_no_crop():
    ns = load(["data", "cache"])
    tf, _ = ns["build_transforms"](128, ["hflip", "vflip", "rot", "color", "trivial", "dark_blur"])
    names = [type(t).__name__ for t in tf.transforms]
    for n in ["RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation", "ColorJitter", "TrivialAugmentWide"]:
        assert n in names
    assert "RandomResizedCrop" not in names


def test_norm_stats_from_clean():
    ns = load(["data", "cache"])
    assert (ns["DATA_DIR"] / "norm_stats_clean.json").exists() and len(ns["FILL"]) == 3
