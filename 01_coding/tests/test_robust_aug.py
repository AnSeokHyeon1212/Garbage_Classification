"""손상 증강 robust: 잡음·JPEG·빛번짐·얼룩·저조도 — 크기를 지키고 실제로 사진을 바꾼다"""
import numpy as np
import torch
from PIL import Image

from nbcells import load

NS = load(["data", "cache"])


def _img(seed=0, size=128):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(40, 220, (size, size, 3), dtype=np.uint8))


def test_robust_keeps_shape():
    tf = NS["build_transforms"](128, ["robust"])[0]
    assert all(n in repr(tf) for n in ["GaussianNoise", "JpegCompress", "Glare", "Stain", "LowLightNoise"])
    for i in range(20):
        x = tf(_img(i))
        assert isinstance(x, torch.Tensor) and x.shape == (3, 128, 128) and x.dtype == torch.float32


def test_each_robust_op_changes_image():
    base = _img(1)
    a = np.asarray(base, dtype=np.float32)
    for name in ["GaussianNoise", "JpegCompress", "Glare", "Stain", "LowLightNoise"]:
        out = NS[name]()(base)
        assert out.size == base.size, name
        assert np.abs(np.asarray(out, dtype=np.float32) - a).mean() > 1, name


def test_eval_transform_unchanged():
    ev = NS["build_transforms"](128, ["robust"])[1]
    assert "Noise" not in repr(ev) and "Glare" not in repr(ev) and "Stain" not in repr(ev)
