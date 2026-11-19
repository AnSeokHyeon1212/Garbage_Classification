"""Task 2: 직접 구현한 작은 ResNet"""
import torch

from nbcells import load


def test_forward_shapes():
    ns = load(["model"])
    m = ns["SmallResNet"](width=32)
    for s in (128, 160):
        assert m(torch.randn(2, 3, s, s)).shape == (2, 12)


def test_param_counts():
    ns = load(["model"])
    n = lambda **k: sum(p.numel() for p in ns["SmallResNet"](**k).parameters())
    assert n(width=32) == 2_798_124
    assert n(width=32, hidden=(256, 128)) == 2_895_276
    assert n(width=64) == 11_174_988


def test_all_params_trainable_and_no_pretrained():
    ns = load(["model"])
    m = ns["build_model"]({"width": 32, "hidden": ()})
    assert all(p.requires_grad for p in m.parameters())
    assert "models" not in ns or not hasattr(ns.get("models"), "resnet18")
