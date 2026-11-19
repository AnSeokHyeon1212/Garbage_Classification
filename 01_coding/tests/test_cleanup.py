"""Task 1: 사전학습 코드·이전 결과가 지워졌고, 고정 분할 CSV는 남아 있는지 확인"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = [p for p in (ROOT / "01_coding").rglob("*")
        if p.suffix in {".py", ".ipynb"} and "tests" not in p.parts and ".ipynb_checkpoints" not in p.parts]


def test_no_pretrained_code():
    for p in CODE:
        t = p.read_text(encoding="utf-8")
        assert "torchvision.models" not in t and "from torchvision import models" not in t, p
        assert "ResNet18_Weights" not in t and "weights=" not in t.replace("weights = list", ""), p


def test_old_results_removed():
    assert not list((ROOT / "01_coding/RESULT/experiments").glob("*"))
    assert not (ROOT / "01_coding/RESULT/ensemble").exists()
    assert not (ROOT / "01_coding/RESULT/final_test/final_test.json").exists()


def test_fixed_splits_kept():
    for f in ["split", "train_mix", "val_ext", "val_hard", "test_ext"]:
        assert (ROOT / f"01_coding/RESULT/data_split/{f}.csv").exists()
