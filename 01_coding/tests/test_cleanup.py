"""Task 1: 사전학습 코드·이전 결과가 지워졌고, 고정 분할 CSV는 남아 있는지 확인"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = [p for p in (ROOT / "01_coding").rglob("*")
        if p.suffix in {".py", ".ipynb"} and "tests" not in p.parts and ".ipynb_checkpoints" not in p.parts]


def _code_text(p: Path) -> str:
    """.py는 전체, .ipynb는 코드 셀만 (설명 마크다운의 '쓰지 않는다' 문장은 검사 대상 아님)"""
    if p.suffix == ".ipynb":
        import nbformat
        return "\n".join(c.source for c in nbformat.read(p, as_version=4).cells if c.cell_type == "code")
    return p.read_text(encoding="utf-8")


def test_no_pretrained_code():
    for p in CODE:
        t = _code_text(p)
        assert "torchvision.models" not in t and "from torchvision import models" not in t, p
        assert "ResNet18_Weights" not in t and "weights=" not in t.replace("weights = list", ""), p


def test_old_results_removed():
    """experiments/에는 직접 만든 CNN 사다리 결과(R1~R12, M1~M9와 ladder 표)만 있어야 한다 (사전학습 시절 결과 없음)"""
    import re
    exp = ROOT / "01_coding/RESULT/experiments"
    for p in exp.glob("*"):
        assert re.fullmatch(r"[RM]\d+|ladder(_m)?\.csv", p.name), p
        if p.is_dir():
            assert set(json.load(open(p / "cfg.json", encoding="utf-8"))) >= {"width", "aug", "train_data"}, p
    assert not (ROOT / "01_coding/RESULT/ensemble").exists()
    assert not (ROOT / "01_coding/RESULT/final_test/final_test.json").exists()


def test_fixed_splits_kept():
    for f in ["split", "train_mix", "val_ext", "val_hard", "test_ext"]:
        assert (ROOT / f"01_coding/RESULT/data_split/{f}.csv").exists()


NB = ROOT / "01_coding/notebooks/final_model.ipynb"


def test_cell_purpose_and_no_jargon():
    import re
    import nbformat
    nb = nbformat.read(NB, 4)
    assert all(c.source.lstrip().startswith("# [셀 목적]") for c in nb.cells if c.cell_type == "code")
    text = "\n".join(c.source for c in nb.cells)
    assert not re.search(r"\b(S3|E[1-9])\b", text)
