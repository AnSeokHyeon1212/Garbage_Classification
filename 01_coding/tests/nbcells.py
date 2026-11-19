"""final_model.ipynb의 셀을 태그로 골라 한 namespace에서 실행하는 테스트 도우미"""
import os
from pathlib import Path

import matplotlib
import nbformat

matplotlib.use("Agg")
NB = Path(__file__).resolve().parents[1] / "notebooks" / "final_model.ipynb"


def load(tags, **overrides) -> dict:
    """setup 셀 실행 → overrides로 전역 값 덮어쓰기 → tags에 해당하는 셀을 노트북 순서대로 실행한 namespace"""
    nb = nbformat.read(NB, as_version=4)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    ns = {"display": lambda *a, **k: None, "__name__": "__nbtest__"}
    cwd = os.getcwd()
    os.chdir(NB.parent)  # 노트북은 01_coding/notebooks에서 실행된다고 가정
    try:
        for c in cells:
            if "setup" in c.metadata.get("tags", []):
                exec(c.source, ns)
        ns.update(overrides)
        for c in cells:
            t = c.metadata.get("tags", [])
            if "setup" not in t and any(x in tags for x in t):
                exec(c.source, ns)
    finally:
        os.chdir(cwd)
    return ns
