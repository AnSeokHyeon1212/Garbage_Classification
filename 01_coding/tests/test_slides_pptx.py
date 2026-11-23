"""발표 PPTX(build_slides_pptx.build) 구조 검사: 도형 위치, 글자 크기, 노트, 숫자 출처, 글꼴."""
import re
import sys
from pathlib import Path

import pytest
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Pt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "01_coding/tools"))

NUM = re.compile(r"\d+(?:[.,]\d+)*")
SPEC_DOCS = [ROOT / "docs/presentation/revision_prompt.md", ROOT / "docs/presentation/slides.md"]
SYMBOL_FONT = "Segoe UI Symbol"


@pytest.fixture(scope="module")
def deck(tmp_path_factory):
    from build_slides_pptx import build
    out = build(tmp_path_factory.mktemp("deck") / "t.pptx")
    return Presentation(out)


def text_frames(slide):
    for sh in slide.shapes:
        if sh.has_text_frame:
            yield sh, sh.text_frame
        if sh.has_table:
            for row in sh.table.rows:
                for cell in row.cells:
                    yield sh, cell.text_frame


def runs(slide):
    """(shape, run) for every run on the slide, table cells included."""
    for sh, tf in text_frames(slide):
        for p in tf.paragraphs:
            for r in p.runs:
                yield sh, r


def run_color(r):
    try:
        return str(r.font.color.rgb)
    except AttributeError:
        return None


def slide_text(slide, skip_page_no=True):
    parts = []
    for sh, tf in text_frames(slide):
        if skip_page_no and sh.name == "PageNo":
            continue
        parts.append(tf.text)
    return "\n".join(parts)


def notes(slide):
    return slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ""


def spec_numbers():
    return {m for doc in SPEC_DOCS for m in NUM.findall(doc.read_text(encoding="utf-8"))}


def test_shapes_inside_slide(deck):
    W, H = deck.slide_width, deck.slide_height
    for i, s in enumerate(deck.slides, 1):
        for sh in s.shapes:
            assert sh.left >= 0 and sh.top >= 0, (i, sh.name)
            assert sh.left + sh.width <= W and sh.top + sh.height <= H, (i, sh.name)


def test_min_font_size(deck):
    for i, s in enumerate(deck.slides, 1):
        for sh, r in runs(s):
            assert r.font.size is not None and r.font.size >= Pt(12), (i, r.text)


def test_every_slide_has_notes(deck):
    for i, s in enumerate(deck.slides, 1):
        assert notes(s).strip(), i


def test_numbers_come_from_spec(deck):
    allowed = spec_numbers()
    for i, s in enumerate(deck.slides, 1):
        for tok in NUM.findall(slide_text(s) + "\n" + notes(s)):
            # 01~14 같은 작은 정수(번호·개수)는 순서 표시로 허용
            if tok.isdigit() and int(tok) <= 20:
                continue
            assert tok in allowed, (i, tok)


def test_fonts_are_noto(deck):
    for i, s in enumerate(deck.slides, 1):
        for sh, r in runs(s):
            rPr = r._r.find(qn("a:rPr"))
            faces = {rPr.find(qn(t)).get("typeface") for t in ("a:latin", "a:ea")}
            assert faces in ({"Noto Sans KR"}, {SYMBOL_FONT}), (i, r.text, faces)
