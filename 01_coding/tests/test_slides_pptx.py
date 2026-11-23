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


# ---- 장별 필수 문구 (revision_prompt.md 1차 수정) --------------------------------
REQUIRED = {
    1: ["쓰레기 사진 12종 분류", "사전학습 없이 직접 만든 CNN(SmallResNet)"],
    3: ["깨끗한 배경만 학습하면 실제 사진은 14~19%", "Kaggle 사진은 단색 배경 → 바닥·매립지 사진을 거의 못 맞힘",
        "아스팔트·매립지·실내 등 촬영 환경이 다른 사진을 섞음", "clothes 5,325 vs brown-glass 607", "AI Hub 직접 촬영"],
    4: ["중복 제거", "라벨 오류 제거", "192 → 128px 크기 통일", "정규화", "정규화 후: 모델이 보는 값(색이 조금 달라 보임)"],
    5: ["유리병 3색(갈색·초록·흰색)은 색이 정답 단서라 색을 흔들면 헷갈림", "letterbox: 비율 유지 + 빈 곳 채움 (신발 사진)"],
}

# 이유 줄: 14pt #707070 런이어야 하는 문구
REASONS = {
    3: ["Kaggle 사진은 단색 배경 → 바닥·매립지 사진을 거의 못 맞힘", "아스팔트·매립지·실내 등 촬영 환경이 다른 사진을 섞음"],
    4: ["같은 사진이 학습과 테스트에 함께 있으면 점수가 부풀려짐", "값의 범위가 일정해야 학습이 안정적이고 빠름"],
    5: ["그늘·실내 사진, 초점이 안 맞은 사진 대비", "변형이 너무 강해 오히려 하락 (0.848 → 0.836)"],
}


def test_required_text(deck):
    for n, phrases in REQUIRED.items():
        t = slide_text(deck.slides[n - 1])
        for ph in phrases:
            assert ph in t, (n, ph)


def test_reason_style(deck):
    for n, phrases in REASONS.items():
        rs = [r for _, r in runs(deck.slides[n - 1])]
        for ph in phrases:
            hit = [r for r in rs if r.text == ph]
            assert hit, (n, ph)
            assert all(r.font.size == Pt(14) and run_color(r) == "707070" for r in hit), (n, ph)


def test_cover_has_no_result(deck):
    s = deck.slides[0]
    t = slide_text(s)
    assert "0.8823" not in t and "정확도" not in t
    pics = [sh for sh in s.shapes if sh.shape_type == 13]
    assert len(pics) >= 13          # 클래스 사진 12 + 아이콘
