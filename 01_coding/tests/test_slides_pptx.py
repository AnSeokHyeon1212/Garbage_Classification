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
        "아스팔트·매립지·실내 등 촬영 환경이 다른 사진을 섞음", "clothes 5,325 vs brown-glass 607", "AI Hub\n직접 촬영"],
    4: ["중복 제거", "라벨 오류 제거", "192 → 128px 크기 통일", "정규화", "정규화 후: 모델이 보는 값(색이 조금 달라 보임)"],
    5: ["유리병 3색(갈색·초록·흰색)은 색이 정답 단서라 색을 흔들면 헷갈림", "letterbox: 비율 유지 + 빈 곳 채움 (신발 사진)"],
    6: ["val_avg = (val + val_ext + val_hard) ÷ 3", "M2 비율 유지(letterbox)", "M5 TrivialAugment", "M9 입력 160px",
        "폭 48 · cosine 학습률 · 어둡게/흐림 증강 · 30 epoch", "val (2,327장)", "→ 기본 실력"],
    7: ["y축은 0.70부터 시작", "30 epoch에서는 아직 오르는 중이었음"],
    8: ["차이 0.168 → 반전·회전을 넣은 M3는 0.069로 줄어듦", "검증 점수((val+val_ext)/2)가 가장 높았던 epoch의 가중치를 저장"],
    9: ["사진은 점점 작아지고, 특징(채널)은 점점 많아진다", "128×128", "8×8", "1~4단계는 각각 잔차 블록 2개",
        "2~4단계 첫 블록에서 크기 절반·채널 두 배"],
    10: ["test (Kaggle 사진 2,328장, 1회 평가)", "학습·검증에 안 쓴 외부 실사 사진 360장"],
    11: ["epoch 50: train 0.982 vs val 0.879", "epoch 26 (0.477) 최저 → epoch 50 0.537",
         "데이터 다양화 · 반전/회전 증강 · weight decay · BatchNorm · 작은 출력층 · 최고 epoch 저장"],
    13: ["사전학습 없이 직접 만든 작은 CNN으로 달성", "더러운 사진 0.773"],
    14: ["은닉층 2개로 키우면 train은 오르고(0.996 → 0.998) 검증은 낮아짐(0.878 → 0.875)", "모델을 키우면 더 외운다"],
}

# 이유 줄: 14pt #707070 런이어야 하는 문구
REASONS = {
    3: ["Kaggle 사진은 단색 배경 → 바닥·매립지 사진을 거의 못 맞힘", "아스팔트·매립지·실내 등 촬영 환경이 다른 사진을 섞음"],
    4: ["같은 사진이 학습과 테스트에 함께 있으면 점수가 부풀려짐", "값의 범위가 일정해야 학습이 안정적이고 빠름"],
    5: ["그늘·실내 사진, 초점이 안 맞은 사진 대비", "변형이 너무 강해 오히려 하락 (0.848 → 0.836)"],
    7: ["사진을 외우던 문제가 줄어듦", "0.01 기준에 못 미치거나 하락"],
    8: ["반전·회전 증강으로 train과 val이 같이 오름"],
    9: ["처음부터 학습하므로 작게 설계", "학습률을 점점 줄여 마지막에 안정적으로 수렴"],
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


def test_flowchart_branches(deck):
    """6번: M4~M7 은 M3 에서 각각 갈라진다 — 회색 점선이 각 상자 왼쪽 끝에 닿고, 상자끼리 잇는 선은 없다."""
    s = deck.slides[5]
    boxes = {sh.text_frame.text[:2]: sh for sh in s.shapes
             if sh.has_text_frame and sh.text_frame.text[:2] in ("M4", "M5", "M6", "M7")}
    assert set(boxes) == {"M4", "M5", "M6", "M7"}
    lines = [sh for sh in s.shapes if sh.shape_type == 9 or sh.__class__.__name__ == "Connector"]
    gray = [c for c in lines if str(c.line.color.rgb) == "C7C7C7" and c.line.dash_style is not None]
    tol = Pt(2)
    for k, b in boxes.items():
        mid = b.top + b.height // 2
        assert any(abs(max(c.begin_x, c.end_x) - b.left) <= tol and abs(c.begin_y - mid) <= tol
                   and abs(c.end_y - mid) <= tol for c in gray), k
    for c in gray:      # 가지 상자 사이를 잇는 선이 없음
        ends = [b for b in boxes.values() if abs(min(c.begin_x, c.end_x) - (b.left + b.width)) <= tol]
        assert not ends, c.name


TITLES = ["쓰레기 사진 12종 분류", "목차", "데이터: 9개 출처를 모으다", "전처리 ① 정리와 크기 맞춤",
          "전처리 ② 증강: 쓴 것과 뺀 것", "튜닝 ① 한 번에 하나씩", "튜닝 ② 두 번 크게 올랐다", "튜닝 ③ 학습 곡선으로 본 변화",
          "모델: SmallResNet 직접 구현(사전학습 없음)", "결과", "과적합 점검: 가볍게 있지만 일반화는 유지",
          "결과: 비슷한 쌍을 헷갈렸다", "결론: 배운 점과 한계", "예비 자료"]


def test_slide_titles(deck):
    got = [next(sh.text_frame.text for sh in s.shapes if sh.name == "Title") for s in deck.slides]
    assert got == TITLES


def test_warn_color_only_conclusion(deck):
    hits = {(i, r.text) for i, s in enumerate(deck.slides, 1) for _, r in runs(s) if run_color(r) == "FF2201"}
    assert hits == {(13, "0.28"), (13, "−0.122")}


def test_overfit_notes(deck):
    spec = (ROOT / "docs/presentation/revision_prompt.md").read_text(encoding="utf-8")
    script = next(l.strip()[2:] for l in spec.splitlines() if l.strip().startswith("> 과적합을 세 가지로"))
    assert notes(deck.slides[10]).strip() == script


def test_spare_has_no_ext_confusion(deck):
    s = deck.slides[13]
    assert "test_ext 혼동행렬" not in slide_text(s)
    pics = [sh for sh in s.shapes if sh.shape_type == 13]
    assert len(pics) == 1           # 제목 아이콘만
