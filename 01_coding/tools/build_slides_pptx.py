"""발표 PPTX를 만든다: docs/presentation/slides.md(설계) + revision_prompt.md(1차 수정).

실행: python 01_coding/tools/build_slides_pptx.py  ->  docs/presentation/slides.pptx
그림은 01_coding/RESULT/figures*/ 의 PNG, 아이콘·잘라 낸 사진은 01_coding/RESULT/slide_assets/ 를 쓴다.
"""
import os
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "01_coding/RESULT"
FS = RESULT / "figures_slides"
FG = RESULT / "figures"
ASSETS = RESULT / "slide_assets"
OUT = ROOT / "docs/presentation/slides.pptx"

# ---- 디자인 토큰 ------------------------------------------------------------
WHITE, SURF = "FFFFFF", "FAFAFA"
INK, INK2, INK3, LINE = "171717", "707070", "9A9A9A", "DFDFDF"
PRI, PRI_D, OFF, WARN = "3ECF8E", "24B47E", "C7C7C7", "FF2201"
FONT = "Noto Sans KR"          # Pretendard 미설치 -> 설계 문서의 대체 글꼴
SYMBOL_FONT = "Segoe UI Symbol"
T_TITLE, T_KEY, T_REASON, T_CAP, T_BIG = 32, 20, 14, 12, 60

SW, SH = 13.333, 7.5
M = 0.6
CW = SW - 2 * M
TOP, BOT = 1.45, 6.75


def rgb(h):
    return RGBColor.from_string(h)


# ---- 아이콘 (Material Symbols Rounded) ----------------------------------------
def icon_png(name, px=512):
    """ASSETS/icons/{name}.png. 없으면 MATERIAL_FONT(글꼴 경로)와 같은 폴더의 .codepoints로 렌더."""
    p = ASSETS / "icons" / f"{name}.png"
    if p.exists():
        return p
    font_path = Path(os.environ["MATERIAL_FONT"])
    cps = dict(l.split() for l in font_path.with_suffix(".codepoints").read_text().splitlines() if l.strip())
    font = ImageFont.truetype(str(font_path), int(px * 0.9))
    ch = chr(int(cps[name], 16))
    im = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    b = d.textbbox((0, 0), ch, font=font)
    d.text(((px - (b[2] - b[0])) / 2 - b[0], (px - (b[3] - b[1])) / 2 - b[1]), ch, font=font,
           fill=(0x17, 0x17, 0x17, 255))
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p)
    return p


# ---- 그림 가공 ------------------------------------------------------------------
def crop_top(src, name):
    """그림 맨 위의 작은 제목 줄을 잘라 ASSETS/{name}.png 로 저장."""
    out = ASSETS / f"{name}.png"
    im = Image.open(src).convert("RGB")
    g = im.convert("L")
    W, H = g.size
    px = g.load()

    def dark(y):
        return any(px[x, y] < 200 for x in range(0, W, 2))

    y = 0
    while y < H and not dark(y):
        y += 1
    while y < H and dark(y):
        y += 1
    while y < H and not dark(y):
        y += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    im.crop((0, max(0, y - 4), W, H)).save(out)
    return out


# ---- 도형 도우미 ------------------------------------------------------------------
def strip_style(shape):
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def style_run(run, size, color=INK, bold=False, font=FONT):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = rgb(color)
    f.name = font
    rPr = run._r.get_or_add_rPr()
    prev = rPr.find(qn("a:latin"))
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.Element(qn(tag))
            prev.addnext(el)
        el.set("typeface", font)
        prev = el


def add_runs(p, para, size, color, bold):
    runs = [(para, {})] if isinstance(para, str) else para
    for t, o in runs:
        r = p.add_run()
        r.text = t
        style_run(r, o.get("size", size), o.get("color", color), o.get("bold", bold), o.get("font", FONT))


def text(slide, x, y, w, h, paras, size=T_KEY, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space_after=0, name=None):
    """paras: str 또는 [(text, {size,color,bold,font})] 의 목록. 한 원소가 한 문단."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if space_after:
            p.space_after = Pt(space_after)
        add_runs(p, para, size, color, bold)
    return tb


def card(slide, x, y, w, h, border=LINE, fill=WHITE, dash=False, radius=0.125, width_pt=0.75, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    strip_style(s)
    s.adjustments[0] = min(0.5, radius / min(w, h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    s.line.color.rgb = rgb(border)
    s.line.width = Pt(width_pt)
    if dash:
        s.line.dash_style = MSO_LINE.DASH
    if name:
        s.name = name
    return s


def box_text(shape, lines, size=14, color=INK):
    """lines: str 또는 런 목록의 목록. 가운데 정렬."""
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        add_runs(p, ln, size, color, False)


def image_fit(slide, path, x, y, w, h):
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    px, py = x + (w - pw) / 2, y + (h - ph) / 2
    slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    return px, py, pw, ph


def image_card(slide, path, x, y, w, h, pad=0.15, border=LINE):
    card(slide, x, y, w, h, border=border)
    return image_fit(slide, path, x + pad, y + pad, w - 2 * pad, h - 2 * pad)


def arrow(slide, x1, y1, x2, y2, color=PRI, width=1.5, dash=False, head=True, name=None):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    strip_style(c)
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = MSO_LINE.DASH
    if head:
        te = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"))
        te.set("type", "triangle")
        te.set("w", "med")
        te.set("len", "med")
    if name:
        c.name = name
    return c


def new_slide(prs, title, icon, notes, centered=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    n = len(prs.slides)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(WHITE)
    if not centered:
        isz = 0.44
        s.shapes.add_picture(str(icon_png(icon)), Inches(M), Inches(0.55 + (0.62 - isz) / 2), Inches(isz), Inches(isz))
        text(s, M + isz + 0.16, 0.55, CW - isz - 0.16, 0.62, [title], size=T_TITLE, bold=True,
             anchor=MSO_ANCHOR.MIDDLE, name="Title")
    text(s, SW - M - 1.0, 6.95, 1.0, 0.25, [str(n)], size=T_CAP, color=INK3, align=PP_ALIGN.RIGHT, name="PageNo")
    s.notes_slide.notes_text_frame.text = notes
    return s


def keywords(slide, x, y, w, items, gap=10, color=INK, h=None):
    h = h or (len(items) * 0.5 + 0.2)
    return text(slide, x, y, w, h, items, size=T_KEY, color=color, space_after=gap)


def keyword_band(slide, y, items, h=0.5, weights=None):
    n = len(items)
    gap = 0.3
    weights = weights or [1] * n
    unit = (CW - gap * (n - 1)) / sum(weights)
    x = M
    for it, wt in zip(items, weights):
        text(slide, x, y, unit * wt, h, [it], size=T_KEY, anchor=MSO_ANCHOR.MIDDLE)
        x += unit * wt + gap


def _est_lines(s, w_in, size_pt):
    """한 문단이 폭 w_in 에서 차지할 대략의 줄 수 (한글 1em, 그 밖 0.55em)."""
    if not isinstance(s, str):
        s = "".join(t for t, _ in s)
    em = sum(1.0 if ord(c) > 0x2E80 else 0.55 for c in s) * size_pt / 72
    return max(1, int(em / w_in * 1.04) + 1) if em > w_in * 0.96 else 1


def keyword_block(slide, x, y, w, items, gap_pt=14):
    """items: (kw, [reason...]) 또는 (label, kw, [reason...]).
    kw는 20pt #171717, reason은 14pt #707070, label은 12pt #9A9A9A. 항목마다 텍스트 상자 하나.
    마지막 상자의 아래 y(in)를 돌려준다."""
    for it in items:
        label, kw, reasons = it if len(it) == 3 else (None, *it)
        paras, h = [], 0.0
        if label:
            paras.append([(label, {"size": T_CAP, "color": INK3})])
            h += 0.26
        paras.append(kw)
        h += 0.42 * _est_lines(kw, w, T_KEY)
        for r in reasons:
            paras.append([(r, {"size": T_REASON, "color": INK2})] if isinstance(r, str) else r)
            h += 0.29 * _est_lines(r, w, T_REASON)
        tb = text(slide, x, y, w, h, paras, size=T_KEY, space_after=2)
        tb.text_frame.paragraphs[0 if not label else 1].space_after = Pt(4)
        y += h + gap_pt / 72
    return y - gap_pt / 72


# ==========================================================================
# 슬라이드
def slide_01_cover(prs):
    s = new_slide(prs, "", "recycling",
                  "안녕하세요. 저희는 쓰레기 사진 한 장을 보고 12종류 중 무엇인지 맞히는 모델을 만들었습니다. "
                  "사전학습 모델 없이 작은 ResNet을 직접 구현해서 처음부터 학습했고, 최종 test 정확도는 0.8823입니다.",
                  centered=True)
    isz = 0.7
    s.shapes.add_picture(str(icon_png("recycling")), Inches((SW - isz) / 2), Inches(1.15), Inches(isz), Inches(isz))
    text(s, M, 2.0, CW, 0.7, ["쓰레기 사진 12종 분류"], size=T_TITLE, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, name="Title")
    text(s, M, 2.85, CW, 1.0, ["사전학습 없이 처음부터 학습", "직접 만든 작은 ResNet"], size=T_KEY, color=INK2,
         align=PP_ALIGN.CENTER, space_after=4)
    text(s, M, 4.15, CW, 1.1, ["0.8823"], size=T_BIG, color=PRI_D, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, M, 5.3, CW, 0.3, ["test 정확도"], size=T_CAP, color=INK3, align=PP_ALIGN.CENTER)


def slide_02_toc(prs):
    s = new_slide(prs, "목차", "list",
                  "발표는 다섯 부분입니다. 데이터를 어떻게 모으고 다듬었는지, 튜닝을 어떤 규칙으로 했는지, "
                  "최종 모델과 결과, 그리고 배운 점과 한계 순서로 말씀드리겠습니다.")
    items = ["데이터", "전처리", "튜닝", "모델·결과", "결론"]
    gap = 0.3
    w = (CW - gap * 4) / 5
    for i, nm in enumerate(items):
        x = M + i * (w + gap)
        card(s, x, 3.0, w, 2.2)
        text(s, x + 0.3, 3.25, w - 0.6, 0.6, [f"{i + 1:02d}"], size=T_TITLE, color=INK3)
        text(s, x + 0.3, 4.4, w - 0.6, 0.5, [nm], size=T_KEY, anchor=MSO_ANCHOR.BOTTOM)


def slide_03_data(prs):
    s = new_slide(prs, "데이터: 9개 출처를 모으다", "dataset",
                  "처음에는 Kaggle 사진만 학습했는데, 대부분 깨끗한 배경 위 물건이라 바닥이나 매립지에서 찍은 외부 사진은 "
                  "0.14에서 0.19밖에 맞히지 못했습니다. 그래서 촬영 환경이 다른 9개 출처를 모아 27,050장으로 정리했습니다. "
                  "덕분에 가장 많은 클래스와 적은 클래스의 차이도 8.8배에서 2.4배로 줄었습니다.")
    keyword_band(s, TOP, ["Kaggle만 학습 → 외부 0.14~0.19", "9개 출처 27,050장", "불균형 8.8배 → 2.4배"],
                 weights=[1.5, 1, 1])
    cy = TOP + 0.75
    cw = (CW - 0.3) / 2
    image_card(s, crop_top(FG / "source_examples.png", "source_examples"), M, cy, cw, BOT - cy)
    image_card(s, FS / "class_counts.png", M + cw + 0.3, cy, cw, BOT - cy)


def slide_04_preprocess(prs):
    s = new_slide(prs, "전처리 ① 정리와 크기 맞춤", "cleaning_services",
                  "먼저 사진마다 지문을 만들어 거의 같은 사진을 지웠습니다. 같은 사진이 학습과 테스트에 함께 있으면 "
                  "점수가 부풀려지기 때문입니다. 라벨과 맞지 않는 사진도 직접 보고 뺐고, 모든 사진을 192px로 맞춘 뒤 "
                  "모델에는 128px로 넣고 정규화했습니다.")
    fw = CW * 0.6
    fx = M + CW - fw
    keywords(s, M, TOP + 0.3, fx - M - 0.4, ["중복 제거(사진 지문 비교)", "라벨 오류 제거", "192→128px · 정규화"], gap=18)
    image_card(s, crop_top(FG / "preprocess_steps.png", "preprocess_steps"), fx, TOP, fw, BOT - TOP)


def slide_05_augment(prs):
    s = new_slide(prs, "전처리 ② 증강: 쓴 것과 뺀 것", "flip",
                  "증강은 학습할 때만 사진마다 무작위로 넣었습니다. 쓰레기는 어느 방향으로든 놓이니까 반전과 회전을 넣었고, "
                  "그늘이나 초점이 안 맞은 사진에 대비해 어둡게와 흐림을 넣었습니다. 반대로 색 변화는 뺐는데, "
                  "유리병 세 종류는 색이 바로 정답 단서이기 때문입니다. 비율을 유지하는 letterbox도 val_hard가 "
                  "0.766에서 0.640으로 떨어져서 뺐습니다.")
    keyword_band(s, TOP, ["쓴 것: 반전 · 회전 · 어둡게/흐림", "뺀 것: 색 변화(색이 정답 단서)", "뺀 것: letterbox 0.766→0.640"])
    y1 = TOP + 0.75
    hh = (BOT - y1 - 0.25) / 2
    for j, (lab, fn, bd) in enumerate([("쓴 것", "aug_used", PRI), ("뺀 것", "aug_rejected", LINE)]):
        yy = y1 + j * (hh + 0.25)
        card(s, M, yy, CW, hh, border=bd)
        text(s, M + 0.25, yy + 0.2, 0.8, 0.3, [lab], size=T_CAP, color=INK2)
        image_fit(s, crop_top(FG / f"{fn}.png", fn), M + 1.1, yy + 0.15, CW - 1.35, hh - 0.3)


def slide_06_tuning_method(prs):
    s = new_slide(prs, "튜닝 ① 한 번에 하나씩", "tune",
                  "튜닝은 한 단계에 한 가지만 바꿨고, 점수가 지금까지 최고보다 0.01 이상 오를 때만 반영했습니다. "
                  "점수는 val_avg로, Kaggle 검증 사진, 학습에 안 쓴 외부 실사 사진, 그리고 흐림·어둡게·빛번짐을 입힌 "
                  "어려운 사진, 이 세 셋의 평균입니다. 그림에서 초록 실선이 반영된 단계, 회색 점선이 시험만 하고 버린 단계입니다.")
    left_w = 5.2
    keywords(s, M, TOP, left_w, ["한 번에 하나만 바꾼다", "val_avg +0.01 이상이면 반영", "val_avg = 세 검증 셋 평균"], gap=12)
    cards = ["val: Kaggle 검증 2,327장", "val_ext: 학습에 안 쓴 외부 실사 498장", "val_hard: 흐림·어둡게·빛번짐 1,198장"]
    ch_, cg = 0.72, 0.2
    cy0 = BOT - 3 * ch_ - 2 * cg
    for i, t in enumerate(cards):
        yy = cy0 + i * (ch_ + cg)
        card(s, M, yy, left_w, ch_)
        text(s, M + 0.25, yy, left_w - 0.5, ch_, [t], size=T_CAP, color=INK2, anchor=MSO_ANCHOR.MIDDLE)
    stem = ["출발 설정", "M1 0.813", "M3 0.848", "M8 0.877", "최종 M8"]
    sx, sw_, sh_ = 6.2, 1.8, 0.6
    gap_y = (BOT - TOP - 5 * sh_) / 4
    ys = [TOP + i * (sh_ + gap_y) for i in range(5)]
    bx0, bw, bg = 8.4, 0.98, 0.13
    branches = {1: ["M2 0.776"], 2: ["M4 0.845", "M5 0.836", "M6 0.842", "M7 0.857"], 3: ["M9 0.860"]}
    for k, labs in branches.items():
        yc = ys[k] + sh_ / 2
        xend = bx0 + (len(labs) - 1) * (bw + bg) + bw / 2
        arrow(s, sx + sw_, yc, xend, yc, color=OFF, width=1.0, dash=True, head=False)
    for i in range(4):
        arrow(s, sx + sw_ / 2, ys[i] + sh_, sx + sw_ / 2, ys[i + 1], color=PRI, width=1.5)
    for i, lab in enumerate(stem):
        b = card(s, sx, ys[i], sw_, sh_, border=PRI_D if i == 4 else PRI, width_pt=1.5 if i == 4 else 1.0, radius=0.1)
        box_text(b, [lab], size=14, color=INK)
    for k, labs in branches.items():
        for j, lab in enumerate(labs):
            b = card(s, bx0 + j * (bw + bg), ys[k] + 0.06, bw, sh_ - 0.12, border=OFF, dash=True, radius=0.08)
            box_text(b, [lab], size=T_CAP, color=INK2)


def slide_07_tuning_result(prs):
    s = new_slide(prs, "튜닝 ② 두 번 크게 올랐다", "bar_chart",
                  "막대에서 초록이 반영된 단계입니다. 가장 크게 오른 건 M3의 반전·회전 증강으로 +0.035, "
                  "그 다음이 M8에서 epoch을 30에서 60으로 늘린 +0.029였습니다. 라벨 스무딩 같은 나머지 시도는 "
                  "0.01 기준에 못 미치거나 오히려 떨어져서 기각했습니다.")
    kw = 3.4
    keywords(s, M, TOP + 0.3, kw, ["M3 반전·회전 +0.035", "M8 epoch 30→60 +0.029", "나머지는 기각"], gap=18)
    image_card(s, FS / "ladder_val_avg.png", M + kw + 0.3, TOP, CW - kw - 0.3, BOT - TOP)


def slide_08_curves(prs):
    s = new_slide(prs, "튜닝 ③ 학습 곡선으로 본 변화", "show_chart",
                  "왼쪽 M1은 학습 정확도가 1.0까지 오르는데 val은 0.83에서 멈춥니다. 사진을 통째로 외운 겁니다. "
                  "반전·회전을 넣은 M3부터는 학습과 검증이 함께 오르고, M8은 60 epoch 중 기준 점수가 가장 좋았던 "
                  "epoch 50의 가중치를 저장했습니다.")
    keyword_band(s, TOP, ["M1: train 1.0 / val 0.83 → 외움", "M3부터 함께 오름", "M8은 epoch 50 저장"])
    image_card(s, FS / "curves_m1_m3_m8.png", M, TOP + 0.75, CW, BOT - TOP - 0.75)


def slide_09_model(prs):
    s = new_slide(prs, "모델: SmallResNet 직접 구현(사전학습 없음)", "account_tree",
                  "모델은 작은 ResNet을 torch로 직접 만들었고, 사전학습 가중치는 쓰지 않았습니다. 잔차 블록 네 단계를 지나며 "
                  "채널이 48에서 384까지 늘고, 평균 풀링 뒤 Linear 하나로 12개 클래스 점수를 냅니다. 파라미터는 "
                  "6,289,212개이고, AdamW와 cosine 학습률로 학습해 가장 좋은 epoch을 저장했습니다.")
    keywords(s, M, TOP, CW, ["파라미터 629만 개", "잔차 블록 4단계 48→384", "AdamW · cosine"], gap=12)
    blocks = [(["입력", "3×128×128"], 1.1), (["줄기", "48"], 1.1), (["1단계", "48"], 1.2), (["2단계", "96"], 1.4),
              (["3단계", "192"], 1.6), (["4단계", "384"], 1.8), (["평균 풀링"], 1.1), (["Linear", "12"], 1.1)]
    bws = [1.6] + [1.2] * 7
    bg = 0.26
    x0 = M + (CW - (sum(bws) + 7 * bg)) / 2
    yc = 4.6
    for i, (lines, h) in enumerate(blocks):
        x = x0 + sum(bws[:i]) + i * bg
        bw = bws[i]
        last = i == len(blocks) - 1
        b = card(s, x, yc - h / 2, bw, h, border=PRI if last else LINE, width_pt=1.5 if last else 0.75, radius=0.12)
        box_text(b, lines, size=14, color=INK)
        if not last:
            arrow(s, x + bw + 0.03, yc, x + bw + bg - 0.03, yc, color=INK3, width=1.0)
    st_x = x0 + sum(bws[:2]) + 2 * bg
    text(s, st_x, yc + 1.05, 4 * 1.2 + 3 * bg, 0.3, ["BasicBlock ×2, 잔차 연결"], size=T_CAP, color=INK2,
         align=PP_ALIGN.CENTER)


def slide_10_result(prs):
    s = new_slide(prs, "결과와 과적합 점검", "fact_check",
                  "최종 TEST는 한 번만 했고, test 0.8823, 외부 실사 test_ext 0.9028이 나왔습니다. 과적합을 점검해 보면, "
                  "저장한 epoch에서 train이 0.982, val 0.879, test 0.882로 약 0.10 차이가 나서 가벼운 과적합은 있습니다. "
                  "하지만 검증과 TEST 점수가 거의 같아서 처음 보는 사진에도 일반화는 유지됩니다. 외부 사진은 val_ext 0.922에서 "
                  "test_ext 0.903으로 약 0.02 낮지만 큰 차이는 아닙니다.")
    bn_w, bn_h = 3.9, 1.75
    card(s, M, TOP, bn_w, bn_h)
    text(s, M + 0.3, TOP + 0.15, bn_w - 0.6, 1.0, ["0.8823"], size=T_BIG, color=PRI_D, bold=True,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, M + 0.3, TOP + 1.2, bn_w - 0.6, 0.3, ["test, 2,328장 1회 평가"], size=T_CAP, color=INK3)
    keywords(s, M + bn_w + 0.5, TOP + 0.05, CW - bn_w - 0.5,
             ["test_ext 0.9028", "train−val 차이 0.10 → 가벼운 과적합", "검증 ≈ TEST → 일반화 유지"], gap=6)
    ry = TOP + bn_h + 0.3
    rw = (CW - 0.3) / 2
    image_card(s, FS / "final_curve_gap.png", M, ry, rw, BOT - ry)
    image_card(s, FS / "test_scores.png", M + rw + 0.3, ry, rw, BOT - ry)


def slide_confusion(prs):
    s = new_slide(prs, "결과: 비슷한 쌍을 헷갈렸다", "grid_on",
                  "혼동행렬은 어떤 클래스끼리 헷갈리는지 보여 줍니다. 가장 많이 틀린 쌍은 투명 유리→플라스틱 16장, "
                  "옷↔신발 16장·12장, 건전지→금속 8장으로, 눈으로 봐도 비슷한 것들입니다. 그래서 다음 과제도 투명 재질, "
                  "옷·신발처럼 비슷한 쌍을 구분하는 쪽입니다.")
    cm_h = BOT - TOP
    cm_w = cm_h * 1501 / 1450 + 0.1
    cm_x = M + CW - cm_w
    keywords(s, M, TOP + 0.3, cm_x - M - 0.4, ["white-glass→plastic 16", "clothes↔shoes 16·12", "battery→metal 8"],
             gap=18)
    px, py, pw, ph = image_card(s, FS / "confusion_test.png", cm_x, TOP, cm_w, cm_h)
    sc = pw / 1501
    gx0, gy0, cell = 343, 49, (1451 - 343) / 12      # confusion_test.png 의 칸 격자(픽셀)
    for r, c in [(11, 8), (4, 9), (9, 4), (0, 6)]:
        b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(px + (gx0 + c * cell) * sc),
                               Inches(py + (gy0 + r * cell) * sc), Inches(cell * sc), Inches(cell * sc))
        strip_style(b)
        b.fill.background()
        b.line.color.rgb = rgb(INK)
        b.line.width = Pt(1.0)


def slide_conclusion(prs):
    s = new_slide(prs, "결론: 배운 점과 한계", "lightbulb",
                  "가장 크게 도움이 된 건 데이터를 다양하게 모은 것이었고, 그 다음이 증강, 그리고 충분한 epoch이었습니다. "
                  "한계도 있습니다. 더러운 사진 평가에서는 0.773으로, 특히 학습에 없던 강한 빛번짐 0.28, 저조도와 잡음 0.51, "
                  "얼룩 0.60에 약했고, 배경을 지우면 0.122 떨어져 배경에도 기대고 있었습니다. 그래서 지금은 노이즈 증강과 "
                  "배경 교란 증강을 실험하고 있습니다. 감사합니다.")
    fig_h = BOT - TOP - 0.45
    fig_w = fig_h * 1473 / 1314 + 0.1
    fig_x = M + CW - fig_w
    R = {"color": WARN}
    text(s, M, TOP + 0.3, fig_x - M - 0.4, 3.0, [
        "데이터 다양화 > 증강 > epoch",
        [("약점: 빛번짐 ", {}), ("0.28", R), (" · 저조도 ", {}), ("0.51", R), (" · 얼룩 ", {}), ("0.60", R)],
        "다음: 노이즈·배경 증강(실험 중)",
    ], size=T_KEY, space_after=18)
    image_card(s, FS / "dirty_by_corruption.png", fig_x, TOP, fig_w, fig_h)
    text(s, fig_x, TOP + fig_h + 0.12, fig_w, 0.3,
         ["더러운 사진 0.773 (원본 0.826 / 손상 0.720) · 배경을 지우면 −0.122"],
         size=T_CAP, color=INK2, align=PP_ALIGN.CENTER)


TUNING_ROWS = [
    ["단계", "바꾼 것", "val", "val_ext", "val_hard", "val_avg", "결과"],
    ["M1", "정리한 혼합 데이터", "0.834", "0.839", "0.766", "0.813", "반영"],
    ["M2", "letterbox", "0.850", "0.839", "0.640", "0.776", "기각"],
    ["M3", "반전·회전", "0.859", "0.900", "0.786", "0.848", "반영"],
    ["M4", "밝기·색 변화", "0.856", "0.894", "0.785", "0.845", "기각"],
    ["M5", "TrivialAugmentWide", "0.838", "0.884", "0.787", "0.836", "기각"],
    ["M6", "클래스 가중치", "0.853", "0.896", "0.777", "0.842", "기각"],
    ["M7", "라벨 스무딩 0.1", "0.869", "0.908", "0.793", "0.857", "기각"],
    ["M8", "epoch 30→60", "0.879", "0.922", "0.830", "0.877", "최종"],
    ["M9", "입력 160px", "0.885", "0.914", "0.782", "0.860", "기각"],
]


def cell_border(cell, sides, color=LINE, w_emu=9525):
    tcPr = cell._tc.get_or_add_tcPr()
    for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(side))
        if side in sides:
            ln.set("w", str(w_emu))
            sf = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
        else:
            ln.set("w", "0")
            etree.SubElement(ln, qn("a:noFill"))
    for f in tcPr.findall(qn("a:solidFill")) + tcPr.findall(qn("a:noFill")):
        tcPr.remove(f)
        tcPr.append(f)          # 스키마 순서: 테두리 다음에 채움


def tuning_table(s, x, y, w, h, col_w, size):
    rows = TUNING_ROWS
    rh = h / len(rows)
    tbl = s.shapes.add_table(len(rows), 7, Inches(x), Inches(y), Inches(w), Inches(h)).table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"):
        tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"      # No Style, No Grid
    for j, cw_ in enumerate(col_w):
        tbl.columns[j].width = Inches(cw_)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(rh)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = 0
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(SURF if i == 0 or row[0] in ("M3", "M8") else WHITE)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j <= 1 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            best = row[0] == "M8" and j == 5
            style_run(r, size, PRI_D if best else INK2 if i == 0 else INK, bold=(i == 0 or best))
            cell_border(c, ("a:lnB",) + (("a:lnT",) if i == 0 else ()))


def slide_spare(prs):
    s = new_slide(prs, "예비 자료", "inventory_2",
                  "(질문이 나올 때만) 전체 튜닝 표와 외부 사진 혼동행렬입니다. 출력층을 은닉층 두 개로 키워 봐도 0.875로 "
                  "Linear 하나의 0.878보다 낫지 않아서, 작은 출력층을 유지했습니다.")
    tw = 7.4
    text(s, M, TOP, tw, 0.45, ["M1~M9 전체 표"], size=T_KEY, anchor=MSO_ANCHOR.MIDDLE)
    ty = TOP + 0.65
    tuning_table(s, M, ty, tw, BOT - ty, [0.65, 2.35, 0.85, 0.9, 0.95, 0.9, 0.8], T_CAP)
    rx = M + tw + 0.4
    rw = M + CW - rx
    text(s, rx, TOP, rw, 0.45, ["test_ext 혼동행렬"], size=T_KEY, anchor=MSO_ANCHOR.MIDDLE)
    image_card(s, FG / "confusion_test_ext.png", rx, ty, rw, 2.75)
    oy = ty + 2.75 + 0.3
    card(s, rx, oy, rw, BOT - oy)
    text(s, rx + 0.25, oy + 0.2, rw - 0.5, 0.8, ["출력층: Linear 0.878", "vs 은닉층 2개 0.875"], size=T_KEY)
    text(s, rx + 0.25, BOT - 0.42, rw - 0.5, 0.3, ["→ 작은 출력층 유지"], size=T_CAP, color=INK2)


SLIDES = [slide_01_cover, slide_02_toc, slide_03_data, slide_04_preprocess, slide_05_augment,
          slide_06_tuning_method, slide_07_tuning_result, slide_08_curves, slide_09_model, slide_10_result,
          slide_confusion, slide_conclusion, slide_spare]


def build(out: Path = OUT) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    for f in SLIDES:
        f(prs)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


if __name__ == "__main__":
    print("saved", build())
