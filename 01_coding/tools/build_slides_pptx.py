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
    rPr.set("lang", "ko-KR")      # 한글을 단어 단위로 줄바꿈
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


def arrow(slide, x1, y1, x2, y2, color=PRI, width=1.5, dash=False, head=True, name=None, small=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    strip_style(c)
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = MSO_LINE.DASH
    if head:
        te = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"))
        te.set("type", "triangle")
        te.set("w", "sm" if small else "med")
        te.set("len", "sm" if small else "med")
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


def _item(it):
    return it if len(it) == 3 else (None, *it)


def block_height(items, w, gap_pt=14):
    """keyword_block 이 차지할 높이(in) 추정."""
    h = 0.0
    for it in items:
        label, kw, reasons = _item(it)
        h += (0.26 if label else 0) + 0.38 * _est_lines(kw, w, T_KEY)
        h += sum(0.26 * _est_lines(r, w, T_REASON) for r in reasons) + gap_pt / 72
    return h - gap_pt / 72


def keyword_block(slide, x, y, w, items, gap_pt=14):
    """items: (kw, [reason...]) 또는 (label, kw, [reason...]).
    kw는 20pt #171717, reason은 14pt #707070, label은 12pt #9A9A9A. 항목마다 텍스트 상자 하나.
    마지막 상자의 아래 y(in)를 돌려준다."""
    for it in items:
        label, kw, reasons = _item(it)
        paras = []
        if label:
            paras.append([(label, {"size": T_CAP, "color": INK3})])
        paras.append(kw)
        for r in reasons:
            paras.append([(r, {"size": T_REASON, "color": INK2})] if isinstance(r, str) else r)
        h = block_height([it], w)
        tb = text(slide, x, y, w, h, paras, size=T_KEY, space_after=2)
        tb.text_frame.paragraphs[1 if label else 0].space_after = Pt(4)
        y += h + gap_pt / 72
    return y - gap_pt / 72


def keyword_columns(slide, y, columns, weights, gap=0.3):
    """여러 칸의 keyword_block 을 가로로 놓고, 가장 낮은 칸의 아래 y를 돌려준다."""
    unit = (CW - gap * (len(columns) - 1)) / sum(weights)
    x, bottom = M, y
    for items, wt in zip(columns, weights):
        bottom = max(bottom, keyword_block(slide, x, y, unit * wt, items))
        x += unit * wt + gap
    return bottom


# ---- 표지·출처 사진 ------------------------------------------------------------
CLASSES = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass",
           "metal", "paper", "plastic", "shoes", "trash", "white-glass"]
COVER_OVERRIDE: dict = {}      # {"battery": "archive/.../x.jpg"} 처럼 지정하면 그 사진을 씀


def square_crop(src, out, px=400):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    k = min(w, h)
    im = im.crop(((w - k) // 2, (h - k) // 2, (w - k) // 2 + k, (h - k) // 2 + k)).resize((px, px), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return out


def pick_cover_photos() -> dict:
    """클래스마다 Kaggle 학습 사진 1장을 정사각형 400px로 ASSETS/cover/{label}.png 에 저장(있으면 재사용)."""
    import pandas as pd
    outs = {c: ASSETS / "cover" / f"{c}.png" for c in CLASSES}
    if all(o.exists() for o in outs.values()):
        return outs
    d = pd.read_csv(RESULT / "data_split/train_mix_clean.csv")
    d = d[(d["split"] == "train") & (d["source"] == "kaggle")]
    for c in CLASSES:
        src = COVER_OVERRIDE.get(c) or d[d["label"] == c].iloc[0]["path"]
        square_crop(ROOT / src, outs[c])
    return outs


# source_examples.png 안의 사진 10장(영문 라벨 제외) 픽셀 상자: 위 줄 왼→오, 아래 줄 왼→오
SOURCE_BOXES = [(42, 83, 152, 230), (200, 108, 373, 205), (403, 83, 549, 230), (615, 80, 715, 232),
                (788, 115, 923, 198), (11, 292, 184, 389), (225, 275, 340, 405), (403, 285, 548, 405),
                (578, 276, 752, 404), (799, 268, 909, 414)]
SOURCE_NAMES = ["Kaggle 원본", "AI Hub\n직접 촬영", "RealWaste\n매립지", "TriCascade", "생활 쓰레기\n(실제 환경)",
                "Garbage V2", "신발 제품 사진", "생활 쓰레기\n(기본 배경)", "다양한 금속", "clothing\n옷·신발"]


def crop_source_photos() -> list:
    im = Image.open(FG / "source_examples.png").convert("RGB")
    outs = []
    for i, b in enumerate(SOURCE_BOXES):
        o = ASSETS / "sources" / f"{i:02d}.png"
        o.parent.mkdir(parents=True, exist_ok=True)
        im.crop(b).save(o)
        outs.append(o)
    return outs


def aug_rejected_4() -> Path:
    """aug_rejected.png 에서 제목 줄과 마지막(두 번째 TrivialAugment) 칸을 잘라 4칸으로."""
    src = crop_top(FG / "aug_rejected.png", "aug_rejected")
    im = Image.open(src)
    o = ASSETS / "aug_rejected_4.png"
    im.crop((0, 0, 760, im.size[1])).save(o)
    return o


# ==========================================================================
# 슬라이드
def slide_01_cover(prs):
    s = new_slide(prs, "", "recycling",
                  "안녕하세요. 저희는 쓰레기 사진 한 장을 보고 12종류 중 무엇인지 맞히는 모델을 만들었습니다. "
                  "사전학습 모델 없이 작은 CNN인 SmallResNet을 직접 구현해서 처음부터 학습했습니다.",
                  centered=True)
    isz = 0.6
    s.shapes.add_picture(str(icon_png("recycling")), Inches((SW - isz) / 2), Inches(0.7), Inches(isz), Inches(isz))
    text(s, M, 1.45, CW, 0.7, ["쓰레기 사진 12종 분류"], size=T_TITLE, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, name="Title")
    text(s, M, 2.2, CW, 0.45, ["사전학습 없이 직접 만든 CNN(SmallResNet)"], size=T_KEY, color=INK2,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    photos = pick_cover_photos()
    sq, gap, cap = 1.25, 0.2, 0.35
    x0 = (SW - (6 * sq + 5 * gap)) / 2
    for i, c in enumerate(CLASSES):
        x = x0 + (i % 6) * (sq + gap)
        y = 3.0 + (i // 6) * (sq + cap + gap)
        s.shapes.add_picture(str(photos[c]), Inches(x), Inches(y), Inches(sq), Inches(sq))
        text(s, x - 0.1, y + sq + 0.06, sq + 0.2, 0.25, [c], size=T_CAP, color=INK3, align=PP_ALIGN.CENTER)


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
    bottom = keyword_columns(s, TOP, [
        [("깨끗한 배경만 학습하면 실제 사진은 14~19%", ["Kaggle 사진은 단색 배경 → 바닥·매립지 사진을 거의 못 맞힘"])],
        [("9개 출처 27,050장", ["아스팔트·매립지·실내 등 촬영 환경이 다른 사진을 섞음"])],
        [("불균형 8.8배 → 2.4배", ["가장 많은 클래스와 적은 클래스의 차이를 줄임",
                                 "(clothes 5,325 vs brown-glass 607 → 외부 데이터로 보충)"])],
    ], weights=[1.7, 0.9, 1.2])
    cy = bottom + 0.3
    cw = (CW - 0.3) / 2
    ch = BOT - cy
    card(s, M, cy, cw, ch)
    pad, cap = 0.15, 0.42
    cell_w = (cw - 2 * pad) / 5
    cell_h = (ch - 2 * pad) / 2
    for i, (ph, nm) in enumerate(zip(crop_source_photos(), SOURCE_NAMES)):
        x = M + pad + (i % 5) * cell_w
        y = cy + pad + (i // 5) * cell_h
        image_fit(s, ph, x + 0.06, y + 0.04, cell_w - 0.12, cell_h - cap - 0.08)
        text(s, x, y + cell_h - cap, cell_w, cap, nm.split("\n"), size=T_CAP, color=INK2, align=PP_ALIGN.CENTER)
    image_card(s, FS / "class_counts.png", M + cw + 0.3, cy, cw, ch)


def slide_04_preprocess(prs):
    s = new_slide(prs, "전처리 ① 정리와 크기 맞춤", "cleaning_services",
                  "먼저 사진마다 지문을 만들어 거의 같은 사진을 지웠습니다. 같은 사진이 학습과 테스트에 함께 있으면 "
                  "점수가 부풀려지기 때문입니다. 라벨과 맞지 않는 사진도 직접 보고 뺐고, 모든 사진을 192px로 맞춘 뒤 "
                  "모델에는 128px로 넣고 정규화했습니다. 정규화 값은 사전학습을 쓰지 않으므로 ImageNet 값 대신 "
                  "우리 학습 사진 27,050장의 평균과 표준편차를 썼습니다. 우리 데이터는 밝은 배경이 많아 평균이 높습니다"
                  "(평균 0.645, 0.621, 0.593).")
    half = (CW - 0.4) / 2
    keyword_block(s, M, TOP, half, [
        ("중복 제거 (사진 지문 비교)", ["같은 사진이 학습과 테스트에 함께 있으면 점수가 부풀려짐"]),
        ("라벨 오류 제거 (직접 보고 제외)", ["틀린 정답으로 학습하면 모델이 헷갈림. 예: 우유팩이 섞인 cardboard"]),
        ("192 → 128px 크기 통일", ["크기가 같아야 묶어서 학습 · 128px는 속도와 정확도의 균형",
                                  "160px(M9)은 흐린 사진에서 오히려 낮았음"]),
        ("정규화 (색 값을 평균 0, 퍼짐 1 근처로)", ["값의 범위가 일정해야 학습이 안정적이고 빠름",
                                           "ImageNet 값 대신 우리 학습 사진 27,050장의 평균·표준편차 사용"]),
    ], gap_pt=16)
    fx = M + half + 0.4
    image_card(s, crop_top(FG / "preprocess_steps.png", "preprocess_steps"), fx, TOP, half, BOT - TOP - 0.4)
    text(s, fx, BOT - 0.28, half, 0.28, ["정규화 후: 모델이 보는 값(색이 조금 달라 보임)"], size=T_CAP, color=INK2,
         align=PP_ALIGN.CENTER)


def slide_05_augment(prs):
    s = new_slide(prs, "전처리 ② 증강: 쓴 것과 뺀 것", "flip",
                  "증강은 학습할 때만 사진마다 무작위로 넣었습니다. 쓰레기는 어느 방향으로든 놓이니까 반전과 회전을 넣었고, "
                  "그늘이나 초점이 안 맞은 사진에 대비해 어둡게와 흐림을 넣었습니다. 반대로 색 변화는 뺐는데, "
                  "유리병 세 종류는 색이 바로 정답 단서이기 때문입니다. 비율을 유지하는 letterbox도 val_hard가 "
                  "0.766에서 0.640으로 떨어져서 뺐습니다. TrivialAugment는 변형이 너무 강해 0.848에서 0.836으로 떨어져서 뺐습니다.")
    used = [("반전·회전(±20°)", ["쓰레기는 어느 방향으로든 놓임. 사진을 외우지 않게 함 (M3에서 +0.035)"]),
            ("어둡게·흐림(둘 중 하나, 50%)", ["그늘·실내 사진, 초점이 안 맞은 사진 대비"])]
    rejected = [("색 변화(M4)", ["유리병 3색(갈색·초록·흰색)은 색이 정답 단서라 색을 흔들면 헷갈림"]),
                ("letterbox(M2)", ["흐리거나 어두운 사진 정확도(val_hard) 0.766 → 0.640으로 하락"]),
                ("TrivialAugment(M5)", ["변형이 너무 강해 오히려 하락 (0.848 → 0.836)"])]
    # 그림 안 칸 제목은 너무 작아서 칸마다 슬라이드 글자 라벨을 다시 단다(칸: 왼쪽 10px, 폭 172px, 간격 189.3px)
    panels = {"쓴 것": ["원본", "좌우 반전", "상하 반전", "회전(±20°)", "어둡게", "흐림"],
              "뺀 것": ["원본", "letterbox(M2)", "색 변화(M4)", "TrivialAugment(M5)"]}
    lw, tw, pad, gap_pt, lab_h = 0.7, 4.6, 0.15, 6, 0.28
    ix = M + 0.2 + lw + tw + 0.3
    iw = M + CW - 0.2 - ix
    rows = []
    for lab, items, img, bd in [("쓴 것", used, crop_top(FG / "aug_used.png", "aug_used"), PRI),
                                ("뺀 것", rejected, aug_rejected_4(), LINE)]:
        W, H = Image.open(img).size
        extra = lab_h * (2 if lab == "뺀 것" else 1)
        h = max(block_height(items, tw, gap_pt), iw * H / W + extra) + 2 * pad
        rows.append((lab, items, img, bd, h, W, extra))
    y = TOP
    for lab, items, img, bd, h, W, extra in rows:
        card(s, M, y, CW, h, border=bd)
        text(s, M + 0.2, y + pad + 0.08, lw, 0.3, [lab], size=T_CAP, color=INK2)
        keyword_block(s, M + 0.2 + lw, y + pad, tw, items, gap_pt=gap_pt)
        px, py, pw, ph = image_fit(s, img, ix, y + pad, iw, h - 2 * pad - extra)
        sc = pw / W
        for i, name in enumerate(panels[lab]):
            cx = px + (10 + 86 + 189.3 * i) * sc
            text(s, cx - 0.6, py + ph + 0.03, 1.2, lab_h - 0.03, [name], size=T_CAP, color=INK2,
                 align=PP_ALIGN.CENTER)
        if lab == "뺀 것":       # letterbox 칸(4칸 중 2번째) 설명
            cx = px + (10 + 86 + 189.3) * sc
            text(s, cx - 1.8, py + ph + lab_h + 0.02, 3.6, lab_h - 0.03,
                 ["letterbox: 비율 유지 + 빈 곳 채움 (신발 사진)"], size=T_CAP, color=INK3, align=PP_ALIGN.CENTER)
        y += h + 0.2


CHECK = {"size": T_CAP, "color": PRI_D, "font": SYMBOL_FONT}
CROSS = {"size": T_CAP, "color": INK3, "font": SYMBOL_FONT}


def slide_06_tuning_method(prs):
    s = new_slide(prs, "튜닝 ① 한 번에 하나씩", "tune",
                  "튜닝은 한 단계에 한 가지만 바꿨고, 점수가 지금까지 최고보다 0.01 이상 오를 때만 반영했습니다. "
                  "점수는 val_avg로, Kaggle 검증 사진, 학습에 안 쓴 외부 실사 사진, 그리고 흐림·어둡게·빛번짐을 입힌 "
                  "어려운 사진, 이 세 셋의 평균입니다. 그림에서 초록 실선이 반영된 단계, 회색 점선이 시험만 하고 버린 단계입니다. "
                  "M4부터 M7은 모두 M3 설정에서 하나씩 따로 바꿔 본 것입니다.")
    left_w = 5.2
    rule = {"size": T_REASON, "color": INK}
    keyword_block(s, M, TOP, left_w, [
        ("val_avg = (val + val_ext + val_hard) ÷ 3",
         ["깨끗한 사진, 실제 환경 사진, 흐리거나 어두운 사진을 모두 잘 맞혀야 점수가 오름. 한 가지에만 맞춘 설정을 걸러냄",
          [("한 번에 하나만 바꾸고, val_avg가 지금까지 최고보다 ", rule), ("0.01 이상", {**rule, "bold": True}),
           (" 오르면 반영", rule)]]),
    ])
    cards = [("val (2,327장)", "Kaggle 원본 검증용 → 기본 실력"),
             ("val_ext (498장)", "학습에 안 쓴 외부 실사 → 다른 촬영 환경"),
             ("val_hard (1,198장)", "흐림·어둡게·빛번짐 → 어려운 사진")]
    ch_, cg = 0.72, 0.15
    cy0 = BOT - 3 * ch_ - 2 * cg
    for i, (a, b) in enumerate(cards):
        yy = cy0 + i * (ch_ + cg)
        card(s, M, yy, left_w, ch_)
        text(s, M + 0.25, yy, left_w - 0.5, ch_, [[(a, {"size": T_REASON})], [(b, {"size": T_CAP, "color": INK2})]],
             anchor=MSO_ANCHOR.MIDDLE)

    # 줄기(반영): 중심 x, 각 상자 (중심 y, 높이, 줄들)
    cx, sw_ = 7.45, 2.6
    stem = [(1.85, 0.8, ["출발 설정", "폭 48 · cosine 학습률 · 어둡게/흐림 증강 · 30 epoch"]),
            (2.85, 0.6, ["M1 정리한 혼합 데이터", [("0.813 ", {}), ("✓", CHECK)]]),
            (4.15, 0.6, ["M3 반전·회전", [("0.848 ", {}), ("✓", CHECK)]]),
            (5.5, 0.6, ["M8 epoch 30→60", [("0.877 ", {}), ("✓", CHECK)]]),
            (6.45, 0.5, ["최종 M8"])]
    for (y1, h1, _), (y2, h2, _) in zip(stem, stem[1:]):
        arrow(s, cx, y1 + h1 / 2, cx, y2 - h2 / 2, color=PRI, width=1.5)
    for i, (yc, h, lines) in enumerate(stem):
        last = i == len(stem) - 1
        b = card(s, cx - sw_ / 2, yc - h / 2, sw_, h, border=PRI_D if last else PRI, width_pt=1.5 if last else 1.0,
                 radius=0.1)
        box_text(b, lines, size=T_CAP, color=INK)

    # 가지(반영 안 함): 회색 점선, 상자마다 따로
    bx, bw, bh, spine = 9.1, 2.8, 0.42, 8.92
    right = cx + sw_ / 2

    def branch(yc, label, score):
        b = card(s, bx, yc - bh / 2, bw, bh, border=OFF, dash=True, radius=0.08)
        box_text(b, [[(f"{label} {score} ", {}), ("✗", CROSS)]], size=T_CAP, color=INK2)

    def dash(x1, y1, x2, y2):
        arrow(s, x1, y1, x2, y2, color=OFF, width=1.0, dash=True, head=False)

    for yc, label, score in [(2.85, "M2 비율 유지(letterbox)", "0.776"), (5.5, "M9 입력 160px", "0.860")]:
        dash(right, yc, bx, yc)
        branch(yc, label, score)
    m3 = [(3.37, "M4 밝기·색 변화", "0.845"), (3.89, "M5 TrivialAugment", "0.836"),
          (4.41, "M6 클래스 가중치", "0.842"), (4.93, "M7 라벨 스무딩", "0.857")]
    dash(right, 4.15, spine, 4.15)
    dash(spine, m3[0][0], spine, m3[-1][0])
    for yc, label, score in m3:
        dash(spine, yc, bx, yc)
        branch(yc, label, score)


def slide_07_tuning_result(prs):
    s = new_slide(prs, "튜닝 ② 두 번 크게 올랐다", "bar_chart",
                  "막대에서 초록이 반영된 단계입니다. 가장 크게 오른 건 M3의 반전·회전 증강으로 +0.035, "
                  "그 다음이 M8에서 epoch을 30에서 60으로 늘린 +0.029였습니다. 라벨 스무딩 같은 나머지 시도는 "
                  "0.01 기준에 못 미치거나 오히려 떨어져서 기각했습니다. M1은 출발점이라 회색으로 표시했습니다.")
    kw = 4.2
    keyword_block(s, M, TOP + 0.3, kw, [
        ("M3 반전·회전 +0.035", ["사진을 외우던 문제가 줄어듦"]),
        ("M8 epoch 30→60 +0.029", ["30 epoch에서는 아직 오르는 중이었음"]),
        ("나머지는 기각", ["0.01 기준에 못 미치거나 하락"]),
    ], gap_pt=20)
    fx = M + kw + 0.3
    image_card(s, FS / "ladder_val_avg.png", fx, TOP, M + CW - fx, BOT - TOP - 0.35)
    text(s, fx, BOT - 0.25, M + CW - fx, 0.25, ["y축은 0.70부터 시작"], size=T_CAP, color=INK3, align=PP_ALIGN.RIGHT)


def slide_08_curves(prs):
    s = new_slide(prs, "튜닝 ③ 학습 곡선으로 본 변화", "show_chart",
                  "왼쪽 M1은 학습 정확도가 1.0까지 오르는데 val은 0.83에서 멈춥니다. 사진을 통째로 외운 겁니다. "
                  "반전·회전을 넣은 M3부터는 학습과 검증이 함께 오르고, M8은 60 epoch 중 기준 점수가 가장 좋았던 "
                  "epoch 50의 가중치를 저장했습니다. M1의 train과 val 차이는 0.168이었는데 M3에서는 0.069로 줄어, "
                  "증강이 과적합을 줄였다는 근거가 됩니다.")
    bottom = keyword_columns(s, TOP, [
        [("M1: train 1.0 / val 0.83 → 외움", ["차이 0.168 → 반전·회전을 넣은 M3는 0.069로 줄어듦"])],
        [("M3부터 함께 오름", ["반전·회전 증강으로 train과 val이 같이 오름"])],
        [("M8은 epoch 50 저장", ["검증 점수((val+val_ext)/2)가 가장 높았던 epoch의 가중치를 저장"])],
    ], weights=[1, 1, 1])
    cy = bottom + 0.3
    image_card(s, FS / "curves_m1_m3_m8.png", M, cy, CW, BOT - cy)


def slide_09_model(prs):
    s = new_slide(prs, "모델: SmallResNet 직접 구현(사전학습 없음)", "account_tree",
                  "모델은 작은 ResNet을 torch로 직접 만들었고, 사전학습 가중치는 쓰지 않았습니다. 잔차 블록 네 단계를 지나며 "
                  "채널이 48에서 384까지 늘고, 평균 풀링 뒤 Linear 하나로 12개 클래스 점수를 냅니다. 파라미터는 "
                  "6,289,212개이고, AdamW와 cosine 학습률로 학습해 가장 좋은 epoch을 저장했습니다. "
                  "사진 크기는 128에서 8까지 작아지고, 채널은 3에서 384까지 늘어납니다.")
    bottom = keyword_columns(s, TOP, [
        [("파라미터 629만 개", ["처음부터 학습하므로 작게 설계"])],
        [("잔차 블록 4단계 48→384", ["2~4단계마다 크기는 절반, 채널은 두 배"])],
        [("AdamW · cosine", ["학습률을 점점 줄여 마지막에 안정적으로 수렴"])],
    ], weights=[1, 1, 1])
    text(s, M, bottom + 0.25, CW, 0.45, ["사진은 점점 작아지고, 특징(채널)은 점점 많아진다"], size=T_KEY,
         anchor=MSO_ANCHOR.MIDDLE)
    # (한 변 in, 채움, 테두리 pt, 라벨 줄들) — 크기가 작아질수록 상자는 작게, 채널이 많을수록 진하게
    blocks = [(2.0, "FFFFFF", 0.75, ["입력", "128×128 × 3 (RGB)"]),
              (1.75, "FAFAFA", 1.0, ["줄기", "3×3 합성곱, stride 2", "64×64 × 48"]),
              (1.75, "FAFAFA", 1.0, ["1단계", "64×64 × 48"]),
              (1.5, "DFDFDF", 1.25, ["2단계", "32×32 × 96"]),
              (1.25, "C7C7C7", 1.5, ["3단계", "16×16 × 192"]),
              (1.0, "9A9A9A", 2.0, ["4단계", "8×8 × 384"]),
              (0.9, "9A9A9A", 2.0, ["평균 풀링", "1×1 × 384"]),
              (0.9, "FFFFFF", 1.5, ["Linear", "12", "(클래스 점수)"])]
    gap = 0.15
    x = M + (CW - (sum(b[0] for b in blocks) + gap * (len(blocks) - 1))) / 2
    base = BOT - 0.45 - 0.1 - 0.7          # 상자 아래 끝 y
    ay = base - 0.45
    for i, (side, fill, bw_, label) in enumerate(blocks):
        last = i == len(blocks) - 1
        card(s, x, base - side, side, side, border=PRI if last else INK2, fill=fill, width_pt=bw_, radius=0.08)
        lw = max(side, 1.05)
        text(s, x + side / 2 - lw / 2, base + 0.08, lw, 0.7, [[(t, {"color": INK if j == 0 else INK2})]
                                                             for j, t in enumerate(label)],
             size=T_CAP, align=PP_ALIGN.CENTER)
        if not last:
            arrow(s, x + side + 0.02, ay, x + side + gap - 0.02, ay, color=INK3, width=1.0, small=True)
        x += side + gap
    text(s, M, BOT - 0.45, CW, 0.45, ["1~4단계는 각각 잔차 블록 2개", "2~4단계 첫 블록에서 크기 절반·채널 두 배"],
         size=T_CAP, color=INK2)


def slide_10_result(prs):
    s = new_slide(prs, "결과", "fact_check",
                  "최종 TEST는 한 번만 했고, test 0.8823, 외부 실사 test_ext 0.9028이 나왔습니다. test_ext는 학습과 검증에 "
                  "쓰지 않은 외부 실사 사진 360장입니다. 검증과 TEST 점수가 거의 같아서 처음 보는 사진에도 성능이 유지됩니다. "
                  "외부 사진은 val_ext 0.922에서 test_ext 0.903으로 약 0.02 낮지만 큰 차이는 아닙니다.")
    lw, bn_h = 5.0, 1.9
    card(s, M, TOP, lw, bn_h)
    text(s, M + 0.3, TOP + 0.2, lw - 0.6, 1.05, ["0.8823"], size=T_BIG, color=PRI_D, bold=True,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, M + 0.3, TOP + 1.35, lw - 0.6, 0.3, ["test (Kaggle 사진 2,328장, 1회 평가)"], size=T_CAP, color=INK3)
    keyword_block(s, M, TOP + bn_h + 0.4, lw, [
        ("test_ext 0.9028", ["학습·검증에 안 쓴 외부 실사 사진 360장"]),
        ("검증 ≈ TEST", ["val 0.879 / test 0.882, val_ext 0.922 / test_ext 0.903", "처음 보는 사진에서도 성능 유지"]),
    ], gap_pt=20)
    fx = M + lw + 0.4
    image_card(s, FS / "test_scores.png", fx, TOP, M + CW - fx, BOT - TOP)


OVERFIT_NOTES = ("과적합을 세 가지로 점검했습니다. 학습 정확도와 검증 정확도는 0.10 차이가 나서 과적합이 조금 있습니다. "
                 "검증 손실은 26 epoch에서 가장 낮고 이후 조금 오르는데, 정확도는 떨어지지 않아서 틀린 사진을 더 확신하게 되는 "
                 "가벼운 과적합으로 봤습니다. 그래서 정확도가 가장 높았던 50 epoch의 가중치를 저장했고, 검증과 TEST 점수가 "
                 "거의 같아 처음 보는 사진에서도 성능이 유지됩니다. 참고로 60 epoch으로 늘리면서 차이는 M3의 0.069보다 "
                 "커졌지만, 검증 점수 자체가 0.848에서 0.877로 올라 이 설정을 택했습니다.")


def slide_11_overfit(prs):
    s = new_slide(prs, "과적합 점검: 가볍게 있지만 일반화는 유지", "query_stats", OVERFIT_NOTES)
    checks = [("정확도 차이 0.10", "epoch 50: train 0.982 vs val 0.879", "학습 사진을 더 잘 맞힘 → 과적합은 있다"),
              ("val loss는 epoch 26부터 상승", "epoch 26 (0.477) 최저 → epoch 50 0.537", "정확도는 유지 → 가벼운 과적합"),
              ("검증 ≈ TEST", "val 0.879 / test 0.882", "처음 보는 사진에서도 성능 유지 → 일반화는 됨")]
    gap, ch = 0.3, 2.15
    cw = (CW - 2 * gap) / 3
    for i, (kw, ev, why) in enumerate(checks):
        x = M + i * (cw + gap)
        card(s, x, TOP, cw, ch)
        text(s, x + 0.25, TOP + 0.18, cw - 0.5, 0.25, [f"{i + 1:02d}"], size=T_CAP, color=INK3)
        text(s, x + 0.25, TOP + 0.42, cw - 0.5, 0.8, [kw], size=T_KEY)
        text(s, x + 0.25, TOP + 1.22, cw - 0.5, 0.85,
             [[(ev, {"size": T_REASON, "color": INK2})], [(why, {"size": T_REASON, "color": INK})]], space_after=2)
    text(s, M, TOP + ch + 0.12, CW, 0.28,
         ["막은 방법: 데이터 다양화 · 반전/회전 증강 · weight decay · BatchNorm · 작은 출력층 · 최고 epoch 저장"],
         size=T_CAP, color=INK2)
    fy = TOP + ch + 0.55
    fw = (CW - gap) / 2
    for j, (fn, cap) in enumerate([("loss_curve.png", "val loss 최저 epoch 26 (0.477) → 저장 epoch 50 (0.537)"),
                                   ("final_curve_gap.png", "epoch 50: train 0.982 vs val 0.879")]):
        x = M + j * (fw + gap)
        image_card(s, FS / fn, x, fy, fw, BOT - 0.32 - fy, pad=0.1)
        text(s, x, BOT - 0.25, fw, 0.25, [cap], size=T_CAP, color=INK2, align=PP_ALIGN.CENTER)


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
                  "정리하면, 사전학습 없이 직접 만든 작은 CNN으로 test 0.882, 외부 실사 test_ext 0.903을 얻었습니다. "
                  "가장 크게 도움이 된 건 실제 환경 사진을 섞어 데이터를 다양하게 만든 것이었고, 그 다음이 반전·회전 증강 "
                  "+0.035, 충분한 학습 epoch 60 +0.029였습니다. 한계도 있습니다. 더러운 사진 평가에서는 0.773으로, 특히 "
                  "학습에 없던 강한 빛번짐 0.28, 저조도와 잡음 0.51, 얼룩 0.60에 약했고, 배경을 지우면 0.122 떨어져 배경에도 "
                  "기대고 있었습니다. 그래서 지금은 노이즈 증강과 배경 교란 증강을 실험하고 있습니다. 감사합니다.")
    fig_h = BOT - TOP - 0.45
    fig_w = fig_h * 1473 / 1314 + 0.1
    fig_x = M + CW - fig_w
    R = {"color": WARN}
    reason = {"size": T_REASON, "color": INK2}
    keyword_block(s, M, TOP + 0.1, fig_x - M - 0.4, [
        ("01 성과", "test 0.882 · 외부 실사 test_ext 0.903", ["사전학습 없이 직접 만든 작은 CNN으로 달성"]),
        ("02 배운 점", "데이터 다양화 → 반전·회전 → 충분한 학습",
         ["실제 환경 사진을 섞은 것이 가장 컸고, 증강 +0.035, epoch 60 +0.029"]),
        ("03 한계와 다음 과제", [("강한 빛번짐 ", {}), ("0.28", R), (" · 배경 의존", {})],
         [[("배경을 지우면 정확도 ", reason), ("−0.122", {**reason, "color": WARN}),
           (" → 노이즈·배경 증강 실험 중", reason)]]),
    ], gap_pt=40)
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
                  "(질문이 나올 때만) 전체 튜닝 표입니다. 출력층을 은닉층 두 개로 키우면 train은 0.996에서 0.998로 오르지만 "
                  "검증은 0.878에서 0.875로 낮아져서, 모델을 키우면 더 외운다고 보고 작은 출력층을 유지했습니다.")
    tw = 8.0
    text(s, M, TOP, tw, 0.45, ["M1~M9 전체 표"], size=T_KEY, anchor=MSO_ANCHOR.MIDDLE)
    ty = TOP + 0.65
    tuning_table(s, M, ty, tw, BOT - ty, [0.7, 2.6, 0.9, 0.95, 1.0, 0.95, 0.9], T_REASON)
    rx = M + tw + 0.4
    rw = M + CW - rx
    card(s, rx, ty, rw, BOT - ty)
    keyword_block(s, rx + 0.25, ty + 0.25, rw - 0.5, [
        ("출력층: Linear 0.878 vs 은닉층 2개 0.875",
         ["→ 작은 출력층 유지", "은닉층 2개로 키우면 train은 오르고(0.996 → 0.998) 검증은 낮아짐(0.878 → 0.875)",
          "모델을 키우면 더 외운다"]),
    ])


SLIDES = [slide_01_cover, slide_02_toc, slide_03_data, slide_04_preprocess, slide_05_augment,
          slide_06_tuning_method, slide_07_tuning_result, slide_08_curves, slide_09_model, slide_10_result,
          slide_11_overfit, slide_confusion, slide_conclusion, slide_spare]


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
