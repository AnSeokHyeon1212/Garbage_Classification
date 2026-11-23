"""README.md → 보고서 Word(.docx).
- README의 mermaid 흐름도 3개는 Word에서 그려지지 않으므로 matplotlib로 같은 내용을 PNG로 그려 바꿔 넣는다.
- 이미지 경로는 저장소 루트 기준 그대로 쓴다(pandoc --resource-path).
- PDF 변환은 Word에서 한다(build_report.ps1 참고).
실행: python docs/report/build_report.py
"""
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
TEXT, SUB, LINE = "#171717", "#707070", "#9A9A9A"
YES, NO, FIN = ("#cfe8ff", "#1f6fb2"), ("#eeeeee", "#999999"), ("#d4f5d4", "#2e8b57")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})


def box(ax, x, y, w, h, text, style=YES, dashed=False, fs=10):
    fc, ec = style
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4, ls="--" if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TEXT if not dashed else "#555555")


def arrow(ax, x1, y1, x2, y2, dashed=False):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3,
                                                         ls="--" if dashed else "-", shrinkA=0, shrinkB=0))


def lr_chain(labels, name, per_row=4, w=2.6, h=1.15, gap=0.7, vgap=0.7, fs=11, final=None):
    """상자를 한 줄에 per_row개씩 지그재그 없이 두 줄로 놓는다(줄 끝 → 다음 줄 처음은 꺾인 화살표)"""
    rows = [labels[i:i + per_row] for i in range(0, len(labels), per_row)]
    fig, ax = plt.subplots(figsize=(per_row * (w + gap) * 0.75, len(rows) * (h + vgap) * 0.75))
    pos = []
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            k = r * per_row + c
            x, y = c * (w + gap), -r * (h + vgap)
            box(ax, x, y, w, h, t, FIN if final == k else YES, fs=fs)
            pos.append((x, y))
    for k in range(1, len(pos)):
        (x1, y1), (x2, y2) = pos[k - 1], pos[k]
        if y1 == y2:
            arrow(ax, x1 + w / 2, y1, x2 - w / 2, y2)
        else:  # 줄 바꿈: 오른쪽 끝 상자 아래 → 다음 줄 첫 상자 위
            ax.plot([x1, x1, x2, x2], [y1 - h / 2, y1 - h / 2 - vgap / 2, y1 - h / 2 - vgap / 2, y2 + h / 2 + 0.05],
                    color=LINE, lw=1.3)
            arrow(ax, x2, y2 + h / 2 + 0.06, x2, y2 + h / 2)
    ax.set_xlim(-w / 2 - 0.1, (per_row - 1) * (w + gap) + w / 2 + 0.1)
    ax.set_ylim(-(len(rows) - 1) * (h + vgap) - h / 2 - 0.1, h / 2 + 0.1)
    ax.axis("off")
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline():
    lr_chain(["데이터 수집\n9개 출처", "정리\n중복·라벨 오류 제거\n27,050장", "전처리\n192→128px\n정규화",
              "증강\n반전·회전\n어둡게·흐림", "SmallResNet\n처음부터 학습", "튜닝 M1~M9\nval_avg로 선택",
              "최종 모델\nM8", "TEST 1회\n0.8823 / 0.9028"], "pipeline.png", final=7)


def model():
    lr_chain(["입력\n3×128×128", "줄기\n3×3 합성곱\nstride 2, 48채널", "1단계\nBasicBlock ×2\n48채널",
              "2단계\nBasicBlock ×2\n96채널", "3단계\nBasicBlock ×2\n192채널", "4단계\nBasicBlock ×2\n384채널",
              "전역 평균\n풀링", "Linear 384→12\n12개 클래스 점수"], "model.png", final=7)


def tuning():
    fig, ax = plt.subplots(figsize=(9, 8))
    W, H, BW = 4.2, 0.75, 2.9
    main = {"S": (0, 9.0, "출발 설정\n폭 48 · cosine · 어둡게/흐림 증강 · 30 epoch"),
            "M1": (0, 7.6, "M1 정리한 혼합 데이터 27,050장\nval_avg 0.813 — 반영"),
            "M3": (0, 5.0, "M3 좌우·상하 반전 + 회전\nval_avg 0.848 — 반영"),
            "M8": (0, 2.2, "M8 epoch 30 → 60\nval_avg 0.877 — 반영"),
            "F": (0, 0.8, "최종 모델 M8\ntest 0.8823 · test_ext 0.9028")}
    for k, (x, y, t) in main.items():
        box(ax, x, y, W, H, t, FIN if k == "F" else YES, fs=11)
    order = ["S", "M1", "M3", "M8", "F"]
    for a, b in zip(order, order[1:]):
        arrow(ax, 0, main[a][1] - H / 2, 0, main[b][1] + H / 2)
    branches = {"M1": [(7.6, "M2 비율 유지(letterbox)\n0.776 — 반영 안 함")],
                "M3": [(6.3, "M4 밝기·색 변화\n0.845 — 반영 안 함"), (5.43, "M5 TrivialAugmentWide\n0.836 — 반영 안 함"),
                       (4.57, "M6 클래스 가중치\n0.842 — 반영 안 함"), (3.7, "M7 라벨 스무딩 0.1\n0.857 — 반영 안 함")],
                "M8": [(2.2, "M9 입력 160px\n0.860 — 반영 안 함")]}
    bx = W / 2 + 1.4 + BW / 2
    for k, items in branches.items():
        y0 = main[k][1]
        for y, t in items:
            box(ax, bx, y, BW, H * 0.95, t, NO, dashed=True, fs=11)
            arrow(ax, W / 2, y0, bx - BW / 2, y, dashed=True)
    ax.text(-W / 2, -0.15, "실선 = 반영되어 다음 단계의 출발점, 점선 = 시험했지만 반영하지 않음", fontsize=9, color=SUB)
    ax.set_xlim(-W / 2 - 0.2, bx + BW / 2 + 0.2)
    ax.set_ylim(-0.4, 9.6)
    ax.axis("off")
    fig.savefig(FIG / "tuning_flow.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build():
    FIG.mkdir(exist_ok=True)
    pipeline(), model(), tuning()
    md = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n.*?```", md, flags=re.S)
    assert len(blocks) == 3, len(blocks)
    imgs = [("전체 흐름", "pipeline.png", 16), ("모델 구조", "model.png", 16), ("튜닝 흐름", "tuning_flow.png", 15)]
    for b, (alt, f, w) in zip(blocks, imgs):
        md = md.replace(b, f"![{alt}](docs/report/figures/{f}){{width={w}cm}}", 1)
    # README 그림은 페이지 폭에 맞춘다 (정사각형 혼동행렬은 한 쪽을 다 차지하지 않게 작게)
    md = re.sub(r"!\[([^\]]*)\]\((01_coding/[^)]+\.png)\)",
                lambda m: f"![{m[1]}]({m[2]}){{width={11 if 'confusion' in m[2] else 16}cm}}", md)
    # 실선/점선 설명 문장은 그림 안에 이미 있음
    md = md.replace("실선 = 반영되어 다음 단계의 출발점이 됨, 점선 = 시험했지만 반영하지 않음.\n", "")
    tmp = HERE / "_report.md"
    tmp.write_text(md, encoding="utf-8")
    out = HERE / "garbage_classification_report.docx"
    subprocess.run(["pandoc", str(tmp), "-o", str(out), "--resource-path", str(ROOT),
                    "--reference-doc", str(HERE / "reference.docx"), "-f", "gfm+attributes",
                    "--metadata", "lang=ko-KR"], check=True, cwd=ROOT)
    tmp.unlink()
    print("saved", out)


if __name__ == "__main__":
    build()
