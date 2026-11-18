import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import time
    import random
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from sklearn.model_selection import train_test_split

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    return (
        DataLoader, Dataset, Image, Path, json, mo, models, nn, np, pd, plt,
        random, time, torch, train_test_split, transforms,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # T-01 데이터 확인 + 분할 고정 + CPU 속도 실측

    - 데이터: Kaggle `mostafaabla/garbage-classification` (12 클래스) → `archive/garbage_classification/`
    - 산출물: `01_coding/results/split.csv`, `01_coding/results/class_counts.csv`, `01_coding/figures/class_dist.png`, `01_coding/results/cpu_timing.json`
    - **split.csv는 한 번 만들면 다시 만들지 않는다.** 파일이 있으면 읽어서 검증만 한다.
    - test 데이터는 이 노트북에서 학습·평가 어디에도 쓰지 않는다.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 설정(CFG)과 경로
    모든 설정값은 `CFG` 한 곳에 모은다. 경로는 한글·공백이 있어서 `pathlib.Path`로만 다룬다.
    """)
    return


@app.cell
def _(Path, mo):
    ROOT = Path(mo.notebook_dir()).resolve().parents[1]  # 프로젝트 루트 (01_coding/notebooks 의 두 단계 위)
    CFG = {
        "seed": 42,
        "data_dir": ROOT / "archive" / "garbage_classification",
        "results_dir": ROOT / "01_coding" / "results",
        "figures_dir": ROOT / "01_coding" / "figures",
        "split_ratio": (0.70, 0.15, 0.15),  # train / val / test
        "source": "kaggle",
        # CPU 속도 실측용
        "timing_model": "resnet18",
        "timing_img_size": 224,
        "timing_batch": 32,
        "timing_lr": 1e-3,
        "timing_num_workers": 0,  # Windows + 노트북에서 멀티프로세스 로더 문제 회피
    }
    CFG["results_dir"].mkdir(parents=True, exist_ok=True)
    CFG["figures_dir"].mkdir(parents=True, exist_ok=True)
    {k: str(v) for k, v in CFG.items()}
    return CFG, ROOT


@app.cell
def _(np, random, torch):
    def seed_everything(seed: int):
        """random / numpy / torch 의 난수 시드를 모두 고정해 재현성을 확보한다."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return (seed_everything,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 파일 목록 수집
    클래스 폴더 이름 = 라벨. 경로는 프로젝트 루트 기준 상대경로(`/` 구분)로 저장해서 Colab 등 다른 환경에서도 그대로 쓸 수 있게 한다.
    """)
    return


@app.cell
def _(CFG, ROOT, pd):
    _rows = []
    for _cls_dir in sorted(p for p in CFG["data_dir"].iterdir() if p.is_dir()):
        for _f in sorted(_cls_dir.iterdir()):
            if _f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                _rows.append({
                    "path": _f.relative_to(ROOT).as_posix(),
                    "label": _cls_dir.name,
                    "source": CFG["source"],
                })
    df_all = pd.DataFrame(_rows)
    CLASSES = sorted(df_all["label"].unique())
    print(f"전체 이미지 수: {len(df_all):,} / 클래스 수: {len(CLASSES)}")
    print(CLASSES)
    return CLASSES, df_all


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 이미지 상태 점검
    모든 파일을 열어 깨진 파일, 크기, 색 모드(RGB/RGBA/L 등)를 확인한다. RGB가 아닌 파일은 학습 때 `.convert("RGB")`로 처리한다.
    """)
    return


@app.cell
def _(Image, ROOT, df_all, pd):
    _info = []
    for _p in df_all["path"]:
        try:
            with Image.open(ROOT / _p) as _im:
                _info.append((_im.size[0], _im.size[1], _im.mode, False))
        except Exception:
            _info.append((None, None, None, True))
    img_info = pd.DataFrame(_info, columns=["width", "height", "mode", "broken"])
    print("깨진 파일 수:", int(img_info["broken"].sum()))
    print("색 모드 분포:", img_info["mode"].value_counts().to_dict())
    img_info[["width", "height"]].describe().round(1)
    return (img_info,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Stratified 분할 (70/15/15, seed=42) — 1회 고정
    `train_test_split(stratify=label)`을 두 번 사용: 전체 → train 70% / 나머지 30%, 나머지 → val 15% / test 15%.
    클래스 비율이 세 세트에서 같게 유지된다. **`split.csv`가 이미 있으면 새로 만들지 않고 읽기만 한다.**
    """)
    return


@app.cell
def _(CFG, df_all, pd, seed_everything, train_test_split):
    split_path = CFG["results_dir"] / "split.csv"
    if split_path.exists():
        df_split = pd.read_csv(split_path)
        print("기존 split.csv 사용 (재생성 안 함)")
    else:
        seed_everything(CFG["seed"])
        _train, _rest = train_test_split(
            df_all, test_size=0.30, stratify=df_all["label"], random_state=CFG["seed"]
        )
        _val, _test = train_test_split(
            _rest, test_size=0.50, stratify=_rest["label"], random_state=CFG["seed"]
        )
        df_split = pd.concat([
            _train.assign(split="train"),
            _val.assign(split="val"),
            _test.assign(split="test"),
        ])[["path", "label", "split", "source"]].sort_values("path").reset_index(drop=True)
        df_split.to_csv(split_path, index=False, encoding="utf-8")
        print("split.csv 새로 생성:", split_path)

    # 검증: 행 수 = 전체 이미지 수, 중복 없음, 누락 없음
    assert len(df_split) == len(df_all), "split.csv 행 수가 전체 이미지 수와 다름"
    assert df_split["path"].is_unique, "중복 경로 존재"
    assert set(df_split["path"]) == set(df_all["path"]), "파일 목록 불일치"
    print(df_split["split"].value_counts().to_dict())
    return (df_split,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. 클래스별 개수 표
    """)
    return


@app.cell
def _(CFG, df_split, pd):
    class_counts = pd.crosstab(df_split["label"], df_split["split"])[["train", "val", "test"]]
    class_counts["total"] = class_counts.sum(axis=1)
    class_counts["ratio(%)"] = (class_counts["total"] / class_counts["total"].sum() * 100).round(2)
    class_counts = class_counts.sort_values("total", ascending=False)
    class_counts.loc["합계"] = class_counts.sum(numeric_only=True).round(2)
    _int_cols = ["train", "val", "test", "total"]
    class_counts[_int_cols] = class_counts[_int_cols].astype(int)  # 합계 행 추가 시 float 변환 방지
    class_counts.to_csv(CFG["results_dir"] / "class_counts.csv", encoding="utf-8-sig")
    _imb = class_counts.drop("합계")["total"]
    print(f"불균형 비율(최다/최소) = {_imb.max()} / {_imb.min()} = {_imb.max() / _imb.min():.2f}배")
    class_counts
    return (class_counts,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 클래스 분포 그래프 → `figures/class_dist.png`
    """)
    return


@app.cell
def _(CFG, class_counts, plt):
    plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트
    plt.rcParams["axes.unicode_minus"] = False

    _cc = class_counts.drop("합계").sort_values("total")
    _fig, _ax = plt.subplots(figsize=(9, 6))
    _left = 0
    for _sp, _color in zip(["train", "val", "test"], ["#4C72B0", "#DD8452", "#55A868"]):
        _ax.barh(_cc.index, _cc[_sp], left=_left, color=_color, label=_sp)
        _left = _left + _cc[_sp]
    for _i, _t in enumerate(_cc["total"]):
        _ax.text(_t + 40, _i, f"{int(_t):,}", va="center", fontsize=9)
    _ax.set_xlabel("이미지 수")
    _ax.set_ylabel("클래스")
    _ax.set_title(f"클래스별 이미지 수 (전체 {int(_cc['total'].sum()):,}장, train/val/test = 70/15/15)")
    _ax.set_xlim(0, _cc["total"].max() * 1.12)
    _ax.legend(title="split", loc="lower right")
    _fig.tight_layout()
    _fig.savefig(CFG["figures_dir"] / "class_dist.png", dpi=200)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. CPU 속도 실측: resnet18(pretrained), 224px, train 1 epoch

    처음 나오는 학습 루프라 줄마다 주석을 달았다. 핵심 개념:
    - **Dataset / DataLoader**: Dataset은 "i번째 이미지와 라벨"을 돌려주는 객체이고, DataLoader는 이를 batch 단위로 묶고 섞어 준다.
    - **전이학습**: ImageNet으로 미리 학습된 resnet18을 가져와 마지막 분류층(fc)만 12 클래스용으로 바꾼다.
    - **학습 1 step**: 예측(forward) → 손실 계산 → 기울기 초기화 → 역전파(backward) → 가중치 갱신(step).

    측정 조건: CPU, fp32, batch 32, num_workers 0, train만 사용(val/test 미사용). 시간 측정 목적이라 정확도는 참고용이다.
    결과는 `results/cpu_timing.json`에 저장하고, 파일이 있으면 다시 측정하지 않는다(재실행 시 30분 이상 걸릴 수 있음).
    """)
    return


@app.cell
def _(Dataset, Image, ROOT):
    class GarbageDataset(Dataset):
        def __init__(self, df, class_to_idx, transform=None):
            self.paths = df["path"].tolist()                           # 이미지 상대경로 목록
            self.labels = [class_to_idx[c] for c in df["label"]]     # 클래스 이름 → 정수 라벨
            self.transform = transform                                # 전처리(크기 조정, 텐서 변환 등)

        def __len__(self):
            return len(self.paths)                                    # 데이터 개수

        def __getitem__(self, i):
            img = Image.open(ROOT / self.paths[i]).convert("RGB")     # 이미지 열기 + RGB로 통일
            if self.transform:
                img = self.transform(img)                             # 전처리 적용 → Tensor
            return img, self.labels[i]                                # (이미지, 라벨) 반환
    return (GarbageDataset,)


@app.cell
def _(CFG, CLASSES, DataLoader, GarbageDataset, json, models, nn, df_split, seed_everything, time, torch, transforms):
    timing_path = CFG["results_dir"] / "cpu_timing.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        print("기존 측정 결과 사용:", timing_path)
    else:
        seed_everything(CFG["seed"])                                         # 재현성
        _device = torch.device("cpu")                                        # 이번 측정은 CPU 고정
        _size = CFG["timing_img_size"]
        _tf = transforms.Compose([
            transforms.Resize((_size, _size)),                               # 224x224로 크기 맞춤
            transforms.ToTensor(),                                           # PIL → Tensor (0~1)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet 평균/표준편차로 정규화
        ])
        _class_to_idx = {c: i for i, c in enumerate(CLASSES)}                # 클래스 → 번호 매핑
        _train_df = df_split[df_split["split"] == "train"]                   # train만 사용 (test 금지)
        _ds = GarbageDataset(_train_df, _class_to_idx, _tf)
        _loader = DataLoader(_ds, batch_size=CFG["timing_batch"], shuffle=True,
                             num_workers=CFG["timing_num_workers"])          # batch 단위로 섞어서 꺼냄

        _model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)    # ImageNet 사전학습 가중치
        _model.fc = nn.Linear(_model.fc.in_features, len(CLASSES))           # 마지막 층을 12 클래스용으로 교체
        _model = _model.to(_device)
        _criterion = nn.CrossEntropyLoss()                                   # 분류용 손실 함수
        _optimizer = torch.optim.Adam(_model.parameters(), lr=CFG["timing_lr"])  # 가중치 갱신 방법

        _model.train()                                                       # 학습 모드 (BatchNorm/Dropout 동작)
        _n, _correct, _loss_sum, _batch_times = 0, 0, 0.0, []
        _t0 = time.perf_counter()
        _tb = time.perf_counter()
        for _x, _y in _loader:                                               # batch 하나씩 반복
            _x, _y = _x.to(_device), _y.to(_device)
            _out = _model(_x)                                                # 1) forward: 예측
            _loss = _criterion(_out, _y)                                     # 2) 손실 계산
            _optimizer.zero_grad()                                           # 3) 이전 기울기 지우기
            _loss.backward()                                                 # 4) backward: 기울기 계산
            _optimizer.step()                                                # 5) 가중치 갱신
            _loss_sum += _loss.item() * _x.size(0)                           # 평균 손실 계산용 누적
            _correct += (_out.argmax(1) == _y).sum().item()                  # 맞힌 개수 누적
            _n += _x.size(0)
            _now = time.perf_counter()
            _batch_times.append(_now - _tb)                                  # batch 1개 걸린 시간 기록
            _tb = _now
        _epoch_sec = time.perf_counter() - _t0

        timing = {
            "device": "cpu",
            "torch": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "model": CFG["timing_model"], "img_size": _size,
            "batch": CFG["timing_batch"], "num_workers": CFG["timing_num_workers"],
            "precision": "fp32",
            "n_train_images": _n, "n_batches": len(_batch_times),
            "epoch_sec": round(_epoch_sec, 1),
            "epoch_min": round(_epoch_sec / 60, 2),
            "sec_per_batch_median": round(sorted(_batch_times)[len(_batch_times) // 2], 3),
            "images_per_sec": round(_n / _epoch_sec, 1),
            "train_loss_1ep": round(_loss_sum / _n, 4),
            "train_acc_1ep": round(_correct / _n, 4),
        }
        timing_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    timing
    return (timing,)


@app.cell
def _(mo, timing):
    mo.md(f"""
    ## 8. 요약
    - train 1 epoch (resnet18, 224px, batch 32, CPU {timing['num_threads']}스레드): **{timing['epoch_min']}분** ({timing['images_per_sec']} img/s)
    - 1 epoch 후 train acc {timing['train_acc_1ep']:.4f} (참고용, val/test 미평가)
    """)
    return


if __name__ == "__main__":
    app.run()
