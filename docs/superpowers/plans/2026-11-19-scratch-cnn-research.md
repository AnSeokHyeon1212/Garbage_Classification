# 처음부터 학습하는 작은 ResNet 재연구 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사전학습 모델 없이 직접 구현한 작은 ResNet을 Colab GPU에서 처음부터 학습한다. 한 번에 한 변수씩, epoch를 늘려 가며 튜닝 과정을 기록하고, 최종 모델을 TEST로 한 번만 평가한다. 모든 코드는 제출 노트북 `final_model.ipynb` 하나에 담는다.

**Architecture:**
- 노트북 하나에 다음 셀을 둔다: 데이터 준비(기존 고정 분할 재사용) → 이미지 캐시 → 모델 → 학습 함수 → 실험 사다리 → 튜닝 과정 시각화 → 최종 평가·TEST.
- 셀마다 metadata 태그(`setup`, `data`, `cache`, `model`, `train`, `ladder`, `report`, `final`)를 단다. pytest가 이 태그로 셀을 골라 실행해 검증한다.
- 실험 결과는 Colab에서 내 Drive(`garbage_classification/experiments/`)에 바로 저장한다. 세션이 끊겨도 완료된 실험은 건너뛴다.

**Tech Stack:** Python 3, PyTorch·torchvision(transforms만 사용, `torchvision.models` 금지), pandas, numpy, PIL, matplotlib, pytest, Google Colab(T4 GPU), Google Drive

**Spec:** 사용자 지시 (2026-11-19 대화)
- 사전학습 미세조정은 허용되지 않음 → 작은 ResNet을 직접 구현해 처음부터 학습
- Colab GPU에서 학습
- 증강(좌우반전·RandomResizedCrop)과 cosine 학습률 추가
- 마감은 11/20. epoch를 늘려 가며 재연구하고, 튜닝 과정을 보여 줌
- 이전 사전학습 연구(E1~E9)는 모두 삭제
- 전처리 방식은 유지

## Global Constraints
- 사용 패키지: PyTorch, torchvision의 `transforms`만. `torchvision.models`·사전학습 가중치·timm·YOLO·Keras/TF·fastai·Lightning 금지.
- 고정 분할 CSV(`01_coding/RESULT/data_split/`의 split·train_mix·val_ext·val_hard·test_ext)는 재생성·수정 금지.
- 한 번에 한 변수만 바꾼다. 비교 기준은 그 시점 최고 설정이다.
- seed 42
- 선택 지표: **val_avg = (val + val_ext + val_hard) / 3**
- 채택 조건: val_avg가 최고 설정보다 +0.01 이상, clothes·shoes의 val 정확도 하락이 각각 0.03 이내
- 조기 종료 점수: (val acc + val_ext acc) / 2, 가장 높은 epoch의 가중치를 저장
- TEST(Kaggle test 2,328 + test_ext 360)는 최종 모델로 **한 번만** 평가한다. 결과 파일이 있으면 다시 평가하지 않는다.
- 노트북 문구에 내부 은어(S3, D번호, 결정 A 등)를 쓰지 않는다. 실험 ID는 `R1…`로 쓰고 처음 나올 때 뜻을 설명한다.
- 코드 셀은 모두 `# [셀 목적]` 주석으로 시작한다.
- 커밋은 각 Task의 마지막 단계에서 **사용자 확인 후** 한다. 외부 이미지와 `.pth`는 커밋하지 않는다.

## Review Focus
1. Colab 세션이 끊긴 뒤 다시 실행하는 경우 → 완료된 실험은 건너뛰고, 중단된 실험만 처음부터 다시 돈다. 테스트: Task 4 `test_resume_skips_finished`.
2. AI Hub 이미지가 없는 환경(공개 링크로 받은 경우) → 학습·평가 셀이 죽지 않고, 빠진 장수를 출력한다. 테스트: Task 3 `test_cache_skips_missing_files`.
3. 입력 크기를 128에서 160으로 바꾼 실험 → 모델이 그대로 동작하고, 캐시(192px)에서 잘라 쓴다. 테스트: Task 2 `test_forward_shapes`, Task 3 `test_cache_size`.
4. cosine 학습률에서 조기 종료가 일찍 걸리는 경우 → 남은 epoch가 있어도 학습을 멈추고, 저장된 가중치는 최고 epoch의 것이다. 테스트: Task 4 `test_best_checkpoint_matches_best_epoch`.
5. 실수로 TEST 셀을 두 번 실행하는 경우 → 두 번째는 `AssertionError`로 막는다. 테스트: Task 6 `test_final_test_runs_once`.

---

## 파일 구조

| 파일 | 역할 | 작업 |
|---|---|---|
| `01_coding/notebooks/final_model.ipynb` | 제출 노트북 하나. 데이터 준비~최종 TEST까지 모든 코드 | 수정 (모델·학습·실험·보고 셀 교체) |
| `01_coding/tests/nbcells.py` | 태그로 노트북 셀을 골라 namespace에서 실행하는 테스트 도우미 | 생성 |
| `01_coding/tests/test_model.py`, `test_cache.py`, `test_train.py`, `test_ladder.py`, `test_final.py` | 각 Task의 테스트 | 생성 |
| `01_coding/RESULT/experiments/` | Drive에서 받아온 실험 결과(`ladder.csv`, `R*/history.csv`, `R*/cfg.json`, `R*/metrics.json`) | 비운 뒤 새로 채움 |
| `01_coding/RESULT/final_test/` | 새 최종 모델 TEST 1회 결과 | 비운 뒤 새로 채움 |
| `README.md`, `01_coding/README.md`, `01_coding/RESULT/README.md`, `00_lead/decisions.md` | 문서 | 수정 |

**삭제 대상 (Task 1)**
- 이전 연구 코드: `01_coding/notebooks/`의 `02_experiments.py`, `bench_threads.py`, `run_exp.py`, `overnight.py`, `overnight2.py`, `eval_val_hard.py`, `stacking.py`, `ensemble_compare.py`, `plot_compare.py`, `final_test.py`, `gc_lib.py`
- 데이터 구성 스크립트: `01_data.py`, `build_mix.py`, `build_val_hard.py`, `build_holdouts.py` (같은 코드가 노트북 2~5절에 있고, 결과가 완전히 같음을 11/19에 확인함)
- 이전 결과: `01_coding/RESULT/`의 `experiments/*`, `ensemble/`, `final_test/*`, `logs/`, `overnight_summary.md`, `final_model/*`, 그림 중 `class_dist.png`를 제외한 전부
- 로컬 체크포인트 `01_coding/checkpoints/*.pth`, `docs/superpowers/plans/2026-11-19-overnight-experiments.md`
- Drive: `06_final_model/final_model.pth`

---

### Task 1: 이전 사전학습 연구 삭제와 문서 정리

**Files:**
- Delete: 위 "삭제 대상" 전부
- Modify: `README.md`, `01_coding/README.md`, `01_coding/RESULT/README.md`, `00_lead/decisions.md`
- Test: `01_coding/tests/test_cleanup.py`

**Interfaces:**
- Produces: 비어 있는 `RESULT/experiments/`, `RESULT/final_test/`. 사전학습 관련 코드가 저장소에 없는 상태.

- [ ] **Step 1: 실패하는 테스트 작성**
```python
# 01_coding/tests/test_cleanup.py
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CODE = [p for p in (ROOT / "01_coding").rglob("*") if p.suffix in {".py", ".ipynb"} and "tests" not in p.parts]

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
```
- [ ] **Step 2: 실행해서 실패 확인** — `cd 01_coding && python -m pytest tests/test_cleanup.py -q` → 3 failed
- [ ] **Step 3: 삭제 실행**
  - `git rm`으로 추적 파일을 지우고, 추적하지 않는 `.pth`와 로그는 일반 삭제한다.
  - 노트북 7절(모델)의 `models.resnet18` 셀은 Task 2에서 교체한다. 그 전까지 이 테스트의 노트북 검사는 실패해도 된다. Step 4는 Task 2 뒤에 다시 돌린다.
  - Drive `06_final_model/final_model.pth`는 `trash_file`로 휴지통에 보낸다.
- [ ] **Step 4: 문서 수정**
  - decisions.md에 D34를 추가한다.
    - 사전학습 미세조정은 규칙 위반으로 판단해 E1~E9를 폐기했다.
    - 이전 TEST 1회 평가는 폐기된 모델에만 쓰였고, 모델 선택에는 쓰이지 않았다.
    - 새 모델로 TEST를 1회 다시 평가한다.
  - 루트 README의 "최종 결과"·실험표는 "재연구 중"으로 바꾼다.
- [ ] **Step 5: 검증** — `python -m pytest tests/test_cleanup.py -q -k "old_results or fixed_splits"` → 2 passed
- [ ] **Step 6: 커밋 (사용자 확인 후)** — `git commit -m "Remove pretrained-ResNet research (rule violation); keep fixed data splits"`

---

### Task 2: 작은 ResNet 직접 구현 (노트북 7절 교체)

**Files:**
- Modify: `01_coding/notebooks/final_model.ipynb`의 `model` 태그 셀 (7절 설명 마크다운 포함)
- Create: `01_coding/tests/nbcells.py`, `01_coding/tests/test_model.py`

**Interfaces:**
- Produces:
  - `nbcells.load(tags: list[str], **overrides) -> dict`: setup 셀을 실행한 뒤 overrides로 전역 값을 덮어쓰고, 지정한 태그 셀을 순서대로 실행한 namespace를 돌려준다. 앞으로 모든 테스트가 쓴다.
  - `BasicBlock(cin: int, cout: int, stride: int)`
  - `SmallResNet(width: int = 32, hidden: tuple = (), num_classes: int = 12)`
    - 줄기: Conv3x3 s2 → BN → ReLU
    - 4단계 채널 `[w, 2w, 4w, 8w]`, 각 단계 BasicBlock 2개, 단계 첫 블록 stride `[1, 2, 2, 2]`
    - 전역 평균 풀링 → 헤드(`hidden` 층마다 Linear+ReLU, 마지막 Linear 12)
    - 합성곱은 kaiming_normal로 초기화
  - `build_model(cfg: dict) -> nn.Module`: `cfg["width"]`, `cfg["hidden"]`를 읽는다.

- [ ] **Step 1: 실패하는 테스트 작성**
```python
# 01_coding/tests/test_model.py
import torch
from nbcells import load

def test_forward_shapes():
    ns = load(["model"])
    m = ns["SmallResNet"](width=32)
    for s in (128, 160):
        assert m(torch.randn(2, 3, s, s)).shape == (2, 12)

def test_param_counts():
    ns = load(["model"])
    n = lambda **k: sum(p.numel() for p in ns["SmallResNet"](**k).parameters())
    assert n(width=32) == 2_798_124
    assert n(width=32, hidden=(256, 128)) == 2_895_276
    assert n(width=64) == 11_174_988

def test_all_params_trainable_and_no_pretrained():
    ns = load(["model"])
    m = ns["build_model"]({"width": 32, "hidden": ()})
    assert all(p.requires_grad for p in m.parameters())
    assert "models" not in ns or not hasattr(ns.get("models"), "resnet18")
```
- [ ] **Step 2: 실행해서 실패 확인** — `python -m pytest tests/test_model.py -q` → FAIL (`SmallResNet` 없음)
- [ ] **Step 3: 구현** — NotebookEdit로 `model` 셀을 위 Interfaces대로 교체한다. setup 셀에서 `from torchvision import models` import를 지운다. 7절 마크다운은 "ResNet 구조를 참고해 직접 설계, 처음부터 학습"으로 바꾼다.
- [ ] **Step 4: 통과 확인** — `python -m pytest tests/test_model.py tests/test_cleanup.py -q` → 6 passed
- [ ] **Step 5: 커밋 (사용자 확인 후)**

---

### Task 3: 전처리·증강·이미지 캐시 (노트북 6절)

**Files:**
- Modify: `final_model.ipynb`의 `cache` 태그 셀(새로 추가)과 6절 transform 셀
- Create: `01_coding/tests/test_cache.py`

**Interfaces:**
- Produces:
  - `CACHE_ROOT` (setup 셀의 전역): Colab `/content/cache`, 로컬 `<ROOT>/external_data/cache`. 테스트는 덮어쓴다.
  - `SUBSET_FOR_TEST = None` (setup 셀의 전역): 테스트 전용.
  - `val_df`, `test_df`는 `data` 태그 셀(2절)에서 정의한다. 지금은 6절에 있어서 옮긴다.
  - `CACHE_SIZE = 192`
  - `cache_resized(df: pd.DataFrame, cache_root: Path, size: int = CACHE_SIZE) -> pd.DataFrame`
    - 각 이미지를 RGB `size×size`(기존과 같은 정사각 리사이즈)로 줄여 `cache_root/<path>.png`에 저장한다. 이미 있으면 건너뛴다.
    - 반환 df에는 `file` 열(절대경로 문자열)을 붙인다. 원본이 없는 행은 빼고 출처별 장수를 출력한다.
  - `build_transforms(img_size: int, aug: list[str]) -> tuple[train_tf, eval_tf]`
    - `aug` 원소: `"hflip"` → RandomHorizontalFlip, `"rrc"` → RandomResizedCrop(img_size, scale=(0.6, 1.0)) (없으면 Resize), `"dark_blur"` → 기존 어둡게(밝기 0.3~1.0)·흐림(k5, σ0.5~2.0) 중 하나, p=0.5
    - 끝에 ToTensor와 정규화를 붙인다. 정규화 값은 학습 데이터 RGB 평균·표준편차를 `cache` 셀에서 계산해 `MEAN`, `STD`로 둔다(ImageNet 값 대신 사용).
    - eval_tf = Resize(img_size) + ToTensor + 정규화
  - `GarbageDataset(df, tf)`는 `file` 열을 읽는다.

- [ ] **Step 1: 실패하는 테스트 작성**
```python
# 01_coding/tests/test_cache.py
import pandas as pd
from PIL import Image
from nbcells import load

def test_cache_size(tmp_path):
    ns = load(["data", "cache"])
    out = ns["cache_resized"](ns["val_hard"].head(5), tmp_path)
    assert len(out) == 5 and all(Image.open(f).size == (192, 192) for f in out.file)

def test_cache_idempotent(tmp_path):
    ns = load(["data", "cache"])
    a = ns["cache_resized"](ns["val_hard"].head(3), tmp_path)
    t = [pd.Timestamp(__import__("os").path.getmtime(f), unit="s") for f in a.file]
    b = ns["cache_resized"](ns["val_hard"].head(3), tmp_path)
    assert t == [pd.Timestamp(__import__("os").path.getmtime(f), unit="s") for f in b.file]

def test_cache_skips_missing_files(tmp_path):
    ns = load(["data", "cache"])
    df = pd.concat([ns["val_hard"].head(2), pd.DataFrame([{"path": "no/such.jpg", "label": "metal", "source": "aihub"}])])
    assert len(ns["cache_resized"](df, tmp_path)) == 2

def test_transforms_output():
    ns = load(["data", "cache"])
    tr, ev = ns["build_transforms"](128, ["hflip", "rrc", "dark_blur"])
    img = Image.new("RGB", (192, 192))
    assert tr(img).shape == (3, 128, 128) and ev(img).shape == (3, 128, 128)
```
- [ ] **Step 2: 실행해서 실패 확인** — `python -m pytest tests/test_cache.py -q` → 4 failed
- [ ] **Step 3: 구현** — `cache` 셀과 6절 transform 셀을 Interfaces대로 작성하고, 예시 그림 셀이 `file` 열을 쓰도록 바꾼다.
- [ ] **Step 4: 통과 확인** — `python -m pytest tests/test_cache.py -q` → 4 passed
- [ ] **Step 5: 커밋 (사용자 확인 후)**

---

### Task 4: 학습 함수와 실험 실행기 (노트북 8절)

**Files:**
- Modify: `final_model.ipynb`의 `train` 태그 셀
- Create: `01_coding/tests/test_train.py`

**Interfaces:**
- Consumes: `build_model`(Task 2), `cache_resized`·`build_transforms`·`GarbageDataset`(Task 3)
- Produces:
  - `BASE_CFG`: `dict(width=32, hidden=(), img_size=128, aug=[], train_data="kaggle", max_per_class=1000, epochs=30, patience=7, batch=64, lr=1e-3, weight_decay=1e-4, scheduler="none", label_smoothing=0.0, seed=42)`
  - `experiment_root(in_colab: bool) -> Path`: Colab에서 `/content/drive/MyDrive/garbage_classification`이 있으면 그 아래 `experiments/`, 아니면 `RESULT/experiments/`.
  - `run_experiment(exp_id: str, cfg: dict, exp_root: Path) -> dict`
    - 진행 중에는 `exp_root/exp_id/`에 `history.csv`, `best.pth`, `cfg.json`을 쓰고, 끝나면 `metrics.json`(`val`, `val_ext`, `val_hard`, `val_avg`, `per_class_val`, `by_corruption`, `best_epoch`, `epochs_run`, `minutes`)을 쓴다.
    - `metrics.json`이 이미 있으면 학습하지 않고 그 내용을 돌려준다.
    - 옵티마이저는 AdamW. `scheduler="cosine"`이면 CosineAnnealingLR(T_max=epochs). CUDA에서는 AMP를 쓰고, num_workers는 Colab 2, 로컬 0.
    - `history.csv` 열: `epoch, train_loss, train_acc, val_loss, val_acc, val_ext_loss, val_ext_acc, score, lr, sec`. `lr`은 그 epoch 학습에 쓴 값(scheduler.step 이전)이다.
    - `SUBSET_FOR_TEST`가 정수면 학습·검증 df를 각각 그 장수로 자른다(`sample(n, random_state=seed)`).

- [ ] **Step 1: 실패하는 테스트 작성** (아주 작은 설정, CPU)
```python
# 01_coding/tests/test_train.py
import json, pandas as pd, torch
from nbcells import load
TINY = dict(epochs=2, patience=5, batch=8, img_size=64, max_per_class=3)

def _run(tmp_path, **kw):
    ns = load(["data", "cache", "model", "train"], CACHE_ROOT=tmp_path / "cache", SUBSET_FOR_TEST=24)
    cfg = {**ns["BASE_CFG"], **TINY, **kw}
    return ns, ns["run_experiment"]("T1", cfg, tmp_path / "exp")

def test_run_writes_outputs(tmp_path):
    ns, m = _run(tmp_path)
    d = tmp_path / "exp/T1"
    assert {"history.csv", "best.pth", "cfg.json", "metrics.json"} <= {p.name for p in d.iterdir()}
    assert abs(m["val_avg"] - (m["val"] + m["val_ext"] + m["val_hard"]) / 3) < 1e-9

def test_resume_skips_finished(tmp_path):
    ns, m = _run(tmp_path)
    mt = (tmp_path / "exp/T1/best.pth").stat().st_mtime
    _, m2 = _run(tmp_path)
    assert m2 == m and (tmp_path / "exp/T1/best.pth").stat().st_mtime == mt

def test_best_checkpoint_matches_best_epoch(tmp_path):
    ns, m = _run(tmp_path, epochs=3)
    h = pd.read_csv(tmp_path / "exp/T1/history.csv")
    assert m["best_epoch"] == int(h.loc[h.score.idxmax(), "epoch"])

def test_cosine_reaches_zero(tmp_path):
    ns, m = _run(tmp_path, scheduler="cosine", epochs=3)
    h = pd.read_csv(tmp_path / "exp/T1/history.csv")
    assert h.lr.iloc[0] == 1e-3 and h.lr.iloc[-1] < 1e-3
```
  - `SUBSET_FOR_TEST`: 이 값이 있으면 각 셋을 그 장수로 자르는 테스트 전용 전역 변수. 기본값 `None`.
- [ ] **Step 2: 실행해서 실패 확인** — `python -m pytest tests/test_train.py -q` → 4 failed
- [ ] **Step 3: 구현** — `train` 셀을 Interfaces대로 작성한다. 조기 종료는 Global Constraints를 따른다. 최고 epoch 가중치로 val·val_ext·val_hard를 평가해 metrics를 만든다.
- [ ] **Step 4: 통과 확인** — `python -m pytest tests/test_train.py -q` → 4 passed
- [ ] **Step 5: 커밋 (사용자 확인 후)**

---

### Task 5: 실험 사다리 (노트북 새 9절) — 튜닝 과정 자동 실행·기록

**Files:**
- Modify: `final_model.ipynb`에 `ladder` 태그 셀과 설명 마크다운 추가
- Create: `01_coding/tests/test_ladder.py`

**Interfaces:**
- Consumes: `run_experiment`, `BASE_CFG`, `experiment_root`(Task 4)
- Produces:
  - `adopt(cand: dict, best: dict) -> bool`: Global Constraints의 채택 조건
  - `LADDER`: 아래 표의 리스트. 각 원소는 `{"id", "change": dict, "why": str}`
  - `run_ladder(ladder: list, exp_root: Path, stop_before: str | None = None) -> pd.DataFrame`
    - 각 단계의 설정 = 현재 최고 설정 + `change`. `run_experiment`를 실행한 뒤 `adopt`로 최고 설정을 갱신한다.
    - 단계마다 `exp_root/ladder.csv`에 한 줄을 쓴다: `id, change, why, val, val_ext, val_hard, val_avg, best_epoch, epochs_run, minutes, adopted, best_after`
    - `RUN_EXPERIMENTS=False`면 실행하지 않고 `ladder.csv`를 읽기만 한다.

| ID | 바꾸는 것 (최고 설정 대비) | 이유 |
|---|---|---|
| R1 | 기준: Kaggle만(클래스당 ≤1,000), 증강 없음, 128px, width 32, 선형 헤드, lr 고정, 30 epoch | 출발점 |
| R2 | `train_data="mix"` (외부 혼합 12,000장) | 외부 사진 일반화 |
| R3 | `aug=["dark_blur"]` | 어둡고 흐린 사진 대비 |
| R4 | `aug=["hflip","rrc","dark_blur"]` | 처음부터 학습할 때의 과적합 억제 |
| R5 | `scheduler="cosine"` | 후반 학습 안정화 |
| R6 | `epochs=60, patience=10` | epoch 증가 1 |
| R7 | `epochs=100, patience=15` | epoch 증가 2 (R6이 마지막 epoch까지 오를 때) |
| R8 | `hidden=(256,128)` | 헤드 용량 |
| R9 | `width=48` | 모델 폭 |
| R10 | `img_size=160` | 해상도 (작은 물체) |
| R11 | `lr=3e-3` | 학습률 |
| R12 | `weight_decay=5e-4` | 규제 |

- [ ] **Step 1: 실패하는 테스트 작성**
```python
# 01_coding/tests/test_ladder.py
from nbcells import load

def _m(v, pc_c=0.9, pc_s=0.9):
    return {"val_avg": v, "per_class_val": {"clothes": pc_c, "shoes": pc_s}}

def test_adopt_rule():
    adopt = load(["model", "train", "ladder"])["adopt"]
    assert adopt(_m(0.711), _m(0.70)) is True
    assert adopt(_m(0.709), _m(0.70)) is False
    assert adopt(_m(0.80, pc_c=0.86), _m(0.70)) is False

def test_ladder_ids_unique_and_one_change():
    L = load(["model", "train", "ladder"])["LADDER"]
    assert [s["id"] for s in L] == [f"R{i}" for i in range(1, 13)]
    assert all(len(s["change"]) <= 2 for s in L[1:])   # epochs+patience만 2개

def test_run_ladder_tiny(tmp_path):
    ns = load(["data", "cache", "model", "train", "ladder"], CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=24)
    tiny = [{**s, "change": {**s["change"], "epochs": 1, "batch": 8, "img_size": 64}} for s in ns["LADDER"][:2]]
    df = ns["run_ladder"](tiny, tmp_path / "exp")
    assert list(df.id) == ["R1", "R2"] and (tmp_path / "exp/ladder.csv").exists()
```
- [ ] **Step 2: 실행해서 실패 확인** — `python -m pytest tests/test_ladder.py -q` → 3 failed
- [ ] **Step 3: 구현** — `ladder` 셀을 작성한다. 마크다운에는 위 표와 채택 조건을 쉬운 말로 적는다.
- [ ] **Step 4: 통과 확인** — `python -m pytest tests -q` → 전부 통과
- [ ] **Step 5: 커밋 (사용자 확인 후)** 후 **GitHub push (사용자 확인 후)**. Colab이 노트북을 GitHub에서 열 수 있게 하기 위해서다.

---

### Task 6: 튜닝 과정 시각화와 최종 평가·TEST (노트북 10~12절)

**Files:**
- Modify: `final_model.ipynb`의 `report`·`final` 태그 셀
- Create: `01_coding/tests/test_final.py`

**Interfaces:**
- Consumes: `ladder.csv`, `R*/history.csv`, `R*/metrics.json`(Task 5), `build_model`
- Produces:
  - `report` 셀
    - 사다리 표(채택/기각)
    - val_avg 추이 막대
    - 실험별 val loss·val_ext acc 곡선 겹침
    - **epoch 증가 효과 그림**(R5·R6·R7의 epoch별 val_avg 대용 점수)
    - 최종 설정의 손실·정확도 곡선과 과적합 지표
  - `FINAL_ID`: 사다리가 끝난 뒤의 최고 ID (`ladder.csv`의 마지막 `best_after`)
  - `final_test_once(model, test_sets: dict, out_dir: Path) -> dict`: `out_dir/final_test.json`이 있으면 `AssertionError("TEST는 이미 1회 평가함")`. 없으면 평가 후 json과 혼동행렬 CSV 2개를 저장한다.
  - 검증 혼동행렬, 클래스별 정밀도·재현율·F1 표(기존 셀 재사용)

- [ ] **Step 1: 실패하는 테스트 작성**
```python
# 01_coding/tests/test_final.py
import pytest
from nbcells import load

def test_final_test_runs_once(tmp_path):
    ns = load(["data", "cache", "model", "train", "final"], CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=8)
    m = ns["build_model"]({"width": 32, "hidden": ()}).eval()
    sets = {"test": ns["test_df"].head(8), "test_ext": ns["test_ext"].head(8)}
    r = ns["final_test_once"](m, sets, tmp_path)
    assert set(r) == {"test", "test_ext"} and (tmp_path / "final_test.json").exists()
    with pytest.raises(AssertionError, match="이미 1회"):
        ns["final_test_once"](m, sets, tmp_path)
```
- [ ] **Step 2: 실패 확인 → Step 3: 구현 → Step 4: 통과 확인** — `python -m pytest tests -q` → 전부 통과
- [ ] **Step 5: 커밋 (사용자 확인 후)**

---

### Task 7: Colab 실행 (사용자) — 실험 사다리 R1~R12

**Files:** 없음 (실행 결과는 Drive `garbage_classification/experiments/`)

- [ ] **Step 1: 스모크 실행** (사용자, Colab T4)
  - Colab에서 GitHub의 `final_model.ipynb`를 연다.
  - 런타임 → GPU를 고른다.
  - `RUN_EXPERIMENTS=True`, `stop_before="R2"`로 R1만 실행한다.
  - 기대 결과: 마지막에 `ladder.csv`의 R1 행이 출력되고, epoch당 시간이 표시된다.
- [ ] **Step 2: 시간 확인 후 범위 결정**
  - R1의 epoch당 시간 × 남은 epoch 합계로 총시간을 추정한다.
  - Colab 하루 GPU 한도 안에 들지 않으면 R9(width) → R12 → R11 순으로 뺀다.
- [ ] **Step 3: 전체 실행** — `stop_before=None`. 세션이 끊기면 같은 셀을 다시 실행한다(완료된 실험은 건너뜀).
- [ ] **Step 4: 결과 회수** (팀장)
  - Drive `experiments/`의 `ladder.csv`, `R*/history.csv`, `R*/cfg.json`, `R*/metrics.json`을 `RESULT/experiments/`로 받는다. 크기가 작아 Drive 도구로 받는다.
  - 최고 모델 `best.pth`는 사용자가 Drive에서 `01_coding/checkpoints/final_model.pth`로 내려받는다.
  - 검증: `ladder.csv` 12행(또는 제외분 반영), `best_after` 마지막 값 = `FINAL_ID`

---

### Task 8: 최종 TEST 1회, 노트북 출력 확정, Drive·문서 갱신

**Files:**
- Modify: `final_model.ipynb`(출력 포함), `RESULT/final_test/*`, `RESULT/README.md`, `README.md`, `00_lead/decisions.md`
- Drive: `06_final_model/final_model.pth` 교체, `split_and_results.zip` 갱신

- [ ] **Step 1: TEST 1회** — 노트북을 `RUN_EXPERIMENTS=False`, `RUN_FINAL_TEST=True`로 한 번 실행한다(로컬 CPU 가능). 실행 후 곧바로 `RUN_FINAL_TEST=False`로 되돌린다.
- [ ] **Step 2: 출력 확정 실행** — `RUN_FINAL_TEST=False`로 전체를 다시 실행한다(`jupyter nbconvert --execute --inplace`). 오류와 stderr가 0인지, 모든 코드 셀이 `# [셀 목적]`으로 시작하는지 확인한다.
- [ ] **Step 3: 테스트 전체** — `python -m pytest tests -q` → 전부 통과
- [ ] **Step 4: Drive 갱신**
  - 새 `final_model.pth`를 올린다. 10MB를 넘으면 사용자가 직접 올린다.
  - `split_and_results.zip`을 새 실험 결과로 다시 만들어 올린다(10MB 미만).
- [ ] **Step 5: 문서**
  - README 최종 결과와 사다리 표를 갱신한다.
  - decisions.md에 D35(최종 모델·TEST 결과)를 추가한다.
  - 보고서용 메모에 이전 TEST 사용 경위를 적는다.
- [ ] **Step 6: 커밋·push (사용자 확인 후)**

---

## 시간 (추정, 11/20 마감)

| 작업 | 시간 |
|---|---|
| Task 1~6 (코드·테스트, 로컬) | 약 2~3시간 |
| Task 7 (Colab T4) | R1 30 epoch 약 10~15분 기준, 전체 약 3~5시간. 캐시 덕분에 이미지 읽기는 짧음 |
| Task 8 | 약 1시간 |
