# 2차 튜닝 (정리된 데이터·비율 유지·새 증강) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 정리된 외부 데이터로 2차 실험 사다리(M1~M9)를 실행할 수 있게 노트북을 고치고, 1차→2차 튜닝 과정과 배경 외우기 점검을 보여 준다.

**Architecture:** 기존 `final_model.ipynb`의 태그별 셀(setup·data·cache·model·train·ladder·report·final·eval)을 고친다. 테스트는 `tests/nbcells.py`로 태그 셀을 실행해 함수 단위로 검증한다. 1차 결과(`ladder.csv`, R1~R12)는 읽기만 하고, 2차는 `ladder_m.csv`에 기록한다.

**Tech Stack:** Python, PyTorch, torchvision.transforms, pandas, pytest, nbformat

**Spec:** `docs/superpowers/specs/2026-11-19-scratch-cnn-round2-design.md`

## Global Constraints
- 사전학습 가중치·`torchvision.models` 사용 금지. 모델 구조(SmallResNet) 변경 없음.
- 무작위 자르기(RandomResizedCrop)를 2차 사다리에 쓰지 않는다 (`rrc` 코드는 1차 기록용으로 유지).
- 모든 코드 셀 첫 줄은 `# [셀 목적] ...` 주석. 은어(E9, S3 등) 금지, 지표 이름은 val / val_ext / val_hard / val_avg.
- 학습·검증·테스트 사진은 겹치지 않는다. TEST는 `final_test_once`로 1회만.
- 1차 실험(R1~R12)은 다시 실행하지 않는다. `BASE_CFG`는 바꾸지 않는다 (저장된 R의 cfg.json과 같아야 함). 새 cfg 키는 `cfg.get(키, 기본값)`으로 읽는다.
- 커밋은 이 브랜치(`scratch-cnn`)에만. push는 사용자 승인 후.
- 전체 테스트 명령: `cd 01_coding && python -m pytest tests -q` (현재 30개 통과).

## Review Focus
1. 이미 압축을 푼 Colab 런타임(`.prepared` 있음)에서 새 zip이 추가되면 다시 풀어야 한다 → Task 1 `test_unpack_reextracts_when_zip_set_changes`.
2. 세로로 긴/가로로 긴 사진이 비율 유지 캐시에서 늘어나지 않고 채움색으로 패딩되어야 한다 → Task 2 `test_letterbox_keeps_aspect`.
3. 클래스 가중치가 검증 손실에 섞이면 실험 간 val loss 비교가 깨진다 → Task 3 `test_eval_loss_unweighted`.
4. 2차 사다리가 1차 R의 cfg 검사(설정 불일치 오류)를 건드리지 않아야 한다 → Task 4 `test_round2_does_not_touch_round1`.
5. `ladder_m.csv`가 없을 때(2차 실행 전) 보고 셀이 1차 결과만으로 끝까지 돌아야 한다 → Task 5 `test_report_round1_only`.

---

### Task 1: 정리된 데이터 분할 읽기 + Drive zip 갱신

**Files:**
- Modify: `01_coding/notebooks/final_model.ipynb` (setup 셀 `46d38c4d`, 데이터 셀 뒤에 새 셀 `clean-md`, `clean-load`, `clean-tables`; 데이터셋 정리 표 `3c72ed97`·`fd0b5b85` 갱신)
- Create: `01_coding/tests/test_data_clean.py`
- Add (기록용): `01_coding/tools/{build_extra_external.py, glass_check.py, clean_splits.py, household_selected.json, extra_rejects.csv, split_rejects.csv}`, `01_coding/RESULT/data_split/{*_clean.csv, extra_external.csv, extra_counts.csv, clean_report.csv}`

**Interfaces:**
- Produces: 전역 `train_mix`(27,050), `val_ext`(498), `test_ext`(360), `val_hard`(1,198) = 정리본. `train_mix_v1`, `val_ext_v1`, `test_ext_v1`, `val_hard_v1` = 1차 분할(표시용). `DATA_ZIPS`에 `aihub_extra.zip`, `extra_public.zip` 포함. `unpack_drive_data(src, root)`는 `.prepared`에 zip 이름 목록을 기록하고, 목록이 다르면 다시 푼다.

- [ ] **Step 1: 실패하는 테스트 작성** — `test_data_clean.py`
```python
def test_clean_splits_loaded():
    ns = load(["data"])
    assert (len(ns["train_mix"]), len(ns["val_ext"]), len(ns["test_ext"]), len(ns["val_hard"])) == (27050, 498, 360, 1198)
    assert len(ns["train_mix_v1"]) == 12000

def test_no_overlap_between_splits():
    ns = load(["data"])
    tr, ve, te = set(ns["train_mix"].path), set(ns["val_ext"].path), set(ns["test_ext"].path)
    assert not tr & ve and not tr & te and not ve & te
    assert not set(ns["val_hard"].orig_path) & tr

def test_data_zips_include_extra():
    ns = load([])
    assert {"aihub_extra.zip", "extra_public.zip"} <= ns["DATA_ZIPS"]

def test_unpack_reextracts_when_zip_set_changes(tmp_path):
    ns = load([])
    src, root = tmp_path / "src", tmp_path / "root"; src.mkdir()
    _zip(src / "split_and_results.zip", "a.txt")
    ns["unpack_drive_data"](src, root)
    _zip(src / "extra_public.zip", "b.txt")
    ns["unpack_drive_data"](src, root)
    assert (root / "b.txt").exists()
```
(`_zip`은 `test_review_fixes.py`와 같은 도우미)

- [ ] **Step 2: 실행해 실패 확인** — `python -m pytest tests/test_data_clean.py -q` → 4 FAIL (`train_mix_v1` 없음, 장수 12000, DATA_ZIPS 누락, b.txt 없음)
- [ ] **Step 3: 구현**
  - setup: `DATA_ZIPS`에 두 zip 추가. `unpack_drive_data`: `.prepared` 내용 = 정렬된 zip 이름 줄바꿈 목록. 내용이 현재 zip 목록과 같을 때만 건너뜀.
  - 새 셀 `clean-load`(태그 data, 기존 데이터 셀들 뒤): 1차 분할을 `*_v1` 이름으로 옮기고 `*_clean.csv`를 읽어 네 전역에 넣는다. 파일이 없으면 `clean_splits.py`를 실행하라는 메시지와 함께 오류.
  - 새 셀 `clean-tables`(태그 data): `extra_counts.csv`, `clean_report.csv` 요약표, 학습·val_ext·test_ext의 클래스×출처 장수(비율) 표 출력.
  - `clean-md`: 정리 기준 3가지, 보충 방식, 결과 장수, 스크립트 위치, 알려진 한계 설명. 데이터셋 정리 표(`3c72ed97`)와 장수 표(`fd0b5b85`)를 정리본 숫자로 갱신.
- [ ] **Step 4: 통과 확인** — `python -m pytest tests -q` → 전부 통과 (새 테스트 포함) (기존 테스트 중 장수 12,000이나 val_ext 500을 가정한 것이 있으면 `*_v1`을 쓰도록 고친다)
- [ ] **Step 5: 커밋** — `git add ...; git commit -m "Use cleaned splits (train 27,050 / val_ext 498 / test_ext 360 / val_hard 1,198) and new Drive zips"`

### Task 2: 비율 유지 캐시 + 정규화 통계 + 새 증강

**Files:**
- Modify: 캐시 셀 `2349ad66`, 증강 예시 셀 `5f20a341`, 전처리 설명 `ffc5145e`
- Test: `01_coding/tests/test_preprocess.py`

**Interfaces:**
- Produces: `letterbox(img: Image.Image, size: int, fill: tuple) -> Image.Image`; `cache_resized(df, cache_root, size=192, name="데이터", mode="squash"|"letterbox") -> DataFrame` (letterbox는 `cache_root/192_lb/...`); `MEAN`, `STD` = `norm_stats_clean.json` 값(없으면 정리된 학습 데이터 2,000장으로 계산·저장); `FILL` = MEAN×255 정수 RGB; `build_transforms(img_size, aug) -> (train_tf, eval_tf)`가 `hflip`, `vflip`, `rot`(RandomRotation 20, fill=FILL), `color`(ColorJitter 0.2/0.2/0.2/0.03), `trivial`(TrivialAugmentWide), `dark_blur`, `rrc`를 이해.

- [ ] **Step 1: 실패하는 테스트**
```python
def test_letterbox_keeps_aspect():
    ns = load(["data", "cache"])
    img = Image.new("RGB", (300, 100), (255, 0, 0))
    out = ns["letterbox"](img, 192, (10, 20, 30))
    assert out.size == (192, 192)
    assert out.getpixel((96, 5)) == (10, 20, 30) and out.getpixel((96, 96)) == (255, 0, 0)

def test_letterbox_cache_separate_dir(tmp_path):
    ns = load(["data", "cache"])
    a = ns["cache_resized"](ns["val_ext"].head(2), tmp_path, mode="letterbox")
    assert all("192_lb" in f for f in a.file)

def test_new_aug_flags_no_crop():
    ns = load(["data", "cache"])
    tf, _ = ns["build_transforms"](128, ["hflip", "vflip", "rot", "color", "trivial", "dark_blur"])
    names = [type(t).__name__ for t in tf.transforms]
    for n in ["RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation", "ColorJitter", "TrivialAugmentWide"]:
        assert n in names
    assert "RandomResizedCrop" not in names

def test_norm_stats_from_clean():
    ns = load(["data", "cache"])
    assert (ns["DATA_DIR"] / "norm_stats_clean.json").exists() and len(ns["FILL"]) == 3
```
- [ ] **Step 2: 실패 확인** — `python -m pytest tests/test_preprocess.py -q` → 4 FAIL
- [ ] **Step 3: 구현** — letterbox: 긴 변을 size로 BILINEAR 축소/확대 후 size×size 캔버스(FILL) 가운데에 붙임. 증강 순서: (기하) hflip·vflip·rot → (색) color·trivial → dark_blur → ToTensor·Normalize. 증강 예시 셀은 `["hflip","vflip","rot","color","dark_blur"]`, letterbox 캐시 이미지로 보여 준다. 설명 셀에 비율 유지 방식과 무작위 자르기를 쓰지 않는 이유(물체 일부가 잘려 다른 클래스처럼 보일 수 있음)를 적는다.
- [ ] **Step 4: 통과 확인** — `python -m pytest tests -q` → 전부 통과 (새 테스트 포함)
- [ ] **Step 5: 커밋** — `"Add letterbox cache, clean normalization stats and flip/rotate/color/trivial augmentations"`

### Task 3: 학습 — 정리 데이터, 비율 유지, 클래스 가중치, 배경 점검 지표

**Files:**
- Modify: 학습 셀 `0234f7a3`, 학습 설명 `2b7f52e3`
- Test: `01_coding/tests/test_train_round2.py`

**Interfaces:**
- Consumes: Task 2 `cache_resized(mode=)`, `build_transforms`
- Produces: `load_train_df(cfg)` — `train_data="mix2"` → 정리된 `train_mix`, `"mix"` → `train_mix_v1`, `"kaggle"` 그대로. `class_weights(df) -> torch.Tensor` (클래스 순서 = CLASSES, w_c = sqrt(N/n_c) 평균 1). `run_experiment`가 `cfg.get("resize","squash")`로 캐시 모드, `cfg.get("class_weight","none")=="sqrt_inv"`이면 학습 손실에 가중치, 평가 손실은 가중치 없음. metrics에 `val_ext_by_source: {source: acc}`, `aihub_metal_as_glass_battery: float`, `per_class_val_ext: {class: acc}` 추가. `make_criteria(cfg, train_df) -> (train_criterion, eval_criterion)`: 학습용은 label_smoothing과 (sqrt_inv면) 클래스 가중치, 평가용은 항상 `nn.CrossEntropyLoss()`.

- [ ] **Step 1: 실패하는 테스트**
```python
def test_load_train_mix2():
    ns = load(TAGS)
    assert len(ns["load_train_df"]({**ns["BASE_CFG"], "train_data": "mix2"})) == 27050

def test_class_weights_mean_one_and_order():
    ns = load(TAGS)
    w = ns["class_weights"](ns["train_mix"])
    assert len(w) == 12 and abs(float(w.mean()) - 1) < 1e-6
    i = ns["CLASSES"].index
    assert w[i("white-glass")] > w[i("shoes")]

def test_eval_loss_unweighted(tmp_path):
    ns = load(TAGS, CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    base = {**ns["BASE_CFG"], "epochs": 1, "batch": 8, "img_size": 64, "train_data": "mix2", "resize": "letterbox"}
    a = ns["run_experiment"]("A", base, tmp_path / "e")
    b = ns["run_experiment"]("B", {**base, "class_weight": "sqrt_inv"}, tmp_path / "e")
    for m in (a, b):
        assert {"val_ext_by_source", "aihub_metal_as_glass_battery", "per_class_val_ext"} <= set(m)
        assert 0 <= m["aihub_metal_as_glass_battery"] <= 1
```
```python
def test_eval_criterion_unweighted():
    ns = load(TAGS)
    tc, ec = ns["make_criteria"]({**ns["BASE_CFG"], "class_weight": "sqrt_inv"}, ns["train_mix"])
    assert tc.weight is not None and ec.weight is None
    tc0, _ = ns["make_criteria"](ns["BASE_CFG"], ns["train_mix"])
    assert tc0.weight is None
```
(`TAGS = ["data","cache","model","train"]`. 위 `test_eval_loss_unweighted`는 이름을 `test_round2_metrics_keys`로 한다.)
- [ ] **Step 2: 실패 확인** — 3 FAIL
- [ ] **Step 3: 구현** — `aihub_metal_as_glass_battery` = val_ext 중 source=="aihub" & label=="metal" 행의 예측이 {battery, brown-glass, green-glass, white-glass}인 비율 (해당 행이 0개면 NaN 대신 0.0이 아니라 `None`). 출처별 정확도는 `sets["val_ext"].source`로 묶는다.
- [ ] **Step 4: 통과 확인** — `python -m pytest tests -q` → 전부 통과 (새 테스트 포함)
- [ ] **Step 5: 커밋** — `"Train on cleaned mix with optional letterbox and class-weighted loss; record per-source val_ext accuracy"`

### Task 4: 채택 규칙 수정 + 2차 사다리

**Files:**
- Modify: 사다리 설명 `ladder-md`, `ladder-def`, `ladder-run`
- Modify: `01_coding/tests/test_ladder.py` (clothes 조건 단언 제거)
- Test: `01_coding/tests/test_ladder_round2.py`

**Interfaces:**
- Produces: `adopt(cand, best) -> bool` = `round(cand.val_avg - best.val_avg, 6) >= 0.01` (클래스 조건 없음). `M_BASE` = `{**BASE_CFG, "width": 48, "aug": ["dark_blur"], "scheduler": "cosine"}` (1차 최종 R9 설정). `LADDER_M` = 설계 5절 M1~M9 (M1 `{"train_data":"mix2"}`, M2 `{"resize":"letterbox"}`, M3 `{"aug":["dark_blur","hflip","vflip","rot"]}`, M4 aug에 `"color"` 추가, M5 aug에 `"trivial"` 추가, M6 `{"class_weight":"sqrt_inv"}`, M7 `{"label_smoothing":0.1}`, M8 `{"epochs":60,"patience":10}`, M9 `{"img_size":160}`). aug 변경은 "그때 최고 설정의 aug + 추가"가 되도록 change에 `"aug_add": [...]`를 쓰고 run_ladder가 합친다. `run_ladder(ladder, exp_root, stop_before=None, base_cfg=BASE_CFG, csv_name="ladder.csv")`; ladder 행에 `per_class_delta`(최고 대비 클래스별 val 변화 JSON) 추가. `ladder-run`: RUN_EXPERIMENTS면 `run_ladder(LADDER_M, EXP_ROOT, STOP_BEFORE, M_BASE, "ladder_m.csv")`; `ladder_df_r1` = `ladder.csv` 읽기, `ladder_df` = `ladder_m.csv` 읽기(없으면 빈 표).

- [ ] **Step 1: 실패하는 테스트**
```python
def test_adopt_no_class_guard():
    adopt = load(TAGS)["adopt"]
    assert adopt(_m(0.80, pc_c=0.50), _m(0.70)) is True

def test_ladder_m_ids_and_aug_add(tmp_path):
    ns = load(TAGS, CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    assert [s["id"] for s in ns["LADDER_M"]] == [f"M{i}" for i in range(1, 10)]
    tiny = [{**s, "change": {**s["change"], "epochs": 1, "batch": 8, "img_size": 64}} for s in ns["LADDER_M"][:3]]
    df = ns["run_ladder"](tiny, tmp_path / "exp", base_cfg=ns["M_BASE"], csv_name="ladder_m.csv")
    cfg3 = json.load(open(tmp_path / "exp/M3/cfg.json"))
    assert set(cfg3["aug"]) == {"dark_blur", "hflip", "vflip", "rot"} and "aug_add" not in cfg3
    assert (tmp_path / "exp/ladder_m.csv").exists() and "per_class_delta" in df

def test_round2_does_not_touch_round1(tmp_path):
    ns = load(TAGS, CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    (tmp_path / "exp/R1").mkdir(parents=True)
    (tmp_path / "exp/R1/cfg.json").write_text('{"old": 1}'); (tmp_path / "exp/R1/metrics.json").write_text("{}")
    tiny = [{**ns["LADDER_M"][0], "change": {"train_data": "mix2", "epochs": 1, "batch": 8, "img_size": 64}}]
    ns["run_ladder"](tiny, tmp_path / "exp", base_cfg=ns["M_BASE"], csv_name="ladder_m.csv")   # 오류 없이 끝나야 함
```
- [ ] **Step 2: 실패 확인** — 3 FAIL
- [ ] **Step 3: 구현** — `ladder-md`에 채택 규칙 변경 이유(설계 3절)와 M1~M9 표, 예상 시간, 1차와 숫자를 직접 비교하지 않는 이유를 적는다. `assert_full_data`는 정리된 네 세트로 검사.
- [ ] **Step 4: 통과 확인** — `python -m pytest tests -q` → 전부 통과 (새 테스트 포함)
- [ ] **Step 5: 커밋** — `"Drop class guard from adoption; add round-2 ladder M1-M9 with aug_add and ladder_m.csv"`

### Task 5: 보고·평가·TEST 셀 — 1차→2차 과정, 배경 점검

**Files:**
- Modify: `report-md`, `report-table`, `report-curves`, `report-epochs`, `report-final-curve`, `eval-md`, `eval-run`, `test-md`, `test-run`
- Modify: `01_coding/tests/test_final.py` (작은 사다리를 `LADDER_M`·`ladder_m.csv`로)

**Interfaces:**
- Consumes: Task 4 `ladder_df_r1`, `ladder_df`, metrics의 새 키
- Produces: `FINAL_ID` = `ladder_df.best_after.iloc[-1]`(2차), 없으면 None. `report-table`은 1차 표(“이전 데이터 기준”)와 2차 표·막대를 함께. `report-epochs`는 1차 R5/R6/R7과 2차 M7/M8을 각각 그린다(있는 것만). `eval-run`은 val_ext 출처별 정확도와 `aihub_metal_as_glass_battery`를 출력하고, AI Hub metal의 예측 분포 표를 보여 준다. `test-run`은 정리된 `test_ext`를 사용.

- [ ] **Step 1: 실패하는 테스트**
```python
def test_report_round1_only(tmp_path):
    (tmp_path / "ladder.csv").write_text(R1_CSV)      # 1차 표 한 줄짜리 고정 문자열
    ns = load(TAGS + ["report"], EXP_ROOT=tmp_path)
    assert ns["FINAL_ID"] is None and len(ns["ladder_df_r1"]) == 1

def test_report_and_eval_round2_tiny(tmp_path):
    # 기존 test_report_and_eval_cells_with_tiny_results를 LADDER_M[:2], M_BASE, ladder_m.csv로 바꾼 것
    ...
    assert ns2["FINAL_ID"] in {"M1", "M2"} and "aihub_metal_as_glass_battery" in ns2
```
- [ ] **Step 2: 실패 확인** — 2 FAIL
- [ ] **Step 3: 구현** — 결론 셀(`conclusion-md`)은 자리표시 그대로.
- [ ] **Step 4: 통과 확인** — `python -m pytest tests -q` → 전부 통과 (새 테스트 포함)
- [ ] **Step 5: 커밋** — `"Report round-1 and round-2 tuning, per-source val_ext and background-shortcut check"`

### Task 6: 문서·노트북 정리

**Files:**
- Modify: 노트북 첫 셀 `f3d81790`(전체 흐름·데이터 장수), `README.md`, `01_coding/README.md`, `00_lead/decisions.md`(D35: 2차 튜닝 — 데이터 정리·채택 규칙 변경·비율 유지·증강), `01_coding/RESULT/README.md`(새 CSV·tools 설명), 데이터 출처에 TrashBusters(MIT)·Recyclable and Household Waste(Kaggle)·ShoeSandalBoot(라이선스 표기 없음) 추가
- Test: 기존 `test_cleanup.py` (사전학습 문자열 없음, 셀 목적 주석)

- [ ] **Step 1: 실패하는 테스트** — `test_cleanup.py`에 추가
```python
def test_cell_purpose_and_no_jargon():
    nb = nbformat.read(NB, 4)
    assert all(c.source.lstrip().startswith("# [셀 목적]") for c in nb.cells if c.cell_type == "code")
    text = "\n".join(c.source for c in nb.cells)
    assert not re.search(r"\b(S3|E[1-9])\b", text)
```
- [ ] **Step 2: 실패 확인** — 주석이 빠진 셀이나 은어가 있으면 FAIL (없으면 바로 PASS — 그 경우 테스트가 검사 범위를 지키는지 셀 하나의 주석을 지워 FAIL을 확인한 뒤 되돌린다)
- [ ] **Step 3: 문서 갱신**
- [ ] **Step 4: 통과 확인** — `python -m pytest tests -q` → 전부 통과
- [ ] **Step 5: 커밋** — `"Document round-2 data cleaning, sources and tuning plan"`

### Task 7 (사용자, Colab): 2차 사다리 실행
1. GitHub push 승인 → Colab에서 노트북 열기 → 런타임 **T4 GPU**.
2. `RUN_EXPERIMENTS=True`, `STOP_BEFORE="M2"`로 M1만 실행해 epoch당 시간 확인.
3. `STOP_BEFORE=None`으로 나머지 실행 (끊기면 같은 셀 재실행).
4. 끝나면 Drive `experiments/ladder_m.csv`와 M*/metrics.json을 로컬 `RESULT/experiments/`로 가져와 분석.

### Task 8: 최종 TEST 1회 + 결론
`RUN_FINAL_TEST=True`로 한 번 → False. 결론 셀 작성, README·decisions 결과 반영, Drive에 `final_model.pth` 업로드, push(승인 후).
