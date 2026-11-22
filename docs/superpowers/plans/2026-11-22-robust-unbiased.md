# 편향을 줄이고 노이즈에 강건한 모델 (3차 사다리 N1~N3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M8을 출발점으로, 한쪽 출처·모습에 치우친 학습을 줄이고(편향) 빛번짐·잡음·얼룩 같은 손상에도 덜 흔들리는(강건) 모델을 3차 사다리로 찾는다.

**Architecture:** 노트북 `01_coding/notebooks/final_model.ipynb`에 (1) 편향 점검 표, (2) 강건 증강 `robust`, (3) 손상 검증셋 `val_dirty`와 선택 지표 `val_avg4`, (4) 클래스 안 출처 균형 샘플링 `source_cap`, (5) 글자 없는 종이 보강 데이터 `mix3`를 더한다. 실험은 M8 설정에서 한 번에 하나씩 바꾸는 3차 사다리(N1~N3, `ladder_n.csv`)로 Colab GPU에서 돌리고, 채택은 `val_avg4`로 한다. TEST·test_dirty는 사다리가 끝난 뒤 사용자가 승인할 때만 평가한다.

**Tech Stack:** PyTorch(사전학습 없음), torchvision transforms, Pillow, numpy, pandas, imagehash, pytest, nbformat/nbconvert, Colab GPU

**Spec:** 아래 "방향" 절 (사용자 지시 2026-11-22: "한쪽으로 편향되지 않지만 노이즈에도 강건한 모델"). 근거 자료: 노트북 12-2절(test_dirty v2 결과), D36~D39, 이 대화의 종이 점검 결과

## 방향
- **강건성 근거**: test_dirty v2에서 같은 사진 집단의 원본 0.826 → 손상 0.720(-10.6%p). 강한 빛번짐 0.28, 강한 저조도+잡음 0.51, 강한 얼룩 0.60. 학습 증강에 있던 어둡게·흐림 계열은 상대적으로 버팀 → **학습 증강에 손상 종류를 더한다.**
- **편향 근거(지금까지 확인한 것)**:
  - paper: 학습 2,522장 중 약 75~80%가 글자 인쇄물. 검증 사진에서 글자 인쇄물 0.950(141장) vs 글자 거의 없는 종이 0.864(66장); Kaggle 배경만 보면 0.967 vs 0.865(37장) → **글자 없는 종이 데이터를 새로 더한다.**
  - 한 출처 비중이 큰 클래스(학습 27,050장 기준): battery AI Hub 75%, green-glass AI Hub 71%, brown-glass AI Hub 70%, white-glass AI Hub 57%(모두 아스팔트·보도블록 배경), shoes ShoeSandalBoot 54%(흰 배경 제품 사진), clothes Kaggle 95%(한 출처). AI Hub 금속 25장 중 3장을 battery·유리로 예측(배경 단서 의심).
  - trash: 마스크·칫솔·기저귀(Kaggle) + 스티로폼(Household) + 잡다한 쓰레기(RealWaste·V2)가 섞인 정의.
- **새 데이터는 paper(글자 없는 종이)만 기본으로 찾는다.** 다른 클래스는 Task 1 점검 결과를 보고 사용자가 정한다. 나머지 편향은 새 데이터 없이 **출처 균형 샘플링**으로 먼저 줄인다.
- test_dirty v2와 TEST 사진은 학습·선택에 쓰지 않는다. 새 데이터는 이 두 셋과 겹치면 안 된다.

## Global Constraints
- PyTorch만, 사전학습 가중치·`torchvision.models` 금지(transforms만 사용).
- 노트북 코드 셀은 `# [셀 목적]`으로 시작, 내부 용어 금지, 지표 이름 val / val_ext / val_hard / val_avg (새 지표 `val_dirty`, `val_avg4`, `paper_plain`은 노트북 markdown에서 처음 나올 때 뜻을 적는다). 문서·답변 한국어.
- 3차 사다리 채택 규칙: `val_avg4 = (val + val_ext + val_hard + val_dirty) / 4`가 현재 최고보다 **+0.01 이상**이면 채택. 출발점(현재 최고)은 M8.
- 한 단계에 한 가지만 바꾼다. 실험 ID는 `N1`, `N2`, `N3`; 결과 표 `experiments/ladder_n.csv`. 1·2차 결과(`ladder.csv`, `ladder_m.csv`, R*/M* 폴더)는 바꾸지 않는다.
- 손상 종류 `["noise", "jpeg", "motion_blur", "low_light_noise", "glare", "stain", "dark", "blur"]`, 세기 `mild`/`strong` — test_dirty와 같은 정의. `val_dirty`는 검증 사진(val_hard와 같은 1,198장의 원본)에만 적용, seed `7`.
- 새 외부 데이터: phash hamming ≤ 4 중복 제거 대상에 학습·val·val_ext·val_hard·TEST·test_ext·**test_dirty v2의 원본(origin)** 모두 포함. 라이선스를 README 데이터 출처 표에 적는다. 이미지는 저장소에 올리지 않고 Drive zip으로.
- TEST(`final_test.json`)와 test_dirty(`dirty_test_predictions.csv`)는 이 계획 안에서 **사용자 승인 없이는 다시 평가하지 않는다**(Task 7의 게이트).
- `RUN_EXPERIMENTS`, `RUN_FINAL_TEST`, `RUN_DIRTY_TEST`는 저장되는 노트북에서 모두 `False`.

## Review Focus
- **1·2차 판정이 바뀜**: 채택 지표를 바꾸면 M1~M9 재판정 위험 → 3차는 별도 사다리(`LADDER_N`, `ladder_n.csv`)와 `run_ladder(..., key="val_avg4", best_init=...)`로 분리. Task 4 테스트가 `ladder_m.csv` 재계산 결과가 그대로인지 확인.
- **M8을 다시 학습함**: 출발점 M8에 `val_dirty`가 없으면 비교가 안 된다 → `rescore_experiment`가 저장된 `best.pth`로 평가만 해서 지표를 채운다(학습 없음). Task 4 테스트가 학습 함수가 호출되지 않음을 확인.
- **손상 검증셋이 test_dirty와 같은 사진을 씀**: `val_dirty`는 val_hard 원본 목록만 쓴다 → Task 4 테스트가 test_dirty·TEST 경로와 겹침 0을 확인.
- **강건 증강이 이미지 크기·형식을 깨뜨림**: 증강 뒤 텐서 크기 `(3, img_size, img_size)`와 값 범위 → Task 3 테스트.
- **출처 균형 가중치가 클래스 비율까지 바꿈**: `source_cap`은 클래스 **안**의 출처 비율만 조정해야 한다 → Task 5 테스트가 클래스별 기대 합이 원래 장수와 같음을 확인.

---

### Task 1: 편향 점검 표와 글자 없는 종이 목록 고정

**Files:**
- Modify: `01_coding/notebooks/final_model.ipynb` — 3절 정리표 셀 다음에 markdown + code 셀 `bias-audit`(태그 `data`)
- Create: `01_coding/RESULT/data_split/paper_plain_val.csv` (열 `path,set`) — 이 대화에서 사람이 고른 글자 없는 종이 66장(val 37 + val_ext 29). 원본 분류표: 스크래치 `paper_probe/paper_preds_labeled.csv`의 `kind == "글자 거의 없음"` 행
- Test: `01_coding/tests/test_bias_audit.py`

**Interfaces:**
- Produces: `source_share(df: pd.DataFrame) -> pd.DataFrame` (행 = 클래스, 열 = 출처, 값 = 클래스 안 비율, 행 합 1.0); 전역 `PAPER_PLAIN: set[str]` (paper_plain_val.csv의 path)

- [ ] **Step 1: 테스트** — `test_source_share_rows_sum_to_one`(학습 데이터로 계산한 표의 행 합이 1.0 ± 1e-9, battery의 aihub 비율이 0.75 ± 0.01), `test_paper_plain_list`(66행, 모두 val 또는 val_ext의 paper 경로, TEST·test_dirty 경로와 겹침 0)
- [ ] **Step 2: 실행해 FAIL 확인** — `cd 01_coding/tests && python -m pytest test_bias_audit.py -v`
- [ ] **Step 3: 구현** — `source_share`와 셀 출력: 출처 비율 표, 한 출처가 50%를 넘는 클래스 목록, `PAPER_PLAIN` 로드. markdown에 "방향" 절의 편향 근거를 표로 적는다
- [ ] **Step 4: 통과 확인** 후 전체 테스트 한 번
- [ ] **Step 5: 커밋** — `bias audit table and fixed list of plain-paper validation photos`

### Task 2: 글자 없는 종이 보강 데이터 (사용자 승인 게이트 포함)

**Files:**
- Create: `01_coding/tools/build_extra_paper.py` (기록용), `01_coding/RESULT/data_split/extra_paper.csv`(열 `path,label,source,origin`), `drive_upload/extra_paper.zip`
- Modify: `README.md` 데이터 출처 표
- Test: `01_coding/tests/test_extra_paper.py`

**Interfaces:**
- Produces: `extra_paper.csv` — Task 6의 `train_data="mix3"`가 `train_mix`에 이어 붙인다

- [ ] **Step 1: 출처 조사** — 공개 데이터셋에서 "흰 종이·구긴 종이·A4 묶음·노트·봉투·영수증" 단일 물체 사진을 찾는다(예: TACO의 Normal paper, Kaggle·Hugging Face의 crumpled paper/office paper 데이터). 출처마다 URL·라이선스·장수·배경을 표로 `docs/superpowers/plans/2026-11-22-paper-sources.md`에 적는다
- [ ] **Step 2: 사용자 승인 대기(게이트)** — 표를 사용자에게 보여 주고 쓸 출처를 승인받는다. 승인 없이 다음 단계로 가지 않는다. 승인된 출처가 없으면 Task 2를 "데이터 없음"으로 끝내고 Task 6에서 N3를 뺀다
- [ ] **Step 3: 테스트** — `test_extra_paper_clean`: 모든 행 `label == "paper"`; 400 ≤ 행 수 ≤ 1,000; 학습·val·val_ext·val_hard·TEST·test_ext·test_dirty origin과 경로 겹침 0; 사람이 본 검토 결과 파일(`tools/extra_paper_review.json`)에서 뺀 사진이 없음
- [ ] **Step 4: 도구 구현·실행** — 내려받기 → 짧은 변 384px 저장 → phash 중복 제거(Global Constraints의 대상 전부) → 검토 시트 → 사람이 검토(종이컵·종이봉투·상자·글자 인쇄물은 제외) → 확정·zip. 목표 600장, 배경은 한 출처가 60%를 넘지 않게
- [ ] **Step 5: 통과 확인, README 출처 표 갱신, 커밋** — `extra plain-paper data for training (reviewed)`

### Task 3: 강건 증강 `robust`

**Files:**
- Modify: 노트북 6절 증강 셀(`build_transforms`가 있는 셀, 태그 `cache`)
- Test: `01_coding/tests/test_robust_aug.py`

**Interfaces:**
- Produces: `build_transforms(img_size, aug)`가 `"robust"`를 받으면 학습 변환에 `RandomApply([RandomChoice([GaussianNoise, JpegCompress, Glare, Stain, LowLightNoise])], p=0.5)`를 `dark_blur` 뒤, `ToTensor` 앞에 넣는다. 각 클래스는 PIL 이미지를 받아 같은 크기의 PIL 이미지를 돌려준다: `GaussianNoise(sigma=(5, 25))`, `JpegCompress(quality=(8, 40))`, `Glare(strength=(80, 200))`(밝은 원형 반점, 위치·반경 무작위), `Stain(n=(1, 4), alpha=(100, 180))`(어두운 반투명 타원), `LowLightNoise(scale=(0.25, 0.5), sigma=(5, 12))`

- [ ] **Step 1: 테스트** — `test_robust_keeps_shape`: 128px 무작위 이미지 20장에 `build_transforms(128, ["robust"])[0]` 적용 → 모두 `(3, 128, 128)` float 텐서; `test_each_robust_op_changes_image`: 다섯 클래스 각각 고정 이미지에 적용하면 원본과 평균 절대 차이 > 1, 크기 동일; `test_eval_transform_unchanged`: 평가 변환에는 `robust`가 들어가지 않음
- [ ] **Step 2: FAIL 확인**
- [ ] **Step 3: 구현** — 노트북 셀에 다섯 클래스(`torch`·`numpy`·`PIL`만 사용, 무작위는 `random` 모듈로 학습 seed를 따름)와 `build_transforms` 분기. 셀 끝 `print(build_transforms(...))` 예시에 `robust`를 추가
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `robust augmentation (noise, jpeg, glare, stain, low light)`

### Task 4: 손상 검증셋 `val_dirty`, 지표 `val_avg4`·`paper_plain`, 3차 사다리 실행기

**Files:**
- Modify: 노트북 — 4절 다음에 `val_dirty` 생성 셀(태그 `data`), 8절 `run_experiment`의 평가·지표 부분, 9절 `adopt`·`run_ladder`
- Create: `drive_upload/val_dirty_generated.zip`(이미지 + `01_coding/RESULT/data_split/val_dirty.csv`); `DATA_ZIPS`에 추가
- Test: `01_coding/tests/test_val_dirty.py`

**Interfaces:**
- Consumes: Task 1의 `PAPER_PLAIN`
- Produces:
  - `corrupt(img: Image.Image, kind: str, strong: bool, rng: np.random.Generator) -> Image.Image` — `tools/build_test_dirty.py`의 `corrupt`와 같은 정의(도구는 기록용으로 그대로 둔다)
  - `build_val_dirty(src: pd.DataFrame, out_root: Path, seed: int = 7) -> pd.DataFrame` — 열 `path,label,source,corruption,severity`; 원본 1,198장 모두 손상(종류는 돌아가며, 세기는 종류마다 번갈아); 있으면 다시 만들지 않음; 전역 `val_dirty`
  - `run_experiment`의 metrics에 `val_dirty`, `val_avg4`, `paper_plain`(PAPER_PLAIN 사진만의 정확도) 추가. 기존 키는 그대로
  - `rescore_experiment(exp_id: str, exp_root: Path) -> dict` — 저장된 `best.pth`·`cfg.json`으로 평가만 해서 빠진 키를 `metrics.json`에 채움(학습 없음)
  - `adopt(cand, best, key: str = "val_avg") -> bool`; `run_ladder(ladder, exp_root, stop_before=None, base_cfg=BASE_CFG, csv_name="ladder.csv", key="val_avg", best_init: tuple | None = None)` — `best_init=(id, cfg, metrics)`이면 그 실험을 처음 최고로 두고 시작

- [ ] **Step 1: 테스트**
  - `test_val_dirty_built_from_val_hard_sources`: 1,198행, `corruption != "none"` 100%, 각 종류 ≥ 140행, 원본 경로가 val_hard 원본 목록과 같고 TEST·test_ext·test_dirty 경로와 겹침 0
  - `test_adopt_key`: `adopt({"val_avg4": .80}, {"val_avg4": .79}, key="val_avg4") is True`, `.795`면 False
  - `test_ladder_m_unchanged`: 기존 `ladder_m.csv`를 `run_ladder(LADDER_M, ..., key="val_avg")`로 다시 계산(저장된 결과만 읽음)했을 때 `adopted`·`best_after` 열이 그대로
  - `test_rescore_does_not_train`: `train_one_epoch`를 예외를 던지는 함수로 바꾼 뒤 `rescore_experiment`를 작은 가짜 실험 폴더(테스트용 작은 모델 가중치)에 실행 → 예외 없이 `val_dirty` 키가 생김
- [ ] **Step 2: FAIL 확인**
- [ ] **Step 3: 구현** — 위 Interfaces 그대로. markdown에 `val_dirty`·`val_avg4`·`paper_plain` 뜻과 "test_dirty와 손상 종류는 같지만 사진은 검증 사진"임을 적는다
- [ ] **Step 4: 로컬 실행** — `val_dirty` 생성(CPU), `rescore_experiment("M8", EXP_ROOT)`로 M8의 `val_dirty`·`val_avg4`·`paper_plain`을 채움. 기대: `paper_plain` ≈ 0.864(이 대화의 수동 점검과 같은 값)
- [ ] **Step 5: 통과 확인, zip 생성, 커밋** — `damaged validation set, val_avg4 selection metric, rescoring without training`

### Task 5: 클래스 안 출처 균형 샘플링 `source_cap`

**Files:**
- Modify: 노트북 8절 `run_experiment`의 학습 로더 생성, 6절 `make_loader`
- Test: `01_coding/tests/test_source_cap.py`

**Interfaces:**
- Produces: `source_weights(df: pd.DataFrame, cap: float) -> np.ndarray` — 클래스마다 한 출처의 기대 비율이 `cap`을 넘지 않도록 사진별 가중치를 정하고, **클래스별 가중치 합은 그 클래스의 원래 장수와 같게** 맞춘다. 출처가 하나뿐인 클래스는 가중치 1. `make_loader(df, tf, batch, shuffle, weights=None)` — weights가 있으면 `WeightedRandomSampler(weights, num_samples=len(df), replacement=True)`. cfg 키 `source_cap`(기본 `None` = 기존과 같음)

- [ ] **Step 1: 테스트** — `test_source_cap_limits_share`: 학습 데이터에 `cap=0.5` → 클래스마다 출처별 가중치 합 / 클래스 합 ≤ 0.5 + 1e-6(출처가 2개 이상인 클래스); `test_class_totals_preserved`: 클래스별 가중치 합 == 원래 장수(±1e-6); `test_cap_none_keeps_loader`: `source_cap` 없으면 `make_loader`가 `shuffle=True`로 동작(기존과 같음)
- [ ] **Step 2: FAIL 확인**
- [ ] **Step 3: 구현** — 한 출처가 cap을 넘으면 그 출처 사진의 가중치를 줄이고 나머지 출처에 늘려 합을 맞춘다(출처 2개 이상일 때만). 넘는 출처가 없으면 가중치 1
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `source-balanced sampling within each class`

### Task 6: 3차 사다리 N1~N3 정의·실행·정리

**Files:**
- Modify: 노트북 9절(`LADDER_N`, 실행 셀), 10절(3차 표·곡선), `load_train_df`(`mix3`), `DATA_ZIPS`; `README.md`; `00_lead/decisions.md`(D40)
- Test: `01_coding/tests/test_ladder_n.py`

**Interfaces:**
- Consumes: Task 3 `robust`, Task 4 `run_ladder(..., key="val_avg4", best_init=("M8", M8_cfg, M8_metrics))`, Task 5 `source_cap`, Task 2 `extra_paper.csv`
- Produces: `LADDER_N = [N1 {"aug_add": ["robust"]}, N2 {"source_cap": 0.5}, N3 {"train_data": "mix3"}]`(Task 2에서 데이터가 없으면 N3 없음); `experiments/N*/`, `experiments/ladder_n.csv`

- [ ] **Step 1: 테스트** — `test_ladder_n_definition`: id 순서 `N1, N2, N3`, 각 change가 위와 같음; `test_mix3_adds_extra_paper`: `load_train_df({"train_data": "mix3", ...})` 행 수 = 27,050 + extra_paper 행 수, 추가분은 모두 paper; `test_run_n_uses_val_avg4_from_m8`: 가짜 결과 폴더로 `run_ladder(LADDER_N[:1], ..., key="val_avg4", best_init=...)` 실행 시 첫 비교 상대가 M8
- [ ] **Step 2: FAIL 확인 → 구현 → 통과** — 실행 셀: `RUN_EXPERIMENTS`가 True면 M8 rescoring 확인 후 3차 사다리 실행. `RESUME_FROM`은 `LADDER_N`에도 동작
- [ ] **Step 3: 커밋 후 Colab 실행 안내** — 사용자가 Drive에 `val_dirty_generated.zip`·`extra_paper.zip`을 올리고, T4 GPU에서 `RUN_EXPERIMENTS=True`로 N1~N3 실행(각 약 1.7시간). 사용자 작업
- [ ] **Step 4: 결과 회수·분석** — `ladder_n.csv`와 N*/metrics.json을 Drive에서 받아 `RESULT/experiments/`에 둔다. 단계별 val·val_ext·val_hard·val_dirty·val_avg4·paper_plain·AI Hub 금속 배경 지표·출처별 val_ext 표와 해석을 10절 끝 markdown에 쓴다
- [ ] **Step 5: 노트북 재실행·테스트·문서(README 현재 상태, D40)·커밋** — `third ladder N1-N3 results`

### Task 7: 최종 모델 평가 (사용자 승인 게이트)

**Files:**
- Modify: 노트북 12·12-2·13절, `README.md`, `00_lead/decisions.md`(D41)

- [ ] **Step 1: 게이트** — 3차 최고가 M8이면 이 Task는 "최종 모델 그대로"를 기록하고 끝. 다른 실험이 최고면, TEST를 **두 번째로** 평가하는 것과 test_dirty v2를 다시 평가하는 것을 사용자에게 설명하고 승인받는다(보고서에 "첫 TEST를 본 뒤 모델을 바꿨다"는 경위를 적어야 함)
- [ ] **Step 2: 승인 시 평가** — 이전 `final_test/` 결과를 `final_test/M8/`로 옮겨 보존한 뒤 새 최종 모델로 TEST·test_dirty를 각 1회 평가
- [ ] **Step 3: 13절·README·D41 갱신, 노트북 재실행, 테스트, 커밋** — `final model evaluation after third ladder`
