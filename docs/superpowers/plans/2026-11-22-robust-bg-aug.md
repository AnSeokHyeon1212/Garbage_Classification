# 노이즈 강건 증강 + 배경 교란 증강으로 M8 재학습 (N1·N2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 새 데이터 없이 학습 증강만 바꿔 M8을 다시 학습(N1: 손상 증강, N2: N1 + 배경 교란)하고, 손상 검증셋·배경 의존도로 M8과 비교해 최종 모델을 정한다. 마감 2026-11-23.

**Architecture:** 노트북 `01_coding/notebooks/final_model.ipynb` 안에서 (1) 손상 증강 `robust`, (2) 사진마다 추정한 물체 상자 바깥을 바꾸는 배경 교란 `bg_swap`, (3) 손상 검증셋 `val_dirty`·배경 의존도 `bg_drop`·선택 지표 `val_avg4`, (4) M8을 출발점으로 하는 3차 사다리 `LADDER_N`을 더한다. 증강을 걸지 말지·종류는 **모든 사진에 같은 확률**로 정하고, 사진마다 다른 것은 물체 상자(배경 범위)뿐이다. 학습은 Colab GPU, 나머지는 로컬.

**Tech Stack:** PyTorch(사전학습 없음), torchvision transforms, Pillow, numpy, pandas, pytest, nbformat/nbconvert

**Spec:** 아래 "방향" 절 (사용자 지시 2026-11-22). 이전 계획 `2026-11-22-robust-unbiased.md`는 마감 때문에 보류(출처 균형 샘플링·종이 새 데이터는 보고서 "향후 과제"로)

## 방향
- 근거: test_dirty v2에서 원본 0.826 → 손상 0.720. 강한 빛번짐 0.28, 강한 저조도+잡음 0.51, 강한 얼룩 0.60. 한 출처가 절반을 넘는 클래스(battery AI Hub 75%, 유리 57~71% 아스팔트 배경, shoes 흰 배경 제품 54%, clothes Kaggle 95%) → 배경 단서 의존 의심(AI Hub 금속 25장 중 3장을 battery·유리로 예측).
- **교란 규칙(사용자 질문에 대한 결정)**: 교란 확률·종류는 클래스·출처와 무관하게 모든 학습 사진에 같게. 사진마다 다른 것은 "어디가 배경인가"(물체 상자)뿐. 상자는 사진에서 추정하고, 추정이 안 되면 가운데 60% 상자.
- 배경 교란은 다른 무작위 학습 사진의 가장자리로 바꾸기(배경 섞기)를 주로 쓰고, 잡음·흐림·단색 덮기를 함께 쓴다.
- 새 데이터는 쓰지 않는다. TEST·test_dirty는 사다리가 끝난 뒤 사용자 승인 때만 평가한다.

## Global Constraints
- PyTorch만, 사전학습 가중치·`torchvision.models` 금지.
- 노트북 코드 셀은 `# [셀 목적]`으로 시작, 내부 용어 금지, 지표 이름 val / val_ext / val_hard / val_avg; 새 지표 `val_dirty`, `val_avg4`, `bg_drop`은 markdown에서 처음 나올 때 뜻을 적는다. 한국어.
- 채택: `val_avg4 = (val + val_ext + val_hard + val_dirty) / 4`가 현재 최고보다 +0.01 이상. 출발점은 M8(학습 없이 다시 채점). `bg_drop`은 채택에 쓰지 않고 함께 보고.
- 실험 ID `N1`(M8 + `aug_add: ["robust"]`), `N2`(N1 설정 + `aug_add: ["bg_swap"]`, N1이 기각돼도 M8 + robust + bg_swap); 결과 `experiments/ladder_n.csv`. 1·2차 결과 파일은 바꾸지 않는다.
- 손상 종류 `["noise", "jpeg", "motion_blur", "low_light_noise", "glare", "stain", "dark", "blur"]`, 세기 mild/strong — test_dirty와 같은 정의. `val_dirty`는 val_hard 원본 1,198장 전부에 적용, seed `7`.
- 증강 확률: `robust` p=0.5(다섯 종류 중 하나), `bg_swap` p=0.5(배경 섞기 0.6 / 잡음 0.15 / 흐림 0.15 / 단색 0.1). 클래스·출처로 분기하는 코드가 없어야 한다.
- 물체 상자 추정이 안 될 때 기본 상자: 가운데 60%(각 변 20% 여백). 상자는 바깥쪽으로 10% 넓혀서 보호한다.
- `RUN_EXPERIMENTS`, `RUN_FINAL_TEST`, `RUN_DIRTY_TEST`는 저장되는 노트북에서 모두 `False`.

## Review Focus
- **교란이 정답 힌트가 됨**: 클래스·출처별 분기 → Task 2 테스트가 교란 함수 인자에 label·source가 없고, 클래스별 교란 비율이 같음(시뮬레이션 2,000장, 클래스 간 차이 < 0.05)을 확인.
- **물체를 지움**: 상자 추정이 물체보다 작으면 물체가 지워진다 → Task 2 테스트가 흰 배경 위 합성 물체(가장자리에 걸친 경우 포함)에서 물체 픽셀의 95% 이상이 보존됨을 확인.
- **1·2차 판정이 바뀜**: 지표 변경으로 M1~M9 재판정 → Task 3 테스트가 `ladder_m.csv`의 `adopted`·`best_after`가 그대로임을 확인.
- **M8을 다시 학습**: `rescore_experiment`는 평가만 → Task 3 테스트가 학습 함수 호출 없음을 확인.
- **Colab에서 느려짐**: 배경 섞기가 매번 다른 사진을 디스크에서 읽으면 느리다 → 캐시된 192px 이미지에서 읽고, Task 2에서 1 batch(64장) 변환 시간이 로컬 CPU 2초 이하인지 확인.

---

### Task 1: 손상 증강 `robust`

**Files:**
- Modify: 노트북 6절 증강 셀(`build_transforms`가 있는 셀, 태그 `cache`)
- Test: `01_coding/tests/test_robust_aug.py`

**Interfaces:**
- Produces: PIL→PIL 변환 클래스 `GaussianNoise(sigma=(5, 25))`, `JpegCompress(quality=(8, 40))`, `Glare(strength=(80, 200))`(밝은 원형 반점, 위치·반경 무작위), `Stain(n=(1, 4), alpha=(100, 180))`(어두운 반투명 타원), `LowLightNoise(scale=(0.25, 0.5), sigma=(5, 12))`. `build_transforms(img_size, aug)`가 `"robust"`를 받으면 `RandomApply([RandomChoice([...5개])], p=0.5)`를 `dark_blur` 뒤·`ToTensor` 앞에 넣는다. 무작위는 `random` 모듈(학습 seed를 따름)

- [ ] **Step 1: 테스트** — `test_robust_keeps_shape`(128px 무작위 이미지 20장 → 모두 `(3, 128, 128)` float 텐서), `test_each_robust_op_changes_image`(다섯 클래스 각각 원본과 평균 절대 차이 > 1, 크기 동일), `test_eval_transform_unchanged`(평가 변환에 robust 없음)
- [ ] **Step 2: FAIL 확인** — `cd 01_coding/tests && python -m pytest test_robust_aug.py -v`
- [ ] **Step 3: 구현** — 셀 끝 예시 `print(build_transforms(...))`에 `robust` 추가
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `robust augmentation: noise, jpeg, glare, stain, low light`

### Task 2: 물체 상자 추정과 배경 교란 `bg_swap`

**Files:**
- Modify: 노트북 6절 — 캐시 셀(`cache_resized`)과 Dataset 셀(`GarbageDataset`, `make_loader`), 증강 markdown
- Test: `01_coding/tests/test_bg_swap.py`

**Interfaces:**
- Produces:
  - `object_box(img: Image.Image) -> tuple[float, float, float, float]` — 0~1 비율 `(x0, y0, x1, y1)`. 사진 테두리 띠(각 변 8%)의 색 중앙값을 배경색으로 보고, 배경색과 차이가 큰 픽셀(가우시안 흐림 후 RGB 거리 > 40)의 범위를 상자로. 그 픽셀 비율이 0.5% 미만이거나 85% 초과면 추정 실패 → 기본 상자 `(0.2, 0.2, 0.8, 0.8)`. 결과 상자를 각 변 10% 넓혀 0~1로 자른다
  - `cache_resized`가 캐시 만들 때 사진마다 상자를 계산해 `box` 열(문자열 `"x0,y0,x1,y1"`)을 붙이고 캐시 폴더의 `boxes.csv`에 저장(있으면 읽기만)
  - `bg_replace(img: Image.Image, box, donor: Image.Image | None, mode: str) -> Image.Image` — 상자 바깥만 `mode ∈ {"swap", "noise", "blur", "solid"}`로 바꾸고 경계는 8px 페더로 섞음. `"swap"`은 donor(같은 크기로 맞춘 다른 학습 사진)의 같은 위치 픽셀
  - `GarbageDataset(df, tf, bg_swap: bool = False)` — 켜져 있으면 `__getitem__`에서 p=0.5로 mode를 고르고(swap 0.6 / noise 0.15 / blur 0.15 / solid 0.1), swap이면 `df`에서 무작위 다른 행의 캐시 이미지를 donor로. 이 단계는 다른 증강보다 **먼저**. `make_loader(..., bg_swap=False)`로 전달; cfg의 aug에 `"bg_swap"`이 있으면 학습 로더만 켠다
- [ ] **Step 1: 테스트**
  - `test_object_box_finds_object`: 회색 바탕(128,128,128)에 빨간 사각형(0.4~0.6) → 상자가 그 사각형을 포함하고 넓이 < 0.25
  - `test_object_box_fallback`: 전체가 잡음인 이미지 → `(0.2, 0.2, 0.8, 0.8)`을 10% 넓힌 상자
  - `test_bg_replace_keeps_object`: 흰 바탕 위 물체(가장자리에 걸친 경우 포함) → 네 mode 모두 물체 픽셀 95% 이상 그대로, 상자 바깥 평균 절대 차이 > 10
  - `test_bg_swap_is_label_free`: `bg_replace`·`GarbageDataset`의 교란 결정 코드에 `label`·`source` 참조가 없음(소스 문자열 검사) + 12클래스 가짜 df 2,000행으로 교란이 걸린 비율의 클래스 간 최대 차이 < 0.05
  - `test_bg_swap_speed`: 64장 변환 ≤ 2초(로컬 CPU)
- [ ] **Step 2: FAIL 확인**
- [ ] **Step 3: 구현** — 위 Interfaces 그대로. markdown에 "교란은 모든 사진에 같은 확률, 사진마다 다른 것은 배경 범위뿐" 이유를 적는다
- [ ] **Step 4: 통과 확인 + 육안 확인** — 클래스마다 2장씩 `bg_swap` 결과 시트를 스크래치에 만들어 물체가 남는지 본다
- [ ] **Step 5: 커밋** — `per-photo object box and label-free background perturbation (bg_swap)`

### Task 3: 손상 검증셋 `val_dirty`, 배경 의존도 `bg_drop`, 선택 지표 `val_avg4`, 3차 사다리 실행기

**Files:**
- Modify: 노트북 — 4절 다음 `val_dirty` 생성 셀(태그 `data`), 8절 `run_experiment` 지표, 9절 `adopt`·`run_ladder`·`LADDER_N`·실행 셀, `DATA_ZIPS`(`val_dirty_generated.zip`)
- Create: `drive_upload/val_dirty_generated.zip`(이미지 + `01_coding/RESULT/data_split/val_dirty.csv`)
- Test: `01_coding/tests/test_val_dirty.py`

**Interfaces:**
- Consumes: Task 1 `robust`, Task 2 `object_box`·`bg_replace`·`boxes.csv`
- Produces:
  - `corrupt(img, kind, strong, rng) -> Image.Image` (`tools/build_test_dirty.py`의 정의와 같음; 도구는 기록용으로 그대로)
  - `build_val_dirty(src: pd.DataFrame, out_root: Path, seed: int = 7) -> pd.DataFrame` — 열 `path,label,source,corruption,severity`, 1,198장 모두 손상(종류 순환, 세기 종류마다 번갈아), 있으면 다시 만들지 않음; 전역 `val_dirty`
  - metrics 추가 키: `val_dirty`, `val_avg4`, `bg_drop`(= val·val_ext 정확도 − 같은 사진의 상자 바깥을 회색 128로 채웠을 때 정확도), `aihub_metal_as_glass_battery`는 기존대로
  - `rescore_experiment(exp_id: str, exp_root: Path) -> dict` — 저장된 `best.pth`·`cfg.json`으로 평가만, 빠진 키를 `metrics.json`에 채움
  - `adopt(cand, best, key="val_avg")`; `run_ladder(..., key="val_avg", best_init=None)` — `best_init=(id, cfg, metrics)`이면 그것을 첫 최고로
  - `LADDER_N = [{"id": "N1", "change": {"aug_add": ["robust"]}}, {"id": "N2", "change": {"aug_add": ["bg_swap"]}}]`, 실행: `run_ladder(LADDER_N, EXP_ROOT, STOP_BEFORE, M8_cfg, "ladder_n.csv", key="val_avg4", best_init=("M8", M8_cfg, rescore_experiment("M8", EXP_ROOT)))`. N2는 그 시점 최고 설정에 bg_swap을 더한다(run_ladder의 aug_add 동작)
- [ ] **Step 1: 테스트** — `test_val_dirty_sources`(1,198행, 전부 손상, 종류마다 ≥ 140행, 원본 = val_hard 원본 목록, TEST·test_ext·test_dirty 경로와 겹침 0), `test_adopt_key`(.80 vs .79 → True, .795 → False), `test_ladder_m_unchanged`(저장된 M1~M9를 `key="val_avg"`로 다시 판정 → `adopted`·`best_after` 동일), `test_rescore_does_not_train`(`train_one_epoch`를 예외 함수로 바꾸고 작은 가짜 실험 폴더에 rescore → 예외 없이 `val_dirty`·`bg_drop` 키 생성), `test_ladder_n_definition`
- [ ] **Step 2: FAIL 확인**
- [ ] **Step 3: 구현** — markdown에 세 지표 뜻과 "val_dirty는 test_dirty와 손상 종류는 같고 사진은 검증 사진" 한계를 적는다
- [ ] **Step 4: 로컬 실행** — `val_dirty` 생성, `rescore_experiment("M8", EXP_ROOT)` → M8의 `val_dirty`·`val_avg4`·`bg_drop`을 기록. zip 만들기
- [ ] **Step 5: 전체 테스트, 커밋** — `damaged validation set, background dependency, val_avg4, third ladder runner`

### Task 4: Colab 재학습과 결과 정리

**Files:**
- Modify: 노트북 10절(3차 표·곡선·해석), 13절, `README.md`, `00_lead/decisions.md`(D40), `RESULT/experiments/N*`·`ladder_n.csv`(Drive에서 회수)

- [ ] **Step 1: 사용자 실행 안내** — Drive에 `val_dirty_generated.zip` 업로드, 최신 노트북, T4 GPU, `RUN_EXPERIMENTS=True`, `STOP_BEFORE="N2"`로 N1 먼저(약 2시간). GPU가 남으면 `STOP_BEFORE=None`으로 N2. 다른 계정이면 `RESUME_FROM`·결과 폴더 옮기기 안내
- [ ] **Step 2: 결과 회수·분석** — M8·N1·N2의 val / val_ext / val_hard / val_dirty / val_avg4 / bg_drop / AI Hub 금속 배경 지표 / 손상 종류별 / 클래스별 표, train−val 차이(과적합), 채택 결과
- [ ] **Step 3: 문서** — 10절 끝 해석, README 현재 상태, D40(교란 규칙과 이유 포함), 13절 갱신. 노트북 재실행, 전체 테스트, 커밋 `third ladder N1-N2 results`
- [ ] **Step 4: 최종 평가 게이트** — 최고가 M8이면 그대로 제출. N1·N2가 최고면 TEST 두 번째 평가와 test_dirty 재평가를 사용자에게 설명하고 승인받은 뒤, 기존 `final_test/`를 `final_test/M8/`로 옮겨 보존하고 새 모델로 각 1회 평가 → 12·12-2·13절, README, D41, 커밋
