# 더러운 사진 추가 평가셋 v2 (학습과 같은 종류의 물체 + 노이즈) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 학습 데이터와 **같은 종류의 물체**만으로, 어려움은 **더러움·노이즈로만** 주는 추가 평가셋(test_dirty v2)을 다시 만들고 최종 모델 M8로 한 번 평가한다.

**Architecture:** 기존 기록용 도구 `01_coding/tools/build_test_dirty.py`의 후보 출처와 PLAN을 "학습한 종류" 기준으로 바꾸고, 사람이 검토 시트로 고른 결과(`test_dirty_review.json`)로 확정한다. 노트북 12-2절은 같은 파일명(`test_dirty.csv`, `dirty_test_predictions.csv`)을 그대로 읽으므로 코드는 설명만 바뀐다. v1은 git 기록에만 남기고 저장소에서는 v2로 교체한다.

**Tech Stack:** Python, pandas, Pillow, imagehash, PyTorch(평가만), pytest, nbformat/nbconvert

**Spec:** 아래 "방향" 절 (사용자 지시, 2026-11-22). 이전 결정은 `00_lead/decisions.md` D38

## 방향 (사용자 지시, 2026-11-22)
- 실제 시연은 **Kaggle 클래스에 있는 종류의 물체**를 더럽거나 노이즈가 섞인 사진으로 준다. 학습에 없던 완전히 다른 물체는 나오지 않는다.
- v1(D38)은 학습에 없던 종류(잎·풀, 커피 찌꺼기, 옷 더미·매장 진열, 과자 봉지·휴지, 웹 인포그래픽)를 넣어 평가 목적에서 벗어났다 → **폐기하고 v2로 다시 만든다.**
- 이 평가셋의 사진은 **학습에 추가하지 않는다.** 최종 모델 M8은 그대로 둔다(용도 A).
- 구성 목표는 그대로: 1,200장 = 12클래스 × 100장, 클래스마다 Kaggle test 40 + 외부 60. 같은 종류의 외부 사진이 모자란 클래스는 Kaggle test로 채운다.

### "같은 종류"의 기준 (Kaggle 학습 사진을 직접 보고 정함)
| 클래스 | Kaggle·학습 데이터의 물체 | v2에 쓰는 외부 사진 | v2에서 빼는 것 |
|---|---|---|---|
| battery | 원통·각형 건전지 | AI Hub 건전지 | — |
| brown / green / white-glass | 유리병·유리 단지 | AI Hub 유리병(색 확인), 투명 유리 단지 1개짜리 사진(white) | 여러 병이 섞인 사진 |
| metal | 캔·통·스프레이 캔 | AI Hub 철캔 | 웹 사진(캔 더미·인포그래픽) |
| plastic | 병·용기·봉지 | RealWaste 플라스틱, Household 병·용기 1~3개 사진 | 공장·바다·포스터 |
| cardboard | 상자·골판지 | TriCascade 골판지·상자 | 사람이 주인공인 사진 |
| shoes | 신발 한 켤레 | ShoeSandalBoot | 웹 신발 사진(사람·광고) |
| trash | 마스크·칫솔·기저귀 (+ 2차 학습에 넣은 스티로폼 컵·용기) | Household 스티로폼, TrashBusters 중 마스크·칫솔류 | 과자 봉지·휴지·종이 조각·사과 심 |
| biological | 과일·채소·음식 | TriCascade 음식(A_Foods) | 잎·풀(RealWaste Vegetation), 커피 찌꺼기 |
| paper | 신문·잡지·종이 | Household 신문·종이 더미 | 사무실 풍경·광고·인물 |
| clothes | 옷 한 벌(펼침·걸어 둠) | 옷 한 벌만 보이는 사진 | 옷 더미·매장 진열·사람이 입은 사진·인포그래픽 |

## Global Constraints
- 사전학습 모델 사용 금지(평가만 하므로 해당 없음). 최종 모델은 M8 그대로, **이 평가셋으로 모델을 고르거나 학습하지 않는다.**
- 평가셋 사진은 학습(`train_mix_clean.csv` 등)·검증·TEST 목록과 **경로가 겹치지 않고**, 기존 사진과 phash hamming ≤ 4인 사진은 쓰지 않는다 (`HAM = 4`, `SEED = 42`).
- 클래스당 100장, 총 1,200장. 묶음(클래스 × kaggle/external)마다 절반은 원본(`corruption == "none"`), 절반은 손상.
- 손상 종류는 v1과 같다: `["noise", "jpeg", "motion_blur", "low_light_noise", "glare", "stain", "dark", "blur"]`, 세기 `mild`/`strong`.
- 평가는 M8로 **1회만**. `dirty_test_predictions.csv`가 있으면 다시 평가하지 않는다(노트북 `dirty_test_once`의 보호 유지).
- AI Hub 이미지는 재배포 금지 → `drive_upload/test_dirty.zip`은 Drive에 비공개로 둔다(시연 기간만 공개).
- 노트북 코드 셀은 `# [셀 목적]`으로 시작, 내부 용어 금지, 지표 이름은 val / val_ext / val_hard / val_avg 그대로. 답변·문서는 한국어.

## Review Focus
- **v1 결과가 남아 v2 평가가 막힘**: 이전 `dirty_test_predictions.csv`가 있으면 `dirty_test_once`가 중단된다 → Task 3에서 v1 파일을 지운 뒤 평가한다(지운 경위는 D39에 기록).
- **v1에서 걸러낸 사진이 다시 들어옴**: 후보를 다시 만들면 시트 번호(`sheet_id`)가 바뀔 수 있다 → Task 1 테스트가 금지 출처(`Vegetation/`, `coffee_grounds`, `clothing` 웹 시트, `tb_trash` 중 마스크·칫솔 외)가 없는지 `origin`으로 확인한다.
- **외부가 모자란 클래스의 비율**: clothes·paper는 외부가 60장에 못 미친다 → 합계 100장은 Kaggle로 채우고, 실제 Kaggle:외부 비율을 README·D39에 그대로 적는다.
- **AI Hub 같은 물건**: 물건 번호가 없어 같은 병이 학습에 있을 수 있다 → 결과를 출처별로 보고한다(노트북 12-2절 표 유지).
- **trash 정의 차이**: Kaggle trash(마스크·칫솔·기저귀)와 학습에 넣은 스티로폼이 섞인다 → 클래스별 표에서 trash를 출처별로 나눠 적는다.

---

### Task 1: 후보 출처와 PLAN을 "학습한 종류" 기준으로 바꾸기

**Files:**
- Modify: `01_coding/tools/build_test_dirty.py` (`HH_MAP`, `PLAN`, `candidates()`의 보충 출처)
- Create: `01_coding/tests/test_test_dirty.py`

**Interfaces:**
- Consumes: `build_extra_external.py`의 `AIHUB, AI_ZIPS, HAM, ROOT, SEED, SPLIT, min_ham, phash`
- Produces: `candidates()` → `external_data/test_dirty/candidates.csv`, `sheet_index.csv`, `sheets/*.jpg`; `finalize()` → `01_coding/RESULT/data_split/test_dirty.csv`(열 `path,label,source,group,corruption,severity,origin`), `drive_upload/test_dirty.zip`

- [ ] **Step 1: 데이터 테스트 작성** — `test_test_dirty.py` (CSV가 없으면 `pytest.skip`)
  - `test_size_and_balance`: 1,200행, `label`별 100행, 묶음(`label, group`)마다 `corruption == "none"` 비율이 0.5 ± 1행
  - `test_no_overlap_with_used`: `path`·`origin`이 `split.csv`·`train_mix_clean.csv`·`val_ext_clean.csv`·`test_ext_clean.csv`·`val_hard_clean.csv`·`extra_external.csv`의 `path`와 겹치지 않음 (Kaggle 행은 `split == "test"`인 경로만)
  - `test_only_trained_kinds`: 외부 행의 `origin`에 `Vegetation/`, `coffee_grounds`, `/clothing/`(Household 웹 옷), `office_paper` 광고, `TrashBusters/` 중 `trash` 외 라벨이 없음; `source` ∈ {`aihub`, `realwaste`, `tricascade`, `household_real`, `household_default`, `hf_shoes`, `tb_trash`}; `realwaste` 행의 라벨은 `plastic`만
- [ ] **Step 2: 실행 → v1 CSV로 FAIL 확인** — `cd 01_coding/tests && python -m pytest test_test_dirty.py -v` (Vegetation·coffee 때문에 `test_only_trained_kinds` 실패)
- [ ] **Step 3: 도구 수정**
  - `HH_MAP`에서 `coffee_grounds`, `office_paper`, `clothing`(real·default 모두), Household `aluminum_*`/`steel_*`/`aerosol_cans` 제거. 유지: `newspaper`(paper), 플라스틱 4종, 스티로폼 2종, 유리 3종(white 지정용)
  - RealWaste `Vegetation` 보충 블록 삭제
  - `biological` 보충 출처: `external_data/hf_extra/biological/tricascade_*`(학습에 안 쓴 것)
  - `PLAN` = `battery {aihub 60}`, `brown/green-glass {aihub 60}`, `white-glass {aihub 30, household 30}`, `metal {aihub 60}`, `plastic {realwaste 40, household 20}`, `cardboard {tricascade 60}`, `shoes {hf_shoes 60}`, `trash {household 50, tb_trash 10}`, `biological {tricascade 60}`, `paper {household 60}`, `clothes {household 60}` (모자라면 `finalize`가 Kaggle로 채움 — 기존 동작)
  - `candidates()`의 AI Hub 시트를 클래스당 최대 3장(180장)까지 만들도록 유지
- [ ] **Step 4: 후보 다시 만들기** — `cd 01_coding/tools && python -W ignore build_test_dirty.py candidates` → 출력 표에 `realwaste biological`, `coffee`, `clothing`이 없고 `tricascade biological`이 있음
- [ ] **Step 5: 커밋** — `git add 01_coding/tools/build_test_dirty.py 01_coding/tests/test_test_dirty.py && git commit -m "test_dirty v2: limit candidates to object kinds seen in training"`

### Task 2: 검토 시트로 사진 고르고 평가셋 확정

**Files:**
- Modify: `01_coding/tools/test_dirty_review.json` (v2 시트 기준으로 새로 작성, `_설명`에 v2 기준 표의 요약 한 줄)
- Output: `01_coding/RESULT/data_split/test_dirty.csv`, `drive_upload/test_dirty.zip`

**Interfaces:**
- Consumes: Task 1의 `sheets/*.jpg`, `sheet_index.csv`; 시트 번호 = `candidates.csv`의 `sheet_id`
- Produces: Task 3이 읽는 `test_dirty.csv`

- [ ] **Step 1: 시트 검토** — "방향" 표의 기준으로 시트마다 `{"reject": [...]}` 또는 `{"accept": [...]}` 작성. 새로 볼 시트: `tricascade_biological_0~2`, `aihub_metal_1`, `household_real_paper_*`(신문·종이만 accept), `household_real_plastic_*`, `tb_trash_trash_0`(마스크·칫솔·기저귀만 accept), Household 유리(투명 단지 1개짜리만 accept + `household_glass_color`). 시트 번호가 바뀌었으면 v1 검토 번호를 재사용하지 말고 시트를 다시 본다
- [ ] **Step 2: 확정** — `python -W ignore build_test_dirty.py finalize` → 클래스별 `external/kaggle` 표와 손상 표 출력, zip 경로 출력
- [ ] **Step 3: 샘플 시트로 육안 확인** — 클래스마다 외부 원본 2장 + 손상 4장 시트(스크래치)를 만들어 "학습에 없던 종류"가 없는지 확인
- [ ] **Step 4: 데이터 테스트 통과** — `cd 01_coding/tests && python -m pytest test_test_dirty.py -v` → 3 passed
- [ ] **Step 5: 커밋** — `git add 01_coding/tools/test_dirty_review.json 01_coding/RESULT/data_split/test_dirty.csv && git commit -m "test_dirty v2: reviewed selection (trained object kinds only)"`

### Task 3: M8로 v2 한 번 평가하고 기록

**Files:**
- Delete: `01_coding/RESULT/final_test/dirty_test_predictions.csv` (v1 결과, git 기록에 남음)
- Modify: `01_coding/notebooks/final_model.ipynb` (12-2절 markdown의 구성 표·주의 문단, 13절 결론의 "더러운 사진" 문단), `README.md`, `00_lead/decisions.md` (D39 추가)

**Interfaces:**
- Consumes: Task 2의 `test_dirty.csv`; 노트북 `dirty_test_once(model, df, out_dir, img_size, mode)`와 `RUN_DIRTY_TEST` 스위치(변경 없음)
- Produces: v2 `dirty_test_predictions.csv`(열 = `test_dirty.csv` 열 + `pred, ok`)

- [ ] **Step 1: v1 결과 삭제 후 1회 평가** — `git rm 01_coding/RESULT/final_test/dirty_test_predictions.csv`, 그다음 노트북 셀을 `RUN_DIRTY_TEST=True`로 한 번 실행(스크래치 실행기: `nbcells.load([...], RUN_DIRTY_TEST=True)`; `RUN_FINAL_TEST`는 False 유지) → "더러운 사진 1회 평가 완료: 정확도 …"
- [ ] **Step 2: 결과 분석** — 전체, 원본/손상, Kaggle/외부 × 원본/손상, 손상 종류 × 세기, 외부 출처별, 클래스 × 묶음, 주요 오답. 핵심 비교는 **같은 사진 집단의 원본 대비 손상 하락폭**
- [ ] **Step 3: 문서 반영**
  - 12-2절: v2 구성 표(실제 장수), "학습과 같은 종류만, 어려움은 손상으로만" 문장, v1 폐기 한 줄
  - 13절: v1 문단을 v2 결과로 교체
  - README 현재 상태 줄을 v2 결과로 교체
  - D39: v1 폐기 이유(사용자 지시) + v1 결과(0.753)는 평가 목적과 달라 쓰지 않음 + v2 구성·결과
- [ ] **Step 4: 노트북 전체 재실행·검증** — `jupyter nbconvert --to notebook --execute --inplace final_model.ipynb` → 오류 0, 출력에 사용자 경로 없음; `python -m pytest -q` → 모두 통과
- [ ] **Step 5: 커밋** — `git add -A 01_coding/RESULT/final_test 01_coding/notebooks/final_model.ipynb README.md 00_lead/decisions.md && git commit -m "Evaluate M8 once on test_dirty v2 and replace v1 records"`

### 이 계획 밖 (결과를 본 뒤 사용자 결정)
- v2에서 **같은 사진의 손상 하락폭이 크면**(예: 원본 대비 10%p 이상), 추가 데이터 없이 **학습 증강에 손상 종류를 넣는 재학습**을 검토한다. 이때 채택 판단은 test_dirty가 아니라 **val·val_ext 사진에 같은 손상을 입힌 검증셋**으로 하고, test_dirty v2는 마지막에 한 번만 다시 본다.
