# 쓰레기 이미지 분류 해커톤 (12 classes)

> 전체 현황판 — **팀장만 수정**. 마지막 갱신: 2026-11-19(목)

## 현재 상태 (11/19 재연구 중)
- 대회 규칙상 사전학습 모델 미세조정이 허용되지 않아, ImageNet 사전학습 ResNet18로 진행한 이전 연구를 **전부 폐기**했다 (`00_lead/decisions.md` D34).
- 작은 ResNet을 `torch.nn`으로 직접 구현하고 Colab GPU에서 **처음부터 학습**하는 재연구를 진행 중이다. 계획: [`docs/superpowers/plans/2026-11-19-scratch-cnn-research.md`](docs/superpowers/plans/2026-11-19-scratch-cnn-research.md)
- 데이터 분할·외부 데이터 통합·검증셋 구성(아래)은 그대로 쓴다.
- 최종 노트북: [`01_coding/notebooks/final_model.ipynb`](01_coding/notebooks/final_model.ipynb) (데이터 준비 → 모델 → 학습 → 튜닝 과정 → 최종 평가)
- 데이터: Google Drive 공개 폴더 https://drive.google.com/drive/folders/19R4yPCNI2B204UuINfGlBx4IZ7N2j9XB (AI Hub 폴더는 약관상 비공개)

## 데이터 출처
학습·평가에 쓴 이미지는 아래 4개 공개 데이터셋에서 가져왔다. **이미지 파일은 저장소에 올리지 않는다** (라이선스·용량). 저장소에는 파일 경로 목록 CSV(`01_coding/RESULT/data_split/`)만 있다.

| 데이터셋 | 출처 | 라이선스·이용 조건 | 사용 범위 | 사용 장수 |
|---|---|---|---|---|
| **Garbage Classification (12 classes)** — 원본 | Kaggle `mostafaabla/garbage-classification` (https://www.kaggle.com/datasets/mostafaabla/garbage-classification) | Kaggle 데이터셋 페이지의 라이선스를 따름 (보고서 제출 전 페이지에서 재확인) | 12클래스 전체. 고정 분할 70/15/15 (seed 42) | 전체 15,515 → train 10,860 / val 2,327 / test 2,328. 학습에는 6,396장 사용 |
| **Garbage Classification V2** | Kaggle, 작성자 sumn2u (`original/` 폴더 사용) | Kaggle 데이터셋 페이지의 라이선스를 따름 (재확인) | cardboard·metal·paper·plastic·trash·biological만. 원본과 md5·phash 중복(약 70%) 제거 후 사용. glass는 색 라벨이 없어 제외 | train 1,360 / meta_train 188 |
| **RealWaste** | UCI Machine Learning Repository (id 908). 인용: Single et al., *RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning*, Information 14(12):633, 2023 (https://www.mdpi.com/2078-2489/14/12/633) | **CC BY-NC-SA 4.0** (데이터셋 README 원문) — 비상업·출처 표기·동일 조건 | Cardboard·Metal·Paper·Plastic → 동명, Food Organics → biological, Misc Trash → trash. Glass·Vegetation·Textile 제외 | train 1,959 / val_ext 275 / test_ext 180 / meta_train 115 |
| **AI Hub 「재활용품 분류 및 선별 데이터」** | AI Hub (aihub.or.kr), 직접촬영 원천 데이터(TS_2/VS_2) | AI Hub 이용약관 (재배포 금지 → 이미지 미포함) | 유리병 갈색·녹색·투명(→ white-glass), 건전지(→ battery), 철캔(→ metal). 영상 추출 이미지는 여러 물체가 섞여 제외 | train 2,285 / val_ext 225 / test_ext 180 / meta_train 323 |

- 검토 후 **사용하지 않은** 데이터: clothing-dataset-small (alexeygrigorev, CC0) — Kaggle clothes·shoes와 99.3% 중복이라 제외
- 중복 제거: 외부 이미지는 perceptual hash(hamming ≤ 4)로 Kaggle 전체(train·val·test) 및 외부끼리 비교해 중복·라벨 충돌을 제거했다 (`01_coding/RESULT/data_split/dedup_report.csv`)
- 파생 평가셋: **val_hard** 1,200장 = Kaggle val 700장 + val_ext 500장에 흐림·어둡게·빛번짐을 적용해 만든 고정 셋 (새 출처 아님)

| 셋 | 장수 | 구성 | 용도 |
|---|---|---|---|
| train_mix | 12,000 | Kaggle 6,396 · AI Hub 2,285 · RealWaste 1,959 · V2 1,360 (클래스당 1,000) | 학습 |
| val (Kaggle) | 2,327 | Kaggle 고정 분할 | 조기 종료·모델 선택 |
| val_ext | 500 | RealWaste 275 · AI Hub 225 (10클래스, clothes·shoes 없음) | 모델 선택 (외부 사진) |
| val_hard | 1,200 | val·val_ext 손상 버전 | 모델 선택 (어려운 사진) |
| meta_train | 1,200 | Kaggle 574 · AI Hub 323 · V2 188 · RealWaste 115 | 스태킹 메타 학습 (미채택) |
| test (Kaggle) | 2,328 | Kaggle 고정 분할 | **최종 1회** |
| test_ext | 360 | AI Hub 180 · RealWaste 180 (9클래스) | **최종 1회** |

## 규칙
- PyTorch만 사용 (torchvision은 이미지 변환 transforms만; 사전학습 모델·`torchvision.models`·YOLO·timm 등 금지) · 고정 분할 · 한 번에 한 변수 변경 · 최종 TEST 1회 (`final_test.py`가 재실행 차단)
- 로컬 CPU 학습 (GPU 없음)

## 폴더
- `00_lead/` 팀장: decisions, plan, prompt_log, protocol
- `01_coding/notebooks/` 코드: `final_model.ipynb`(최종 노트북: 전처리~학습~평가), `gc_lib.py`(데이터셋·모델·학습 루프), `run_exp.py`·`overnight*.py`(실험 실행), `build_*.py`(데이터 구성), `final_test.py`(TEST 1회), `ensemble_compare.py`·`stacking.py`(대안 비교)
- `01_coding/RESULT/` 결과: 아래 하위 폴더 ([`RESULT/README.md`](01_coding/RESULT/README.md))
  - `data_split/` 분할·데이터 목록 CSV · `final_model/` 최종 모델 설정·학습 기록 · `experiments/` 실험표·epoch 기록 · `final_test/` 최종 TEST · `ensemble/` 대안 비교 · `figures/` 그림 · `logs/` 학습 로그
- `02_research/` 리서치·보고서
- 저장소 제외: 원본·외부 이미지(`archive/`, `external_data/` 등), 체크포인트(`*.pth`)

## 일정
| 날짜 | 마감 | 상태 |
|---|---|---|
| 11/16(월) | GitHub 주소 등록 | ✅ |
| 11/19(목) | **최종 제출** (.ipynb + GitHub + Word 보고서) | ⏳ Word 보고서 |
