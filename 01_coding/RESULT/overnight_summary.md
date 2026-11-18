# 야간 실험 요약 (2026-11-19, 01:59–06:40)

## 결론
- **최종 모델: E9** — ResNet18(ImageNet), layer4 + MLP 헤드(512→256→128→12, ReLU) 학습, 나머지 백본 동결, 입력 224px, degrade 증강(어둡게 0.3~1.0 / 흐림 σ0.5~2.0, p=0.5), train_mix 12,000장
- **최종 TEST (1회, 06:37):** Kaggle test **0.9394** (n=2,328) · 외부 test_ext **0.9083** (n=360; AI Hub 0.967, RealWaste 0.850)
- 스태킹 미채택 (S3 +1%p 미달) → 단일 모델

## 실험 비교 (S3 = (val + val_ext + val_hard)/3)
| 실험 | 바꾼 변수 (base) | val | val_ext | val_hard | S3 | best/실행 ep | 분 | 판정 |
|---|---|---|---|---|---|---|---|---|
| E1 | baseline, Kaggle만 | 0.932 | 0.388 | 0.613 | 0.644 | 6/9 | 36.6 | — |
| E2 | train_data → mix (E1) | 0.896 | 0.724 | 0.689 | 0.770 | 3/6 | 33.2 | 채택 |
| E3 | aug → degrade (E2) | 0.893 | 0.744 | 0.715 | 0.784 | 5/8 | 43.0 | 채택 |
| E4 | 은닉 1층 512→256 (E3) | 0.912 | 0.784 | 0.763 | 0.820 | 6/9 | 34.8 | 채택 |
| E5 | 은닉 2층 512→256→128 (E3) | 0.909 | 0.820 | 0.768 | 0.832 | 8/10 | 37.7 | 채택 |
| E6 | GELU (E5) | 0.916 | 0.790 | 0.762 | 0.823 | 7/10 | 37.7 | 유지 |
| E7 | degrade_strong 밝기 0.15~1.0 (E5) | 0.914 | 0.796 | 0.767 | 0.826 | 8/10 | 37.9 | 유지 (val_ext 하락) |
| E8 | layer4 동결 해제 (E5) | 0.927 | 0.854 | 0.829 | 0.870 | 6/9 | 43.6 | 채택 |
| **E9** | **224px (E8)** | **0.949** | **0.890** | **0.834** | **0.891** | 10/10 | 73.5 | **채택** ((val+val_ext)/2 0.890→0.920) |

val_hard 손상별 (E3 → E5 → E8 → E9): 흐림 0.658→0.719→0.792→0.780 (n=395), 어둡게 0.691→0.738→0.824→0.828 (n=408), 빛번짐 0.796→0.849→0.872→0.894 (n=397)

## 앙상블 (meta_train 1,200장으로 학습, VAL·TEST 미사용)
| 방식 | val | val_ext | val_hard | S3 |
|---|---|---|---|---|
| 단일 E9 | 0.9493 | 0.890 | 0.8342 | 0.8912 |
| 평균 E9+E8 | 0.9562 | 0.892 | 0.8617 | 0.9033 |
| 스태킹 E9+E8 (Linear 24→12) | 0.9566 | 0.870 | 0.8475 | 0.8914 |

- 파트너 E8 (오답 겹침 0.57). 사전 규칙(스태킹 S3 +1%p)에 따라 미채택.
- 확률 평균은 S3 +1.2%p지만 상승분 대부분이 val_hard(160px, E8에 유리)에서 나옴. (val+val_ext)/2 기준으로는 +0.4%p라 E9 선택 기준과 맞춰 단일 E9 유지.

## 해석
1. 헤드 용량 부족(과소적합)이 E3의 주된 한계였음 — 은닉층 추가로 S3 +4.8%p (E3 train 0.836 < val 0.893)
2. layer4 동결 해제가 손상 사진에 가장 효과적 (어둡게 +8.6%p, 흐림 +7.3%p, plastic val_hard 0.50→0.83)
3. 224px는 깨끗한·외부 사진에서 추가 개선 (metal 0.80→0.91, brown-glass 0.88→0.97)
4. GELU·더 강한 어둡게 증강은 val_ext를 낮춰 기각
5. 남은 약점: white-glass (TEST 0.759), cardboard (0.888), test_ext의 plastic 0.775·trash·cardboard 0.825, RealWaste 출처 0.850

## 한계
- val_ext에 clothes·shoes 없음, test_ext에 biological·clothes·shoes 없음
- 작은 물체·넓은 배경, 오염·찌그러짐 조건은 태그가 없어 측정하지 못함 (흐림·어둡게·빛번짐만 측정)
- E9는 10 epoch 상한에서 best → 추가 학습 여지. 실험별 단일 시드(42)라 1~2%p 차이는 노이즈 범위일 수 있음

## 산출물
- 체크포인트 `01_coding/checkpoints/E1~E9.pth`, `STACK.pth` · epoch 기록 `RESULT/experiments/history/E*.csv` · val_hard `history/E*_val_hard.json`
- 그림 `RESULT/figures/overnight_compare.png`, `RESULT/figures/E9_curve.png` (+기존 E1~E3 곡선)
- 스태킹 `RESULT/experiments/stacking_results.json` · 최종 `RESULT/final_test/final_test.json`, `final_test_confusion.csv`, `final_test_ext_confusion.csv`
- 제출 노트북 `notebooks/04_final_analysis.ipynb` (저장된 결과만 읽어 실행, 출력 포함)
- 로그 `RESULT/logs/overnight_log.txt`

## 남은 제출 작업
- 04_final_analysis.ipynb로 제출본 확정 (결정 A: E9 단일)
- Word 보고서, GitHub 커밋 (외부 이미지·체크포인트 업로드 금지 확인)
- decisions.md·01_coding README에 val_hard·S3 규칙·E4~E9 결과 반영
- TEST는 사용 완료 — 다시 평가하지 말 것 (`final_test.py`가 차단)
