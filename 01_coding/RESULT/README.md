# RESULT — 코드 외 산출물

| 폴더 | 내용 |
|---|---|
| `data_split/` | 고정 분할·데이터 목록 CSV: `split.csv`(Kaggle 70/15/15), `train_mix.csv`(학습 12,000), `val_ext.csv`, `val_hard.csv`, `test_ext.csv`, `class_counts.csv`, `mix_counts.csv`, `dedup_report.csv` — 이미지 경로·라벨·출처 (이미지 자체는 미포함, Drive에 있음) |
| `experiments/` | 재연구 실험 사다리 결과: `ladder.csv`, `R*/history.csv`, `R*/cfg.json`, `R*/metrics.json` (Colab 실행 후 Drive에서 회수) |
| `final_test/` | 최종 모델 TEST 1회 결과 |
| `figures/` | 그림 |

이전 사전학습 ResNet18 연구 결과는 규칙 위반으로 삭제했다 (decisions D34, git 기록에는 남아 있음).
