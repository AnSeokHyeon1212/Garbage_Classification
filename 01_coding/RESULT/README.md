# RESULT — 코드 외 산출물

| 폴더 | 내용 |
|---|---|
| `data_split/` | 고정 분할·데이터 목록 CSV: `split.csv`(Kaggle 70/15/15), `train_mix.csv`(학습 12,000), `val_ext.csv`, `val_hard.csv`, `test_ext.csv`, `class_counts.csv`, `mix_counts.csv`, `dedup_report.csv` — 이미지 경로·라벨·출처 (이미지 자체는 미포함, Drive에 있음). 1차(정리 전) 분할이다 |
| `data_split/` (2차, 정리본) | `train_mix_clean.csv`(27,050), `val_ext_clean.csv`(498), `test_ext_clean.csv`(360), `val_hard_clean.csv`(1,198), `norm_stats_clean.json`(정리본 정규화 통계), `clean_report.csv`(제외 사유별 장수), `extra_counts.csv`(추가 외부 데이터 출처별·클래스별 장수), `extra_external.csv`(추가 외부 사진 목록) |
| `tools/` 설명 | 위 `01_coding/tools/`에 데이터 정리·추가 수집 스크립트(`build_extra_external.py`, `clean_splits.py`, `glass_check.py`)와 제외 목록(`split_rejects.csv`, `extra_rejects.csv`, `household_selected.json`)이 있다. 기록용이며 AI Hub 원본(196GB)이 있어야 실행된다. 노트북은 만들어진 `*_clean.csv`만 읽는다 |
| `experiments/` | 재연구 실험 사다리 결과: `ladder.csv`(1차 R1~R12, 정리 전 데이터), `ladder_m.csv`(2차 M1~M9, 정리본; `per_class_delta` 포함), `R*/`·`M*/history.csv`, `cfg.json`, `metrics.json` (Colab 실행 후 Drive에서 회수) |
| `final_test/` | 최종 모델 TEST 1회 결과 |
| `figures/` | 그림 |

이전 사전학습 ResNet18 연구 결과는 규칙 위반으로 삭제했다 (decisions D34, git 기록에는 남아 있음).
