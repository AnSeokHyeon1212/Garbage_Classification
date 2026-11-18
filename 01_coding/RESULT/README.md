# RESULT — 코드 외 산출물

모든 경로는 `01_coding/notebooks/gc_lib.py`의 상수(`DATA_DIR`, `EXP_DIR`, `FINAL_DIR`, `ENS_DIR`, `LOG_DIR`, `FIGURES`)로 참조한다.

| 폴더 | 내용 | 만든 코드 |
|---|---|---|
| `data_split/` | `split.csv`(Kaggle 고정 분할), `train_mix.csv`(학습 12,000), `val_ext.csv`, `val_hard.csv`, `test_ext.csv`, `meta_train.csv`, `class_counts.csv`, `mix_counts.csv`, `dedup_report.csv` — 이미지 경로·라벨·출처 목록 (이미지 자체는 미포함) | `01_data.py`, `build_mix.py`, `build_val_hard.py`, `build_holdouts.py` |
| `experiments/` | `experiments.csv`(전 실험 설정·점수), `history/`(epoch별 기록 `E*.csv`, 설정 `E*_cfg.json`, 클래스별 정확도, val_hard 평가 json), `overnight_state.json`, `stacking_results.json`, `thread_bench.json`, `cpu_timing.json` | `run_exp.py`, `overnight*.py`, `eval_val_hard.py`, `stacking.py` |
| `final_test/` | **최종 TEST 1회 결과 (E9)**: `final_test.json`, Kaggle test·test_ext 혼동행렬 CSV | `final_test.py` |
| `ensemble/` | E9 단일 / E9+E8 확률 평균 / 스태킹 비교: 이미지별 예측, 혼동행렬 9개, 오답 수·McNemar | `ensemble_compare.py` |
| `figures/` | 학습 곡선(E1·E2·E3·E9), 전체 실험 비교, 클래스 분포, 대안 혼동행렬 | 각 스크립트 |
| `logs/` | 학습·야간 실험 로그 | — |
| `overnight_summary.md` | 야간 실험(E4~E9) 요약과 해석 | — |
