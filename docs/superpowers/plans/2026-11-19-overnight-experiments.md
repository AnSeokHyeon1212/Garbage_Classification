# 야간 실험 (11/19) 실행 계획

> 같은 세션에서 바로 실행 (사용자 지시: 짧은 계획 → 즉시 구현·학습·검증). 탐색 마감 07:50, T-06·제출 준비 07:50–09:00.

**목표:** E3(ResNet18 백본 동결, mix 12,000장, 160px, degrade 증강)를 기준으로 헤드 구조·활성화·증강 강도·스태킹을 비교해 최종 모델을 고르고, TEST를 1회 평가한다.

## 고정 규칙
- 데이터 분할 고정: split.csv / train_mix.csv / val_ext.csv / val_hard.csv / test_ext.csv / meta_train.csv 재생성 금지
- 한 번에 한 변수 (base_exp 대비). seed 42, AdamW 1e-3, wd 1e-4, batch 32, 최대 10 epoch, patience 3 (점수 = (val+val_ext)/2)
- 최종 선택 기준 **S3 = (val + val_ext + val_hard)/3**. 채택 = S3 +1%p 이상 & clothes·shoes val −3%p 이내. E7은 val_ext 하락 없음 추가
- TEST·test_ext는 탐색 중 사용 금지, 최종 1회 (`final_test.py`가 가드)
- PNG: 개별 곡선은 E3·최종 모델만, 전체 비교 그림 1장 (`overnight_compare.png`). epoch CSV는 전 실험
- val_ext에 clothes·shoes 없음, test_ext에 biological·clothes·shoes 없음 (외부 사진 부족) — 보고에 명시

## 작업
- [x] T1 헤드 구조 코드: `gc_lib.build_head/model_from_cfg` (hidden=(256,), (256,128), act relu|gelu), `degrade_strong`(밝기 0.15~1.0)
- [x] T2 `overnight.py`: E4(은닉1 ReLU) → E5(은닉2 ReLU) → E6(E4/E5 중 S3 승자 + GELU) → E7(최고 설정 + degrade_strong). 각 실험 후 val_hard 평가, 07:50 넘길 실험은 시작 안 함, 오류 1회 재시도. 상태 `results/overnight_state.json`
- [x] T3 `build_holdouts.py`: 남은 TRAIN 풀에서 test_ext(9클래스×40) 먼저, meta_train(12클래스×100) 고정 — 실행 완료 (360 / 1,200)
- [ ] T4 `stacking.py`: best + 오답 겹침 최소 파트너(S3 best−3%p 이내) → meta_train 확률로 nn.Linear(24,12) → 단일·평균·스태킹 S3 비교. 스태킹은 +1%p 이상일 때만 채택
- [ ] T5 `plot_compare.py` + 최종 모델 `G.plot_curve`
- [ ] T6 `final_test.py <ID|stack>` (TEST 1회) → 제출 노트북 export → `results/overnight_summary.md`
- 검증: 각 단계 후 experiments.csv·history·json 값 확인, 요약 수치는 파일에서만 인용
