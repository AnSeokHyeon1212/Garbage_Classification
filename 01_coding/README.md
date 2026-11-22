# 01_coding README

## 📥 팀장 지시
- [11/16] T-01: 데이터 확인 + 분할 고정 + CPU 속도 실측
  - 도구: marimo (`01_coding/notebooks/01_data.py`). 데이터는 `data/` (git 제외)
  - 산출물: results/split.csv (path,label,split; stratified 70/15/15, seed=42), figures/class_dist.png, 클래스별 개수 표
  - 추가: torchvision resnet18(pretrained) 224px로 train 1 epoch 소요 시간 실측해서 보고 (CPU)
  - 완료 기준: split.csv 행 수 = 전체 이미지 수, 1 epoch 시간 보고
  - 금지: torch/torchvision 외 학습 패키지. split.csv 생성 후 변경 금지. test 사용 금지
  - 마감: 11/16(월) 밤
  - 추가: 데이터 경로 = `archive/garbage_classification` (12폴더, 15,515장). split.csv에 `source` 컬럼(kaggle) 포함
- [11/16] T-04 예고 (T-02 베이스라인 후): 증강 1개씩 추가 실험
  - 순서: 좌우반전 → 회전(±15~30°) → RandomResizedCrop → ColorJitter → GaussianBlur
  - 각 실험은 직전 최고 설정 대비 1변수만 변경, experiments.csv에 `changed_var=aug` 기록
  - 증강 예시 이미지 그리드 1장 저장: figures/aug_examples.png (보고서용)

- [11/18] ⚡ 오늘 일정 (D19·D20 승인). 제출 = 내일. 컷오프: 11/19 09시
- [11/18] T-02: 베이스라인 E1 (Kaggle만)
  - 데이터: split.csv의 train에서 클래스당 최대 1,000장 샘플(seed=42, 부족하면 전부). val = Kaggle val 그대로
  - 설정: resnet18(pretrained), 160px, 백본 동결(fc만 학습), AdamW lr=1e-3, wd=1e-4, batch 32, epochs 10, early stop patience 3, 증강 없음(Resize+Normalize)
  - 먼저 torch.set_num_threads(12) 효과 측정(1 epoch 시간) → 빠르면 적용
  - 산출물: experiments.csv에 E1 행, figures/E1_curve.png(train/val loss·acc), 클래스별 val 정확도
- [11/18] T-03: 외부 데이터 통합 (T-02 학습 도는 동안 병행)
  - 소스: V2 `archive(1)/original`, RealWaste `realwaste/realwaste-main/RealWaste`, AI Hub 직접촬영 zip(TS_2/VS_2: 유리병 갈색·녹색·투명, 건전지, 철캔_3) — zip은 풀지 말고 필요한 장수만 zipfile로 꺼내 `external_data/`에 저장
  - 클래스 매핑: Food Organics→biological, Misc Trash→trash, Cardboard/Metal/Paper/Plastic→동명, 투명→white-glass, 건전지→battery, 철캔→metal. 제외: V2·RealWaste glass, Vegetation, Textile(샘플 확인 전까지 제외)
  - 중복 제거: imagehash(phash)로 Kaggle **전체(train·val·test)** 대비 + 외부끼리. 기준 hamming≤4 & 같은 클래스 → 제거, 다른 클래스 근접은 제외 처리 후 개수 보고
  - 순서: ① val_ext 먼저: RealWaste·AI Hub에서 클래스당 50장 고정 ② 나머지에서 D19 표대로 train 샘플(seed=42)
  - D19 표 (train): battery K500+AI500 / biological K579+V2 60+RW361 / brown K425+AI575 / green K440+AI560 / white K450+AI550 / cardboard K400+V2 300+RW300 / clothes K1000 / metal K400+V2 200+RW300+AI100 / paper K400+V2 300+RW300 / plastic K400+V2 300+RW300 / shoes K1000 / trash K400+V2 200+RW400. 부족하면 Kaggle로 채움
  - 산출물: results/train_mix.csv, results/val_ext.csv (path,label,split,source), 출처별·클래스별 개수 표. **split.csv는 수정 금지**
- [11/18] 이후 순서 (각 1변수, 직전 최고 대비): E2 데이터=혼합 → T-04 증강(좌우반전→회전→RandomResizedCrop→ColorJitter(hue≤0.05)→GaussianBlur) → T-05 lr(1e-2,1e-3,1e-4) → wd(1e-4,1e-3,1e-2) → 과적합 대응(layer4 동결 해제, dropout)
  - 모델 선택 기준: val_acc와 val_ext_acc 평균. 모든 실험에 val_ext_acc 기록
  - T-06: 최고 설정 224px로 최종 학습 → **Kaggle test 1회** → `marimo export ipynb --include-outputs` → git 커밋은 팀장 승인 후
- [11/19] 추가 검증셋·선택 규칙 (D24·D25): `build_val_hard.py` → val_hard 1,200장(흐림·어둡게·빛번짐), `build_holdouts.py` → test_ext 360장·meta_train 1,200장. 최종 선택 = S3 = (val+val_ext+val_hard)/3, +1%p & clothes·shoes −3%p 가드
- [11/19] 야간 실험 (팀장 직접 실행 승인): E4 은닉1층 → E5 은닉2층 → E6 GELU → E7 강한 어둡게 → E8 layer4 해제 → E9 224px → 스태킹 비교 → T-06 TEST 1회

- [11/19] 📁 폴더 정리: `results/`·`figures/` → `RESULT/` 하위(`data_split/`, `experiments/`, `final_test/`, `ensemble/`, `figures/`, `logs/`)로 이동. 아래 보고의 `results/…` 경로는 이동 전 기준. 경로 상수는 `gc_lib.py`. 최종 노트북은 `notebooks/final_model.ipynb` 하나 (전처리~최종 모델 학습~평가, 비교 실험 내용 제외. `03_results`·`04_final_analysis`는 삭제)

- [11/19] ⚠️ **사전학습 미세조정 금지 → 이전 연구(E1~E9) 전부 폐기 (D34).** 아래 E1~E9·T-06 보고는 폐기된 기록. 재연구 계획: `docs/superpowers/plans/2026-11-19-scratch-cnn-research.md`

- [11/20] 2차 튜닝 준비 (D35): 데이터 정리(`tools/`, `RESULT/data_split/*_clean.csv`) → 학습 27,050 / val_ext 498 / test_ext 360 / val_hard 1,198. 채택 규칙에서 clothes·shoes 조건 제거(val_avg +0.01 이상), letterbox 전처리·증강·클래스 가중치·배경 점검 지표, 사다리 M1~M9(`ladder_m.csv`). Colab에서 실행 예정, 최종 TEST는 아직 하지 않음
- [11/22] 2차 튜닝 완료 (D36): 최종 설정 M8 (epoch 60; val_avg 0.877 = val 0.879 / val_ext 0.922 / val_hard 0.830). 다음: 최종 TEST 1회 → 13절 결론 작성

## 📤 작업 보고
- [11/16] T-01: ✅ 완료 — `notebooks/01_data.py`(marimo), 실행 결과 `results/01_data.html`
  - `results/split.csv` (path,label,split,source): 15,515행 = 전체 이미지 수. train 10,860 / val 2,327 / test 2,328. seed=42 stratified. source=kaggle. 경로는 루트 기준 상대경로(`archive/garbage_classification/...`)
  - `results/class_counts.csv`, `figures/class_dist.png`: 최다 clothes 5,325(34.3%), 최소 brown-glass 607(3.9%) → 불균형 8.77배
  - 이미지 점검: 깨진 파일 0, 색 모드 RGB 15,481 / P(팔레트) 34 → 로딩 시 `.convert("RGB")` 처리
  - CPU 실측 `results/cpu_timing.json`: resnet18(pretrained) 224px, batch 32, fp32, num_workers 0, torch 6스레드 → **train 1 epoch 899초(15.0분), 12.1 img/s**. (참고: 1 epoch 후 train acc 0.771, val/test 미평가)
  - 환경: Python 3.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu. pandas 3.1.1은 Windows 애플리케이션 제어 정책이 DLL을 막아 실패 → **pandas 3.0.6으로 설치**
  - test 미사용. split.csv는 노트북 재실행 시에도 재생성되지 않도록 되어 있음(있으면 읽기만 함)
- [11/18] T-02: ✅ 완료 — 코드 `notebooks/gc_lib.py`(train_one_epoch/evaluate/run_experiment), `notebooks/run_exp.py`(실험 CFG), 결과 노트북 `notebooks/02_experiments.py`
  - 스레드 측정 `results/thread_bench.json` (E1 조건 train 1 epoch, 7,749장): 6스레드 218초 → **12스레드 139초(-36%) → 적용**. num_workers 4는 144초로 이득 없음 → 0 유지
  - E1 (Kaggle train 클래스당 ≤1,000 = 7,749장, resnet18 160px 백본 동결, AdamW 1e-3, wd 1e-4, batch 32, 증강 없음): ep9에서 early stop, **best ep6 val_acc 0.9317** (train 0.9423, val loss 0.2197). 36.6분(T-03과 CPU를 같이 써서 epoch당 190~410초)
  - 산출물: `results/experiments.csv` E1 행, `figures/E1_curve.png`, `results/history/E1.csv`, 클래스별 val 정확도 `results/history/E1_val_per_class.csv`
  - 클래스별 val: 낮음 white-glass 0.80 / plastic 0.81 / metal 0.84 / cardboard 0.85, 높음 biological 0.99 / clothes 0.98 / shoes 0.96
  - **E1 val_ext 0.388** (학습 후 best ckpt로 사후 평가, `results/history/E1_val_ext.json`). Kaggle val 0.93과 차이가 큼 → Kaggle만으로는 외부 사진에 일반화가 안 됨. 클래스별: white-glass 0.10, cardboard 0.12, trash 0.16 최저 / brown-glass 0.72 최고
  - 곡선: ep4 이후 train acc는 계속 오르는데 val은 0.92~0.93에서 정체 → 약한 과적합 시작
- [11/18] T-03: ✅ 완료 — `notebooks/build_mix.py` → `results/train_mix.csv` 12,000장, `results/val_ext.csv` 500장 (path,label,split,source). split.csv 수정 없음
  - AI Hub: zip 6개에서 필요량의 약 1.35배만 꺼내 짧은 변 384px로 줄여 `external_data/aihub/<클래스>/`에 저장(원본 2.5~6MB/장). 깨진 파일 1장(녹색) 제외
  - 중복 제거 `results/dedup_report.csv` (phash, hamming≤4, 외부 후보 13,493장): Kaggle과 같은 클래스 중복 **3,427**(거의 전부 V2), Kaggle과 다른 클래스 근접 19, 외부끼리 중복 16, 외부끼리 다른 클래스 근접 26 → 모두 제외, 남은 10,005장 중에서 샘플
  - val_ext: 클래스당 50장 × 10 클래스 = 500 (AI Hub 225 / RealWaste 275, metal은 RW 25+AI 25). **clothes·shoes는 RealWaste·AI Hub에 없어 val_ext에 없음**
  - train_mix 출처: Kaggle 6,396 / AI Hub 2,285 / RealWaste 1,959 / V2 1,360. 클래스당 1,000장. 부족분: biological RealWaste 359/361 → Kaggle 2장으로 채움(Kaggle 581)
  - 출처별·클래스별 표: `results/mix_counts.csv`
  - 샘플 육안 확인: 유리 3색 라벨은 대체로 맞음. AI Hub는 바닥(보도블록·아스팔트) 배경이라 Kaggle(단색 배경)과 도메인이 다름
- [11/18] E2 (E1 대비 train_data=mix만 변경): ✅ 완료(사용자 실행) — best ep3, val 0.8956 / val_ext 0.724 / val_hard 0.689. 채택
- [11/19] E3 (aug=degrade): ✅ 완료(사용자 실행) — val 0.893 / val_ext 0.744 / val_hard 0.715 (S3 0.784). 채택
- [11/19] 야간 실험 E4~E9 (팀장 실행, `overnight.py`·`overnight2.py`): ✅ 완료. 상세 `results/overnight_summary.md`, 로그 `results/overnight_log.txt`
  - E4 은닉1층 S3 0.820 채택 / **E5 은닉2층 0.832 채택** / E6 GELU 0.823 유지 / E7 degrade_strong 0.826 유지(val_ext 하락) / **E8 layer4 해제 0.870 채택** / **E9 224px 0.891 채택** (val 0.949, val_ext 0.890, val_hard 0.834)
  - val_hard 손상별 E3→E9: 흐림 0.658→0.780, 어둡게 0.691→0.828, 빛번짐 0.796→0.894
  - 스태킹 E9+E8: S3 0.8914 (단일 0.8912) → 미채택. `results/stacking_results.json`
- [11/19] T-06: ✅ 완료 — **E9 최종 TEST 1회: Kaggle test 0.9394 (n=2,328), test_ext 0.9083 (n=360; AI Hub 0.967, RealWaste 0.850)**. 약점 white-glass 0.759, cardboard 0.888 (Kaggle test), plastic 0.775 (test_ext). `results/final_test.json`, `final_test_confusion.csv`, `final_test_ext_confusion.csv`
  - 제출 노트북 `notebooks/final_model.ipynb` (출력 포함, 셀별 주석), 그림 `figures/overnight_compare.png`, `figures/E9_curve.png`

- [11/20] 문서·노트북 정리 ✅ — 1차 튜닝 R1~R12(정리 전 데이터) 최고 R9 val_avg 0.502 (val 0.816 / val_ext 0.184 / val_hard 0.506). 외부 혼합 R2는 val_ext 0.144 → 0.772였으나 clothes·shoes 조건으로 기각. 이후 데이터 정리(`tools/`는 기록용, 196GB 원본 필요)와 2차 사다리 M1~M9 준비를 마쳤고 실행은 Colab에서 할 예정. 새 외부 출처(TrashBusters MIT, Kaggle Recyclable and Household Waste 라이선스 미확인, ShoeSandalBoot 라이선스 표기 없음)는 루트 README 데이터 출처 표에 기재
- [11/22] 2차 사다리 M1~M9 ✅ — 채택 M1 0.813 → M3 0.848 → **M8 0.877** (val 0.879 / val_ext 0.922 / val_hard 0.830). 기각 M2(letterbox, val_hard 0.640)·M4·M5·M6·M7(+0.008)·M9(160px, 흐림 0.687). M8 과적합 점검: epoch 50 train 0.982 / val 0.879, val loss 최저 epoch 26 → 가벼운 과적합, val 정확도 유지. 결과 `RESULT/experiments/`(가중치 제외), 해석은 노트북 10절 끝

## ❓ 질문
- [11/18] val_ext에 clothes·shoes가 없어서 (val+val_ext)/2 기준은 이 두 클래스를 val로만 평가합니다. 이대로 진행할지, V2 clothes/shoes 비중복분(거의 없음)이나 다른 소스를 찾을지 결정 부탁드립니다.
- [11/18] (11/19 갱신: 최종 노트북은 `final_model.ipynb` 하나, 학습 코드는 `.py` 모듈로 함께 제출 — D32) 제출 노트북: 학습 코드가 `gc_lib.py` 모듈에 있습니다(Windows DataLoader 때문). T-06에서 제출용 노트북에 모듈 코드를 셀로 펼쳐 넣을지, 모듈을 같이 제출할지 결정 부탁드립니다.
- ~~[11/16] 데이터 위치:~~ (해결: D6) 지시 경로대로 `archive/garbage_classification`을 직접 참조했고 `data/`로 복사하지 않음(archive/도 .gitignore 대상). 복사가 필요하면 지시 부탁드립니다.
- ~~[11/16] CPU로는 1 epoch에~~ (해결: D20) 15분 → 비교 실험 1개(예: 5 epoch + val)마다 약 1.5시간 걸림. 제안(미실행·미실측): ① Colab GPU로 전환 ② CPU 유지 시 비교 실험은 img 128px + 백본 동결로 하고, 유망 설정만 224px로 학습. 결정 부탁드립니다.
- ~~[11/16] torch가 논리 코어~~ (해결: T-02 측정, 12스레드 적용) 12개 중 6스레드만 사용 중. `torch.set_num_threads(12)`와 DataLoader `num_workers>0`(Dataset을 .py 모듈로 분리해야 함) 효과는 미측정입니다. T-02 전에 측정할지 확인 부탁드립니다.

## 📝 보고서용 메모
- 분할: stratified 70/15/15 (seed=42) → 세 세트의 클래스 비율이 거의 같음 (`results/class_counts.csv`)
- 불균형 8.77배(clothes/brown-glass) → Accuracy만 보면 다수 클래스에 유리할 수 있음. 클래스별 정확도·혼동행렬을 함께 제시할 것
- 도메인 차이: Kaggle만 학습한 E1이 Kaggle val 0.932 vs 외부 사진 val_ext 0.388 → 외부 데이터 혼합(E2)의 근거. E1↔E2 비교가 보고서 핵심 표
- 외부 데이터 정제: V2 후보의 약 53%(3,427/6,426)가 Kaggle과 phash 중복 → 중복 제거 없이 합치면 val/test 누출 위험
- CPU 최적화: torch 스레드 6→12로 epoch 시간 36% 단축, 백본 동결+160px로 224px 전체 학습 대비 약 6.5배 빠름(899초 → 139초, 조건이 달라 단순 비교 주의)
