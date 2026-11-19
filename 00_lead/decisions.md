# Decisions

| # | 날짜 | 결정 | 근거 | 상태 |
|---|---|---|---|---|
| D1 | 11/16 | 팀 폴더 구조·protocol 확정 | 역할 분리, 소통 = 폴더 README | ✅ |
| D2 | 11/16 | 노트북 도구 = **marimo**(.py), 제출용 .ipynb는 `marimo export ipynb`로 변환 | 사용자 결정. marimo는 편집기일 뿐 학습 패키지 아님 (허용 여부 공식 확인은 미확인) | ✅ |
| D4 | 11/16 | GitHub = https://github.com/AnSeokHyeon1212/Garbage_Classification | 사용자 결정, 로컬 git 연결 완료 | ✅ |
| D5 | 11/16 | 로컬 CPU 학습 위험 → T-01에서 1 epoch 시간 실측 후 모델·이미지 크기 결정 | GPU 없음 확인 | ⏳ 실측 대기 |
| D3 | 11/16 | 외부 데이터·TTA·앙상블은 시간 남을 때만 | 일정 3일, 필수 우선 | ⏳ 사용자 확인 대기 |
| D6 | 11/16 | 원본 데이터 위치 = `archive/garbage_classification` (12폴더, 총 15,515장), git 제외 | 274MB, 저장소에 올리지 않음 | ✅ |
| D7 | 11/16 | 외부 데이터 = **공개 이미지 데이터셋에서 수집** (직접 생성한 AI 이미지는 허용되나 우선순위 낮음) | 사용자 결정. 실제 test = 생성형·애매한 이미지 | ✅ |
| D8 | 11/16 | 외부 데이터 일부(소량·고정)를 `val_ext`로 분리해 실제 test 근사 평가에 사용. 나머지 외부 데이터는 train만 | 사용자 승인. Kaggle val만으로는 실제 점수 예측 어려움 | ✅ |
| D9 | 11/16 | 증강 후보: 회전, 좌우/상하 반전, 흐림(blur), 색 변화, 잘라내기. 하나씩 추가하며 val로 비교 | 사용자 요청 + 1변수 규칙 | ✅ |
| D10 | 11/16 | 순서: Kaggle만 베이스라인 → 증강 → 외부 데이터 추가(Kaggle만 vs +외부 비교) | 일정·CPU 제약, 비교 실험이 보고서 가점 | ✅ |
| D11 | 11/17 | V2(sumn2u) original 12,259장 중 **8,620장(70.3%)이 archive와 md5 완전 동일** → 중복 전부 제외, 신규 약 3,400~3,600장만 후보 | 팀장 직접 측정(dHash 근접 일치는 라벨 불일치가 많아 hamming≤2 + 같은 라벨만 중복 처리 권장) | ✅ |
| D12 | 11/17 | V2 glass 중 archive와 겹치는 것은 색 라벨 확인 가능, 신규 glass 581장은 색 미분류 → 수작업 또는 제외 | 측정 결과 | ⏳ 사용자 결정 |
| D13 | 11/17 | RealWaste: 9클래스 4,752장, 라이선스 **CC BY-NC-SA 4.0**(README 원문) → 비상업·출처표기·동일조건. 이미지는 GitHub에 올리지 않음 | README 확인 | ✅ |
| D14 | 11/17 | T-01 승인: split.csv 15,515행, train 10,860/val 2,327/test 2,328, 금지 패키지 없음 | 산출물 직접 확인 | ✅ |
| D15 | 11/17 | V2에서 가져올 것 = **중복 아닌 이미지 전부**(plastic 915, paper 733, cardboard 586, metal 310, trash 219, biological 60, clothes 1). glass 581은 색 라벨 없어서 D12 결정 전까지 보류. 원본 해상도 `original/` 폴더 사용 | 팀장 측정(근사치), 최종 중복 제거는 코딩 agent가 imagehash로 재확인 | ✅ 제안 |
| D16 | 11/17 | AI Hub 데이터 수집 중(사용자). 이용약관·형식(탐지용 여부)·클래스 대응 확인 후 사용 여부 결정 | 약관·형식 미확인 | ⏳ |
| D17 | 11/18 | AI Hub 232: 원천 zip 24개 중 23개 정상, `TS_1...플라스틱_001.PE.zip`은 0바이트(다운로드 미완료, .irx255 임시파일) | 팀장 직접 확인(zipfile) | ✅ |
| D18 | 11/18 | AI Hub는 **직접촬영(TS_2/VS_2) 유리병 3색 + 건전지만** 사용 (물체 1개 단독 사진 → 라벨링 데이터 불필요). 영상추출(T3)은 한 장에 여러 물체 섞여 있어 라벨링 JSON 없이는 사용 불가 → 제외. 투명→white-glass, 클래스당 샘플링 | 샘플 이미지 직접 확인, 일정(목 제출) | ✅ 11/18 승인 |
| D19 | 11/18 | 학습 구성안: 클래스당 train 1,000장 (Kaggle 약 53% + 외부 약 47%), val_ext = RealWaste·AI Hub에서 클래스당 50장 먼저 떼어 고정. Kaggle val/test는 그대로 | 실제 test가 지저분·애매한 사진 → 외부 비중 확대, CPU 시간 고려 | ✅ 11/18 승인 |
| D20 | 11/18 | CPU 고속 설정: 비교 실험은 160px + 백본 동결(헤드만 학습), torch 스레드 12 측정 후 적용. 최고 설정만 224px·부분 미세조정으로 최종 학습 | 1 epoch 15분 실측(D5 해소) | ✅ |
| D21 | 11/18 | glass 색 문제(D12) 해소: 외부 glass는 AI Hub 직접촬영(색 구분)만 사용, V2·RealWaste glass는 제외 | AI Hub 색 라벨 있음 | ✅ |
| D22 | 11/18 | E2 완료(mix): val 0.896 / val_ext 0.724 → 채택. 외부 혼합으로 val_ext +33.6%p, Kaggle val −3.6%p | experiments.csv | ✅ |
| D23 | 11/19 | 증강 = 학습 중 실시간 degrade(p=0.5, 어둡게 밝기 0.3~1.0 또는 흐림 σ0.5~2.0). E3 채택(val_ext 0.744) | 사용자 요청(어두움·흐림 대비) | ✅ |
| D24 | 11/19 | **val_hard** 1,200장 고정 생성(Kaggle val 700 + val_ext 500에 흐림·어둡게·빛번짐 손상, 160px). 학습 증강보다 강하거나 없는 손상. 최종 선택 기준 **S3 = (val+val_ext+val_hard)/3**, 채택 = S3 +1%p & clothes·shoes val −3%p 이내 | 사용자 요청(검증 변별력). 조기종료는 (val+val_ext)/2 유지 | ✅ |
| D25 | 11/19 | 남은 TRAIN 풀에서 **test_ext**(외부 9클래스 360장, 최종 1회용)와 **meta_train**(12클래스×100, 스태킹 학습용) 고정 분리. phash 중복 검사 통과 | 사용자 요청(test에 외부 사진 포함). VAL·TEST로 메타 학습 금지 | ✅ |
| D26 | 11/19 | clothing-dataset-small은 Kaggle clothes/shoes와 99.3% 중복 → 사용 안 함 | 팀장 측정 | ✅ |
| D27 | 11/19 | 야간 탐색 결과: 은닉 2층 헤드(E5) 채택, GELU(E6)·강한 어둡게(E7) 기각, layer4 동결 해제(E8) 채택, 224px(E9, (val+val_ext)/2 기준 사전 등록) 채택 | `01_coding/results/overnight_summary.md` | ✅ |
| D28 | 11/19 | 스태킹(E9+E8, Linear 24→12) S3 +0.02%p로 미채택. 확률 평균은 S3 +1.2%p지만 상승분이 160px val_hard에 치우침((val+val_ext)/2 +0.4%p) → **최종 = E9 단일 모델** | stacking_results.json, 사전 규칙 | ✅ |
| D29 | 11/19 | **최종 TEST 1회 사용 완료 (E9): Kaggle test 0.9394 (n=2,328), test_ext 0.9083 (n=360)**. 이후 TEST 재평가 금지 | final_test.json | ✅ |
| D30 | 11/19 | ~~제출 노트북 = `01_coding/notebooks/03_results.ipynb`~~ (D32로 대체)(nbformat으로 작성·실행, 저장된 결과만 읽음). marimo 미설치로 D2의 export 대신 사용. 학습 코드는 gc_lib.py 모듈 | 환경 확인 | ⏳ 제출 형식(모듈 vs 셀) 확인 필요 |
| D31 | 11/19 | **결정 A: 최종 제출 모델 = E9 단일** 확정. 검증셋 오답 E9 372 / 확률 평균 322 / 스태킹 349 (4,027장). 평균의 이득은 val(p=0.04)·val_hard(160px, E8에 유리)에 몰리고 val_ext는 55→54로 동일, 사전 기준 미달, TEST 재사용 불가 → 보고서에 대안 분석으로만 기재 | `results/ensemble_compare.json`, `figures/ensemble_confusion.png`, `notebooks/04_final_analysis.ipynb` | ✅ 사용자 결정 |
| D32 | 11/19 | 제출 구성 확정 (D30 대체): 최종 분석 노트북 = **`01_coding/notebooks/04_final_analysis.ipynb` 하나** (셀마다 목적 주석, 저장된 결과만 읽음). 학습·평가 코드는 `notebooks/*.py`. 결과물은 `01_coding/RESULT/`(data_split·experiments·final_test·ensemble·figures·logs)로 정리. 데이터 출처·라이선스는 루트 README | 사용자 요청 | ✅ |
| D33 | 11/19 | 제출 노트북 교체 (D32 대체): **`01_coding/notebooks/final_model.ipynb` 하나** — 데이터 분할·외부 통합·중복 제거·검증셋 생성·전처리·모델·학습·평가 코드를 노트북에 직접 포함, 비교 실험(E1~E8)·앙상블 내용 제외, 내부 용어 제거(S3 → val_avg 등). 데이터 셀을 빈 폴더에서 재실행해 고정 CSV 6개·손상 이미지 1,200장이 원본과 완전히 같음을 확인. 04_final_analysis.ipynb 삭제 | 사용자 요청 | ✅ |
| D34 | 11/19 | **사전학습 미세조정 금지 확인 → 이전 연구(E1~E9, 앙상블, 최종 TEST) 전부 폐기·삭제.** 작은 ResNet을 직접 구현해 Colab GPU에서 처음부터 학습하는 재연구(R1~R12)로 전환. 고정 분할 CSV·전처리는 유지. 이전 TEST 1회 평가는 폐기된 모델에만 쓰였고 모델 선택에는 쓰이지 않음 → 새 최종 모델로 TEST 1회 재평가하며 보고서에 경위 기재. 마감 11/20 | 사용자 확인 (규칙) | ✅ |
| D35 | 11/20 | **2차 튜닝 준비.** ① 사진을 직접 확인해 데이터 정리: AI Hub 유리(병이 작거나 색 불일치) 제외, 육안 제외 목록, v2 cardboard 전체 제외, 부족분은 같은 출처·클래스로 보충 → 학습 27,050 / val_ext 498 / test_ext 360 / val_hard 1,198. 1차 R1~R12는 정리 전 데이터 기준(최고 R9 val_avg 0.502)이라 직접 비교하지 않음. ② 채택 규칙에서 clothes·shoes 조건 제거(val_avg +0.01 이상만 채택) — 1차에서 외부 혼합(R2, val_ext 0.144→0.772)이 이 조건 때문에 기각됐음. ③ 비율 유지(letterbox) 전처리 ④ 증강: 좌우·상하 반전, 회전, 색상, TrivialAugmentWide, 어둡게·흐림 (무작위 자르기 미사용), 클래스 가중치, 배경 외우기 점검 지표. ⑤ 사다리 M1~M9, Colab 실행 예정, 최종 TEST 미실시. 새 출처: TrashBusters/combined(MIT), Recyclable and Household Waste(Kaggle, 라이선스 미확인), ShoeSandalBoot(라이선스 표기 없음), AI Hub 직접촬영 추가분(비공개) | 사용자 결정, `RESULT/data_split/clean_report.csv`·`extra_counts.csv`, 노트북 `clean-md` | ✅ |
