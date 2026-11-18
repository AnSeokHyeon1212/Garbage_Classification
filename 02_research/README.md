# 02_research README

## 📥 팀장 지시
- [11/16] R-01: 데이터 출처·라이선스 + 보고서 목차
  - 산출물: data_sources.md (Kaggle mostafaabla/garbage-classification URL·라이선스), report/outline.md
  - 완료 기준: 라이선스 원문 링크 포함, 모르면 "미확인"
  - 마감: 11/16(월) 밤
- [11/16] R-02: 외부 쓰레기·재활용 이미지 데이터셋 후보 조사 (5개 이상)
  - 각 후보: URL / 라이선스 / 장수 / 클래스 목록 / 우리 12클래스 대응표 / 유리 색(brown·green·white) 구분 여부
  - 우선: 라이선스 명확 + 우리 클래스와 겹침 많음 + 유리 색 구분 가능
  - 산출물: data_sources.md (후보 표 + 추천 순위)
  - 주의: 다운로드는 하지 말고 조사만. 라이선스 불명확 → "미확인" + 제외 후보
  - 마감: 11/17(화) 오전
  - [11/16 추가] 팀장 1차 후보 (라이선스는 각 원본 페이지에서 직접 확인 필수):
    Garbage Classification V2 (sumn2u, Kaggle) / RealWaste (UCI 908) / TrashNet (garythung) / TrashBox / Drinking Waste Classification (arkadiyhacks) / Recyclable and Household Waste Classification (alistairking) / TACO(탐지용, 잘라내기 필요) / AI Hub 생활 폐기물 이미지
  - V2는 mostafaabla 원본과 겹칠 가능성 높음 → 겹침 비율 확인 필요 (코딩 agent imagehash)

- [11/18] R-03: data_sources.md에 실제 사용 3종 확정 기록
  - V2(sumn2u, Kaggle): 라이선스 Kaggle 페이지에서 확인, archive와 70.3% 중복(팀장 측정) 명시
  - RealWaste: CC BY-NC-SA 4.0 (README 원문), 인용 Single et al. 2023, Information 14(12):633
  - AI Hub 232 재활용품 분류 및 선별 데이터: 이용약관 확인(재배포 금지 여부), 직접촬영만 사용
- [11/18] R-04: 보고서 초안 (서론·데이터·방법). 수치는 01_coding/results 파일에서만 가져오고, 없으면 "미확인"
  - 데이터 섹션에 D19 구성표와 선택 이유(실제 test = 애매·지저분한 사진) 포함
  - 프롬프트 부록은 00_lead/prompt_log.md 기반

## 📤 작업 보고
- [11/16] 착수 확인: 팀장 지시 대기 상태 확인 — 작업 없음 (protocol·plan·decisions 숙지 완료)

## ❓ 질문
- [11/16] 제안: plan.md의 R-01(외부 데이터 출처·라이선스 조사, 보고서 목차 골격)은 실행 환경(D2)과 무관하므로 지금 선착수 가능. 지시서(R-01) 발행 요청
  - 조사 대상: TrashNet, Garbage Classification V2(sumn2u), RealWaste(UCI), Recyclable and Household Waste(alistairking), TACO(crop), AI Hub 생활 폐기물
  - 산출물(예정): data_sources.md(URL/라이선스/12클래스 매핑/장수/중복 가능성/용량), report/ 목차 골격(.docx)
- [11/16] 확인 요청: 원본 데이터 = Kaggle mostafaabla/garbage-classification(12 클래스) 맞는지
