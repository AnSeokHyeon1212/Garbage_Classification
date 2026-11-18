# 쓰레기 이미지 분류 해커톤 (12 classes)

> 전체 현황판 — **팀장만 수정**. 마지막 갱신: 2026-11-18(수)

## 일정
| 날짜 | 마감 | 상태 |
|---|---|---|
| 11/16(월) | **GitHub 주소 등록** | ⏳ 진행 중 |
| 11/19(목) | **최종 제출** (.ipynb + GitHub + Word 보고서) | ⏳ |

## 진행 현황
| 항목 | 담당 | 상태 |
|---|---|---|
| 폴더 구조 / 규칙 문서 | 팀장 | ✅ |
| GitHub 저장소 생성 | 사용자 | ✅ https://github.com/AnSeokHyeon1212/Garbage_Classification |
| GitHub 주소 등록 (오늘 마감) | 사용자 | ⏳ |
| 첫 push | 팀장 | ✅ 11/16 (7d38043) |
| 실행 환경 | 사용자 | ✅ marimo, 로컬 CPU (속도 실측 대기) |
| 데이터 분할 CSV (고정) | 코딩 | ✅ 15,515장 70/15/15 |
| 외부 데이터 통합 (V2·RealWaste·AI Hub) | 코딩 | ⏳ T-03 |
| 베이스라인 | 코딩 | ⏳ T-02 |
| 과적합 진단·대응 | 코딩 | ⬜ |
| 데이터 증강 | 코딩 | ⬜ |
| HP 튜닝 + 비교 표·그래프 | 코딩 | ⬜ |
| 최종 test 1회 | 코딩 | ⬜ |
| 외부 데이터 중복 점검 (V2) | 팀장 | ✅ 70.3% 중복 |
| 데이터 출처·참고문헌 | 리서치 | ⬜ |
| Word 보고서 (+프롬프트 부록) | 리서치 | ⬜ |

## 오늘(11/18) 핵심
- T-02 베이스라인 + T-03 외부 데이터 통합 → 증강 → HP → 11/19 오전 최종 test 1회
- 학습 구성: 클래스당 1,000장 (Kaggle 53% + V2·RealWaste·AI Hub 47%), val_ext 클래스당 50장

## 최고 성능 (val 기준)
- 미확인 (실험 전)

## 폴더
- `00_lead/` 팀장: decisions, plan, prompt_log, protocol
- `01_coding/` 코딩: notebooks, results/experiments.csv, figures
- `02_research/` 리서치·보고서: data_sources, references, report
