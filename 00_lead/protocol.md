# Protocol (팀 규칙·양식)

## 1. 규칙 요약
- 프레임워크: PyTorch(torch, torchvision)만. 금지: YOLO, timm, Keras/TF, fastai, Lightning, 기타 탐지·분류 패키지
- 보조 허용: numpy, pandas, matplotlib, seaborn, sklearn(분할·지표), PIL, imagehash, python-docx
- 분할: `01_coding/results/split.csv` 고정 (seed=42, stratified, train/val/test = 70/15/15). 한번 만들면 변경 금지
- 튜닝은 val만. **test는 최종 1회** (실행 후 experiments.csv에 `test_used=1` 기록)
- 실험은 한 번에 한 변수만 변경
- 외부 데이터: 출처·라이선스 기록, train에만 추가, imagehash 중복 제거
- 노트북: marimo `.py`로 작성 → 제출 전 `marimo export ipynb <파일>.py -o <파일>.ipynb --include-outputs`
- 우선순위: 필수(과적합·HP·증강) > HP 비교 표·그래프 > 외부 데이터 > TTA·앙상블
- 각 agent는 자기 폴더에만 쓰기. 팀장은 팀원 README의 "📥 팀장 지시"에만 추가

## 2. experiments.csv 컬럼
`exp_id, date, base_exp, changed_var, value, model, img_size, epochs, batch, lr, optimizer, weight_decay, aug, train_acc, val_acc, train_loss, val_loss, best_epoch, test_acc, test_used, notes`

## 3. 팀원 README 양식
```
# <폴더명> README
## 📥 팀장 지시   (팀장만 추가)
- [날짜] T-번호: 내용
## 📤 작업 보고   (팀원 작성)
- [날짜] T-번호: 완료/진행/막힘 — 산출물 경로 — 핵심 수치
## ❓ 질문
```

## 4. 작업지시서 양식
```
[T-번호] 제목
- 목표:
- 입력:
- 산출물(경로):
- 완료 기준:
- 금지/주의:
- 마감:
```

## 5. 검토 기준 (팀장)
- 보고 수치가 산출물(CSV/PNG/docx)과 일치하는가
- 분할 CSV 사용, test 미사용(최종 전) 확인
- 한 변수만 변경했는가 (`base_exp` 대비)
- 금지 패키지 import 없음 (`import` 줄 확인)
- 결과: ✅ 승인 / 🔧 수정 / ⏸ 보류 → decisions.md 기록
