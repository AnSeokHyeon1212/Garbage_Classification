# 01_coding README

## 📥 팀장 지시
- [11/16] T-01: 데이터 확인 + 분할 고정 + CPU 속도 실측
  - 도구: marimo (`01_coding/notebooks/01_data.py`). 데이터는 `data/` (git 제외)
  - 산출물: results/split.csv (path,label,split; stratified 70/15/15, seed=42), figures/class_dist.png, 클래스별 개수 표
  - 추가: torchvision resnet18(pretrained) 224px로 train 1 epoch 소요 시간 실측해서 보고 (CPU)
  - 완료 기준: split.csv 행 수 = 전체 이미지 수, 1 epoch 시간 보고
  - 금지: torch/torchvision 외 학습 패키지. split.csv 생성 후 변경 금지. test 사용 금지
  - 마감: 11/16(월) 밤

## 📤 작업 보고

## ❓ 질문
