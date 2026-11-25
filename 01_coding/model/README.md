# M8 모델

> M8은 M 시리즈 튜닝의 결과이며, 이후 개선의 출발점이다. **최종 모델(F3+ · S1 앙상블)은 저장소 README 9절과 `01_coding/submission/f3p_s1/`** 에 있다.

쓰레기 사진 12종 분류 모델. `torch.nn`으로 직접 만든 작은 ResNet이며 **사전학습 가중치를 쓰지 않고 처음부터 학습**했다.

| 파일 | 내용 |
|---|---|
| `final_model.pth` | 가중치 (`state_dict`, 실험 M8의 최고 epoch 50) |
| `config.json` | 구조·전처리·학습 설정, 클래스 순서, 정규화 값 |

## 구조 (SmallResNet, 폭 48)
| 부분 | 내용 | 출력 채널 |
|---|---|---|
| 줄기 | 3×3 합성곱(stride 2) + BatchNorm + ReLU | 48 |
| 1단계 | BasicBlock × 2 | 48 |
| 2단계 | BasicBlock × 2 (첫 블록 stride 2) | 96 |
| 3단계 | BasicBlock × 2 (첫 블록 stride 2) | 192 |
| 4단계 | BasicBlock × 2 (첫 블록 stride 2) | 384 |
| 헤드 | 전역 평균 풀링 → Linear(384 → 12) | 12 |

- BasicBlock = 3×3 합성곱 2개 + BatchNorm + 지름길(크기·채널이 바뀌면 1×1 합성곱). He 초기화.
- 학습되는 파라미터 **6,289,212개**.

## 입력 전처리 (학습·평가 공통)
1. RGB로 읽어 192×192로 줄임 (비율 유지 안 함, `squash`)
2. 128×128로 줄임
3. 정규화: mean [0.6448, 0.6212, 0.5932], std [0.2607, 0.2642, 0.2837]

클래스 순서 (출력 번호 0~11): 0 battery, 1 biological, 2 brown-glass, 3 cardboard, 4 clothes, 5 green-glass, 6 metal, 7 paper, 8 plastic, 9 shoes, 10 trash, 11 white-glass

## 학습 설정
| 항목 | 값 |
|---|---|
| 학습 데이터 | 정리한 혼합 데이터 27,050장 (`train_data=mix2`) |
| 증강 | dark_blur, hflip, vflip, rot |
| 최적화 | AdamW, 학습률 0.001, weight decay 0.0001, batch 64 |
| 학습률 조절 | cosine |
| epoch | 최대 60 (조기 종료 patience 10), 저장 epoch 50 |
| 손실 | CrossEntropy (라벨 스무딩 0.0) |
| seed | 42 |

## 성능
| 평가 | 장수 | 정확도 |
|---|---|---|
| val (Kaggle 원본 검증) | 2,327 | 0.879 |
| val_ext (외부 실사) | 498 | 0.922 |
| val_hard (흐림·어둡게·빛번짐) | 1,198 | 0.830 |
| val_dirty (검증 사진에 8가지 손상) | 1,198 | 0.755 |
| val_avg4 = (val + val_ext + val_hard + val_dirty) / 4 | — | 0.847 |
| 배경 의존도 bg_drop (배경을 회색으로 지울 때 떨어지는 정확도) | — | 0.122 |
| **TEST** (Kaggle 원본, 1회 평가) | 2,328 | **0.8823** |
| **test_ext** (외부 실사, 1회 평가) | 360 | **0.9028** |
| 더러운 사진 추가 평가 (final_model.ipynb 12-2절) | 1,200 | **0.773** (원본 0.826 / 손상 0.720) |

## 불러오기
```python
import json, torch
cfg = json.load(open("01_coding/model/config.json", encoding="utf-8"))
model = SmallResNet(width=cfg["width"], hidden=tuple(cfg["hidden"]))   # 정의는 notebooks/final_model.ipynb 7절, test.ipynb에도 있음
model.load_state_dict(torch.load("01_coding/model/final_model.pth", map_location="cpu"))
model.eval()
```
정확도와 혼동행렬을 바로 보려면 `01_coding/notebooks/test.ipynb`를 실행한다.
