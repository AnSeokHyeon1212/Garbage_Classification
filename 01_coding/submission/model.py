"""쓰레기 이미지 12종 분류 — 최종 모델(M8) 제출용 모델 정의.

규칙: 모델 구조는 torch.nn 으로 직접 정의했다(사전학습 모델·가중치 사용 안 함).
      torchvision.transforms 는 전처리에만 쓴다.

가중치: 01_coding/model/final_model.pth 와 같은 state_dict (weights.pt)
평가 서버가 호출하는 함수는 load_model, get_transform 두 개다.
"""

import torch
from torch import nn
from torchvision import transforms

# 모델 출력 인덱스 순서 (학습 때와 같은 알파벳 순)
CLASSES = [
    "battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass",
    "metal", "paper", "plastic", "shoes", "trash", "white-glass",
]

WIDTH = 48          # 첫 단계 채널 수 (단계별 48 → 96 → 192 → 384)
CACHE_SIZE = 192    # 학습 때 원본을 먼저 192×192로 줄여 저장했다
IMG_SIZE = 128      # 모델 입력 크기
MEAN = [0.6448, 0.6212, 0.5932]   # 학습 데이터에서 구한 정규화 통계
STD = [0.2607, 0.2642, 0.2837]


class BasicBlock(nn.Module):
    """3×3 합성곱 2개 + 지름길(identity, 크기·채널이 바뀌면 1×1 합성곱)"""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class SmallResNet(nn.Module):
    """줄기(3×3, stride 2) → 4단계(채널 w, 2w, 4w, 8w × BasicBlock 2개) → 전역 평균 풀링 → Linear(12)"""

    def __init__(self, width=WIDTH, num_classes=12):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        blocks, cin = [], width
        for i, mult in enumerate([1, 2, 4, 8]):
            for j in range(2):
                blocks.append(BasicBlock(cin, width * mult, 2 if (j == 0 and i > 0) else 1))
                cin = width * mult
        self.layers = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(cin, num_classes))   # 학습 때 키 이름(fc.0.*)과 맞춤

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.layers(self.stem(x))), 1))   # softmax 없이 logits


def load_model(weights_path, device):
    model = SmallResNet(width=WIDTH, num_classes=len(CLASSES))
    state = torch.load(weights_path, map_location=device, weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


def get_transform():
    # 학습·검증과 같은 전처리: 192×192로 맞춤(비율 무시) → 128×128 → Tensor → 정규화
    # 입력: PIL 이미지 1장(RGB) → 출력: Tensor [3, 128, 128]
    return transforms.Compose([
        transforms.Resize((CACHE_SIZE, CACHE_SIZE)),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
