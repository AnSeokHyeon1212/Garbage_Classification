"""최종 모델(F3+ + S1 앙상블) 평가. 저장소 루트에서 실행하고, submission/f3p_s1/weights.pt 와 데이터가 필요하다.

python 01_coding/tools/evaluate_final.py test   # TEST 1회 평가 (결과 파일이 이미 있으면 멈춤)
앙상블 비율을 고른 결과는 RESULT/experiments/F3P_S1_ensemble/ 에 있다
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
CODING = ROOT / "01_coding"
SPLIT = CODING / "RESULT" / "data_split"
SUB = CODING / "submission" / "f3p_s1"
spec = importlib.util.spec_from_file_location("final_model", SUB / "model.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
model = m.load_model(str(SUB / "weights.pt"), "cpu").eval()
tf = m.get_transform()
CLASSES = m.CLASSES


def labels(df):
    return np.array([CLASSES.index(c) for c in df.label])


@torch.no_grad()
def predict(net, xs, batch=64):
    return torch.cat([net(xs[i:i + batch]) for i in range(0, len(xs), batch)])


def run_test():
    out = CODING / "RESULT" / "final_test"
    if (out / "final_test_f3p_s1.json").exists():
        sys.exit("이미 TEST를 평가했습니다 (1회 규칙): " + str(out / "final_test_f3p_s1.json"))
    sp = pd.read_csv(SPLIT / "split.csv")
    result = {"model": "F3P_30ep + S1 ensemble (hflip TTA)", "w_weak": m.W_WEAK, "w_other": m.W_OTHER, "temperature": m.TEMPERATURE}
    for name, df in [("test", sp[sp.split == "test"]), ("test_ext", pd.read_csv(SPLIT / "test_ext_clean.csv"))]:
        lost = [p for p in df.path if not (ROOT / p).exists()]
        assert not lost, f"{name}: 없는 사진 {len(lost)}장"
        y, logits = labels(df), []
        for i in range(0, len(df), 64):
            xs = torch.stack([tf(Image.open(ROOT / p).convert("RGB")) for p in df.path[i:i + 64]])
            logits.append(predict(model, xs))
        logits = torch.cat(logits)
        pred = logits.argmax(1).numpy()
        per_class = {c: float((pred[y == k] == k).mean()) for k, c in enumerate(CLASSES) if (y == k).any()}
        result[name] = dict(n=len(df), correct=int((pred == y).sum()), accuracy=float((pred == y).mean()),
                            loss=float(torch.nn.functional.cross_entropy(logits, torch.tensor(y))),
                            macro_accuracy=float(np.mean(list(per_class.values()))), per_class=per_class)
        cm = pd.crosstab(pd.Categorical(y, range(12)), pd.Categorical(pred, range(12)), dropna=False)
        cm.index, cm.columns = CLASSES, CLASSES
        cm.to_csv(out / f"{'final_test' if name == 'test' else 'final_test_ext'}_f3p_s1_confusion.csv", encoding="utf-8-sig")
        print(name, {k: v for k, v in result[name].items() if k != "per_class"})
    json.dump(result, open(out / "final_test_f3p_s1.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    assert sys.argv[1:] == ["test"], "사용법: python 01_coding/tools/evaluate_final.py test"
    run_test()
