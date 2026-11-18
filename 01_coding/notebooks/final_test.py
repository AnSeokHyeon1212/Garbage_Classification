"""T-06 최종 평가 (딱 한 번): python final_test.py <EXP_ID | stack>
- 실행 전 확인: experiments.csv의 test_used가 모두 0, results/final_test.json 없음 → 아니면 중단
- 평가: Kaggle TEST(split.csv, 2,328장) + test_ext(외부 9클래스 360장)을 한 번에 평가
- 기록: final_test.json(전체/클래스별/출처별), final_test_confusion.csv, experiments.csv 해당 행 test_acc·test_used=1
  'stack'이면 stacking_results.json의 best 행에 기록하고 notes에 스태킹임을 남긴다
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gc_lib as G
from stacking import probs


def main(target):
    out = G.RESULTS / "final_test.json"
    exp = pd.read_csv(G.EXP_CSV)
    assert not out.exists(), "final_test.json이 이미 있음: TEST는 이미 1회 사용됨"
    assert (exp.test_used.fillna(0) == 0).all(), "experiments.csv에 test_used=1 행이 있음"
    torch.set_num_threads(12)
    split = pd.read_csv(G.SPLIT_CSV)
    sets = {"test": split[split.split == "test"], "test_ext": pd.read_csv(G.RESULTS / "test_ext.csv")}

    if target == "stack":
        ck = torch.load(G.CKPT / "STACK.pth")
        head = torch.nn.Linear(24, 12)
        head.load_state_dict(ck["state"])
        row_id, desc = ck["best"], f"stacking({ck['best']}+{ck['partner']})"
        def predict(df):
            with torch.no_grad():
                f = torch.cat([probs(ck["best"], df).clamp_min(1e-6).log(),
                               probs(ck["partner"], df).clamp_min(1e-6).log()], 1)
                return head(f).argmax(1).numpy()
    else:
        row_id, desc = target, target
        predict = lambda df: probs(target, df).argmax(1).numpy()

    res = dict(model=desc, evaluated_once=True)
    for k, df in sets.items():
        pred = [G.CLASSES[i] for i in predict(df)]
        d = df.assign(pred=pred, ok=lambda x: x.pred == x.label)
        res[k] = dict(acc=round(d.ok.mean(), 4), n=len(d),
                      by_class={c: dict(acc=round(g.ok.mean(), 4), n=len(g)) for c, g in d.groupby("label")},
                      by_source={s: dict(acc=round(g.ok.mean(), 4), n=len(g)) for s, g in d.groupby("source")})
        pd.crosstab(d.label, d.pred).to_csv(G.RESULTS / f"final_{k}_confusion.csv", encoding="utf-8-sig")
        print(k, res[k]["acc"], "n", len(d))
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    i = exp.index[exp.exp_id == row_id][0]
    exp.loc[i, "test_acc"] = res["test"]["acc"]
    exp.loc[i, "test_used"] = 1
    exp.loc[i, "notes"] = f"{exp.loc[i, 'notes']} / 최종 TEST 1회: {desc}, Kaggle test {res['test']['acc']}, test_ext {res['test_ext']['acc']}"
    exp.to_csv(G.EXP_CSV, index=False, encoding="utf-8")
    print("기록 완료:", out)


if __name__ == "__main__":
    main(sys.argv[1])
