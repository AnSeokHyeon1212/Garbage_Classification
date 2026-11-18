"""저장된 best ckpt를 val_hard(고정 손상 검증셋)로 평가: python eval_val_hard.py E1 E2 ...
출력: results/history/<exp>_val_hard.json (전체 / 손상별 / 출처별 / 클래스별 정확도와 표본 수)
학습 없음, test 사용 안 함.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import gc_lib as G


@torch.no_grad()
def main(exp_ids):
    torch.set_num_threads(12)
    df = pd.read_csv(G.RESULTS / "val_hard.csv")
    for e in exp_ids:
        cfg = json.load(open(G.HISTORY / f"{e}_cfg.json", encoding="utf-8"))
        _, eval_tf = G.build_transforms(cfg["img_size"])  # 224 모델은 160px val_hard를 224로 키워 평가(참고용)
        m = G.model_from_cfg(cfg)
        m.load_state_dict(torch.load(G.CKPT / f"{e}.pth", map_location="cpu"))
        _, acc, preds, labels = G.evaluate(m, G.make_loader(df, eval_tf, 64, False, 0),
                                          torch.nn.CrossEntropyLoss(), torch.device("cpu"))
        d = df.assign(ok=(preds == labels))
        grp = lambda col: {k: dict(acc=round(g.ok.mean(), 4), n=len(g)) for k, g in d.groupby(col)}
        res = dict(exp_id=e, val_hard_acc=round(acc, 4), n=len(d),
                   by_corruption=grp("corruption"), by_source=grp("source"), by_class=grp("label"))
        json.dump(res, open(G.HISTORY / f"{e}_val_hard.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(e, "val_hard", res["val_hard_acc"], "| 손상별", {k: v["acc"] for k, v in res["by_corruption"].items()},
              "| 출처별", {k: v["acc"] for k, v in res["by_source"].items()})


if __name__ == "__main__":
    main(sys.argv[1:])
