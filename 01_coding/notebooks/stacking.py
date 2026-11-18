"""스태킹 비교 (CNN 재학습 없음): python stacking.py [BEST] [PARTNER]
- 후보: 인자가 없으면 overnight_state.json의 best + "best와 오답이 가장 덜 겹치는" 모델(S3가 best-3%p 이내)
  오답 겹침 = val∪val_ext에서 (둘 다 틀림) / (best가 틀림). 후보 선택에만 쓰고 메타 학습에는 쓰지 않음
- 메타 학습: meta_train.csv(TRAIN 내부, 어떤 CNN도 학습에 안 쓴 클래스당 100장)의 두 CNN 확률(log) 24차원
  → nn.Linear(24, 12) (가벼운 PyTorch 로지스틱 회귀, AdamW lr 0.05 wd 1e-3, 300 step 전체배치, seed 42)
- 비교: 같은 val / val_ext / val_hard에서 최고 단일 · 확률 평균 · 스태킹 → S3
- 채택: 스태킹 S3 >= 최고 단일 S3 + 0.01 일 때만. 아니면 단일 모델 유지
- TEST·test_ext는 사용하지 않음
출력: results/stacking_results.json, checkpoints/STACK.pth (메타 가중치)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import gc_lib as G


@torch.no_grad()
def probs(e, df):
    cfg = json.load(open(G.HISTORY / f"{e}_cfg.json", encoding="utf-8"))
    m = G.model_from_cfg(cfg)
    m.load_state_dict(torch.load(G.CKPT / f"{e}.pth", map_location="cpu"))
    m.eval()
    _, tf = G.build_transforms(cfg["img_size"])
    out = [torch.softmax(m(x), 1) for x, _ in G.make_loader(df, tf, 64, False, 0)]
    return torch.cat(out)


def s3_of(e):
    r = pd.read_csv(G.EXP_CSV).set_index("exp_id").loc[e]
    vh = json.load(open(G.HISTORY / f"{e}_val_hard.json", encoding="utf-8"))["val_hard_acc"]
    return (float(r.val_acc) + float(r.val_ext_acc) + vh) / 3


def main(args):
    torch.set_num_threads(12)
    torch.manual_seed(42)
    split = pd.read_csv(G.SPLIT_CSV)
    sets = {"val": split[split.split == "val"], "val_ext": pd.read_csv(G.VAL_EXT_CSV),
            "val_hard": pd.read_csv(G.RESULTS / "val_hard.csv")}
    meta_df = pd.read_csv(G.RESULTS / "meta_train.csv")
    st = json.load(open(G.RESULTS / "overnight_state.json", encoding="utf-8"))
    done = ["E3"] + [e for e in st["done"] if e != "E3"] + ["E2"]
    best = args[0] if args else st["best"]
    lab = {k: torch.tensor([G.CLASS_TO_IDX[c] for c in d.label]) for k, d in sets.items()}
    P = {}

    def get(e, k):
        if (e, k) not in P:
            P[(e, k)] = probs(e, sets[k] if k in sets else meta_df)
        return P[(e, k)]

    if len(args) > 1:
        partner, overlaps = args[1], {}
    else:
        overlaps = {}
        wb = torch.cat([get(best, k).argmax(1) != lab[k] for k in ("val", "val_ext")])
        for e in dict.fromkeys(done):
            if e == best or s3_of(e) < s3_of(best) - 0.03:
                continue
            we = torch.cat([get(e, k).argmax(1) != lab[k] for k in ("val", "val_ext")])
            overlaps[e] = float((wb & we).sum() / wb.sum())
        partner = min(overlaps, key=overlaps.get)
    print("best", best, "partner", partner, "오답 겹침", overlaps)

    feat = lambda k: torch.cat([get(best, k).clamp_min(1e-6).log(), get(partner, k).clamp_min(1e-6).log()], 1)
    y_meta = torch.tensor([G.CLASS_TO_IDX[c] for c in meta_df.label])
    head = nn.Linear(24, 12)
    opt = torch.optim.AdamW(head.parameters(), lr=0.05, weight_decay=1e-3)
    Xm = feat("meta")
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(head(Xm), y_meta)
        loss.backward()
        opt.step()
    torch.save(dict(state=head.state_dict(), best=best, partner=partner), G.CKPT / "STACK.pth")

    res = {}
    with torch.no_grad():
        for name, pred in {
            f"single_{best}": lambda k: get(best, k).argmax(1),
            f"avg_{best}+{partner}": lambda k: ((get(best, k) + get(partner, k)) / 2).argmax(1),
            f"stack_{best}+{partner}": lambda k: head(feat(k)).argmax(1),
        }.items():
            acc = {k: round(float((pred(k) == lab[k]).float().mean()), 4) for k in sets}
            acc["s3"] = round(float(np.mean(list(acc.values()))), 4)
            res[name] = acc
            print(name, acc)
    sb, ss = res[f"single_{best}"]["s3"], res[f"stack_{best}+{partner}"]["s3"]
    decision = "stacking" if ss >= sb + 0.01 else "single"
    out = dict(best=best, partner=partner, error_overlap=overlaps, meta_train_n=len(meta_df),
               meta_train_loss=round(float(loss), 4), results=res, decision=decision,
               rule="스태킹 S3 >= 단일 S3 + 0.01 이면 스태킹 채택")
    json.dump(out, open(G.RESULTS / "stacking_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("결정:", decision)


if __name__ == "__main__":
    main(sys.argv[1:])
