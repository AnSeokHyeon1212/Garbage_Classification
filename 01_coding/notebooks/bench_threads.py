"""T-02 사전 측정: torch 스레드 6 vs 12 (+ DataLoader num_workers) 에 따른 train 1 epoch 시간.
E1 조건: Kaggle train 클래스당 ≤1000장, resnet18 160px, 백본 동결, AdamW 1e-3, batch 32, 증강 없음.
"""
import json
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import gc_lib as G


def one_epoch(threads, workers):
    G.seed_everything(42)
    torch.set_num_threads(threads)
    cfg = dict(train_data="kaggle", max_per_class=1000, seed=42)
    df = G.load_train_df(cfg)
    tf, _ = G.build_transforms(160)
    loader = G.make_loader(df, tf, 32, True, workers)
    model = G.build_model("resnet18", "backbone")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    G.train_one_epoch(model, loader, nn.CrossEntropyLoss(), opt, torch.device("cpu"))
    sec = time.perf_counter() - t0
    r = dict(threads=threads, num_workers=workers, n_images=len(df), epoch_sec=round(sec, 1),
             img_per_sec=round(len(df) / sec, 1))
    print(r, flush=True)
    return r


if __name__ == "__main__":
    print("default threads:", torch.get_num_threads(), flush=True)
    res = [one_epoch(6, 0), one_epoch(12, 0)]
    best_t = min(res, key=lambda r: r["epoch_sec"])["threads"]
    res.append(one_epoch(best_t, 4))
    (G.EXP_DIR / "thread_bench.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
