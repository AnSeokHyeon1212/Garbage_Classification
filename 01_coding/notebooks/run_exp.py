"""실험 실행: python run_exp.py E1 [E2 ...]  (백그라운드 실행용. 결과는 experiments.csv / history/ / figures/)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gc_lib as G

BASE = dict(seed=42, model="resnet18", img_size=160, freeze="backbone", optimizer="adamw", lr=1e-3,
            weight_decay=1e-4, batch=32, epochs=10, patience=3, aug=[], train_data="kaggle",
            max_per_class=1000, threads=12, num_workers=0)

EXPS = {
    "E1": dict(BASE, exp_id="E1", base_exp="", changed_var="baseline", value="kaggle<=1000/cls",
               notes="베이스라인: Kaggle만, 백본 동결, 증강 없음"),
}
EXPS["E2"] = dict(EXPS["E1"], exp_id="E2", base_exp="E1", changed_var="train_data", value="mix(train_mix.csv)",
                  train_data="mix", notes="Kaggle+V2+RealWaste+AI Hub 혼합 12,000장. 선택=val·val_ext 평균")
EXPS["E3"] = dict(EXPS["E2"], exp_id="E3", base_exp="E2", changed_var="aug", value="degrade",
                  aug=["degrade"], notes="E2 + 실시간 증강 degrade(p=0.5: 밝기 0.3~1.0 또는 GaussianBlur σ0.5~2)")
# 야간 실험 (11/19). E6·E7은 결과에 따라 overnight.py가 E4/E5 승자·최고 설정에서 만든다(history/<id>_cfg.json에 기록)
EXPS["E4"] = dict(EXPS["E3"], exp_id="E4", base_exp="E3", changed_var="hidden", value="1층(512→256, ReLU)",
                  hidden=[256], act="relu", notes="E3 + 은닉층 1개(256, ReLU)")
EXPS["E5"] = dict(EXPS["E3"], exp_id="E5", base_exp="E3", changed_var="hidden", value="2층(512→256→128, ReLU)",
                  hidden=[256, 128], act="relu", notes="E3 + 은닉층 2개(256→128, ReLU)")

if __name__ == "__main__":
    for eid in sys.argv[1:]:
        cfg = EXPS[eid]
        G.run_experiment(cfg, log=lambda s: print(s, flush=True))
        G.plot_curve(eid)
