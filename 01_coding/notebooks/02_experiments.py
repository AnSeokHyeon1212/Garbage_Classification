import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(mo.notebook_dir())))
    import gc_lib as G
    from run_exp import EXPS
    return EXPS, G, json, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # T-02 베이스라인 E1 · T-03 외부 데이터 통합

    학습은 시간이 오래 걸려서 `run_exp.py`(같은 폴더)로 백그라운드에서 실행하고, 이 노트북은 저장된 결과를 읽어 보여준다.
    학습 코드(`train_one_epoch` / `evaluate` / `run_experiment`)는 `gc_lib.py`에 있다.
    Windows에서 DataLoader 멀티프로세스를 쓰려면 Dataset이 .py 모듈에 있어야 해서 분리했다.
    결과가 없으면 아래 셀에서 `G.run_experiment(cfg)`로 직접 실행한다.

    - **test는 사용하지 않는다.** 평가는 Kaggle val(2,327장)과 val_ext(외부, T-03)로만 한다.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. CPU 스레드 측정 (E1 조건, train 1 epoch)
    `torch.set_num_threads`는 행렬 연산에 쓰는 CPU 스레드 수다. 기본값은 6이었다(논리 코어 12).
    `num_workers`는 이미지를 읽고 전처리하는 별도 프로세스 수다.
    """)
    return


@app.cell
def _(G, pd):
    pd.read_json(G.RESULTS / "thread_bench.json")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. E1 베이스라인 설정
    - 데이터: split.csv의 train에서 클래스당 최대 1,000장 (seed=42). 부족한 클래스는 전부 사용
    - resnet18(ImageNet 사전학습), 160px, **백본 동결**: 마지막 fc층만 학습한다. 동결된 부분의 BatchNorm은 eval 모드로 고정
    - AdamW lr 1e-3, wd 1e-4, batch 32, 최대 10 epoch, early stopping patience 3 (val acc 기준), 증강 없음
    - best 체크포인트 = val acc 최고 epoch (E1 학습 시작 시점에는 val_ext가 없어서 val만으로 선택)
    """)
    return


@app.cell
def _(EXPS):
    EXPS["E1"]
    return


@app.cell
def _(EXPS, G):
    if not (G.HISTORY / "E1.csv").exists():
        G.run_experiment(EXPS["E1"])
        G.plot_curve("E1")
    return


@app.cell
def _(G, pd):
    pd.read_csv(G.HISTORY / "E1.csv").round(4)
    return


@app.cell
def _(G, mo):
    mo.image(src=str(G.FIGURES / "E1_curve.png"), width=800)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. E1 클래스별 val 정확도 (best 체크포인트)
    """)
    return


@app.cell
def _(G, pd):
    _pc = pd.read_csv(G.HISTORY / "E1_val_per_class.csv", index_col=0)
    _pc.sort_values("acc").round(4)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. T-03 외부 데이터 통합 결과
    `build_mix.py`로 생성. phash hamming ≤ 4 기준으로 중복을 처리했다.
    - Kaggle 전체(train·val·test)와 같은 클래스로 근접 → `dup_kaggle`, 다른 클래스로 근접 → `conflict_kaggle`
    - 외부끼리 같은 클래스로 근접 → `dup_external`(1장만 유지), 다른 클래스로 근접 → `conflict_external`
    - AI Hub 이미지는 zip에서 필요한 만큼만 꺼내 짧은 변 384px로 줄여 `external_data/aihub/`에 저장했다.
    """)
    return


@app.cell
def _(G, pd):
    pd.read_csv(G.RESULTS / "dedup_report.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 출처별·클래스별 개수 (train_mix / val_ext)
    """)
    return


@app.cell
def _(G, pd):
    _tm = pd.read_csv(G.RESULTS / "train_mix.csv")
    _ve = pd.read_csv(G.VAL_EXT_CSV)
    _all = pd.concat([_tm, _ve])
    mix_table = _all.pivot_table(index="label", columns=["split", "source"], values="path",
                                 aggfunc="count", fill_value=0)
    mix_table
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. E1 체크포인트를 val_ext로 평가
    E1은 val_ext가 만들어지기 전에 학습을 시작했다. 그래서 학습이 끝난 뒤 best 체크포인트로 val_ext를 평가해 기록한다.
    val_ext에는 clothes와 shoes가 없다(RealWaste·AI Hub에 해당 클래스 없음).
    """)
    return


@app.cell
def _(EXPS, G, json, pd):
    _p = G.HISTORY / "E1_val_ext.json"
    if _p.exists():
        e1_ext = json.loads(_p.read_text(encoding="utf-8"))
    else:
        _loss, _acc, _pc = G.eval_ckpt("E1", pd.read_csv(G.VAL_EXT_CSV), EXPS["E1"])
        e1_ext = {"val_ext_loss": round(_loss, 4), "val_ext_acc": round(_acc, 4),
                  "per_class": _pc.dropna().round(4).to_dict()}
        _p.write_text(json.dumps(e1_ext, ensure_ascii=False, indent=2), encoding="utf-8")
    e1_ext
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 실험 기록 (results/experiments.csv)
    """)
    return


@app.cell
def _(G, pd):
    pd.read_csv(G.EXP_CSV)
    return


if __name__ == "__main__":
    app.run()
