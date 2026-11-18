"""야간 실험 자동 실행 (11/19): python overnight.py
순서: E4(은닉1) → E5(은닉2) → E6(E4/E5 승자 + GELU) → E7(최고 설정 + degrade_strong)
- 한 번에 하나씩 학습. 각 실험 후 val_hard 평가 → S3 = (val + val_ext + val_hard)/3
- 채택 규칙: 현재 최고보다 S3 +1%p 이상 & clothes/shoes val -3%p 이내. E7은 val_ext도 하락 없어야 함
- 07:50까지 끝나지 않을 실험(예상 55분)은 시작하지 않음. 오류는 1회 재시도 후 건너뜀
- 진행 상태: RESULT/experiments/overnight_state.json (중단 시 다시 실행하면 완료된 실험은 건너뜀)
"""
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gc_lib as G
import eval_val_hard as V
from run_exp import EXPS

CUTOFF = dt.datetime(2026, 11, 19, 7, 50)
EST_MIN = 55
STATE = G.EXP_DIR / "overnight_state.json"


def now():
    return dt.datetime.now()


def say(msg):
    line = f"[{now():%H:%M}] {msg}"
    print(line, flush=True)
    with open(G.LOG_DIR / "overnight_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    return json.load(open(STATE, encoding="utf-8")) if STATE.exists() else {"done": [], "failed": {}, "best": "E3"}


def save_state(st):
    json.dump(st, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def metrics(e):
    r = pd.read_csv(G.EXP_CSV).set_index("exp_id").loc[e]
    vh = json.load(open(G.HISTORY / f"{e}_val_hard.json", encoding="utf-8"))["val_hard_acc"]
    pc = pd.read_csv(G.HISTORY / f"{e}_val_per_class.csv", index_col=0)["acc"]
    return dict(val=float(r.val_acc), val_ext=float(r.val_ext_acc), val_hard=vh,
                s3=(float(r.val_acc) + float(r.val_ext_acc) + vh) / 3, pc=pc)


def beats(c, b, need_ext=False):
    mc, mb = metrics(c), metrics(b)
    guard = all(mc["pc"][k] >= mb["pc"][k] - 0.03 for k in ("clothes", "shoes"))
    ok = mc["s3"] >= mb["s3"] + 0.01 and guard and (not need_ext or mc["val_ext"] >= mb["val_ext"])
    say(f"  비교 {c}(S3 {mc['s3']:.4f}) vs {b}(S3 {mb['s3']:.4f}): guard={guard} → {'채택' if ok else '유지'}")
    return ok


def run(cfg, st):
    e = cfg["exp_id"]
    if e in st["done"]:
        say(f"{e} 이미 완료 — 건너뜀")
        return True
    if now() + dt.timedelta(minutes=EST_MIN) > CUTOFF:
        say(f"{e} 시작 안 함: 마감(07:50) 전에 끝나지 않을 것으로 예상")
        st["failed"][e] = "시간 부족으로 미실행"
        save_state(st)
        return False
    for attempt in (1, 2):
        try:
            say(f"{e} 시작 (시도 {attempt}) — {cfg['changed_var']}={cfg['value']}")
            with open(G.LOG_DIR / f"history_{e}.log", "w", encoding="utf-8") as f:
                def log(s):
                    print(s, flush=True)
                    f.write(s + "\n")
                    f.flush()
                G.run_experiment(cfg, log=log)
            V.main([e])
            st["done"].append(e)
            st["failed"].pop(e, None)
            save_state(st)
            m = metrics(e)
            say(f"{e} 완료: val {m['val']:.4f} / val_ext {m['val_ext']:.4f} / val_hard {m['val_hard']:.4f} / S3 {m['s3']:.4f}")
            return True
        except Exception as ex:  # noqa: BLE001
            say(f"{e} 오류 (시도 {attempt}): {ex!r}")
            with open(G.LOG_DIR / f"error_{e}.txt", "a", encoding="utf-8") as f:
                f.write(traceback.format_exc() + "\n")
            st["failed"][e] = repr(ex)
            save_state(st)
    return False


def cfg_of(e):
    return json.load(open(G.HISTORY / f"{e}_cfg.json", encoding="utf-8"))


def main():
    st = load_state()
    say("야간 실험 시작")
    for e in ("E4", "E5"):
        run(dict(EXPS[e]), st)

    hid = [e for e in ("E4", "E5") if e in st["done"]]
    for e in hid:  # 최고 모델 갱신 (기준 E3)
        if beats(e, st["best"]):
            st["best"] = e
    save_state(st)

    if hid:
        hw = max(hid, key=lambda e: metrics(e)["s3"])
        st["hidden_winner"] = hw
        c = dict(cfg_of(hw), exp_id="E6", base_exp=hw, changed_var="act", value="gelu", act="gelu",
                 notes=f"{hw} 구조 + GELU (ReLU→GELU만 변경)")
        if run(c, st) and beats("E6", st["best"]):
            st["best"] = "E6"
        save_state(st)

    b = st["best"]
    aug = ["degrade_strong" if a == "degrade" else a for a in cfg_of(b).get("aug", [])]
    c = dict(cfg_of(b), exp_id="E7", base_exp=b, changed_var="aug", value="degrade_strong(밝기 0.15~1.0)",
             aug=aug, notes=f"{b} + 어둡게 범위 확대(0.3→0.15)")
    if run(c, st) and beats("E7", b, need_ext=True):
        st["best"] = "E7"
    save_state(st)
    say(f"단일 모델 탐색 종료. 최고 = {st['best']}")


if __name__ == "__main__":
    main()
