"""야간 실험 2차 (overnight.py 종료 후 실행): python overnight2.py
E8: 그 시점 최고 모델 + layer4 동결 해제 (freeze backbone → layer4)   — 채택: S3 +1%p & clothes/shoes 가드
E9: E8 이후 최고 모델 + 입력 224px (img_size 160 → 224)            — 채택: (val+val_ext)/2 +1%p & 가드
    (val_hard는 160px로 고정 생성돼 224 모델에 불리 → 참고로만 보고. 실행 전 사전 등록)
마감 07:50 전에 끝나지 않을 실험은 시작하지 않음 (예상: E8 65분, E9 80분 / layer4+224면 100분)
"""
import pandas as pd

import overnight as O


def beats_s2(c, b):
    mc, mb = O.metrics(c), O.metrics(b)
    s2 = lambda m: (m["val"] + m["val_ext"]) / 2
    guard = all(mc["pc"][k] >= mb["pc"][k] - 0.03 for k in ("clothes", "shoes"))
    ok = s2(mc) >= s2(mb) + 0.01 and guard
    O.say(f"  비교 {c}((val+val_ext)/2 {s2(mc):.4f}, val_hard {mc['val_hard']:.4f}) vs "
          f"{b}({s2(mb):.4f}, val_hard {mb['val_hard']:.4f}): guard={guard} → {'채택' if ok else '유지'}")
    return ok


def main():
    st = O.load_state()
    O.say("야간 실험 2차 시작 (E8 layer4, E9 224px)")
    b = st["best"]
    O.EST_MIN = 65
    c = dict(O.cfg_of(b), exp_id="E8", base_exp=b, changed_var="freeze", value="layer4",
             freeze="layer4", notes=f"{b} + layer4 동결 해제(layer4+헤드 학습)")
    if O.run(c, st) and O.beats("E8", b):
        st["best"] = "E8"
    O.save_state(st)

    b = st["best"]
    base = O.cfg_of(b)
    O.EST_MIN = 100 if base.get("freeze") == "layer4" else 80
    c = dict(base, exp_id="E9", base_exp=b, changed_var="img_size", value="224", img_size=224,
             notes=f"{b} + 입력 224px (선택 기준 (val+val_ext)/2, val_hard 참고)")
    if O.run(c, st) and beats_s2("E9", b):
        st["best"] = "E9"
    O.save_state(st)
    O.say(f"2차 종료. 최고 = {st['best']}")


if __name__ == "__main__":
    main()
