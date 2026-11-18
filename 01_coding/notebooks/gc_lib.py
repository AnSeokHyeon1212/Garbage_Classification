"""쓰레기 분류 실험 공통 코드.

marimo 노트북과 백그라운드 실행 스크립트가 같이 쓴다.
Windows에서 DataLoader(num_workers>0)를 쓰려면 Dataset 클래스가 import 가능한 .py 모듈에 있어야 해서 분리했다.
"""
from __future__ import annotations

import json
import random
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
CODING = ROOT / "01_coding"
RESULTS = CODING / "results"
FIGURES = CODING / "figures"
HISTORY = RESULTS / "history"
CKPT = CODING / "checkpoints"
SPLIT_CSV = RESULTS / "split.csv"
VAL_EXT_CSV = RESULTS / "val_ext.csv"
EXP_CSV = RESULTS / "experiments.csv"

CLASSES = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass",
           "metal", "paper", "plastic", "shoes", "trash", "white-glass"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

EXP_COLUMNS = ["exp_id", "date", "base_exp", "changed_var", "value", "model", "img_size", "epochs",
               "batch", "lr", "optimizer", "weight_decay", "aug", "train_acc", "val_acc", "train_loss",
               "val_loss", "best_epoch", "test_acc", "test_used", "val_ext_acc", "select_score",
               "n_train", "train_data", "freeze", "epochs_run", "time_min", "notes"]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- 데이터
class GarbageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["path"].tolist()
        self.labels = [CLASS_TO_IDX[c] for c in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(ROOT / self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def build_transforms(img_size: int, aug: list[str] | tuple = ()):
    """aug 이름: hflip, rotate, rrc, color, blur. 빈 목록이면 Resize+Normalize만."""
    aug = list(aug)
    tf = []
    if "rrc" in aug:
        tf.append(transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)))
    else:
        tf.append(transforms.Resize((img_size, img_size)))
    if "hflip" in aug:
        tf.append(transforms.RandomHorizontalFlip())
    if "rotate" in aug:
        tf.append(transforms.RandomRotation(20))
    if "color" in aug:  # 유리 색이 핵심이라 hue는 아주 약하게
        tf.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05))
    if "blur" in aug:
        tf.append(transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3))
    tf += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    train_tf = transforms.Compose(tf)
    eval_tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                  transforms.Normalize(MEAN, STD)])
    return train_tf, eval_tf


def sample_per_class(df: pd.DataFrame, max_per_class: int | None, seed: int = 42) -> pd.DataFrame:
    if not max_per_class:
        return df.reset_index(drop=True)
    parts = [g.sample(n=min(len(g), max_per_class), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sort_values("path").reset_index(drop=True)


def load_train_df(cfg: dict) -> pd.DataFrame:
    if cfg["train_data"] == "kaggle":
        split = pd.read_csv(SPLIT_CSV)
        return sample_per_class(split[split["split"] == "train"], cfg.get("max_per_class"), cfg["seed"])
    if cfg["train_data"] == "mix":
        return pd.read_csv(RESULTS / "train_mix.csv")
    raise ValueError(cfg["train_data"])


def make_loader(df, tf, batch, shuffle, workers):
    return DataLoader(GarbageDataset(df, tf), batch_size=batch, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0)


# ---------------------------------------------------------------- 모델
def build_model(name: str = "resnet18", freeze: str = "backbone", dropout: float = 0.0):
    """freeze: 'backbone'(fc만 학습) | 'layer4'(layer4+fc 학습) | 'none'(전체 학습)"""
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(name)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, len(CLASSES))) if dropout > 0 \
        else nn.Linear(in_f, len(CLASSES))
    for pname, p in model.named_parameters():
        if freeze == "none":
            p.requires_grad = True
        elif freeze == "layer4":
            p.requires_grad = pname.startswith(("layer4", "fc"))
        else:
            p.requires_grad = pname.startswith("fc")
    return model


def set_train_mode(model: nn.Module):
    """학습 모드로 바꾸되, 동결된 부분의 BatchNorm은 eval로 둔다(통계값이 바뀌지 않게)."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


# ---------------------------------------------------------------- 학습/평가
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    set_train_mode(model)
    n, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            out = model(x)
            loss = criterion(out, y)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    n, correct, loss_sum = 0, 0, 0.0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        n += x.size(0)
        preds.append(p.cpu())
        labels.append(y.cpu())
    return loss_sum / n, correct / n, torch.cat(preds).numpy(), torch.cat(labels).numpy()


def per_class_acc(preds, labels) -> pd.Series:
    s = {}
    for i, c in enumerate(CLASSES):
        m = labels == i
        s[c] = float((preds[m] == i).mean()) if m.any() else float("nan")
    return pd.Series(s, name="acc")


def append_experiment(row: dict):
    df = pd.read_csv(EXP_CSV) if EXP_CSV.exists() else pd.DataFrame(columns=EXP_COLUMNS)
    df = df[df["exp_id"] != row["exp_id"]]  # 같은 ID 재실행 시 덮어씀
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)[EXP_COLUMNS]
    df.to_csv(EXP_CSV, index=False, encoding="utf-8")


def run_experiment(cfg: dict, log=print) -> dict:
    """cfg 하나로 학습 → best ckpt 저장(선택 점수 기준) → history/클래스별 정확도/experiments.csv 기록."""
    for d in (RESULTS, FIGURES, HISTORY, CKPT):
        d.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_df = load_train_df(cfg)
    split = pd.read_csv(SPLIT_CSV)
    val_df = split[split["split"] == "val"]
    val_ext_df = pd.read_csv(VAL_EXT_CSV) if VAL_EXT_CSV.exists() else None

    train_tf, eval_tf = build_transforms(cfg["img_size"], cfg.get("aug", []))
    w = cfg.get("num_workers", 0)
    train_loader = make_loader(train_df, train_tf, cfg["batch"], True, w)
    val_loader = make_loader(val_df, eval_tf, cfg["batch"] * 2, False, w)
    ext_loader = make_loader(val_ext_df, eval_tf, cfg["batch"] * 2, False, w) if val_ext_df is not None else None

    model = build_model(cfg["model"], cfg.get("freeze", "backbone"), cfg.get("dropout", 0.0)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    else:
        raise ValueError(cfg["optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"]) \
        if cfg.get("scheduler") == "cosine" else None
    scaler = torch.amp.GradScaler() if use_amp else None

    ckpt_path = CKPT / f"{cfg['exp_id']}.pth"
    hist, best, best_ep, bad = [], -1.0, 0, 0
    t0 = time.perf_counter()
    log(f"[{cfg['exp_id']}] train {len(train_df)} / val {len(val_df)} / val_ext "
        f"{0 if val_ext_df is None else len(val_ext_df)} / device {device} / threads {torch.get_num_threads()}")
    for ep in range(1, cfg["epochs"] + 1):
        te = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        ex_loss = ex_acc = float("nan")
        if ext_loader is not None:
            ex_loss, ex_acc, _, _ = evaluate(model, ext_loader, criterion, device)
        if scheduler:
            scheduler.step()
        score = va_acc if ext_loader is None else (va_acc + ex_acc) / 2  # 모델 선택 기준
        hist.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc,
                         val_ext_loss=ex_loss, val_ext_acc=ex_acc, score=score,
                         lr=optimizer.param_groups[0]["lr"], sec=time.perf_counter() - te))
        log(f"  ep{ep:02d} train {tr_loss:.4f}/{tr_acc:.4f}  val {va_loss:.4f}/{va_acc:.4f}  "
            f"val_ext {ex_acc:.4f}  score {score:.4f}  ({hist[-1]['sec']:.0f}s)")
        if score > best:
            best, best_ep, bad = score, ep, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= cfg.get("patience", 999):
                log(f"  early stop (patience {cfg['patience']})")
                break
    total_min = (time.perf_counter() - t0) / 60

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(HISTORY / f"{cfg['exp_id']}.csv", index=False)

    # best ckpt로 클래스별 val 정확도
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, _, preds, labels = evaluate(model, val_loader, criterion, device)
    pc = per_class_acc(preds, labels)
    pc.to_csv(HISTORY / f"{cfg['exp_id']}_val_per_class.csv", header=True)

    b = hist_df.loc[hist_df["epoch"] == best_ep].iloc[0]
    row = dict(exp_id=cfg["exp_id"], date=str(date.today()), base_exp=cfg.get("base_exp", ""),
               changed_var=cfg.get("changed_var", ""), value=cfg.get("value", ""), model=cfg["model"],
               img_size=cfg["img_size"], epochs=cfg["epochs"], batch=cfg["batch"], lr=cfg["lr"],
               optimizer=cfg["optimizer"], weight_decay=cfg["weight_decay"],
               aug="+".join(cfg.get("aug", [])) or "none",
               train_acc=round(b.train_acc, 4), val_acc=round(b.val_acc, 4),
               train_loss=round(b.train_loss, 4), val_loss=round(b.val_loss, 4), best_epoch=best_ep,
               test_acc="", test_used=0,
               val_ext_acc="" if np.isnan(b.val_ext_acc) else round(b.val_ext_acc, 4),
               select_score=round(best, 4), n_train=len(train_df), train_data=cfg["train_data"],
               freeze=cfg.get("freeze", "backbone"), epochs_run=len(hist_df), time_min=round(total_min, 1),
               notes=cfg.get("notes", ""))
    append_experiment(row)
    with open(HISTORY / f"{cfg['exp_id']}_cfg.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    log(f"[{cfg['exp_id']}] done: best ep {best_ep}, score {best:.4f}, {total_min:.1f} min")
    return row


@torch.no_grad()
def eval_ckpt(exp_id: str, df: pd.DataFrame, cfg: dict):
    """저장된 best ckpt를 다른 평가셋(df)으로 평가. test에는 쓰지 말 것(최종 1회 제외)."""
    device = torch.device("cpu")
    model = build_model(cfg["model"], cfg.get("freeze", "backbone"), cfg.get("dropout", 0.0))
    model.load_state_dict(torch.load(CKPT / f"{exp_id}.pth", map_location=device))
    _, eval_tf = build_transforms(cfg["img_size"])
    loader = make_loader(df, eval_tf, 64, False, cfg.get("num_workers", 0))
    loss, acc, preds, labels = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    return loss, acc, per_class_acc(preds, labels)


def plot_curve(exp_id: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    h = pd.read_csv(HISTORY / f"{exp_id}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(h.epoch, h.train_loss, "o-", label="train")
    axes[0].plot(h.epoch, h.val_loss, "s-", label="val")
    if h.val_ext_loss.notna().any():
        axes[0].plot(h.epoch, h.val_ext_loss, "^-", label="val_ext")
    axes[0].set(title=f"{exp_id} Loss", xlabel="epoch", ylabel="loss")
    axes[1].plot(h.epoch, h.train_acc, "o-", label="train")
    axes[1].plot(h.epoch, h.val_acc, "s-", label="val")
    if h.val_ext_acc.notna().any():
        axes[1].plot(h.epoch, h.val_ext_acc, "^-", label="val_ext")
    axes[1].set(title=f"{exp_id} Accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{exp_id}_curve.png", dpi=200)
    plt.close(fig)
    return FIGURES / f"{exp_id}_curve.png"
