"""
Train the facade parser (real CMP photo -> 12 classes) used for the FCN-score.

Train split only (400 photos); checkpoint selected on validation mIoU (100 photos). The test split
is never used here. The parser's own test-split score on REAL photos is reported by the evaluation
as the upper bound of the FCN-score.

    python -m src.training.train_facade_parser --epochs 40
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import MODELS_DIR
from src.data.facade_dataset import CMPFacadesDataset
from src.evaluation.metrics import ConfusionMatrix
from src.models.facade_parser import CMP_CLASSES, build_parser, labels_from_tensor, parse, parser_input

CHECKPOINT = MODELS_DIR / "checkpoints" / "facade_parser_best.pt"
METRICS_DIR = MODELS_DIR / "metrics"


def evaluate_parser(parser, loader, device) -> dict:
    parser.eval()
    cm = ConfusionMatrix(len(CMP_CLASSES), 255, device)
    for masks, photos in loader:
        cm.update(parse(parser, photos.to(device)), labels_from_tensor(masks).to(device))
    return cm.compute(CMP_CLASSES)


def train(epochs: int = 40, batch_size: int = 8, lr: float = 2e-4, workers: int = 2, seed: int = 42,
          time_budget_min: float | None = None) -> dict:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = CMPFacadesDataset("train", augment=True, jitter=True)
    val_ds = CMPFacadesDataset("val", augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    parser = build_parser().to(device)
    opt = torch.optim.AdamW(parser.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    best, best_res, t0 = -1.0, None, time.time()
    for epoch in range(1, epochs + 1):
        parser.train()
        total = 0.0
        for masks, photos in train_dl:
            target = labels_from_tensor(masks).to(device)
            out = parser(parser_input(photos.to(device)))["out"]
            loss = loss_fn(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        res = evaluate_parser(parser, val_dl, device)
        print(f"[Parser] Epoch {epoch}/{epochs} | loss={total / len(train_dl):.4f} | val_miou={res['miou']:.4f} "
              f"| val_acc={res['pixel_acc']:.4f}", flush=True)
        if res["miou"] > best:
            best, best_res = res["miou"], {**res, "epoch": epoch}
            CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": parser.state_dict(), "epoch": epoch, "val": res,
                        "data_source": "real:cmp_facades", "classes": CMP_CLASSES}, CHECKPOINT)
        if time_budget_min and (time.time() - t0) / 60 > time_budget_min:
            print("[Parser] time budget reached", flush=True)
            break
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"model": "deeplabv3_resnet50 (ImageNet backbone)", "split": "val", "best": best_res,
               "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed, "checkpoint": str(CHECKPOINT)}
    (METRICS_DIR / "facade_parser_val.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def load_parser(path: Path = CHECKPOINT, device: str = "cpu"):
    if not Path(path).exists():
        return None
    parser = build_parser(pretrained_backbone=False)
    parser.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    return parser.to(device).eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the CMP facade parser for the FCN-score")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--time-budget-min", type=float, default=None)
    a = ap.parse_args()
    s = train(a.epochs, a.batch_size, workers=a.workers, time_budget_min=a.time_budget_min)
    print(f"Parser best val mIoU={s['best']['miou']:.4f} (epoch {s['best']['epoch']})")
