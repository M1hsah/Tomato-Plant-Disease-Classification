"""Train and evaluate tomato disease classifiers (everything except data prep).

Sections: utils, transforms, dataset, models, training, evaluation, CLI.
Reads the fixed split from data/manifest.csv (built once by src/prepare.py).

Usage:
    python -m src.train --config configs/E1.yaml              # every seed listed in the config
    python -m src.train --config configs/E1.yaml --seed 0     # one seed
    python -m src.train --config configs/E1.yaml --debug      # tiny subset, 1 epoch, nothing logged
    python -m src.train --config configs/E1.yaml --overfit    # sanity check: memorise 32 images
    python -m src.train --evaluate outputs/runs/<run_name>    # re-evaluate a saved run
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display (Colab / servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from src.prepare import LABELS, NUM_CLASSES

# Defaults for every config key; an experiment YAML only needs to override what differs.
DEFAULT_CONFIG = {
    "experiment": "E1",
    "model": "simple_cnn",
    "aug": "none",
    "cap": 250,             # max PlantVillage training images per class; null = uncapped
    "seeds": [0, 1, 2],
    "epochs": 20,
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "patience": 4,          # early stopping: epochs without val macro-F1 improvement
    "num_workers": 4,
    "n_bootstrap": 1000,    # resamples for the macro-F1 confidence interval
    "data_dir": "data/raw",
    "manifest": "data/manifest.csv",
    "output_dir": "outputs",
}

# The three sets every model is evaluated on (CLAUDE.md section 9).
EVAL_SETS = ["plantvillage_test", "plantdoc_field", "plantdoc_official_test"]


# =========================================================================== #
# Utils
# =========================================================================== #
def set_seed(seed: int) -> None:
    """Seed every random number generator we use, so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    """Give each DataLoader worker its own reproducible seed (for random augmentations)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Use a CUDA GPU (Colab), else Apple Silicon GPU, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Wait for queued GPU work to finish, so timings measure the real compute."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_config(path: Path) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    return cfg


def make_run_name(cfg: dict, seed: int) -> str:
    return f"{cfg['experiment']}_{cfg['model']}_aug-{cfg['aug']}_cap-{cfg['cap']}_s{seed}"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =========================================================================== #
# Transforms
# =========================================================================== #
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def to_normalized_tensor() -> list:
    return [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def get_transform(aug: str, train: bool) -> v2.Compose:
    """Image preprocessing by name: 'none', 'standard' or 'strong' (CLAUDE.md section 6).

    Evaluation always uses the plain resize + center crop, whatever `aug` is.
    """
    eval_tf = v2.Compose([v2.Resize(256), v2.CenterCrop(224), *to_normalized_tensor()])
    if not train or aug == "none":
        return eval_tf
    if aug == "standard":
        return v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.6, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(30),
            v2.ColorJitter(0.3, 0.3, 0.3, 0.05),
            *to_normalized_tensor(),
        ])
    if aug == "strong":
        # Goal: stop the model relying on PlantVillage's uniform backgrounds and lighting.
        return v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.6, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(30),
            v2.ColorJitter(0.5, 0.5, 0.5, 0.1),
            v2.RandomGrayscale(p=0.1),
            v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.3),
            *to_normalized_tensor(),
            v2.RandomErasing(p=0.5),
        ])
    raise ValueError(f"unknown aug '{aug}'")


# =========================================================================== #
# Dataset
# =========================================================================== #
def get_split(manifest: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rows of the manifest for one named split.

    PlantVillage train/val are the only data used for fitting and model
    selection; the PlantDoc sets are test-only (never trained or tuned on).
    """
    pv = manifest["source"] == "plantvillage"
    field = manifest["split"] == "field_test"
    masks = {
        "train": pv & (manifest["split"] == "train"),
        "val": pv & (manifest["split"] == "val"),
        "plantvillage_test": pv & (manifest["split"] == "test"),
        "plantdoc_field": field,
        "plantdoc_official_test": field & (manifest["orig_split"] == "test"),
    }
    return manifest[masks[name]].reset_index(drop=True)


def cap_per_class(frame: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    """Limit training images per class by sampling whole leaf groups.

    Sampling groups (not single images) means the cap counts distinct leaves
    rather than several photos of the same leaf. Classes already below the cap
    keep everything.
    """
    if cap is None:
        return frame
    rng = np.random.default_rng(seed)
    parts = []
    for _, sub in frame.groupby("label_idx"):
        sizes = sub.groupby("group_id").size()
        groups = np.sort(sub["group_id"].unique())
        rng.shuffle(groups)
        chosen, total = [], 0
        for g in groups:
            if total >= cap:
                break
            chosen.append(g)
            total += sizes[g]
        parts.append(sub[sub["group_id"].isin(chosen)])
    return pd.concat(parts).reset_index(drop=True)


class ManifestDataset(Dataset):
    """Loads images listed in a manifest frame. Labels come only from the manifest."""

    def __init__(self, frame: pd.DataFrame, data_dir: Path, transform):
        self.paths = [str(Path(data_dir) / p) for p in frame["path"]]
        self.labels = frame["label_idx"].to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as img:
            image = img.convert("RGB")
        return self.transform(image), int(self.labels[i])


def make_loader(frame, cfg, transform, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    return DataLoader(
        ManifestDataset(frame, cfg["data_dir"], transform),
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )


def print_class_counts(name: str, frame: pd.DataFrame) -> None:
    counts = frame["label_idx"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    print(f"{name}: {len(frame)} images  " + ", ".join(f"{LABELS[i]}={n}" for i, n in counts.items()))


# =========================================================================== #
# Models
# =========================================================================== #
def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """conv3x3 -> BatchNorm -> ReLU -> 2x2 maxpool (halves the image size)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Baseline trained from scratch: 4 conv blocks, global average pool, dropout, linear."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, 256),
        )
        # Global average pooling makes the head independent of image size and
        # keeps the parameter count small (less overfitting than a big flatten + FC).
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


def build_model(name: str) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(NUM_CLASSES)
    raise ValueError(f"model '{name}' is not implemented yet (resnet18 / efficientnet_b0 come with E2)")


# =========================================================================== #
# Training
# =========================================================================== #
def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> tuple[float, float]:
    """One pass over the training data. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # Mixed precision (CUDA only) roughly halves memory and speeds up training.
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities and true labels for every image in the loader."""
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        probs.append(torch.softmax(model(images.to(device)), dim=1).float().cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def train_run(cfg: dict, seed: int, debug: bool = False) -> Path:
    """Train one model on PlantVillage, select the best epoch on PlantVillage val,
    save the checkpoint, then evaluate it. Returns the run folder."""
    set_seed(seed)
    device = get_device()
    name = ("debug_" if debug else "") + make_run_name(cfg, seed)
    out = Path(cfg["output_dir"])
    run_dir = out / "runs" / name
    ckpt_path = out / "checkpoints" / f"{name}.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name}  (device: {device})")

    manifest = pd.read_csv(cfg["manifest"])
    train_df = cap_per_class(get_split(manifest, "train"), cfg["cap"], seed)
    val_df = get_split(manifest, "val")
    epochs = cfg["epochs"]
    if debug:  # a few images per class, one epoch: just checks the code runs end to end
        train_df = train_df.groupby("label_idx").head(8)
        val_df = val_df.groupby("label_idx").head(4)
        epochs = 1
    print_class_counts("train", train_df)
    print_class_counts("val", val_df)

    train_loader = make_loader(train_df, cfg, get_transform(cfg["aug"], train=True), True, seed, device)
    val_loader = make_loader(val_df, cfg, get_transform(cfg["aug"], train=False), False, seed, device)

    model = build_model(cfg["model"]).to(device)
    params = count_params(model)
    print(f"model: {cfg['model']}, {params:,} parameters")

    # Uncapped PlantVillage is very imbalanced (yellow leaf curl is ~35%), so
    # weight the loss to stop the model favouring the biggest class.
    weight = None
    if cfg["cap"] is None:
        w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_df["label_idx"])
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Early stopping on PlantVillage val macro-F1. PlantDoc is never looked at
    # here: using it to pick epochs would leak the field test set into training.
    best_f1, best_epoch, history = -1.0, 0, []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_probs, val_labels = predict(model, val_loader, device)
        val_f1 = macro_f1(val_labels, val_probs.argmax(1))
        val_acc = float((val_probs.argmax(1) == val_labels).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_acc": val_acc, "val_macro_f1": val_f1, "lr": scheduler.get_last_lr()[0]})
        scheduler.step()
        improved = val_f1 > best_f1
        if improved:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
        print(f"epoch {epoch:2d}  loss {train_loss:.3f}  train acc {train_acc:.3f}  "
              f"val acc {val_acc:.3f}  val macro-F1 {val_f1:.3f}{'  *' if improved else ''}")
        if epoch - best_epoch >= cfg["patience"]:
            print(f"early stopping: no val improvement for {cfg['patience']} epochs")
            break
    train_time = time.perf_counter() - start

    run_cfg = {**cfg, "seed": seed, "run_name": name, "debug": debug, "checkpoint": str(ckpt_path),
               "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "params": params,
               "train_time_s": round(train_time, 1), "n_train": len(train_df)}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False))
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    print(f"best epoch {best_epoch} (val macro-F1 {best_f1:.3f}), trained in {train_time:.0f}s")

    evaluate_run(run_dir)
    return run_dir


def overfit_check(cfg: dict, seed: int, steps: int = 200) -> None:
    """Sanity check before real experiments: a working model + training loop
    should be able to memorise a single batch of 32 images (~100% accuracy).
    If it cannot, something is broken (labels, transforms, loss, optimizer)."""
    set_seed(seed)
    device = get_device()
    manifest = pd.read_csv(cfg["manifest"])
    batch_df = get_split(manifest, "train").groupby("label_idx").head(4)  # 4 per class = 32
    dataset = ManifestDataset(batch_df, cfg["data_dir"], get_transform("none", train=False))
    images = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    labels = torch.tensor(dataset.labels, device=device)

    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    print(f"Overfitting {len(labels)} images with {cfg['model']} on {device}")
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(images).argmax(1) == labels).float().mean().item()
            print(f"step {step:3d}  loss {loss.item():.4f}  acc {acc:.3f}")
    print("PASS: model can fit a single batch" if acc >= 0.95 else
          "FAIL: could not fit a single batch, check the pipeline before training")


# =========================================================================== #
# Evaluation
# =========================================================================== #
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over all 8 classes (a class with no true or predicted images scores 0).

    Written with numpy instead of sklearn because the bootstrap calls it 1,000
    times per eval set; it matches sklearn's f1_score(average='macro').
    """
    cm = np.bincount(y_true * NUM_CLASSES + y_pred, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)  # predicted + actual per class
    f1 = np.divide(2 * tp, denom, out=np.zeros(NUM_CLASSES), where=denom > 0)
    return float(f1.mean())


def bootstrap_ci(y_true, y_pred, n_boot: int, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for macro-F1 by resampling images with replacement.

    The PlantDoc sets are small, so a single macro-F1 number can move a lot by
    chance; the interval shows how much.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [macro_f1(y_true[idx], y_pred[idx]) for idx in rng.integers(0, n, size=(n_boot, n))]
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, n_boot: int) -> dict:
    preds = probs.argmax(1)
    top2 = np.argsort(-probs, axis=1)[:, :2]
    ci_low, ci_high = bootstrap_ci(labels, preds, n_boot)
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": macro_f1(labels, preds),
        "macro_f1_ci_low": ci_low,
        "macro_f1_ci_high": ci_high,
        "weighted_f1": float(f1_score(labels, preds, labels=range(NUM_CLASSES), average="weighted",
                                      zero_division=0)),
        "top2_acc": float((top2 == labels[:, None]).any(1).mean()),
        "n_images": int(len(labels)),
    }


def save_confusion(labels, preds, run_dir: Path, eval_set: str) -> None:
    """Confusion matrix as CSV (counts) and PNG (colour = row %, so small classes are readable)."""
    cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(run_dir / f"confusion_{eval_set}.csv")
    row_pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if row_pct[i, j] > 0.5 else "black")
    ax.set_xticks(range(NUM_CLASSES), LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), LABELS, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{run_dir.name}\n{eval_set}", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / f"confusion_{eval_set}.png", dpi=120)
    plt.close(fig)


@torch.no_grad()
def time_inference(model, device, n: int = 50) -> float:
    """Milliseconds per image at batch size 1 (as a phone app would run it)."""
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(5):  # warm-up
        model(x)
    sync(device)
    start = time.perf_counter()
    for _ in range(n):
        model(x)
    sync(device)
    return (time.perf_counter() - start) / n * 1000


def append_results(rows: list[dict], path: Path) -> None:
    """Add this run's rows to results.csv, replacing any earlier rows for the same run."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["run_name"].isin(new["run_name"])], new])
    new.to_csv(path, index=False)


def evaluate_run(run_dir: Path) -> dict:
    """Evaluate a saved run on PlantVillage test and both PlantDoc sets."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    device = get_device()
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device)["model"])
    manifest = pd.read_csv(cfg["manifest"])
    eval_tf = get_transform(cfg["aug"], train=False)

    metrics, rows = {}, []
    for eval_set in EVAL_SETS:
        frame = get_split(manifest, eval_set)
        if cfg["debug"]:
            frame = frame.groupby("label_idx").head(4)
        probs, labels = predict(model, make_loader(frame, cfg, eval_tf, False, cfg["seed"], device), device)
        preds = probs.argmax(1)
        m = compute_metrics(probs, labels, cfg["n_bootstrap"])
        m["per_class"] = classification_report(labels, preds, labels=range(NUM_CLASSES), target_names=LABELS,
                                               zero_division=0, output_dict=True)
        (run_dir / f"report_{eval_set}.txt").write_text(classification_report(
            labels, preds, labels=range(NUM_CLASSES), target_names=LABELS, zero_division=0, digits=3))
        save_confusion(labels, preds, run_dir, eval_set)
        metrics[eval_set] = m
        rows.append({
            "run_name": cfg["run_name"], "experiment": cfg["experiment"], "model": cfg["model"],
            "aug": cfg["aug"], "cap": cfg["cap"], "seed": cfg["seed"], "eval_set": eval_set,
            **{k: m[k] for k in ["accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
                                 "weighted_f1", "top2_acc", "n_images"]},
            "params": cfg["params"], "train_time_s": cfg["train_time_s"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    # The headline number: how much macro-F1 is lost going from lab to field images.
    metrics["domain_gap"] = metrics["plantvillage_test"]["macro_f1"] - metrics["plantdoc_field"]["macro_f1"]
    metrics["inference_ms_per_image"] = time_inference(model, device)
    metrics["device"] = str(device)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n{'eval set':<24}{'n':>5}{'acc':>8}{'macro-F1':>10}{'95% CI':>16}{'top-2':>8}")
    for eval_set in EVAL_SETS:
        m = metrics[eval_set]
        print(f"{eval_set:<24}{m['n_images']:>5}{m['accuracy']:>8.3f}{m['macro_f1']:>10.3f}"
              f"{'[' + format(m['macro_f1_ci_low'], '.3f') + ', ' + format(m['macro_f1_ci_high'], '.3f') + ']':>16}"
              f"{m['top2_acc']:>8.3f}")
    print(f"domain gap (PV test - PlantDoc field macro-F1): {metrics['domain_gap']:.3f}")
    print(f"inference: {metrics['inference_ms_per_image']:.1f} ms/image on {device}")

    if not cfg["debug"]:  # debug runs use tiny subsets, so keep them out of the results table
        append_results(rows, Path(cfg["output_dir"]) / "results.csv")
    return metrics


# =========================================================================== #
# CLI
# =========================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="experiment YAML, e.g. configs/E1.yaml")
    parser.add_argument("--seed", type=int, help="run only this seed (default: all seeds in the config)")
    parser.add_argument("--debug", action="store_true", help="tiny subset, 1 epoch, CPU-friendly")
    parser.add_argument("--overfit", action="store_true", help="sanity check: fit one batch of 32 images")
    parser.add_argument("--evaluate", type=Path, metavar="RUN_DIR", help="re-evaluate a saved run")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_run(args.evaluate)
        return
    if not args.config:
        parser.error("--config is required unless --evaluate is given")
    cfg = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    if args.overfit:
        overfit_check(cfg, seeds[0])
        return
    if args.debug:
        cfg["num_workers"] = 0  # avoids worker start-up cost for a tiny run
        seeds = seeds[:1]
    for seed in seeds:
        train_run(cfg, seed, debug=args.debug)


if __name__ == "__main__":
    main()
