"""Part 2: fine-tune DistilBERT on BANKING77.

Usage:
  python src/finetune.py train --lr 5e-5 [--epochs 5] [--batch-size 32] [--seed 42]
                               [--class-weight none|balanced]
      Train on train, score validation after every epoch, keep the epoch with the best
      validation macro F1 (saved to models/<run>/). Logs to MLflow, results/history.md
      and results/finetune_<ts>.json.
  python src/finetune.py test --model-dir models/<run>
      Score a saved model once on test and measure latency. Final numbers only.

Optimiser settings follow the Hugging Face Trainer defaults (AdamW, no weight decay,
no warmup, learning rate falling linearly to 0 over all epochs), so the only things
we change are the ones we decided to change.
"""

import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128  # longest train message is well under this; nothing is cut off
EVAL_BATCH = 128

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
HISTORY = RESULTS_DIR / "history.md"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def versions() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mlflow": mlflow.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
    }


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(path)


def log_history(stamp: str, split: str, acc: float, f1: float,
                latency: str, notes: str) -> None:
    row = (f"| {stamp} | {split} | DistilBERT | {acc:.4f} | {f1:.4f} | "
           f"{latency} | – | {notes} |\n")
    with HISTORY.open("a") as f:
        f.write(row)


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], dev) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted label ids, softmax probabilities) for a list of messages."""
    model.eval()
    probs = []
    for i in range(0, len(texts), EVAL_BATCH):
        batch = tokenizer(texts[i:i + EVAL_BATCH], padding=True, truncation=True,
                          max_length=MAX_LENGTH, return_tensors="pt").to(dev)
        probs.append(torch.softmax(model(**batch).logits, dim=-1).cpu())
    probs = torch.cat(probs).numpy()
    return probs.argmax(axis=1), probs


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Same formula as scikit-learn's class_weight="balanced"."""
    counts = np.bincount(y, minlength=n_classes)
    return torch.tensor(len(y) / (n_classes * counts), dtype=torch.float32)


def fit(train_df: pd.DataFrame, val_df: pd.DataFrame, lr: float, batch_size: int,
        epochs: int, seed: int, class_weight: str = "none", patience: int | None = None,
        min_epochs: int = 0, train_accuracy: bool = True, on_epoch=None):
    """Fine-tune DistilBERT on train_df, score val_df after every epoch, and return
    (model with the best epoch's weights, tokenizer, per-epoch rows, best row).
    patience: stop when val macro F1 hasn't improved for this many epochs, but never
    before min_epochs (small data can sit at chance level for the first epochs)."""
    set_seed(seed)
    dev = device()
    labels = sorted(train_df["label"].unique())
    label_id = {label: i for i, label in enumerate(labels)}
    y_train = train_df["label"].map(label_id).to_numpy()
    y_val = val_df["label"].map(label_id).to_numpy()
    texts = train_df["text"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=label_id,
    ).to(dev)

    steps_per_epoch = int(np.ceil(len(texts) / batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=steps_per_epoch * epochs)
    weight = class_weights(y_train, len(labels)).to(dev) if class_weight == "balanced" else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    rows, best, best_state = [], None, None
    shuffler = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = shuffler.permutation(len(texts))  # new order every epoch
        start, total_loss = time.perf_counter(), 0.0
        for step in range(steps_per_epoch):
            idx = order[step * batch_size:(step + 1) * batch_size]
            batch = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                              max_length=MAX_LENGTH, return_tensors="pt").to(dev)
            logits = model(**batch).logits
            loss = loss_fn(logits, torch.tensor(y_train[idx], device=dev))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        train_seconds = time.perf_counter() - start

        pred, probs = predict(model, tokenizer, val_df["text"].tolist(), dev)
        val_loss = float(-np.log(probs[np.arange(len(y_val)), y_val] + 1e-12).mean())
        row = {"epoch": epoch, "train_loss": total_loss / steps_per_epoch}
        if train_accuracy:
            # Train accuracy with the end-of-epoch model: a big, growing gap to val
            # accuracy would be the sign of overfitting.
            train_pred, _ = predict(model, tokenizer, texts, dev)
            row["train_accuracy"] = accuracy_score(y_train, train_pred)
        row.update({
            "val_loss": val_loss,
            "val_accuracy": accuracy_score(y_val, pred),
            "val_macro_f1": f1_score(y_val, pred, average="macro"),
            "train_seconds": train_seconds,
        })
        rows.append(row)
        if on_epoch:
            on_epoch(row)

        if best is None or row["val_macro_f1"] > best["val_macro_f1"]:
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif patience and epoch >= min_epochs and epoch - best["epoch"] >= patience:
            break

    model.load_state_dict(best_state)
    return model, tokenizer, rows, best


def train(args) -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    stamp = utc_stamp()
    run_name = (f"finetune_lr{args.lr:g}_bs{args.batch_size}_cw-{args.class_weight}"
                f"_seed{args.seed}")
    out_dir = MODELS_DIR / f"{run_name}_{stamp}"
    params = {"base_model": BASE_MODEL, "lr": args.lr, "epochs": args.epochs,
              "batch_size": args.batch_size, "seed": args.seed,
              "class_weight": args.class_weight, "max_length": MAX_LENGTH,
              "device": str(device())}

    def report(row: dict) -> None:
        mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])
        print(f"epoch {row['epoch']}: train loss {row['train_loss']:.3f}  "
              f"train acc {row['train_accuracy']:.4f}  "
              f"val loss {row['val_loss']:.3f}  val acc {row['val_accuracy']:.4f}  "
              f"F1 {row['val_macro_f1']:.4f}  ({row['train_seconds'] / 60:.1f} min)", flush=True)

    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("banking77-distilbert")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        model, tokenizer, epochs, best = fit(
            train_df, val_df, args.lr, args.batch_size, args.epochs, args.seed,
            args.class_weight, on_epoch=report)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        mlflow.log_metrics({"best_epoch": best["epoch"],
                            "best_val_accuracy": best["val_accuracy"],
                            "best_val_macro_f1": best["val_macro_f1"]})
        path = RESULTS_DIR / f"{run_name}_{stamp}.json"
        save_json(path, {"run": run_name, "timestamp_utc": stamp, "split": "val",
                         "params": params, "versions": versions(), "epochs": epochs,
                         "best_epoch": best["epoch"], "model_dir": str(out_dir.relative_to(ROOT))})
        mlflow.log_artifact(str(path))

    log_history(stamp, "val", best["val_accuracy"], best["val_macro_f1"], "–",
                f"lr={args.lr:g}, batch {args.batch_size}, class weights {args.class_weight}, "
                f"seed {args.seed}; best epoch {best['epoch']}/{args.epochs}. {path.name}")
    print(f"\nBest epoch {best['epoch']}: val acc {best['val_accuracy']:.4f}, "
          f"F1 {best['val_macro_f1']:.4f}. Model in {out_dir.relative_to(ROOT)}")


def test(args) -> None:
    model_dir = Path(args.model_dir)
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    texts = test_df["text"].tolist()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    results = {}
    for dev_name in ["mps", "cpu"]:
        dev = torch.device(dev_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev)
        id2label = model.config.id2label

        start = time.perf_counter()
        pred, probs = predict(model, tokenizer, texts, dev)
        batch_ms = (time.perf_counter() - start) * 1000 / len(texts)

        # One message at a time (batch size 1), as a live /classify endpoint sees them.
        predict(model, tokenizer, texts[:20], dev)  # warm-up
        single = []
        for text in texts:
            start = time.perf_counter()
            predict(model, tokenizer, [text], dev)
            single.append((time.perf_counter() - start) * 1000)
        single = np.array(single)
        results[dev_name] = {
            "batch_128_ms_per_query": batch_ms,
            "batch_size_1_mean_ms": float(single.mean()),
            "batch_size_1_p50_ms": float(np.percentile(single, 50)),
            "batch_size_1_p95_ms": float(np.percentile(single, 95)),
        }
        print(f"{dev_name}: batch size 1 mean {single.mean():.2f} ms, "
              f"p95 {np.percentile(single, 95):.2f} ms; batch 128: {batch_ms:.3f} ms/query")

    pred_labels = [id2label[int(p)] for p in pred]
    acc = accuracy_score(test_df["label"], pred_labels)
    f1 = f1_score(test_df["label"], pred_labels, average="macro")
    print(f"Test accuracy {acc:.4f}  macro F1 {f1:.4f}  "
          f"({int((test_df['label'] != pred_labels).sum())} wrong of {len(test_df)})")

    stamp = utc_stamp()
    path = RESULTS_DIR / f"finetune_test_{stamp}.json"
    save_json(path, {
        "run": "finetune_test", "timestamp_utc": stamp, "split": "test",
        "model_dir": str(model_dir), "versions": versions(),
        "metrics": {"accuracy": acc, "macro_f1": f1},
        "latency": results,
        "predictions": [
            {"text": t, "label": y, "pred": p, "confidence": float(c)}
            for t, y, p, c in zip(texts, test_df["label"], pred_labels, probs.max(axis=1))
        ],
    })
    log_history(stamp, "**test**", acc, f1,
                f"{results['cpu']['batch_size_1_mean_ms']:.2f} ms CPU / "
                f"{results['mps']['batch_size_1_mean_ms']:.2f} ms MPS (batch 1, laptop)",
                f"final DistilBERT: {model_dir.name}. {path.name}")
    print(f"Saved {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--lr", type=float, required=True)
    tr.add_argument("--epochs", type=int, default=5)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    te = sub.add_parser("test")
    te.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    train(args) if args.stage == "train" else test(args)


if __name__ == "__main__":
    main()
