"""Part 4: learning curve. How good are the trained models with little labelled data?

Usage:
  python src/learning_curve.py
      Full run: every size x seed, baseline and DistilBERT.
  python src/learning_curve.py --rerun results/learning_curve_<ts>.json 43:5 44:25
      Re-train only the listed DistilBERT runs (seed:per_intent), keep all other results
      from the given file, and save everything as a new file. The old file is kept.

For 5 / 10 / 25 / 50 train messages per intent and 3 seeds, train the baseline and
DistilBERT on that subset and score them on test. No API calls: the LLM is compared
through its saved 5-shot result.

- Subsets are nested (the 10 set starts with the 5 set, ...). With seed 42 the 5-per-
  intent set is exactly the 385 examples the LLM saw in its 5-shot prompt.
- Each seed draws different examples and trains with different randomness.
- Settings are the final ones from Parts 1 and 2, nothing re-tuned. DistilBERT gets about
  as many training steps as the full-data run (5 epochs x 282 steps), at most 30 epochs;
  best epoch on validation, stop after 3 epochs without improvement but never before
  epoch 10 (a first version stopped as early as epoch 4, which ended a run that was
  still at chance level on 5 examples per intent). Caveat: the full
  validation set (1,001) picks the epoch, which a team with 5 labels per intent wouldn't
  have, so small-data DistilBERT numbers are slightly optimistic.
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import baseline
import finetune

SIZES = [5, 10, 25, 50]
SEEDS = [42, 43, 44]
FULL_RUN_STEPS = 5 * 282
MAX_EPOCHS = 30
PATIENCE = 3
MIN_EPOCHS = 10
LR, BATCH_SIZE = 2e-4, 32

ROOT = finetune.ROOT
DATA_DIR = finetune.DATA_DIR
RESULTS_DIR = finetune.RESULTS_DIR


def subset(train: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """k messages per intent. Each intent is shuffled once per seed, so smaller sets are
    the start of larger ones (same rule as the LLM's few-shot examples)."""
    return pd.concat(
        group.sample(frac=1, random_state=seed).head(k)
        for _, group in train.groupby("label")
    ).reset_index(drop=True)


def scores(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def run_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = baseline.make_model("words+chars", 10, "balanced")
    seconds, _ = baseline.fit(model, train)
    return {**scores(test["label"], model.predict(test["text"])), "fit_seconds": seconds}


def run_distilbert(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   seed: int, run_name: str) -> dict:
    steps_per_epoch = math.ceil(len(train) / BATCH_SIZE)
    epochs = min(MAX_EPOCHS, math.ceil(FULL_RUN_STEPS / steps_per_epoch))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"train_messages": len(train), "lr": LR, "batch_size": BATCH_SIZE,
                           "max_epochs": epochs, "patience": PATIENCE,
                           "min_epochs": MIN_EPOCHS, "seed": seed})
        start = time.perf_counter()
        model, tokenizer, rows, best = finetune.fit(
            train, val, LR, BATCH_SIZE, epochs, seed, patience=PATIENCE,
            min_epochs=MIN_EPOCHS, train_accuracy=False,
            on_epoch=lambda r: mlflow.log_metrics(
                {k: v for k, v in r.items() if k != "epoch"}, step=r["epoch"]))
        seconds = time.perf_counter() - start
        ids, _ = finetune.predict(model, tokenizer, test["text"].tolist(), finetune.device())
        preds = [model.config.id2label[int(i)] for i in ids]
        result = {**scores(test["label"], preds), "best_epoch": best["epoch"],
                  "epochs_run": len(rows), "max_epochs": epochs,
                  "best_val_accuracy": best["val_accuracy"], "fit_seconds": seconds}
        mlflow.log_metrics({f"test_{k}": result[k] for k in ["accuracy", "macro_f1"]})
    return result


def summarise(payload: dict, path: Path, stamp: str) -> None:
    table = (pd.DataFrame(payload["results"])
             .groupby(["model", "per_intent"])[["accuracy", "macro_f1"]]
             .agg(["mean", "std"]))
    print("\nTest accuracy / macro F1, mean and std over 3 seeds:\n")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    for (name, k), row in table.iterrows():
        approach = "TF-IDF + LR" if name == "baseline" else "DistilBERT"
        line = (f"| {stamp} | **test** | {approach}, {k}/intent | "
                f"{row[('accuracy', 'mean')]:.4f} | {row[('macro_f1', 'mean')]:.4f} | – | – | "
                f"learning curve, mean of 3 seeds (acc std "
                f"{row[('accuracy', 'std')]:.4f}). {path.name} |\n")
        with finetune.HISTORY.open("a") as f:
            f.write(line)
    print(f"\nSaved {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun", nargs="+", metavar="FILE SEED:K",
                        help="results file, then the DistilBERT runs to re-train")
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(finetune.MLFLOW_URI)
    mlflow.set_experiment("banking77-learning-curve")

    stamp = finetune.utc_stamp()
    path = RESULTS_DIR / f"learning_curve_{stamp}.json"
    payload = {"run": "learning_curve", "timestamp_utc": stamp, "split": "test",
               "sizes_per_intent": SIZES, "seeds": SEEDS,
               "settings": {"baseline": "words+chars, C=10, balanced",
                            "distilbert": {"lr": LR, "batch_size": BATCH_SIZE,
                                           "target_steps": FULL_RUN_STEPS,
                                           "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                                           "min_epochs": MIN_EPOCHS}},
               "versions": finetune.versions(), "results": []}

    if args.rerun:
        old_file, *pairs = args.rerun
        todo = [tuple(int(x) for x in p.split(":")) for p in pairs]
        old = json.loads(Path(old_file).read_text())
        payload["rerun_of"] = Path(old_file).name
        payload["replaced"] = []
        for r in old["results"]:
            if r["model"] == "distilbert" and (r["seed"], r["per_intent"]) in todo:
                payload["replaced"].append(r)
            else:
                payload["results"].append(r)
        jobs = [(seed, k, ["distilbert"]) for seed, k in todo]
    else:
        jobs = [(seed, k, ["baseline", "distilbert"]) for seed in SEEDS for k in SIZES]

    for seed, k, names in jobs:
        sub = subset(train, k, seed)
        for name in names:
            start = time.perf_counter()
            if name == "baseline":
                result = run_baseline(sub, test)
            else:
                result = run_distilbert(sub, val, test, seed, f"lc_{k}per_seed{seed}")
            payload["results"].append(
                {"model": name, "per_intent": k, "train_messages": len(sub),
                 "seed": seed, **result})
            finetune.save_json(path, payload)  # after every run, so nothing is lost
            print(f"seed {seed}  {k:2d}/intent  {name:10s}  test acc "
                  f"{result['accuracy']:.4f}  F1 {result['macro_f1']:.4f}  "
                  f"best epoch {result.get('best_epoch', '-')}  "
                  f"({(time.perf_counter() - start) / 60:.1f} min)", flush=True)

    summarise(payload, path, stamp)


if __name__ == "__main__":
    main()
