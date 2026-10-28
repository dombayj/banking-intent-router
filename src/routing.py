"""Part 5: confidence routing. Answer a message automatically only when the model is
confident enough; send the rest to a human.

Usage:
  python src/routing.py val
      Pick a threshold on validation for each model x method x target precision and
      show how much validation traffic it handles. Nothing is scored on test.
  python src/routing.py test
      Same thresholds (picked on validation), then applied once to test.

Models: the final baseline (words+chars, balanced, C=10, refit on train) and DistilBERT
seed 44. Methods: "confidence" = top probability; "margin" = top minus second
probability; "ensemble" = top probability of the 3 DistilBERT seeds averaged.

Threshold rule: sort validation messages by score, highest first, and keep the largest
top group whose accuracy is still >= the target. The threshold is the lowest score in
that group. On test, a message is handled automatically when its score >= threshold.
"""

import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import baseline
import finetune

TARGETS = [0.95, 0.97, 0.99]
HEADLINE_TARGET = 0.97
BASELINE_SETTING = {"features": "words+chars", "C": 10, "class_weight": "balanced"}
DISTILBERT_DIRS = {
    seed: next(finetune.MODELS_DIR.glob(f"finetune_lr0.0002_bs32_cw-none_seed{seed}_*"))
    for seed in (42, 43, 44)
}
MAIN_SEED = 44

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
HISTORY = RESULTS_DIR / "history.md"


def baseline_probs(splits: dict[str, pd.DataFrame]) -> tuple[list[str], dict]:
    model = baseline.make_model(**BASELINE_SETTING)
    baseline.fit(model, splits["train"])
    return list(model.classes_), {name: model.predict_proba(df["text"])
                                  for name, df in splits.items() if name != "train"}


def distilbert_probs(model_dir: Path, splits: dict[str, pd.DataFrame]) -> tuple[list[str], dict]:
    dev = finetune.device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return labels, {name: finetune.predict(model, tokenizer, df["text"].tolist(), dev)[1]
                    for name, df in splits.items() if name != "train"}


def score(probs: np.ndarray, method: str) -> np.ndarray:
    top2 = np.sort(probs, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0] if method == "margin" else top2[:, 1]


def pick_threshold(scores: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest threshold whose accepted validation group is still >= target accurate."""
    order = np.argsort(-scores, kind="stable")
    s, c = scores[order], correct[order]
    precision = np.cumsum(c) / np.arange(1, len(c) + 1)
    # Only cut between different scores: equal scores are all in or all out.
    cut_ok = np.append(s[1:] < s[:-1], True)
    ok = np.where((precision >= target) & cut_ok)[0]
    return float(s[ok[-1]]) if len(ok) else float("inf")


def routed(scores: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    auto = scores >= threshold
    n_auto = int(auto.sum())
    return {
        "coverage": n_auto / len(scores),
        "precision": float(correct[auto].mean()) if n_auto else float("nan"),
        "n_auto": n_auto,
        "n_wrong_auto": int((~correct[auto]).sum()),
        "n_to_human": int(len(scores) - n_auto),
    }


def versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
    }


def log_history(stamp: str, split: str, approach: str, notes: str) -> None:
    with HISTORY.open("a") as f:
        f.write(f"| {stamp} | {split} | {approach} | – | – | – | – | {notes} |\n")


def run(stage: str) -> None:
    names = ["train", "val"] + (["test"] if stage == "test" else [])
    splits = {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in names}

    print("Computing probabilities ...", flush=True)
    labels_b, probs_b = baseline_probs(splits)
    seed_probs = {seed: distilbert_probs(d, splits) for seed, d in DISTILBERT_DIRS.items()}
    labels_d = seed_probs[MAIN_SEED][0]
    assert labels_b == labels_d == sorted(labels_b), "label order differs between models"
    labels = np.array(labels_b)
    ensemble = {split: np.mean([seed_probs[s][1][split] for s in DISTILBERT_DIRS], axis=0)
                for split in probs_b}

    candidates = [  # (model name, method, probabilities per split)
        ("TF-IDF + LR", "confidence", probs_b),
        ("TF-IDF + LR", "margin", probs_b),
        ("DistilBERT", "confidence", seed_probs[MAIN_SEED][1]),
        ("DistilBERT", "margin", seed_probs[MAIN_SEED][1]),
        ("DistilBERT 3-seed ensemble", "confidence", ensemble),
    ]

    rows = []
    for model_name, method, probs in candidates:
        per_split = {}
        for split, p in probs.items():
            correct = labels[p.argmax(axis=1)] == splits[split]["label"].to_numpy()
            per_split[split] = (score(p, method), correct)
        for target in TARGETS:
            threshold = pick_threshold(*per_split["val"], target)
            row = {"model": model_name, "method": method, "target": target,
                   "threshold": threshold}
            for split, (s, c) in per_split.items():
                row[f"{split}_accuracy_all"] = float(c.mean())
                row.update({f"{split}_{k}": v for k, v in routed(s, c, threshold).items()})
            rows.append(row)

    table = pd.DataFrame(rows)
    show = ["model", "method", "target", "threshold", "val_accuracy_all",
            "val_coverage", "val_precision"]
    if stage == "test":
        show += ["test_accuracy_all", "test_coverage", "test_precision",
                 "test_n_wrong_auto", "test_n_to_human"]
    print(f"\n{stage.upper()}: threshold picked on validation for each target\n")
    print(table[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    run_name = f"routing_{stage}"
    stamp = finetune.utc_stamp()
    path = RESULTS_DIR / f"{run_name}_{stamp}.json"
    finetune.save_json(path, {
        "run": run_name, "timestamp_utc": stamp, "split": stage, "versions": versions(),
        "targets": TARGETS, "baseline_setting": BASELINE_SETTING,
        "distilbert_dirs": {s: str(d.relative_to(ROOT)) for s, d in DISTILBERT_DIRS.items()},
        "main_seed": MAIN_SEED, "results": rows,
    })

    split = "val" if stage == "val" else "**test**"
    for r in rows:
        if r["target"] != HEADLINE_TARGET:
            continue
        s = stage if stage == "val" else "test"
        log_history(stamp, split, f"{r['model']}, routing ({r['method']})",
                    f"target {r['target']:.0%} precision, threshold {r['threshold']:.4f} "
                    f"(from val): handled automatically {r[f'{s}_coverage']:.1%}, "
                    f"precision on those {r[f'{s}_precision']:.4f}. {path.name}")
    print(f"\nSaved {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["val", "test"])
    run(parser.parse_args().stage)


if __name__ == "__main__":
    main()
