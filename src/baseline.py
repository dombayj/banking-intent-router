"""Part 1 baseline: TF-IDF + logistic regression.

Usage:
  python src/baseline.py grid
      Try every setting in the grid (features x C x class weights), train on train,
      score on validation. Prints a table, saves results/baseline_grid_<ts>.json.
  python src/baseline.py test --features words --C 10 --class-weight balanced
      Train the chosen setting on train, score it once on test, measure latency.
      Saves results/baseline_test_<ts>.json (includes every test prediction).

Both append a row to results/history.md. Test is only for the final number: pick the
setting from the grid table, never by looking at test.
"""

import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

SEED = 42
FEATURES = ["words", "words+pairs", "words+chars"]
C_VALUES = [0.1, 1, 10, 100, 1000]
CLASS_WEIGHTS = ["none", "balanced"]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
HISTORY = RESULTS_DIR / "history.md"


def make_features(name: str):
    """TF-IDF vocabulary: single words, plus word pairs, or plus letter sequences."""
    if name == "words":
        return TfidfVectorizer(ngram_range=(1, 1))
    if name == "words+pairs":
        return TfidfVectorizer(ngram_range=(1, 2))
    if name == "words+chars":
        # char_wb: letter sequences of length 3-5 taken inside word boundaries,
        # so "witdrawal" still shares "dra", "raw", "awal" with "withdrawal".
        return FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 1))),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
        ])
    raise ValueError(f"unknown features: {name}")


def make_model(features: str, C: float, class_weight: str) -> Pipeline:
    return Pipeline([
        ("tfidf", make_features(features)),
        ("clf", LogisticRegression(
            C=C,
            class_weight=None if class_weight == "none" else "balanced",
            max_iter=5000,
            random_state=SEED,
        )),
    ])


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def fit(model: Pipeline, train: pd.DataFrame) -> tuple[float, bool]:
    """Train; return (seconds, converged)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        start = time.perf_counter()
        model.fit(train["text"], train["label"])
        seconds = time.perf_counter() - start
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return seconds, converged


def versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "machine": f"{platform.system()} {platform.machine()} ({platform.processor()})",
    }


def save(run_name: str, payload: dict) -> tuple[str, Path]:
    """Write results/<run_name>_<UTC timestamp>.json atomically."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{run_name}_{stamp}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"run": run_name, "timestamp_utc": stamp,
                               "seed": SEED, "versions": versions(), **payload}, indent=2))
    tmp.rename(path)
    return stamp, path


def log_history(stamp: str, split: str, acc: float, f1: float,
                latency: str, notes: str) -> None:
    row = (f"| {stamp} | {split} | TF-IDF + LR | {acc:.4f} | {f1:.4f} | "
           f"{latency} | – | {notes} |\n")
    with HISTORY.open("a") as f:
        f.write(row)


def run_grid() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")

    rows = []
    for features in FEATURES:
        for class_weight in CLASS_WEIGHTS:
            for C in C_VALUES:
                model = make_model(features, C, class_weight)
                seconds, converged = fit(model, train)
                train_scores = score(train["label"], model.predict(train["text"]))
                val_scores = score(val["label"], model.predict(val["text"]))
                rows.append({
                    "features": features, "class_weight": class_weight, "C": C,
                    "train_accuracy": train_scores["accuracy"],
                    "val_accuracy": val_scores["accuracy"],
                    "val_macro_f1": val_scores["macro_f1"],
                    "fit_seconds": seconds, "converged": converged,
                })
                print(f"{features:12s} {class_weight:8s} C={C:<6g} "
                      f"val acc {val_scores['accuracy']:.4f}  "
                      f"F1 {val_scores['macro_f1']:.4f}  ({seconds:.1f}s)")

    table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    print("\nAll settings, best validation macro F1 first:\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    stamp, path = save("baseline_grid", {
        "split": "val",
        "grid": {"features": FEATURES, "C": C_VALUES, "class_weight": CLASS_WEIGHTS},
        "results": rows,
    })
    best = table.iloc[0]
    log_history(stamp, "val", best["val_accuracy"], best["val_macro_f1"], "–",
                f"grid of {len(rows)} settings; best by val F1: {best['features']}, "
                f"C={best['C']:g}, class weights {best['class_weight']}. {path.name}")
    print(f"\nSaved {path.relative_to(ROOT)}")


def run_test(features: str, C: float, class_weight: str) -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    model = make_model(features, C, class_weight)
    seconds, converged = fit(model, train)

    # Batch: all 3,080 test messages in one call.
    start = time.perf_counter()
    preds = model.predict(test["text"])
    batch_seconds = time.perf_counter() - start

    # One message at a time (batch size 1), as a live /classify endpoint would see them.
    single = []
    for text in test["text"]:
        start = time.perf_counter()
        model.predict([text])
        single.append(time.perf_counter() - start)
    single_ms = np.array(single) * 1000

    test_scores = score(test["label"], preds)
    confidence = model.predict_proba(test["text"]).max(axis=1)
    latency = {
        "batch_all_ms_per_query": batch_seconds * 1000 / len(test),
        "batch_size_1_mean_ms": float(single_ms.mean()),
        "batch_size_1_p50_ms": float(np.percentile(single_ms, 50)),
        "batch_size_1_p95_ms": float(np.percentile(single_ms, 95)),
    }

    print(f"Test accuracy {test_scores['accuracy']:.4f}  "
          f"macro F1 {test_scores['macro_f1']:.4f}")
    print(f"Latency, batch size 1: mean {latency['batch_size_1_mean_ms']:.2f} ms, "
          f"p95 {latency['batch_size_1_p95_ms']:.2f} ms; "
          f"batch of {len(test)}: {latency['batch_all_ms_per_query']:.3f} ms/query")

    stamp, path = save("baseline_test", {
        "split": "test",
        "setting": {"features": features, "C": C, "class_weight": class_weight},
        "fit_seconds": seconds, "converged": converged,
        "metrics": test_scores,
        "latency": latency,
        "predictions": [
            {"text": t, "label": y, "pred": p, "confidence": float(c)}
            for t, y, p, c in zip(test["text"], test["label"], preds, confidence)
        ],
    })
    log_history(stamp, "**test**", test_scores["accuracy"], test_scores["macro_f1"],
                f"{latency['batch_size_1_mean_ms']:.2f} ms (batch 1, laptop)",
                f"final baseline: {features}, C={C:g}, class weights {class_weight}. "
                f"{path.name}")
    print(f"Saved {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("grid")
    test = sub.add_parser("test")
    test.add_argument("--features", choices=FEATURES, required=True)
    test.add_argument("--C", type=float, required=True)
    test.add_argument("--class-weight", choices=CLASS_WEIGHTS, required=True)
    args = parser.parse_args()

    if args.stage == "grid":
        run_grid()
    else:
        run_test(args.features, args.C, args.class_weight)


if __name__ == "__main__":
    main()
