"""Part 3: classify BANKING77 messages with an LLM API (zero-shot or few-shot).

Usage:
  python src/llm.py run --split val [--effort none] [--shots 3] [--limit 20]
      Classify every message of a split. The answer is forced to one of the 77 intent
      names (structured output). Saves results/llm_<setup>_<split>_<ts>.json and adds
      a row to results/history.md. Run on val for all prompt choices; test only once
      per final setup.

Every API call is cached in cache/llm/ on a hash of (model, settings, prompt, message),
written atomically, so re-runs are free. Cost comes from the token counts the API
reports, priced with PRICES below. Latency is measured separately: one message at a
time, uncached, on the first LATENCY_SAMPLE messages.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MODEL = "gpt-5.4-mini-2026-03-17"
SEED = 42
# USD per 1M tokens (listed price, Oct 2026). Reasoning tokens are billed as output.
PRICES = {"input": 0.75, "cached_input": 0.075, "output": 4.50}
WORKERS = 4  # 8 hit the 2M tokens/minute limit with the 5-shot prompt
LATENCY_SAMPLE = 100

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache" / "llm"
HISTORY = RESULTS_DIR / "history.md"


def load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def pick_examples(train: pd.DataFrame, k: int) -> dict[str, list[str]]:
    """k random train messages per intent (fixed seed). Each intent's messages are
    shuffled once, so the 5-shot set starts with exactly the 3-shot set."""
    return {
        label: group.sample(frac=1, random_state=SEED)["text"].head(k).tolist()
        for label, group in train.groupby("label")
    }


def build_instructions(labels: list[str], examples: dict[str, list[str]] | None) -> str:
    """The prompt: what the model is told before it sees each customer message."""
    task = (
        "You route customer messages for a bank's support team. Classify the "
        "customer's message into exactly one of the following 77 intents, and answer "
        "with the intent that best matches what the customer is asking about.\n\n"
    )
    if not examples:
        return task + "Intents:\n" + "\n".join(labels)
    blocks = [label + "\n" + "\n".join(f"- {t}" for t in examples[label])
              for label in labels]
    return (task + "Each intent is followed by example customer messages labelled "
            "with it.\n\nIntents:\n\n" + "\n\n".join(blocks))


def answer_format(labels: list[str]) -> dict:
    """Structured output: the reply must be {"intent": <one of the 77 names>}."""
    return {"format": {
        "type": "json_schema",
        "name": "intent",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": labels}},
            "required": ["intent"],
            "additionalProperties": False,
        },
    }}


class Classifier:
    def __init__(self, labels: list[str], effort: str,
                 examples: dict[str, list[str]] | None = None):
        self.client = openai.OpenAI(max_retries=5)
        self.instructions = build_instructions(labels, examples)
        self.text = answer_format(labels)
        self.effort = effort

    def key(self, message: str) -> str:
        blob = json.dumps([MODEL, self.effort, self.instructions, self.text, message])
        return hashlib.sha256(blob.encode()).hexdigest()

    def call(self, message: str) -> dict:
        start = time.perf_counter()
        response = self.client.responses.create(
            model=MODEL,
            instructions=self.instructions,
            input=message,
            reasoning={"effort": self.effort},
            text=self.text,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = response.usage
        return {
            "intent": json.loads(response.output_text)["intent"],
            "input_tokens": usage.input_tokens,
            "cached_tokens": usage.input_tokens_details.cached_tokens,
            "output_tokens": usage.output_tokens,
            "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
            "latency_ms": latency_ms,
        }

    def classify(self, message: str) -> dict:
        """Cached call: each (settings, prompt, message) is paid for only once."""
        path = CACHE_DIR / f"{self.key(message)}.json"
        if path.exists():
            return json.loads(path.read_text())
        result = self.call(message)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(result))
        tmp.rename(path)
        return result


def cost_usd(r: dict) -> float:
    uncached = r["input_tokens"] - r["cached_tokens"]
    return (uncached * PRICES["input"] + r["cached_tokens"] * PRICES["cached_input"]
            + r["output_tokens"] * PRICES["output"]) / 1e6


def log_history(stamp: str, split: str, setup: str, acc: float, f1: float,
                latency_ms: float, cost_1k: float, notes: str) -> None:
    row = (f"| {stamp} | {split} | {setup} | {acc:.4f} | {f1:.4f} | "
           f"{latency_ms:.0f} ms (batch 1, API) | ${cost_1k:.2f} | {notes} |\n")
    with HISTORY.open("a") as f:
        f.write(row)


def run(args) -> None:
    load_env()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    labels = sorted(train["label"].unique())
    df = pd.read_csv(DATA_DIR / f"{args.split}.csv")
    if args.limit:
        df = df.head(args.limit)
    texts = df["text"].tolist()
    examples = pick_examples(train, args.shots) if args.shots else None
    clf = Classifier(labels, args.effort, examples)

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(clf.classify, texts))

    # Latency: one message at a time, never from cache, so it is a real round trip.
    latencies = np.array([clf.call(t)["latency_ms"] for t in texts[:LATENCY_SAMPLE]])

    preds = [r["intent"] for r in results]
    acc = accuracy_score(df["label"], preds)
    f1 = f1_score(df["label"], preds, average="macro")
    total_cost = sum(cost_usd(r) for r in results)
    cost_1k = total_cost / len(texts) * 1000
    tokens = {k: int(sum(r[k] for r in results)) for k in
              ["input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"]}

    shots = f"{args.shots}-shot" if args.shots else "zero-shot"
    setup = f"{MODEL.rsplit('-', 3)[0]} {shots}, effort {args.effort}"
    print(f"{setup} on {args.split} ({len(texts)} messages)")
    print(f"accuracy {acc:.4f}  macro F1 {f1:.4f}  "
          f"({sum(p != y for p, y in zip(preds, df['label']))} wrong)")
    print(f"cost ${total_cost:.3f} total, ${cost_1k:.2f} per 1,000; tokens {tokens}")
    print(f"latency (batch 1, {len(latencies)} msgs): median {np.median(latencies):.0f} ms, "
          f"p95 {np.percentile(latencies, 95):.0f} ms")

    if args.limit:
        print("(--limit used: smoke test, not saved or logged)")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"llm_{shots.replace('-', '')}_{args.effort}_{args.split}"
    path = RESULTS_DIR / f"{run_name}_{stamp}.json"
    payload = {
        "run": run_name, "timestamp_utc": stamp, "split": args.split,
        "model": MODEL, "reasoning_effort": args.effort, "shots_per_intent": args.shots,
        "examples_seed": SEED, "prices_usd_per_1m": PRICES,
        "instructions": clf.instructions,
        "versions": {"python": platform.python_version(), "openai": openai.__version__},
        "metrics": {"accuracy": acc, "macro_f1": f1},
        "tokens": tokens, "cost_usd": total_cost, "cost_per_1k_usd": cost_1k,
        "latency_ms": {"n": len(latencies), "median": float(np.median(latencies)),
                       "mean": float(latencies.mean()),
                       "p95": float(np.percentile(latencies, 95))},
        "predictions": [{"text": t, "label": y, "pred": p}
                        for t, y, p in zip(texts, df["label"], preds)],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(path)
    log_history(stamp, args.split if args.split == "val" else "**test**", setup, acc, f1,
                float(np.median(latencies)), cost_1k, path.name)
    print(f"Saved {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["val", "test"], required=True)
    r.add_argument("--effort", default="none")
    r.add_argument("--shots", type=int, default=0, help="train examples per intent")
    r.add_argument("--limit", type=int, help="first N messages only (smoke test)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
