"""Download BANKING77, carve a stratified validation set out of train, save the splits.

Usage:  python src/data.py

Writes data/train.csv, data/val.csv, data/test.csv (columns: text, label). Every later
experiment reads these files, so all approaches see exactly the same data.

The Hugging Face copy (PolyAI/banking77) only has a loading script that current
`datasets` versions refuse to run, so we download the original CSVs from PolyAI's
GitHub, pinned to one commit and checked against known SHA-256 hashes.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_FRACTION = 0.10

COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{COMMIT}/banking_data"
)
SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data"


def download(name: str) -> Path:
    """Download one raw CSV (once) and verify its hash."""
    path = RAW_DIR / name
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        sys.exit(f"Hash mismatch for {path}: got {digest}. Delete it and re-run.")
    return path


def load_raw(name: str) -> pd.DataFrame:
    df = pd.read_csv(download(name))
    return df.rename(columns={"category": "label"})[["text", "label"]]


def sanity_checks(splits: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if the splits are not what every later experiment assumes."""
    labels = {name: set(df["label"]) for name, df in splits.items()}
    for name, found in labels.items():
        assert len(found) == 77, f"{name} has {len(found)} intents, expected 77"
    assert labels["train"] == labels["val"] == labels["test"], "intent names differ"

    for name, df in splits.items():
        assert df["text"].notna().all() and (df["text"].str.strip() != "").all(), (
            f"{name} has empty messages"
        )

    # Train and val must be disjoint row-wise (val was cut out of train).
    assert len(splits["train"]) + len(splits["val"]) == 10_003, "train/val sizes off"
    assert len(splits["test"]) == 3_080, "test size off"

    # Exact duplicate texts across splits would leak answers. Report, don't hide.
    texts = {name: set(df["text"].str.strip().str.lower()) for name, df in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = texts[a] & texts[b]
        print(f"  identical messages in {a} and {b}: {len(overlap)}")


def main() -> None:
    full_train = load_raw("train.csv")
    test = load_raw("test.csv")

    train, val = train_test_split(
        full_train,
        test_size=VAL_FRACTION,
        stratify=full_train["label"],
        random_state=SEED,
    )
    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test,
    }

    print("Sanity checks:")
    sanity_checks(splits)
    print("  all passed")

    for name, df in splits.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        counts = df["label"].value_counts()
        print(
            f"{name:5s}: {len(df):5d} messages, "
            f"{counts.min()}-{counts.max()} per intent -> data/{name}.csv"
        )


if __name__ == "__main__":
    main()
