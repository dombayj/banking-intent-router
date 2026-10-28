"""Part 6: serve the fine-tuned DistilBERT with confidence routing.

Run locally:   uvicorn app.main:app --port 8000      (then open http://localhost:8000)
In Docker:     see Dockerfile and README.

POST /classify {"text": "..."} returns the intent, its confidence, the two runner-ups
and the routing decision: "auto" when confidence >= THRESHOLD, otherwise "human".
Nothing the user types is stored or logged.
"""

import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get(
    "MODEL_DIR", ROOT / "models" / "finetune_lr0.0002_bs32_cw-none_seed44_20261027T171332Z"))
# Lowest confidence that kept answered validation messages >= 97% correct (Part 5,
# results/routing_val_20261028T095719Z.json). Picked on validation, never on test.
THRESHOLD = 0.8629079461097717
TARGET_PRECISION = 0.97
MAX_LENGTH = 128  # same as training
N_ALTERNATIVES = 2

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
id2label = model.config.id2label

app = FastAPI(
    title="Banking Support Intent Router",
    description="Classifies a bank customer message into one of the 77 BANKING77 intents. "
                f"Messages with confidence below {THRESHOLD:.2f} are routed to a human "
                f"(threshold chosen on validation for {TARGET_PRECISION:.0%} precision).",
)


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=500,
                      examples=["I still haven't received my new card"])


class Alternative(BaseModel):
    intent: str
    confidence: float


class Classification(BaseModel):
    intent: str
    confidence: float
    route: str  # "auto" or "human"
    threshold: float
    alternatives: list[Alternative]
    latency_ms: float


@torch.inference_mode()  # per call: FastAPI runs requests in worker threads
def top_intents(text: str) -> list[tuple[str, float]]:
    batch = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    probs = torch.softmax(model(**batch).logits[0], dim=-1)
    top = torch.topk(probs, 1 + N_ALTERNATIVES)
    return [(id2label[int(i)], float(p)) for p, i in zip(top.values, top.indices)]


top_intents("warm-up")  # the first call is slow; pay it at startup, not on a user


@app.post("/classify", response_model=Classification)
def classify(message: Message) -> Classification:
    start = time.perf_counter()
    ranked = top_intents(message.text)
    intent, confidence = ranked[0]
    return Classification(
        intent=intent,
        confidence=confidence,
        route="auto" if confidence >= THRESHOLD else "human",
        threshold=THRESHOLD,
        alternatives=[Alternative(intent=i, confidence=p) for i, p in ranked[1:]],
        latency_ms=(time.perf_counter() - start) * 1000,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
