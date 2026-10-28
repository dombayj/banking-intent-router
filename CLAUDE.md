# Banking Support Intent Router

Classify bank customer messages into the 77 BANKING77 intents and compare three approaches
on accuracy, latency and cost: a TF-IDF + logistic regression baseline, a fine-tuned
DistilBERT, and a prompted GPT-4o-mini (zero-shot and few-shot). Then add confidence
routing (low-confidence messages go to a human) and serve the best model with FastAPI in
Docker.

The point of the project is the comparison: showing when fine-tuning beats an LLM API,
with real numbers. It is a portfolio project for AI/ML internship applications, so the
documentation matters as much as the code. Companion project: FinanceBench RAG
(github.com/dombayj/financebench-rag), which used the same working method.

## Plan

1. Baseline: TF-IDF + logistic regression (scikit-learn). Accuracy and macro F1.
2. Fine-tune DistilBERT (Hugging Face + PyTorch). Log every training run in MLflow.
3. LLM baseline: GPT-4o-mini on the same test set, zero-shot, then few-shot.
4. Compare: table of accuracy, macro F1, latency per query, cost per 1,000 queries.
   Confusion analysis: which intents get mixed up, and why.
5. Confidence routing: pick a threshold on the validation set; report the share of
   messages handled automatically and the precision on those.
6. Serve: FastAPI `/classify` endpoint in Docker. (FinanceBench was never deployed, so
   this is the first serving setup, built from scratch here.)

## Working rules

- **Baseline first.** Record a simple, clearly defined baseline before any improvement.
  Every later number is reported against it.
- **The test set is touched only for final numbers.** BANKING77 has no validation split:
  carve a stratified validation set out of train with a fixed seed, and use it for all
  tuning (hyperparameters, prompts, few-shot examples, confidence thresholds). Few-shot
  examples come from train, never from test.
- **Same test set for every approach** (all 3,080 examples), so the comparison is fair.
  If the LLM run uses a subset to save cost, say so and report the other models on the
  same subset too.
- **Honest numbers.** Report what was measured, even when it is worse than hoped. Never
  pick a number to match a CV bullet; the CV bullets get filled in from the results.
- **One change per experiment**, so each gain can be explained.
- **Log every run** in `results/history.md` (date, setup, metrics, one-line note) and save
  the raw results as `results/<run_name>_<UTC timestamp>.json`. Training runs also go to
  MLflow.
- **Record what didn't help, and why.** Failed ideas with a diagnosis are good CV and
  interview material.
- **Measure cost and latency the same way for all approaches** (same machine, batch size
  stated, API cost from real token counts).
- **Cache paid API calls** on a hash of (model, prompt, input), with atomic writes, so
  re-runs are free.
- **Fixed seeds** everywhere; note library versions in the results JSON.
- Keep `README.md` current: problem, baseline, results table, what didn't help, how to
  reproduce.

## Environment

- macOS, Python 3.13 as `python3`. Use a venv in `.venv/` for this project.
- API keys go in `.env` (gitignored); `.env.example` shows the names.
- Large files (datasets, model checkpoints, `mlruns/`, caches) stay out of git.
