# Planning notes and decisions

Running log of the design discussion, so work can pick up in any later session.
Decisions are made by Osman; each one records the reason. Experiment results go in
`results/history.md`, not here.

## Status

- **2026-10-21:** Discussion phase. No code yet. Went through the overall plan (parts 0–6)
  and the basic concepts below. Next: decide Part 0 (validation split) and Part 1
  (baseline tuning), then start building Part 0.
- **2026-10-21 (later):** Explained validation, 768, epochs, learning rate. Decided cost
  reporting and the learning-curve experiment. Validation size = 10%. Next: agree on the
  Part 0 build plan, then build it.
- **2026-10-21 (Part 0 built):** `.venv/`, `requirements.txt`, `src/data.py` done. Splits
  saved to `data/` (train 9,002 / val 1,001 / test 3,080).
- **2026-10-22:** Decided to keep the 6 train/test duplicates. **Part 0 is finished.**
  **Next session starts here:** explain TF-IDF and logistic regression in plain words,
  then Osman decides Part 1 (how much baseline tuning), then build Part 1.
- **2026-10-27:** Explained TF-IDF, logistic regression (multinomial, worked example with
  numbers), C, class weights, validation. Decided Part 1: option B grid + class weights
  on/off. **Next session starts here:** build Part 1 (script, ~30 validation runs, logging),
  show Osman the validation table, Osman picks the setting, then one test run.
- **2026-10-27 (grid run):** `src/baseline.py` built; 30-setting grid run on validation
  (`results/baseline_grid_20261027T150910Z.json`). Best: words+chars, class weights
  balanced, C=10 (val acc 0.9051, F1 0.9068). **Waiting for Osman** to pick the setting
  and decide train-only vs train+val for the final fit; then `baseline.py test` once.
- **2026-10-27 (Part 1 finished):** Osman chose words+chars / balanced / C=10, train
  only. Test run once: **accuracy 0.9062, macro F1 0.9063**, 0.53 ms per message
  (`results/baseline_test_20261027T151642Z.json`, includes every prediction for later
  error analysis). README updated. **Next session starts here:** discuss Part 2
  (DistilBERT: MPS vs Colab, which hyperparameters, MLflow), optionally look at the
  baseline's mistakes first.

- **2026-10-27 (Part 2 started):** Osman agreed to the standard start (lr 5e-5, batch 32,
  5 epochs, best epoch by val macro F1). `src/finetune.py` built (manual PyTorch loop,
  HF Trainer default optimiser settings, MLflow in `mlruns/mlflow.db`). Measured 1.6 min
  per epoch on the M2. Run 1 started. Next: show run 1 curve, then lr 3e-5 and 2e-5.
- **2026-10-27 (lr runs done):** best val F1 per lr: 5e-5 → 0.9208 (epoch 4), 3e-5 →
  0.9110, 2e-5 → 0.8761 (both still improving at epoch 5: too slow, underfit in 5
  epochs). Best lr is at the top edge of the tried range. **Waiting for Osman:** take
  5e-5, or first try 1e-4 to check the peak.
- **2026-10-27 (lr search upwards done):** 1e-4 → 0.9214 (epoch 3), 2e-4 → **0.9258**
  (epoch 4), 4e-4 → 0.9006 (unstable start, still catching up at epoch 5). Peak found
  at 2e-4. No harmful overfitting: at 2e-4 train acc 99.5% vs val 92.3%, gap stable,
  val loss flat (0.331 → 0.326) after the best epoch. 5e-5 / 1e-4 / 2e-4 are within
  ~5 val messages. **Waiting for Osman** to pick the lr; then class-weights run, 2 seeds.
- **2026-10-27 (class weights run):** lr 2e-4 + balanced weights → val F1 0.9237 (epoch
  5), vs 0.9258 without. No help (−2 messages, within noise), same as the baseline at
  good C. **Waiting for Osman:** keep weights off or on; then seeds 43 and 44.
- **2026-10-27 (seeds done):** lr 2e-4, no weights, seeds 42/43/44 → val F1 0.9258 /
  0.9283 / 0.9306 = **0.9282 ± 0.0024** (acc 0.9244 ± 0.0023). Seed 42 was the lowest,
  so 2e-4 wasn't lucky. Gain over baseline val (0.9068) is +2.1 points, ~9× the spread.
  **Waiting for Osman:** how to run test (all 3 seeds and report mean ± spread, and which
  model to use for Parts 5–6).
- **2026-10-27 (Part 2 finished):** test, 3 seeds: acc 0.9198 / 0.9273 / 0.9169 =
  **0.921 ± 0.005**, F1 0.921 ± 0.005, 242 wrong on average vs baseline 289 (−16%).
  Latency batch 1: ~7.6 ms Mac GPU, ~18 ms CPU (baseline 0.53 ms). Seed 44 (best on val)
  is lowest on test: picking among seeds by val picks noise. We keep seed 44 anyway;
  switching because of test would be tuning on test. README updated. **Next session
  starts here:** Part 3 discussion (which LLM, prompt, few-shot examples, subset vs full).
- **2026-10-27 (Part 3 finished):** gpt-5.4-mini on test: zero-shot (low reasoning)
  0.811, 5-shot 0.886, both below baseline (0.906) and DistilBERT (0.921). Total API
  spend ≈ $9. README updated. **Next session starts here:** Part 4 (comparison table
  mostly done; decide server price for cost estimate; learning-curve experiment), then
  Part 5 routing for both baseline and DistilBERT.
- **2026-10-27 (learning curve done):** see Part 4 findings. README has chart + table.
  **Next session starts here:** last Part 4 item (server price for the cost column, or
  drop it), then Part 5 routing for both models.
- **2026-10-27 (Part 4 finished):** cost column = laptop capacity (see decisions).
  **Next session starts here:** Part 5 (confidence routing, both models: target
  precision, temperature scaling yes/no). Osman will later turn results into CV bullets.
- **2026-10-28 (Part 5 val run):** target 97% decided. `src/routing.py` built; thresholds
  picked on val (`results/routing_val_20261028T095719Z.json`). At 97%: baseline handles
  80.0% of val automatically, DistilBERT 90.1%, 3-seed ensemble 91.3%. Margin ≈ plain
  confidence (within ~5 messages). **Waiting for Osman:** pick the headline method, then
  `routing.py test` once.
- **2026-10-28 (Part 5 finished):** headline = plain confidence, baseline vs DistilBERT
  seed 44; margin and ensemble reported as extras. Test at 97%
  (`results/routing_test_20261028T100158Z.json`): baseline 79.6% handled / 97.7%
  precision; DistilBERT 88.7% / 96.9% (just under target, val was used for its tuning);
  ensemble 90.6% / 97.4%. Human queue 627 → 347 (−45%). README updated.
  **Next session starts here:** Part 6 (FastAPI + Docker; open question: local only or
  also a cloud deploy).
- **2026-10-28 (Part 6 local build):** decided: repo private now, public at the end;
  `/classify` returns intent + confidence + route + 2 runner-ups; demo page at `/` with 5
  example buttons (sent to the server, no caching), one-line model note, nothing stored.
  Built `app/main.py`, `app/static/index.html`, `Dockerfile` (CPU torch, non-root),
  `tests/test_app.py` (5 pass). Container: image 2.1 GB, 342 MB RAM, ~23 ms per request
  over HTTP on the M2 (batch 1). Local git repo initialised, no commit yet (needs Osman's
  git name/email and `gh auth login`). Azure waits for the GitHub Student Pack approval.

## Part 3 findings (LLM)

- Zero-shot v1, effort none, val: **acc 0.7872, F1 0.7867** (213 wrong; baseline 95,
  DistilBERT ~76). $0.73 per 1,000 messages; latency median 1.0 s, p95 1.7 s (batch 1).
- Main cause: intent names don't say what the dataset means by them. get_physical_card
  has recall 0 — in BANKING77 it holds PIN questions ("How do I set up my card PIN?"),
  so the LLM says change_pin. order_physical_card recall 0.08 (LLM says
  get_physical_card). fiat_currency_support vs supported_cards_and_currencies, etc. The
  labelling conventions can only be learned from examples, which is what fine-tuning
  (and few-shot) gives.
- Zero-shot v1, effort **low**, val: acc 0.8132, F1 0.8134 (187 wrong, −26 vs none).
  $0.90 per 1,000 (+23%; 34 reasoning tokens per message), latency median 1.7 s, p95
  4.3 s. Thinking helps a little but can't reveal the labelling habits.
- Few-shot runs keep effort none, so examples are the only change vs zero-shot.
- 3-shot, val: acc 0.8611, F1 0.8679 (139 wrong); $0.47 per 1,000 (cheaper than
  zero-shot: 96% of input tokens cached at 10% price); latency median 1.4 s.
- 5-shot, val: acc 0.8721, F1 0.8800 (128 wrong); $0.69 per 1,000; median 1.2 s.
  Diminishing returns (0 → 3: −74 wrong; 3 → 5: −11). Still below baseline (95 wrong).
- Caveat for cost: the caching discount assumes steady traffic (cache expires when
  idle). Report the uncached price too.
- **Test (final, once):** zero-shot + effort low: acc 0.8110, F1 0.8032, 582 wrong,
  $0.88 / 1k, median 1.23 s. 5-shot: acc 0.8864, F1 0.8848, 350 wrong, $0.66 / 1k
  ($4.70 uncached), median 1.15 s. 5-shot first hit the 2M tokens/min rate limit with 8
  parallel requests; re-run with 4 (cache kept finished answers).

## Part 4 findings (learning curve, test, 3 seeds)

- Final file: `results/learning_curve_20261027T204854Z.json` (first run with the failed
  rule kept: `learning_curve_20261027T194744Z.json`). Chart: `docs/learning_curve.png`
  (`src/plot_learning_curve.py`; lines = median of seeds).
- Baseline 5/10/25/50: 64.2 / 74.0 / 83.6 / 88.2%. DistilBERT: 68.2 (2 of 3) / 80.0 /
  87.4 / 90.1%. LLM 5-shot with the same 385 examples: 88.6%.
- Crossover ≈ 50 per intent. Seed 43 at 5/intent never learned even with ≥10 epochs
  (train loss 4.37 ≈ ln 77 for all 10 epochs): genuine instability, reported.
- Re-running seed 44 at 25 and 50 with the fixed rule: 0.8614 → 0.8776 and 0.8919 →
  0.9010 (both best at epoch 10; the old rule had stopped them too early).

## Why the baseline might still be the right choice (Osman's question, 2026-10-27)

Written up in README "Which model should a bank use?". In short: the speed gap (0.5 vs
18 ms) doesn't matter in practice; the real trade-off is the cost of a mistake (~1,500
fewer misroutes per 100k with DistilBERT) against simplicity (10 MB, 6 s CPU training,
readable weights) and honest confidence (0.48 when wrong vs DistilBERT's 0.70). The fair
business comparison is Part 5: share handled automatically at a target precision, for
both models.

## Part 2 findings (DistilBERT errors on test, seed 44)

- Fixed 123 of the baseline's 289 mistakes, made 90 new ones; 166 wrong for both, 100
  with the same wrong answer (many look like ambiguous labels).
- Fixed mainly wording-carries-meaning cases (top-up pending vs failed, card arrival).
  verify_my_identity ↔ why_verify_identity got worse (10 → 16).
- Overconfident: mean confidence 0.97 when right, but 0.70 when wrong (baseline: 0.87 /
  0.48). Matters for Part 5: temperature scaling may be worth trying.

## Decisions made

| Date | Part | Decision | Why |
|---|---|---|---|
| 2026-10-21 | All | Work one part at a time; discuss and decide before building each part | Osman wants to understand and own every choice (first fine-tuning project) |
| 2026-10-21 | 6 | Serving is built from scratch (FinanceBench was never deployed) | The original description assumed an existing deployment |
| 2026-10-22 | 0 | Keep the 6 messages that appear in both train and test; mention it in the README | Keeps the standard test set, so results compare with published BANKING77 numbers; max effect 0.2% |
| 2026-10-21 | 4 | Report cost two ways: real laptop latency + estimated server cost per 1,000 queries, with the assumption written out | Laptop cost alone can't be compared with an API price |
| 2026-10-21 | 0 | Validation = 10% of train (~1,000), stratified by intent, fixed seed | Big enough for stable scores, small loss of training data |
| 2026-10-27 | 1 | Test the baseline both with and without class weights (`class_weight="balanced"`), compare on validation | Train is imbalanced (32–168 per intent) but test is balanced (40 each); without weights the model leans towards big intents |
| 2026-10-27 | 1 | Option B: small grid on validation — features (words / words + word pairs / words + char n-grams) × ~5 values of C × class weights on/off; best setting runs on test once | A fair, tuned baseline makes later gains believable; cheap (~30 runs of seconds each) |
| 2026-10-27 | 1 | Final baseline setting: words + chars, class weights balanced, C=10 | Highest validation macro F1 (0.9068); top 8 settings are within noise, so "best on validation" is the stated rule |
| 2026-10-27 | 1 | Final model trained on train only (9,002), not train + validation | Same training data as DistilBERT (needs validation for epochs) and LLM few-shot examples, so the comparison is fair |
| 2026-10-27 | 4 | Report both accuracy and number of wrong messages (error reduction vs baseline); decide later which one to headline | Accuracy is the standard number; error counts show what a gain means in practice |
| 2026-10-27 | 2 | Train DistilBERT on the Mac (M2, MPS) | Runs are short; everything stays local (data, MLflow); latency is measured on the Mac anyway |
| 2026-10-27 | 2 | Run 1: lr 5e-5, batch 32, up to 5 epochs, keep best epoch by val macro F1 | Standard DistilBERT values; same selection metric as the baseline |
| 2026-10-27 | 2 | After run 1, vary only the learning rate (2e-5, 3e-5, 5e-5); epochs read off the validation curve | Learning rate matters most; a bigger grid gives gains within noise |
| 2026-10-27 | 2 | Run the final setting with 3 seeds, report mean ± spread | Fine-tuning scores move ~0.5–1 point between seeds; shows whether a gain is real |
| 2026-10-27 | 2 | Extend lr upwards (1e-4, 2e-4, 4e-4) until val F1 drops; measure train accuracy per epoch as overfitting evidence | Best lr was at the edge of the range; Osman wants proof the model isn't overfitting |
| 2026-10-27 | 2 | Delete model files of losing runs, keep their JSON + MLflow records | Each model is 256 MB; nothing later reads losing models |
| 2026-10-27 | 2 | Learning rate 2e-4 | Highest val macro F1 (0.9258), same rule as the baseline; top three are within noise, the 3-seed runs will show stability |
| 2026-10-27 | 2 | Final DistilBERT setting: lr 2e-4, batch 32, 5 epochs, no class weights | Weights gave val F1 0.9237 vs 0.9258 without; no help, simpler model |
| 2026-10-27 | 2 | Run test on all 3 seeds, report mean ± spread; use seed 44 for Parts 5–6 and error analysis | Same honest method as validation; seed 44 has the best val score (our rule) |
| 2026-10-27 | 3 | LLM = current cheap OpenAI model: `gpt-5.4-mini`, pinned to snapshot `gpt-5.4-mini-2026-03-17` | What a company would try today; GPT-4o-mini is now legacy. Listed price ~$0.75 / $4.50 per 1M input / output tokens (third-party pages; real cost comes from token counts) |
| 2026-10-27 | 3 | Reasoning effort: lowest first, then one experiment with "low" on validation | Lowest fits high-volume routing; the experiment shows whether thinking is worth its cost and delay |
| 2026-10-27 | 3 | Full test set (3,080) for the LLM | Costs a few dollars; same messages as every other approach |
| 2026-10-27 | 3 | Structured output: answer must be one of the 77 intent names | No free-text answers to interpret |
| 2026-10-27 | 3 | Zero-shot prompt v1: one-sentence task + the 77 intent names, nothing else | Plain first try a company would make; improve only after looking at validation mistakes |
| 2026-10-27 | 3 | Next: reasoning effort "low" (same prompt), then few-shot. No hand-written intent descriptions | Descriptions would mix in Claude's own judgement; few-shot tests the same idea with real train data |
| 2026-10-27 | 3 | Few-shot: random train examples per intent, fixed seed; 3 per intent first, then 5 (the 5 include the same 3) | Simple, fair, reproducible; 3 → 5 is one change; 5 links to the learning curve. Retrieval of similar examples is an optional later extra |
| 2026-10-27 | 3 | Stop LLM tuning; test once with best zero-shot (effort low) and best few-shot (5-shot, effort none). Skip 5-shot+low, 10-shot, retrieval | More tuning would polish a number without changing the conclusion (LLM well below both trained models) |
| 2026-10-27 | 4 | Learning curve, full version: 5/10/25/50 per intent × 3 seeds, baseline + DistilBERT, scored on test; nested subsets; seed 42's 5/intent = the LLM's 5-shot examples | Answers "how many labels before training beats the API"; 3 seeds show how much depends on which examples get labelled |
| 2026-10-27 | 4 | Learning curve DistilBERT: ~same number of steps as the full run (1,410), max 30 epochs, best epoch on val, stop after 3 epochs without gain; no re-tuning | With fixed 5 epochs the small sets would lose from too little training, not too little data. Caveat: full val picks the epoch (slightly optimistic) |
| 2026-10-27 | 4 | Learning curve: no early stopping before epoch 10, applied to all runs; re-train the 3 runs this changes (seed 43 at 5/intent, seed 44 at 25 and 50). Don't delete the failed run | First rule (stop after 3 epochs without gain) ended seed 43 / 5 per intent at epoch 4 while still at chance (1.3% test); other seeds needed 10–13 epochs. Dropping the seed would be cherry-picking. Original file kept as evidence |
| 2026-10-27 | 4 | Cost column for trained models = capacity on the laptop (no per-message fee; messages/hour on one CPU core at batch 1), not an assumed cloud price. Hardware wording: "ordinary laptop (Apple M2, 16 GB)"; DistilBERT trained on the M2 GPU, inference on CPU | Measured, checkable numbers instead of a guessed server price; don't claim CPU training |
| 2026-10-27 | 5 | No temperature scaling | It doesn't change routing (order of confidences is unchanged); not worth the extra step |
| 2026-10-28 | 5 | Target precision 97% (headline); also report 95% and 99% | Middle ground between automation and mistakes reaching customers; the 3-row table shows the trade-off |
| 2026-10-28 | 5 | Headline routing = plain confidence, baseline vs DistilBERT seed 44; margin + 3-seed ensemble as extra rows; served model stays single seed 44 | Margin gave no gain on val; ensemble +1 point for 3× compute |
| 2026-10-28 | 6 | Build and test locally in Docker first, then deploy to Azure via Azure for Students ($100, no card, through the GitHub Student Pack) | Live link for the CV at no cost. HF Spaces (Docker) now seems to need a paid plan for new accounts; Render free (512 MB) is too small for DistilBERT + PyTorch. Fallback if Azure fails: ONNX + 8-bit model on Render |
| 2026-10-28 | 6 | Repo private until the README is final; API returns intent, confidence, route, 2 runner-ups; demo page with 5 example buttons, no caching, nothing stored | Osman's choices; runner-ups help the human agent and show the model's doubt |
| 2026-10-28 | 6 | Demo examples, option A: keep "cash out abroad" (confident mistake, 91%, auto) and replace the top-up example with "My money hasn't shown up yet" (56%, human) | Shows both a real error that passes routing and a real banking message routed to a human; no showroom |
| 2026-10-27 | 2 | No class weights first; then one run with weights on | One change per experiment; comparable with the baseline |
| 2026-10-21 | 4 | Do the learning-curve experiment (5/10/25/50 examples per intent) | Answers *when* fine-tuning beats the LLM; Osman wants to learn as much as possible |

## Open decisions

- **Part 2:** which hyperparameters to vary; seeds; class weights; max length.
- **Part 3:** reasoning effort; full test set vs subset; answer format; prompt; how to
  pick few-shot examples.
- **Part 6:** API response format, demo page or not, git/GitHub setup (see Status).

## Part 0 findings (data)

- Source: original CSVs from github.com/PolyAI-LDN/task-specific-datasets, pinned to
  commit 57ec275, SHA-256 checked. The Hugging Face copy no longer loads with `datasets`
  5.x (it relies on a loading script, which is no longer supported).
- Train is **imbalanced**: 32 (contactless_not_working) to 168 (card_payment_fee_charged)
  examples per intent. Test is **perfectly balanced**: 40 per intent.
- Validation has only 3–19 examples per intent, so per-intent validation scores for rare
  intents will be noisy; overall validation scores are fine.
- Messages are short: median 10 words, max 79. Contains typos ("witdrawal") and some
  labels that look debatable (a "place a hold on my account" message is labelled
  cash_withdrawal_not_recognised). Useful later for error analysis.
- 6 messages appear word-for-word in both train and test (0.2% of test), all with the
  same label. 0 overlap between train/val and val/test.

## Part 1 findings (baseline errors on test)

- 289 of 3,080 wrong. Errors are between near-synonym intents, not random: identity
  (verify_my_identity / why_verify_identity / unable_to_verify_identity), top-ups
  (failed / reverted / pending), transfers (pending / balance not updated / not received),
  card arrival vs delivery estimate. Top 10 pairs = 20% of errors.
- Cause: same words, different meaning ("What is the point of an identity check?" is
  "why", but shares the words of "verify my identity"). TF-IDF ignores word order and
  meaning, which is what DistilBERT should fix. Some labels are truly ambiguous ("When
  will I get my card?" labelled card_arrival, predicted card_delivery_estimate), so no
  model reaches 100%.
- Confidence separates right (mean 0.87) from wrong (mean 0.48): good sign for Part 5.
  (Looked at on test for understanding only; the threshold will be picked on validation.)

## Concepts explained (glossary)

- **Train / validation / test split:** train = what the model learns from; validation =
  practice exam used to make choices (settings, prompts, thresholds); test = final exam,
  used once at the end. If you make choices by looking at test, the final score is
  inflated, because you tuned to that exam.
- **Decision:** any choice about how the project is built, in any part, not only for
  fine-tuning.
- **Hyperparameter:** a setting you choose before training (learning rate, epochs, batch
  size). Parameters (the weights) are what the model learns itself.
- **77-class output layer:** DistilBERT turns a message into 768 numbers that describe its
  meaning. A small added layer turns those 768 numbers into 77 scores, one per intent.
  Softmax turns scores into probabilities that sum to 1. Highest probability = prediction;
  that probability is also the confidence used for routing in Part 5.
- **Epoch:** one full pass over all training examples. Too few epochs: the model hasn't
  learned enough (underfitting). Too many: it memorises train and gets worse on new
  messages (overfitting). The validation score shows where to stop.
- **Learning rate:** how big a step the model takes each time it corrects a mistake. Too big
  and training is unstable; too small and it learns slowly.
- **Batch size:** how many examples the model looks at before each correction step.
- **How to know what to tune:** start from standard values for DistilBERT text
  classification (learning rate 2e-5 to 5e-5, batch size 16 or 32, 3–5 epochs), run once,
  read the validation score and the training curve, then change one thing at a time.
- **Why also an LLM:** it is the alternative a company would consider: prompt an API, no
  training and no labelled data needed. The comparison shows the trade-off.
- **"Matched GPT-4o-mini":** the draft CV bullet guessed the fine-tuned model would score
  the *same* as the LLM but be much cheaper and faster. Beating it would be even better.
  Either way, the CV bullet is written from the measured result, not decided in advance.
- **Validation is never trained on:** the split happens *before* training. The validation
  messages are removed from train, so the model never sees them. Train ~9,000 / validation
  ~1,000 / test 3,080, no overlap.
- **Why 768 numbers:** DistilBERT's "hidden size", fixed by its designers (copied from
  BERT-base). More numbers = more room to describe meaning, but a bigger, slower model.
  We can't change it without training the model from scratch.
- **Why repeat epochs:** each correction is deliberately tiny, so one pass moves the model
  only a little way towards good answers. Like going through flashcards several times.
  Examples are shuffled every epoch.
- **2e-5 to 5e-5:** scientific notation, = 0.00002 to 0.00005. Kept small because the
  model already knows English; big steps would overwrite that knowledge ("catastrophic
  forgetting").
- **Where BANKING77 comes from:** not stored in the repo. `src/data.py` downloads it from
  PolyAI's GitHub into `data/raw/` (outside git).
- **Sanity check:** a quick automatic test that nothing silly went wrong (sizes right, all
  77 intents in each split, no empty messages, no leaked duplicates). Catches bugs that
  would quietly make every later number wrong.
- **Why the dataset is only ~1 MB:** 13,083 short messages, ~12 words each, ~1 byte per
  character. Text is tiny compared with images. "Big" for text data means millions of
  examples; BANKING77 is small (~130 training examples per intent). The *model* is the
  big part: DistilBERT is ~260 MB (66 million learned numbers).
- **TF-IDF:** turns a message into one number per vocabulary word (vocabulary built from
  train only). 0 if the word is absent; higher for words that are rare across messages
  ("contactless") than for common ones ("card"). Words never seen in train are ignored.
- **n-grams:** word pairs ("top up") or short letter sequences ("witd") counted as extra
  "words". Character n-grams help with typos.
- **Logistic regression (multinomial):** one weight table, 77 rows (intents) × vocabulary
  columns, plus a starting score (bias) per intent. Score of an intent = sum over words of
  (TF-IDF number × weight) + starting score. Softmax turns the 77 scores into
  probabilities summing to 1; highest = prediction, its value = confidence. Basic logistic
  regression is binary; multinomial trains all 77 rows together so they compete.
  scikit-learn uses multinomial by default.
- **Training:** each training message nudges the weights a little so the model would be
  more right next time. Intents with more messages get more nudges.
- **C:** a number set before training that limits how big the weights can get (C = 1/λ,
  the regularisation strength). Large C = big weights, trusts single words, overfits. Small
  C = small weights, cautious, underfits. C is used only during training; prediction just
  uses the resulting table. Picked by trying several values on validation.
- **Class weights:** make each message of a rare intent count more during training, so
  every intent has equal total influence. Without them the starting scores favour intents
  with many training examples, which is wrong for the balanced test set.
- **words + chars:** two TF-IDF tables glued side by side. One has a column per whole
  word; the other has a column per 3–5-letter piece of each word ("wit", "dra", "rawal").
  A typo breaks the whole word but keeps most of its pieces, so the model still
  recognises it. char_wb = pieces taken within words only, padded with spaces at the edges.
- **How `src/baseline.py` works:** `grid` trains every setting on train and scores it
  on validation (also train accuracy, to see under/overfitting); `test` trains the chosen
  setting on train, scores it once on test, times predictions. Both save a JSON and add
  a row to `results/history.md`.
- **Think in errors, not accuracy:** 90.6% → 98% sounds small but cuts mistakes from
  289 to ~62 (4.7× fewer). Each mistake is a customer sent to the wrong team. Report
  error reduction alongside accuracy.
- **Fine-tuning, the whole picture:** DistilBERT was pre-trained by others on huge amounts
  of English (fill-in-the-missing-word), so it already turns a message into 768 numbers
  that capture meaning. We add a new 77-intent layer (random at first) and train
  everything on our labelled messages: batch of 32 → predict → loss (how wrong) →
  backpropagation (which way to nudge each of the 66M numbers) → small nudge of size
  learning rate. 282 nudges per epoch, 5 epochs. Pre-training teaches language;
  fine-tuning teaches our 77 intents.
- **Seed / 3-seed run:** training uses random numbers (starting values of the new layer,
  shuffle order, dropout). A seed is the starting number of the random generator: same
  seed → same random numbers → repeatable run. Training the same settings with 3 seeds
  (42, 43, 44) gives 3 scores; their mean ± spread shows how much is luck. A gain smaller
  than the spread isn't proven.
- **Why few-shot examples come from train, not test (or validation):** test examples in
  the prompt = showing the model exam answers, so the score is inflated and unfair (the
  other models never saw test). A real bank only has labels for past messages (= train).
  Validation is the practice exam for our choices, so it can't be in the prompt either.
- **Is DistilBERT an LLM?** Same family (a transformer language model, pre-trained on
  lots of text), but not "large" and not generative. DistilBERT is an *encoder*: it reads
  a message and turns it into 768 numbers; it can't write text or follow instructions,
  so it must be fine-tuned for each task. LLMs like GPT are *decoders* with billions of
  parameters that generate text and follow prompts. So the comparison is "small
  specialised model trained on your data vs large general model prompted".
- **Learning curve:** train the same model on growing amounts of data (5/10/25/50 per
  intent, then all) and plot test accuracy against data size. Shows how much labelled
  data training needs before it beats the LLM (which used 5 per intent).
- **Temperature scaling:** divide the model's 77 raw scores by one number T before
  softmax (same maths as LLM sampling temperature). T > 1 flattens overconfident
  probabilities. T is fitted on validation (minimise val loss). It never changes which
  intent wins, so accuracy is unchanged, and it never changes the *order* of confidences,
  so routing coverage/precision at a threshold chosen on validation is unchanged too. It
  only makes the number honest ("0.9" ≈ right 90% of the time). Fixes that can change
  routing: ensemble (average the 3 seed models), top-1 minus top-2 margin, label
  smoothing (retrain), MC dropout. In LLMs a high temperature lets 2nd/3rd words appear
  only because generation *samples* (random draw); our classifier always takes the top
  intent, so T only changes the confidence number.
- **Plan for Part 5 (agreed 2026-10-27):** plain confidence as the main method; margin
  and 3-seed ensemble as free extra experiments; no temperature scaling (Osman decided).
  Decide the target (e.g. 97% precision).
- **Loss:** a single number for how wrong the model is: minus the log of the probability
  it gave the right intent. Right with 90% → 0.1; right with only 10% → 2.3.
- **Validation vs. test:** same kind of data, scored the same way, never trained on.
  Validation is for choosing settings (many times); test is for the final number (once).
  Validation is uneven like train (3–19 per intent); test is balanced (40 per intent).
- **Threshold vs target precision (Osman asked, 2026-10-28):** 0.86 is the per-message
  confidence threshold (below it → human). 97% is the required accuracy of the whole
  auto-handled *group*. They differ because the model is right more often than its
  confidence says (at 90% confidence it's usually right well over 90% of the time), so
  messages at 86–97% confidence still keep the group at 97%. Sending everything below
  97% confidence to a human would give ~99% precision but automate far less.
