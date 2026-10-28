# Experiment log

One row per run, newest last. Raw results: `results/<run_name>_<UTC timestamp>.json`.
Validation numbers for tuning runs; test numbers only for final runs (say which).

| timestamp | split | approach | accuracy | macro F1 | latency / query | cost / 1k | notes |
|---|---|---|---|---|---|---|---|
| 20261027T150910Z | val | TF-IDF + LR | 0.9051 | 0.9068 | – | – | grid of 30 settings; best by val F1: words+chars, C=10, class weights balanced. baseline_grid_20261027T150910Z.json |
| 20261027T151642Z | **test** | TF-IDF + LR | 0.9062 | 0.9063 | 0.53 ms (batch 1, laptop) | – | final baseline: words+chars, C=10, class weights balanced. baseline_test_20261027T151642Z.json |
| 20261027T153436Z | val | DistilBERT | 0.9171 | 0.9208 | – | – | lr=5e-05, batch 32, class weights none, seed 42; best epoch 4/5. finetune_lr5e-05_bs32_cw-none_seed42_20261027T153436Z.json |
| 20261027T154259Z | val | DistilBERT | 0.9071 | 0.9110 | – | – | lr=3e-05, batch 32, class weights none, seed 42; best epoch 5/5. finetune_lr3e-05_bs32_cw-none_seed42_20261027T154259Z.json |
| 20261027T155230Z | val | DistilBERT | 0.8851 | 0.8761 | – | – | lr=2e-05, batch 32, class weights none, seed 42; best epoch 5/5. finetune_lr2e-05_bs32_cw-none_seed42_20261027T155230Z.json |
| 20261027T160627Z | val | DistilBERT | 0.9161 | 0.9214 | – | – | lr=0.0001, batch 32, class weights none, seed 42; best epoch 3/5. finetune_lr0.0001_bs32_cw-none_seed42_20261027T160627Z.json |
| 20261027T161844Z | val | DistilBERT | 0.9231 | 0.9258 | – | – | lr=0.0002, batch 32, class weights none, seed 42; best epoch 4/5. finetune_lr0.0002_bs32_cw-none_seed42_20261027T161844Z.json |
| 20261027T163055Z | val | DistilBERT | 0.9001 | 0.9006 | – | – | lr=0.0004, batch 32, class weights none, seed 42; best epoch 5/5. finetune_lr0.0004_bs32_cw-none_seed42_20261027T163055Z.json |
| 20261027T164716Z | val | DistilBERT | 0.9201 | 0.9237 | – | – | lr=0.0002, batch 32, class weights balanced, seed 42; best epoch 5/5. finetune_lr0.0002_bs32_cw-balanced_seed42_20261027T164716Z.json |
| 20261027T170035Z | val | DistilBERT | 0.9231 | 0.9283 | – | – | lr=0.0002, batch 32, class weights none, seed 43; best epoch 5/5. finetune_lr0.0002_bs32_cw-none_seed43_20261027T170035Z.json |
| 20261027T171332Z | val | DistilBERT | 0.9271 | 0.9306 | – | – | lr=0.0002, batch 32, class weights none, seed 44; best epoch 4/5. finetune_lr0.0002_bs32_cw-none_seed44_20261027T171332Z.json |
| 20261027T172829Z | **test** | DistilBERT | 0.9198 | 0.9196 | 17.84 ms CPU / 8.10 ms MPS (batch 1, laptop) | – | final DistilBERT: finetune_lr0.0002_bs32_cw-none_seed42_20261027T161844Z. finetune_test_20261027T172829Z.json |
| 20261027T173012Z | **test** | DistilBERT | 0.9273 | 0.9272 | 17.15 ms CPU / 7.51 ms MPS (batch 1, laptop) | – | final DistilBERT: finetune_lr0.0002_bs32_cw-none_seed43_20261027T170035Z. finetune_test_20261027T173012Z.json |
| 20261027T173204Z | **test** | DistilBERT | 0.9169 | 0.9167 | 20.02 ms CPU / 7.31 ms MPS (batch 1, laptop) | – | final DistilBERT: finetune_lr0.0002_bs32_cw-none_seed44_20261027T171332Z. finetune_test_20261027T173204Z.json |
| 20261027T181219Z | val | gpt-5.4-mini zero-shot, effort none | 0.7872 | 0.7867 | 1023 ms (batch 1, API) | $0.73 | llm_zeroshot_none_val_20261027T181219Z.json |
| 20261027T182543Z | val | gpt-5.4-mini zero-shot, effort low | 0.8132 | 0.8134 | 1746 ms (batch 1, API) | $0.90 | llm_zeroshot_low_val_20261027T182543Z.json |
| 20261027T183404Z | val | gpt-5.4-mini 3-shot, effort none | 0.8611 | 0.8679 | 1363 ms (batch 1, API) | $0.47 | llm_3shot_none_val_20261027T183404Z.json |
| 20261027T183859Z | val | gpt-5.4-mini 5-shot, effort none | 0.8721 | 0.8800 | 1229 ms (batch 1, API) | $0.69 | llm_5shot_none_val_20261027T183859Z.json |
| 20261027T191057Z | **test** | gpt-5.4-mini zero-shot, effort low | 0.8110 | 0.8032 | 1231 ms (batch 1, API) | $0.88 | llm_zeroshot_low_test_20261027T191057Z.json |
| 20261027T192343Z | **test** | gpt-5.4-mini 5-shot, effort none | 0.8864 | 0.8848 | 1148 ms (batch 1, API) | $0.66 | llm_5shot_none_test_20261027T192343Z.json |
| 20261027T194744Z | **test** | TF-IDF + LR, 5/intent | 0.6422 | 0.6407 | – | – | learning curve, mean of 3 seeds (acc std 0.0092). learning_curve_20261027T194744Z.json |
| 20261027T194744Z | **test** | TF-IDF + LR, 10/intent | 0.7398 | 0.7382 | – | – | learning curve, mean of 3 seeds (acc std 0.0065). learning_curve_20261027T194744Z.json |
| 20261027T194744Z | **test** | TF-IDF + LR, 25/intent | 0.8357 | 0.8350 | – | – | learning curve, mean of 3 seeds (acc std 0.0044). learning_curve_20261027T194744Z.json |
| 20261027T194744Z | **test** | TF-IDF + LR, 50/intent | 0.8815 | 0.8815 | – | – | learning curve, mean of 3 seeds (acc std 0.0068). learning_curve_20261027T194744Z.json |
| 20261027T194744Z | **test** | DistilBERT, 5/intent | 0.4587 | 0.4511 | – | – | learning curve, mean of 3 seeds (acc std 0.3860). learning_curve_20261027T194744Z.json |
| 20261027T194744Z | **test** | DistilBERT, 10/intent | 0.7997 | 0.7997 | – | – | learning curve, mean of 3 seeds (acc std 0.0056). learning_curve_20261027T194744Z.json |
| 20261027T194744Z | **test** | DistilBERT, 25/intent | 0.8682 | 0.8683 | – | – | learning curve, mean of 3 seeds (acc std 0.0063). learning_curve_20261027T194744Z.json |
| 20261027T194744Z | **test** | DistilBERT, 50/intent | 0.8982 | 0.8983 | – | – | learning curve, mean of 3 seeds (acc std 0.0067). learning_curve_20261027T194744Z.json |
| 20261027T204854Z | **test** | TF-IDF + LR, 5/intent | 0.6422 | 0.6407 | – | – | learning curve, mean of 3 seeds (acc std 0.0092). learning_curve_20261027T204854Z.json |
| 20261027T204854Z | **test** | TF-IDF + LR, 10/intent | 0.7398 | 0.7382 | – | – | learning curve, mean of 3 seeds (acc std 0.0065). learning_curve_20261027T204854Z.json |
| 20261027T204854Z | **test** | TF-IDF + LR, 25/intent | 0.8357 | 0.8350 | – | – | learning curve, mean of 3 seeds (acc std 0.0044). learning_curve_20261027T204854Z.json |
| 20261027T204854Z | **test** | TF-IDF + LR, 50/intent | 0.8815 | 0.8815 | – | – | learning curve, mean of 3 seeds (acc std 0.0068). learning_curve_20261027T204854Z.json |
| 20261027T204854Z | **test** | DistilBERT, 5/intent | 0.4587 | 0.4511 | – | – | learning curve, mean of 3 seeds (acc std 0.3860). learning_curve_20261027T204854Z.json |
| 20261027T204854Z | **test** | DistilBERT, 10/intent | 0.7997 | 0.7997 | – | – | learning curve, mean of 3 seeds (acc std 0.0056). learning_curve_20261027T204854Z.json |
| 20261027T204854Z | **test** | DistilBERT, 25/intent | 0.8736 | 0.8737 | – | – | learning curve, mean of 3 seeds (acc std 0.0041). learning_curve_20261027T204854Z.json |
| 20261027T204854Z | **test** | DistilBERT, 50/intent | 0.9012 | 0.9012 | – | – | learning curve, mean of 3 seeds (acc std 0.0039). learning_curve_20261027T204854Z.json |
| 20261028T095719Z | val | TF-IDF + LR, routing (confidence) | – | – | – | – | target 97% precision, threshold 0.6880 (from val): handled automatically 80.0%, precision on those 0.9700. routing_val_20261028T095719Z.json |
| 20261028T095719Z | val | TF-IDF + LR, routing (margin) | – | – | – | – | target 97% precision, threshold 0.5749 (from val): handled automatically 79.5%, precision on those 0.9711. routing_val_20261028T095719Z.json |
| 20261028T095719Z | val | DistilBERT, routing (confidence) | – | – | – | – | target 97% precision, threshold 0.8629 (from val): handled automatically 90.1%, precision on those 0.9701. routing_val_20261028T095719Z.json |
| 20261028T095719Z | val | DistilBERT, routing (margin) | – | – | – | – | target 97% precision, threshold 0.7705 (from val): handled automatically 90.3%, precision on those 0.9701. routing_val_20261028T095719Z.json |
| 20261028T095719Z | val | DistilBERT 3-seed ensemble, routing (confidence) | – | – | – | – | target 97% precision, threshold 0.7279 (from val): handled automatically 91.3%, precision on those 0.9705. routing_val_20261028T095719Z.json |
| 20261028T100158Z | **test** | TF-IDF + LR, routing (confidence) | – | – | – | – | target 97% precision, threshold 0.6880 (from val): handled automatically 79.6%, precision on those 0.9772. routing_test_20261028T100158Z.json |
| 20261028T100158Z | **test** | TF-IDF + LR, routing (margin) | – | – | – | – | target 97% precision, threshold 0.5749 (from val): handled automatically 79.0%, precision on those 0.9795. routing_test_20261028T100158Z.json |
| 20261028T100158Z | **test** | DistilBERT, routing (confidence) | – | – | – | – | target 97% precision, threshold 0.8629 (from val): handled automatically 88.7%, precision on those 0.9693. routing_test_20261028T100158Z.json |
| 20261028T100158Z | **test** | DistilBERT, routing (margin) | – | – | – | – | target 97% precision, threshold 0.7705 (from val): handled automatically 88.9%, precision on those 0.9690. routing_test_20261028T100158Z.json |
| 20261028T100158Z | **test** | DistilBERT 3-seed ensemble, routing (confidence) | – | – | – | – | target 97% precision, threshold 0.7279 (from val): handled automatically 90.6%, precision on those 0.9735. routing_test_20261028T100158Z.json |
