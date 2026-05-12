# Solution Report

## Reproducibility

Environment: Python 3.10+, the packages pinned in `requirements.txt` (PyTorch, transformers, scikit-learn, pandas, numpy, tqdm). No GPU is required; `solution.py` auto-selects CUDA / MPS / CPU.

```bash
git clone <repo-url>
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
python solution.py
```

Running `solution.py` writes `results.json` and `predictions.csv` to the project root. All randomness is seeded with 42 (`StratifiedKFold`, `train_test_split`, `PCA`, and `torch.manual_seed` at the start of `HallucinationProbe.fit`), so the same `predictions.csv` is produced on repeated runs.

## Final Solution

I modified the three expected files: `aggregation.py`, `probe.py`, `splitting.py`. `solution.py` and the rest of the infrastructure are unchanged, including `USE_GEOMETRIC = False`.

### Features (`aggregation.py`)

For each input I take the hidden states at layers 13, 17, 21 and 24. These cover the second half of the 24-layer network, which is where factual recall and the decision about what to emit happen; the embedding layer and the early layers add mostly surface-level noise for this task. From every selected layer I extract two vectors: the mean over all non-padding tokens, and the hidden state at the last real token (the position right before `<|endoftext|>`, which has attended to the whole prompt and response). Concatenating gives a 4 × 2 × 896 = 7168-dimensional feature vector. The token index is found from the attention mask, so the code is correct regardless of which side the tokenizer pads on.

`extract_geometric_features` is implemented (layer-wise activation norms, cosine similarity between consecutive layers' pooled representations, and sequence length) but is not part of the submitted run — see below.

### Probe (`probe.py`)

`StandardScaler` → `PCA(128)` → MLP. With ~470 training examples per fold and 7168 raw features a probe trained directly on the raw vectors overfits immediately, so PCA both decorrelates the features and cuts the input dimension by ~98%. The MLP is one hidden layer of 64 units with BatchNorm and Dropout(0.3) before a single logit. Training is full-batch AdamW (lr 2e-3, weight decay 1e-3) for 400 steps with a cosine schedule, optimising `BCEWithLogitsLoss`. I do not reweight the classes: the data is ~70% hallucinated and the test set follows the same distribution, so an unweighted loss leaves the model's prior aligned with the accuracy metric. `fit_hyperparameters` tunes the decision threshold on the validation fold for accuracy (the competition metric) rather than F1.

### Splitting (`splitting.py`)

`StratifiedKFold(n_splits=5, shuffle=True)`; inside each fold a further stratified 15% of the training portion is held out for validation, giving five (train ≈ 470, val ≈ 83, test ≈ 138) splits with the class ratio preserved everywhere. Every sample lands in exactly one test fold, so when `solution.py` builds the final probe from the union of all train and validation indices it trains on the full 689-sample dataset before predicting on `test.csv`.

### What helped most

Going from the skeleton (last token of the final layer only) to multi-layer pooling plus the last-token state was the largest single gain. PCA was the second: without it the validation accuracy was several points below the training accuracy on every fold.

## Experiments and Discarded Ideas

- **All 25 layers concatenated.** A 22400-dim feature vector. Even after PCA the per-fold metrics were noisier and slightly worse on average; the embedding and first few transformer layers seem to dilute rather than help.
- **Geometric features on (`USE_GEOMETRIC = True`).** Since `solution.py` keeps the flag off, this never enters the official run, but in my own experiments appending the norm/cosine-drift features to the 7168-dim vector did not move validation accuracy in a stable way, so there was nothing to gain by relying on it.
- **`pos_weight` for class balancing.** With a 70/30 split and accuracy as the metric, downweighting the majority class consistently lowered held-out accuracy. Dropped.
- **Bigger probe (two hidden layers, 256→64).** More capacity, faster training-set convergence, no improvement on validation — the PCA-reduced space is close to linearly separable, so the extra layer only adds variance.
- **Threshold tuned for F1 (the skeleton default).** Replaced with accuracy-based tuning to match the ranking metric.
