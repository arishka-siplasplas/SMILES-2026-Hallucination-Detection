# Solution Report

## Reproducibility

Environment: Python 3.10+, the packages pinned in `requirements.txt` (PyTorch, transformers, scikit-learn, pandas, numpy, tqdm). No GPU is required; `solution.py` auto-selects CUDA / MPS / CPU.

```bash
git clone <repo-url>
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
python solution.py
```

Running `solution.py` writes `results.json` and `predictions.csv` to the project root. All randomness is seeded with 42 — `StratifiedKFold`, `train_test_split`, `PCA`, and `torch.manual_seed` (re-set before each weight-fitting pass in `HallucinationProbe.fit`) — so the same `predictions.csv` is produced on repeated runs. The probe has no dropout or batch-norm, so there is no train/eval-mode source of variance either.

## Final Solution

Three files were modified: `aggregation.py`, `probe.py`, `splitting.py`. The rest of the codebase, including `USE_GEOMETRIC = False` in `solution.py`, is untouched.

### Features (`aggregation.py`)

For each input I read the hidden states at layers 8, 12, 16 and 20 — a spread that covers the part of the network where, in this size of model, factual content is consolidated, while staying away from the very last layers which are dominated by next-token surface statistics. From every selected layer I take two vectors: the mean over all non-padding tokens, and the hidden state at the last real token (the `<|endoftext|>` position, which has attended to the whole prompt and response). The concatenation is 4 × 2 × 896 = 7168-dimensional. The last-token index is recovered from the attention mask, so the code is correct whichever side the tokenizer pads on. `extract_geometric_features` (layer-wise activation norms, cosine similarity between consecutive layers' pooled states, sequence length) is implemented but, since `USE_GEOMETRIC` stays off, it does not enter the submitted run.

### Probe (`probe.py`)

`StandardScaler` → `PCA(80)` → a single linear layer (`nn.Linear(d, 1)`), trained with `BCEWithLogitsLoss`, full-batch AdamW (lr 5e-3, weight decay 1e-2) for 300 steps with a cosine schedule. The first version of this probe was a one-hidden-layer MLP; with only ~470 training rows it reached 100% train AUROC but ~66% test AUROC — it was memorising. A linear probe on a PCA-compressed feature space cannot do that: it has ~80 parameters, the train and held-out scores stay close, and it still recovers most of the linearly-decodable hallucination signal, which is what these probes rely on.

`fit` also calibrates the decision threshold: it carves out a stratified 20% of the data passed to it, fits weights on the other 80%, picks the threshold that maximises accuracy on that held-out fifth, then refits the weights on the full data and keeps that threshold. This matters because `solution.py` only calls `fit` (not `fit_hyperparameters`) on the probe that produces `predictions.csv`, so without it the submission would run at a fixed 0.5 cut-off, which on a 70/30-imbalanced set is not where accuracy is maximised. `fit_hyperparameters`, used by `evaluate.py` per fold, re-tunes the same accuracy-based threshold on the official validation split.

### Splitting (`splitting.py`)

`StratifiedKFold(n_splits=5, shuffle=True)`; inside each fold a further stratified 15% of the training portion is held out for validation, giving five (train ≈ 470, val ≈ 83, test ≈ 138) splits with the class ratio preserved everywhere. Every sample lands in exactly one test fold, so when `solution.py` builds the final probe from the union of all train and validation indices it trains on the full 689-row dataset before predicting on `test.csv`.

### What helped most

Replacing the MLP with the linear probe removed the train/test collapse and was the decisive change. Calibrating the threshold inside `fit` was the second: it moves the submitted predictions off the arbitrary 0.5 cut-off, which on this imbalanced data is worth a couple of points of accuracy.

## Experiments and Discarded Ideas

- **One-hidden-layer MLP (64 units).** Train AUROC 1.0, test AUROC ≈ 0.66, test accuracy below the majority baseline — textbook overfitting on 689 rows. Replaced by the linear probe.
- **Late layers only (13, 17, 21, 24).** The first feature set used these; shifting earlier (8–20) gave a cleaner signal, consistent with mid-network layers carrying more of the factual content in a small model.
- **PCA with 128+ components.** More components let the probe fit the training set tighter without improving held-out accuracy; 80 is roughly where the curve flattens.
- **`pos_weight` class balancing.** With a 70/30 split and accuracy as the ranking metric, down-weighting the majority class lowered held-out accuracy. The unbalanced loss plus an accuracy-tuned threshold does better.
- **Geometric features on (`USE_GEOMETRIC = True`).** Not part of the official run (`solution.py` keeps the flag off); in side experiments appending the norm / cosine-drift features did not move validation accuracy in a stable way, so there was nothing to gain from relying on them.
- **Threshold tuned for F1 (the skeleton default).** Replaced with accuracy-based tuning to match the ranking metric.
