# Solution

## Running

```bash
git clone https://github.com/arishka-siplasplas/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
```

Python 3.11, PyTorch 2.x. Everything seeds to 42 so the output should be the same every run.

## What I changed

I modified aggregation.py, probe.py, and splitting.py.

The most important change was in the aggregation. The input is `prompt + response` concatenated, and the prompt is quite long because it has the full context passage in it. The response is usually just one short sentence at the end. If you average hidden states over all tokens you mostly get a noisy average of the prompt, and the response part barely shows up. I added a second pooling vector that takes the mean over only the last 64 non-padding tokens — for most examples those are the response tokens. So each selected layer now gives 3 vectors (full mean, tail mean, last token) instead of 2. Feature size went from 7168 to 10752.

For the probe I used StandardScaler → PCA(80) → a single linear layer, trained with BCEWithLogitsLoss and AdamW for 300 steps. I also find the best decision threshold on a 20% holdout before the final refit. With 70% of labels being 1, the default 0.5 threshold doesn't work well for accuracy.

For splits: StratifiedKFold(5) with a 15% validation split inside each fold.

## What I tried but didn't keep

Started with a small MLP (64 hidden units). It memorized the training data completely — train AUROC was 100%, test was 66%, test accuracy went below the majority baseline. Switched to linear.

Tried using only the late layers (13, 17, 21, 24) first. Moving to layers 8, 12, 16, 20 improved things, I think earlier layers carry more of the factual signal in a small 0.5B model.

Tried LDA (LinearDiscriminantAnalysis, solver lsqr, automatic shrinkage) instead of PCA+linear. LDA should find better directions for classification since it uses the labels, but it was a bit worse in practice, probably because the dataset is small enough that the covariance estimates aren't reliable.

Tried LogisticRegressionCV with a C grid search — slightly better than the fixed-weight AdamW linear but less than what the tail tokens gave.

Adding std-pooling per layer as extra features didn't help consistently across folds.

The tail mean was the one change that clearly improved the number.
