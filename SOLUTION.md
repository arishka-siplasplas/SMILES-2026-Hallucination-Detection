# Solution

## How to run

```bash
git clone <repo-url>
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
```

Python 3.11, PyTorch 2.x. Seed is 42 everywhere so predictions.csv should reproduce exactly.

## What I changed

Three files: aggregation.py, probe.py, splitting.py.

The main thing that actually improved results was changing how I aggregate the hidden states. Originally I was just taking the mean over all non-padding tokens plus the last token per layer, but this mixes the long prompt together with the short response. The prompt has the full context passage plus the system/user template, so averaging over everything buries the response signal. I added a second mean that only covers the last 64 real tokens — for most examples these are almost entirely response tokens since the response is short and sits at the end. Feature dimension went from 7168 to 10752 (each of 4 layers now gives 3 vectors: full mean, tail mean, last token instead of 2).

For the probe I used a linear classifier: StandardScaler → PCA(80) → single linear layer trained with BCEWithLogitsLoss and AdamW for 300 steps with cosine LR schedule. I started with a small MLP but it completely overfitted — train AUROC hit 100% while test AUROC was 66% and test accuracy dropped below the majority baseline. The linear probe generalizes much better. I also added threshold calibration: hold out 20% of training data, search for the threshold that maximizes accuracy on it, then refit weights on everything and keep that threshold. This matters because the default 0.5 threshold doesn't work well with 70/30 class imbalance.

Splitting: StratifiedKFold(5) with a stratified 15% validation split carved out of each fold's training portion.

## What didn't work

MLP with 64 hidden units — train AUROC 100%, test AUROC 66%, test accuracy below baseline. Classic overfitting on ~470 training samples.

Late layers only (13, 17, 21, 24) — tried this first. Layers 8, 12, 16, 20 worked better, mid-network representations seem to carry more factual content in this model size.

LinearDiscriminantAnalysis with automatic shrinkage replacing PCA+linear — in theory better because it uses class labels to find discriminative directions. In practice slightly worse (~68% test AUROC), possibly because with only ~330 samples per class the within-class covariance estimate is noisy even with shrinkage.

LogisticRegressionCV with grid search over C — marginally better than fixed-regularization linear (~69% AUROC) but still worse than the tail tokens aggregation.

Adding std over tokens as an extra feature — did not move test AUROC in a consistent direction across folds.

The tail tokens change was the biggest single improvement: ~68% → ~73% test AUROC.
