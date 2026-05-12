from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
PCA_COMPONENTS = 80


def _best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    grid = np.unique(np.concatenate([probs, np.linspace(0.1, 0.9, 17)]))
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


class HallucinationProbe:
    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._threshold: float = 0.5

    def _build_pipeline(self, n_samples: int, n_features: int) -> Pipeline:
        n_comp = min(PCA_COMPONENTS, n_samples - 1, n_features)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=SEED)),
            ("clf", LogisticRegressionCV(
                Cs=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                cv=5,
                max_iter=2000,
                random_state=SEED,
                scoring="roc_auc",
                n_jobs=1,
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        y = y.astype(int)
        tr_idx, ho_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y
        )
        pipe = self._build_pipeline(len(tr_idx), X.shape[1])
        pipe.fit(X[tr_idx], y[tr_idx])
        ho_probs = pipe.predict_proba(X[ho_idx])[:, 1]
        self._threshold = _best_threshold(y[ho_idx], ho_probs)

        self._pipeline = self._build_pipeline(len(y), X.shape[1])
        self._pipeline.fit(X, y)
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = _best_threshold(y_val.astype(int), probs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)
