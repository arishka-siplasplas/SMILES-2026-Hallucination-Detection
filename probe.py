from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
PCA_COMPONENTS = 80
EPOCHS = 300


def _best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    grid = np.unique(np.concatenate([probs, np.linspace(0.1, 0.9, 17)]))
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Linear | None = None
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def _project(self, X: np.ndarray) -> torch.Tensor:
        X = self._scaler.transform(X)
        X = self._pca.transform(X)
        return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))

    def _train_weights(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self._build_network(inputs.shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        self.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            loss = criterion(self(inputs), targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        torch.manual_seed(SEED)
        y = y.astype(np.int64)

        X_std = self._scaler.fit_transform(X)
        n_components = min(PCA_COMPONENTS, X_std.shape[0] - 1, X_std.shape[1])
        self._pca = PCA(n_components=n_components, random_state=SEED)
        Z = np.ascontiguousarray(self._pca.fit_transform(X_std), dtype=np.float32)
        targets = torch.from_numpy(y.astype(np.float32))

        tr_idx, ho_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y
        )
        self._train_weights(
            torch.from_numpy(np.ascontiguousarray(Z[tr_idx])),
            torch.from_numpy(y[tr_idx].astype(np.float32)),
        )
        with torch.no_grad():
            ho_probs = torch.sigmoid(
                self(torch.from_numpy(np.ascontiguousarray(Z[ho_idx])))
            ).numpy()
        self._threshold = _best_threshold(y[ho_idx], ho_probs)

        torch.manual_seed(SEED)
        self._train_weights(torch.from_numpy(Z), targets)

        with torch.no_grad():
            full_probs = torch.sigmoid(self(torch.from_numpy(Z))).numpy()
        t_acc = _best_threshold(y, full_probs)
        t_prev = float(np.percentile(full_probs, 100.0 * (1.0 - float(y.mean()))))
        self._threshold = max(t_acc, t_prev)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = _best_threshold(y_val.astype(np.int64), probs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            probs = torch.sigmoid(self(self._project(X))).numpy()
        return np.stack([1.0 - probs, probs], axis=1)
