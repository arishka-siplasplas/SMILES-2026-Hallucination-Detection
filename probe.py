"""Hallucination probe: scaling + PCA + a small MLP over hidden-state features."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

SEED = 42
PCA_COMPONENTS = 128


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def _project(self, X: np.ndarray) -> torch.Tensor:
        X = self._scaler.transform(X)
        X = self._pca.transform(X)
        return torch.from_numpy(X).float()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        torch.manual_seed(SEED)

        X_std = self._scaler.fit_transform(X)
        n_components = min(PCA_COMPONENTS, X_std.shape[0] - 1, X_std.shape[1])
        self._pca = PCA(n_components=n_components, random_state=SEED)
        inputs = torch.from_numpy(self._pca.fit_transform(X_std)).float()
        targets = torch.from_numpy(y.astype(np.float32))

        self._build_network(inputs.shape[1])

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)

        self.train()
        for _ in range(400):
            optimizer.zero_grad()
            loss = criterion(self(inputs), targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        grid = np.unique(np.concatenate([probs, np.linspace(0.05, 0.95, 19)]))

        best_threshold, best_acc = 0.5, -1.0
        for t in grid:
            acc = accuracy_score(y_val, (probs >= t).astype(int))
            if acc > best_acc:
                best_acc, best_threshold = acc, float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            probs = torch.sigmoid(self(self._project(X))).numpy()
        return np.stack([1.0 - probs, probs], axis=1)
