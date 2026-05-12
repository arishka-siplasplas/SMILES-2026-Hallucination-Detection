"""Stratified k-fold train/validation/test splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

N_SPLITS = 5


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    y = np.asarray(y)
    indices = np.arange(len(y))
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)

    splits = []
    for train_val_idx, test_idx in folds.split(indices, y):
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=random_state,
            stratify=y[train_val_idx],
        )
        splits.append((train_idx, val_idx, test_idx))
    return splits
