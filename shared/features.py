"""Load shared sparse TF-IDF feature matrices and numeric labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

TASKS = ("aspect", "sentiment")
SPLITS = ("train", "val", "test")


def load_features(
    directory: str | Path,
    *,
    task: str,
    split: str,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Load features generated once by prepare_tfidf.py, for either model team."""
    if task not in TASKS or split not in SPLITS:
        raise ValueError(f"task must be {TASKS}; split must be {SPLITS}")
    directory = Path(directory)
    x = sparse.load_npz(directory / f"{task}_{split}_X.npz").tocsr()
    y = np.load(directory / f"{task}_{split}_y.npy", allow_pickle=False)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{task}_{split}: feature and label row counts do not match")
    return x, y
