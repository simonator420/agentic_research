"""
loader.py — data I/O utilities and dataset fingerprinting.

load_data()          : read a CSV / Parquet / Excel file into a DataFrame.
split_data()         : stratification-free train/test split (preserves row order randomness).
dataset_fingerprint(): convert a DataProfile into a fixed-length numerical vector
                       used by ChromaDB for cosine-similarity search across runs.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _sk_split

from src.models.schemas import DataProfile, TargetType


def load_data(path: str) -> pd.DataFrame:
    """
    Load a tabular dataset from disk.

    Supported formats: .csv, .parquet / .pq, .xlsx / .xls.
    Raises ValueError for any other extension.
    """
    p = Path(path)
    loaders = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".pq": pd.read_parquet,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }
    loader = loaders.get(p.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file format: {p.suffix}")
    return loader(path)


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into train / test sets.

    Returns (X_train, X_test, y_train, y_test).
    The target column is removed from X so it cannot leak into features.
    random_state is fixed to ensure reproducible splits across all experiments.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return _sk_split(X, y, test_size=test_size, random_state=random_state)


def dataset_fingerprint(profile: DataProfile) -> list:
    """
    Convert a DataProfile into a fixed-length (8-dimensional) numerical vector.

    This vector is stored in ChromaDB and retrieved via cosine similarity to find
    datasets from past runs that are structurally similar to the current one.
    When a similar dataset is found, the Planner can warm-start from a previously
    successful configuration rather than searching from scratch.

    Vector layout:
      [0] n_rows             — dataset size
      [1] n_cols             — total feature count
      [2] n_numeric          — number of numeric columns
      [3] n_categorical      — number of categorical columns
      [4] overall_missing    — mean missing rate across all columns
      [5] target_type_enc    — 0.0=binary, 0.5=multiclass, 1.0=regression
      [6] imbalance_ratio    — max_class_freq / min_class_freq, capped at 100
      [7] duplicate_rate     — fraction of duplicate rows
    """
    cols = list(profile.columns.values())
    n_numeric = sum(1 for c in cols if c.dtype == "numeric")
    n_categorical = sum(1 for c in cols if c.dtype == "categorical")
    overall_missing = float(np.mean([c.missing_rate for c in cols]))

    target_enc = {
        TargetType.BINARY: 0.0,
        TargetType.MULTICLASS: 0.5,
        TargetType.REGRESSION: 1.0,
    }[profile.target_type]

    if profile.class_distribution:
        freqs = list(profile.class_distribution.values())
        # cap at 100 to prevent a single extreme dataset from dominating cosine distance
        imbalance = min(max(freqs) / (min(freqs) + 1e-9), 100.0)
    else:
        imbalance = 0.0

    dup_rate = profile.n_duplicates / max(profile.n_rows, 1)

    return [
        float(profile.n_rows),
        float(profile.n_cols),
        float(n_numeric),
        float(n_categorical),
        overall_missing,
        target_enc,
        imbalance,
        dup_rate,
    ]
