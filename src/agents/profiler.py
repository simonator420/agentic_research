"""
profiler.py — Profiler Agent (Step 1 of the agentic loop).

Analyses a raw DataFrame and produces a DataProfile: a structured summary of
column types, missing value rates, numeric statistics, categorical distributions,
duplicate count, and the inferred ML task type.

The DataProfile is the primary input to every subsequent agent:
  - Issue Detector uses it to assign severity scores without re-scanning the data.
  - Planner embeds it verbatim into the LLM prompt as dataset context.
  - dataset_fingerprint() converts it to a vector for cross-run memory retrieval.

Public API
----------
generate_profile(df, target) -> DataProfile
"""

from typing import Dict

import pandas as pd

from src.models.schemas import ColumnProfile, DataProfile, TargetType

# Up to this many sample values are stored per column for the LLM prompt.
_MAX_SAMPLE_VALUES = 5

# Numeric target columns with more unique values than this are treated as regression,
# not multiclass. Chosen conservatively — most classification tasks have ≤ 20 classes.
_MULTICLASS_THRESHOLD = 20

# Number of most-frequent categories stored in ColumnProfile.top_categories.
_TOP_CATEGORIES = 10


def _infer_dtype(series: pd.Series) -> str:
    """
    Map a pandas Series to one of four logical types.

    The order of checks matters: bool must come before numeric because
    pandas considers bool a subtype of int.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _profile_column(series: pd.Series) -> ColumnProfile:
    """
    Compute per-column statistics and return a ColumnProfile.

    Numeric stats (mean, std, min, max, q1, q3) are populated only for numeric
    columns. q1 / q3 are used downstream by the Issue Detector for IQR-based
    outlier detection, so they must be present before detect_issues() is called.
    """
    dtype = _infer_dtype(series)
    non_null = series.dropna()
    missing_rate = float(series.isna().mean())
    n_unique = int(series.nunique())
    sample_values = non_null.unique()[:_MAX_SAMPLE_VALUES].tolist()

    col = ColumnProfile(
        name=str(series.name),
        dtype=dtype,
        missing_rate=missing_rate,
        n_unique=n_unique,
        sample_values=sample_values,
    )

    if dtype == "numeric" and len(non_null) > 0:
        col.mean = float(non_null.mean())
        col.std = float(non_null.std()) if len(non_null) > 1 else 0.0
        col.min = float(non_null.min())
        col.max = float(non_null.max())
        col.q1 = float(non_null.quantile(0.25))
        col.q3 = float(non_null.quantile(0.75))

    elif dtype == "categorical":
        counts = non_null.value_counts()
        col.top_categories = {str(k): int(v) for k, v in counts.head(_TOP_CATEGORIES).items()}

    return col


def _infer_target_type(series: pd.Series) -> TargetType:
    """
    Decide the ML task type from the target column alone — no user input required.

    Rules:
      - numeric AND more than _MULTICLASS_THRESHOLD unique values → REGRESSION
      - exactly 2 unique values → BINARY
      - everything else → MULTICLASS
    """
    n_unique = series.nunique()
    if pd.api.types.is_numeric_dtype(series) and n_unique > _MULTICLASS_THRESHOLD:
        return TargetType.REGRESSION
    if n_unique <= 2:
        return TargetType.BINARY
    return TargetType.MULTICLASS


def generate_profile(df: pd.DataFrame, target: str) -> DataProfile:
    """
    Main entry point for the Profiler Agent.

    Scans every column, infers the task type, computes class distribution
    (classification only), and counts duplicate rows.

    Parameters
    ----------
    df     : raw input DataFrame (train + test combined, before any split).
    target : name of the column to predict.

    Returns
    -------
    DataProfile — used by all downstream agents and stored in memory.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    columns: Dict[str, ColumnProfile] = {col: _profile_column(df[col]) for col in df.columns}

    target_type = _infer_target_type(df[target])

    # Class distribution is only meaningful for classification tasks.
    if target_type == TargetType.REGRESSION:
        class_dist = None
    else:
        counts = df[target].value_counts(normalize=True)
        class_dist = {str(k): float(v) for k, v in counts.items()}

    return DataProfile(
        n_rows=int(len(df)),
        n_cols=int(len(df.columns)),
        target_column=target,
        target_type=target_type,
        class_distribution=class_dist,
        columns=columns,
        n_duplicates=int(df.duplicated().sum()),
    )
