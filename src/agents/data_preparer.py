"""
data_preparer.py — Dataset-level preparation before ML pipeline construction.

Runs ONCE on the full DataFrame (before train/test split) to remove columns and
rows that are structurally useless — i.e. that no amount of imputation, encoding,
or scaling can fix.  The sklearn pipeline then operates on clean data and never
encounters all-NaN columns, zero-variance features, or identical duplicate rows.

This is architecturally distinct from the sklearn preprocessing steps:

  Data Preparer (here)         sklearn pipeline (executor.py)
  ─────────────────────        ──────────────────────────────
  Drop 100%-missing cols       Impute remaining missing values
  Drop zero-variance cols      Scale numeric features
  Drop duplicate rows          Encode categoricals
  Type coercion (str→numeric)  Winsorise outliers
  Report what changed          Fit/predict model

Running these at the DataFrame level (not inside CV folds) avoids the
sklearn "Skipping features without any observed values" warning, prevents
linear-model matmul NaN explosions, and reduces feature-space dimensionality
before the Planner even sees the data.

Public API
----------
prepare_dataset(df, target, config) -> (pd.DataFrame, DataPrepReport)
DataPrepConfig — thresholds controlling what gets dropped
DataPrepReport — human-readable log of every change made
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataPrepConfig:
    """
    Thresholds that control what prepare_dataset() considers removable.

    Attributes
    ----------
    max_missing_rate   : drop columns with a higher fraction of missing values.
                         Default 0.99 — only truly empty columns are dropped
                         automatically; high-but-partial missingness is left for
                         the sklearn imputer which handles it per-fold correctly.
    drop_duplicates    : whether to remove exact duplicate rows.
    coerce_numeric     : attempt pd.to_numeric() on object columns that look
                         numeric (e.g. "3.5" stored as string).
    min_class_samples  : warn (but do not drop) target classes with fewer than
                         this many examples — useful for the pipeline report.
    """
    max_missing_rate: float = 0.99
    drop_duplicates: bool = True
    coerce_numeric: bool = True
    min_class_samples: int = 5


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class DataPrepReport:
    """
    Records every change made by prepare_dataset() so it can be shown to the
    user and embedded in the pipeline report.
    """
    rows_before:       int = 0
    rows_after:        int = 0
    cols_before:       int = 0
    cols_after:        int = 0
    dropped_all_nan:   List[str] = field(default_factory=list)
    dropped_constant:  List[str] = field(default_factory=list)
    dropped_duplicate_rows: int = 0
    coerced_columns:   List[str] = field(default_factory=list)
    rare_classes:      List[str] = field(default_factory=list)   # info only

    @property
    def n_cols_dropped(self) -> int:
        return len(self.dropped_all_nan) + len(self.dropped_constant)

    def summary(self) -> str:
        """Return a plain-English summary for the pipeline report."""
        parts = []
        if self.dropped_all_nan:
            parts.append(
                f"Dropped {len(self.dropped_all_nan)} column(s) with 100% missing values "
                f"({', '.join(self.dropped_all_nan[:5])}"
                + (" …" if len(self.dropped_all_nan) > 5 else "") + ")"
            )
        if self.dropped_constant:
            parts.append(
                f"Dropped {len(self.dropped_constant)} constant column(s) "
                f"({', '.join(self.dropped_constant[:5])}"
                + (" …" if len(self.dropped_constant) > 5 else "") + ")"
            )
        if self.dropped_duplicate_rows:
            parts.append(f"Removed {self.dropped_duplicate_rows} duplicate row(s)")
        if self.coerced_columns:
            parts.append(
                f"Converted {len(self.coerced_columns)} object column(s) to numeric"
            )
        if self.rare_classes:
            parts.append(
                f"⚠️  {len(self.rare_classes)} target class(es) have fewer than "
                f"5 examples — CV results may be unstable: {self.rare_classes[:5]}"
            )
        if not parts:
            return "Dataset already clean — no changes required."
        return "\n".join(f"• {p}" for p in parts)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def prepare_dataset(
    df: pd.DataFrame,
    target: Optional[str] = None,
    config: Optional[DataPrepConfig] = None,
) -> Tuple[pd.DataFrame, DataPrepReport]:
    """
    Clean a raw DataFrame before train/test split and pipeline construction.

    Steps (in order):
    1. Coerce object columns that contain numeric strings to float.
    2. Drop duplicate rows (exact matches across all columns).
    3. Drop feature columns whose missing rate exceeds config.max_missing_rate.
    4. Drop feature columns with zero variance (all values identical).
    5. Report rare target classes (info only — not dropped).

    The target column is NEVER dropped, even if it has high missingness or
    zero variance — those would be reported as critical issues by the Issue
    Detector instead.

    Parameters
    ----------
    df     : raw input DataFrame.
    target : name of the target column (excluded from drop candidates).
    config : thresholds; uses DataPrepConfig defaults when None.

    Returns
    -------
    (cleaned_df, DataPrepReport)
    """
    if config is None:
        config = DataPrepConfig()

    report = DataPrepReport(
        rows_before=len(df),
        cols_before=len(df.columns),
    )

    out = df.copy()

    # ── Step 1: coerce numeric-looking object columns ────────────────────────
    if config.coerce_numeric:
        for col in out.columns:
            if col == target:
                continue
            if out[col].dtype == object:
                converted = pd.to_numeric(out[col], errors="coerce")
                # Only apply if conversion succeeds for >80% of non-null values
                non_null = out[col].notna().sum()
                converted_non_null = converted.notna().sum()
                if non_null > 0 and converted_non_null / non_null >= 0.8:
                    out[col] = converted
                    report.coerced_columns.append(col)

    # ── Step 2: drop duplicate rows ──────────────────────────────────────────
    if config.drop_duplicates:
        n_before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        report.dropped_duplicate_rows = n_before - len(out)

    # ── Step 3: drop columns exceeding the missing-rate threshold ────────────
    feature_cols = [c for c in out.columns if c != target]
    missing_rates = out[feature_cols].isnull().mean()
    to_drop_missing = missing_rates[missing_rates > config.max_missing_rate].index.tolist()
    if to_drop_missing:
        out = out.drop(columns=to_drop_missing)
        report.dropped_all_nan = to_drop_missing

    # ── Step 4: drop zero-variance feature columns ───────────────────────────
    remaining_features = [c for c in out.columns if c != target]
    # Only check numeric columns for variance; categorical constants checked by nunique
    for col in remaining_features:
        if pd.api.types.is_numeric_dtype(out[col]):
            if out[col].dropna().nunique() <= 1:
                report.dropped_constant.append(col)
        else:
            if out[col].nunique(dropna=True) <= 1:
                report.dropped_constant.append(col)
    if report.dropped_constant:
        out = out.drop(columns=report.dropped_constant)

    # ── Step 5: flag rare target classes (info only) ─────────────────────────
    if target and target in out.columns:
        counts = out[target].value_counts()
        rare = counts[counts < config.min_class_samples].index.tolist()
        report.rare_classes = [str(c) for c in rare]

    report.rows_after = len(out)
    report.cols_after = len(out.columns)

    return out, report
