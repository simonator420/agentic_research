"""
issue_detector.py — Issue Detector Agent (Step 2 of the agentic loop).

Scans a DataProfile (and the raw DataFrame for outlier / leakage checks) and
returns a list of Issue objects, each with a severity score (HIGH / MEDIUM / LOW).

The list is sorted HIGH → LOW so the Planner Agent processes the most critical
problems first when constructing ActionPlans.

Detected issue types
--------------------
HIGH_MISSINGNESS   : column has too many NaN values to ignore.
OUTLIERS           : column has extreme values beyond 1.5 × IQR fence.
DUPLICATES         : dataset contains identical rows.
NOISY_CATEGORIES   : categorical column has near-unique values per row (likely ID / free text).
LEAKAGE_CANDIDATE  : numeric feature correlates suspiciously strongly with the regression target.
CLASS_IMBALANCE    : minority class is underrepresented in a classification target.

Public API
----------
detect_issues(profile, df) -> List[Issue]
"""

from typing import List

import pandas as pd

from src.models.schemas import DataProfile, Issue, IssueSeverity, IssueType, TargetType

# --- Severity thresholds for missing values ---
# >30 % missing → HIGH (likely needs dropping or heavy imputation)
# >10 % missing → MEDIUM (imputation strategy matters)
# > 1 % missing → LOW  (minor, but should be noted for the Planner)
_MISSING_HIGH = 0.30
_MISSING_MEDIUM = 0.10
_MISSING_LOW = 0.01

# --- IQR outlier thresholds (fraction of non-null rows that fall outside the fence) ---
_OUTLIER_HIGH = 0.05
_OUTLIER_MEDIUM = 0.02

# --- Duplicate row thresholds (fraction of total rows) ---
_DUPLICATE_HIGH = 0.01
_DUPLICATE_MEDIUM = 0.001

# --- Class imbalance: minority class frequency ---
# <10 % → HIGH  (SMOTE or class weights strongly recommended)
# <20 % → MEDIUM (class weights or mild oversampling recommended)
_IMBALANCE_HIGH = 0.10
_IMBALANCE_MEDIUM = 0.20

# --- Leakage detection: absolute Pearson correlation with the regression target ---
# Only checked for regression tasks; classification leakage is harder to detect
# without building a model, so it is deferred to the Planner's reasoning.
_LEAKAGE_CORR_HIGH = 0.95
_LEAKAGE_CORR_MEDIUM = 0.90

# A categorical column whose unique-value count exceeds this fraction of total rows
# is likely an identifier or free-text field and should not be one-hot encoded.
_HIGH_CARDINALITY_RATIO = 0.50

# Used to sort the final issue list in a single pass.
_SEVERITY_ORDER = {IssueSeverity.HIGH: 0, IssueSeverity.MEDIUM: 1, IssueSeverity.LOW: 2}


def _missing_issues(profile: DataProfile) -> List[Issue]:
    """Flag columns with a missing value rate above any threshold."""
    issues = []
    for col_name, col in profile.columns.items():
        if col.missing_rate >= _MISSING_HIGH:
            sev = IssueSeverity.HIGH
        elif col.missing_rate >= _MISSING_MEDIUM:
            sev = IssueSeverity.MEDIUM
        elif col.missing_rate >= _MISSING_LOW:
            sev = IssueSeverity.LOW
        else:
            continue
        issues.append(Issue(
            issue_type=IssueType.HIGH_MISSINGNESS,
            severity=sev,
            affected_column=col_name,
            description=f"Column '{col_name}' has {col.missing_rate:.1%} missing values",
            evidence={"missing_rate": col.missing_rate},
        ))
    return issues


def _outlier_issues(profile: DataProfile, df: pd.DataFrame) -> List[Issue]:
    """
    Detect outliers using Tukey's IQR fence: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    q1 / q3 come from the pre-computed ColumnProfile, so the raw DataFrame is only
    accessed to count how many values actually fall outside the fence — avoids
    recomputing percentiles on the full dataset.
    Columns with IQR == 0 (constant or near-constant) are skipped to avoid false positives.
    """
    issues = []
    for col_name, col in profile.columns.items():
        if col.dtype != "numeric" or col.q1 is None or col.q3 is None:
            continue
        iqr = col.q3 - col.q1
        if iqr == 0:
            continue
        lower = col.q1 - 1.5 * iqr
        upper = col.q3 + 1.5 * iqr
        series = df[col_name].dropna()
        n_outliers = int(((series < lower) | (series > upper)).sum())
        outlier_rate = n_outliers / max(len(series), 1)

        if outlier_rate >= _OUTLIER_HIGH:
            sev = IssueSeverity.HIGH
        elif outlier_rate >= _OUTLIER_MEDIUM:
            sev = IssueSeverity.MEDIUM
        else:
            continue

        issues.append(Issue(
            issue_type=IssueType.OUTLIERS,
            severity=sev,
            affected_column=col_name,
            description=f"Column '{col_name}' has {outlier_rate:.1%} outliers (IQR method)",
            evidence={
                "outlier_rate": outlier_rate,
                "n_outliers": n_outliers,
                "lower_bound": lower,
                "upper_bound": upper,
            },
        ))
    return issues


def _duplicate_issues(profile: DataProfile) -> List[Issue]:
    """Flag datasets with a high proportion of fully identical rows."""
    dup_rate = profile.n_duplicates / max(profile.n_rows, 1)
    if dup_rate >= _DUPLICATE_HIGH:
        sev = IssueSeverity.HIGH
    elif dup_rate >= _DUPLICATE_MEDIUM:
        sev = IssueSeverity.MEDIUM
    else:
        return []
    return [Issue(
        issue_type=IssueType.DUPLICATES,
        severity=sev,
        affected_column=None,   # dataset-level issue — no single column responsible
        description=f"Dataset has {profile.n_duplicates} duplicate rows ({dup_rate:.1%})",
        evidence={"n_duplicates": profile.n_duplicates, "duplicate_rate": dup_rate},
    )]


def _noisy_category_issues(profile: DataProfile) -> List[Issue]:
    """
    Flag categorical columns that are likely identifiers or free-text fields.

    A column whose unique-value count exceeds _HIGH_CARDINALITY_RATIO × n_rows
    cannot be one-hot encoded without creating an explosion of dummy columns and
    will likely degrade model performance.  The target column is excluded.
    """
    issues = []
    for col_name, col in profile.columns.items():
        if col.dtype != "categorical" or col_name == profile.target_column:
            continue
        cardinality_ratio = col.n_unique / max(profile.n_rows, 1)
        if cardinality_ratio >= _HIGH_CARDINALITY_RATIO:
            issues.append(Issue(
                issue_type=IssueType.NOISY_CATEGORIES,
                severity=IssueSeverity.MEDIUM,
                affected_column=col_name,
                description=(
                    f"Column '{col_name}' has very high cardinality "
                    f"({col.n_unique} unique values, {cardinality_ratio:.1%} of rows) "
                    "— likely an ID or free-text field"
                ),
                evidence={"n_unique": col.n_unique, "cardinality_ratio": cardinality_ratio},
            ))
    return issues


def _leakage_issues(profile: DataProfile, df: pd.DataFrame) -> List[Issue]:
    """
    Detect features that correlate almost perfectly with a regression target.

    Only regression targets are checked here because Pearson correlation is
    meaningful for continuous targets.  For classification, leakage is typically
    detected by the Planner via reasoning about column names and semantics.

    Alignment with .loc ensures that rows with NaN in either series are excluded
    from the correlation calculation (avoids inflated coefficients from NaN patterns).
    """
    issues = []
    target = profile.target_column

    if profile.target_type != TargetType.REGRESSION:
        return issues

    target_series = df[target]
    for col_name, col in profile.columns.items():
        if col_name == target or col.dtype != "numeric":
            continue
        col_series = df[col_name].dropna()
        aligned_target = target_series.loc[col_series.index].dropna()
        aligned_col = col_series.loc[aligned_target.index]
        if len(aligned_target) < 10:   # too few rows to compute a reliable correlation
            continue
        corr = abs(aligned_col.corr(aligned_target))
        if corr >= _LEAKAGE_CORR_HIGH:
            sev = IssueSeverity.HIGH
        elif corr >= _LEAKAGE_CORR_MEDIUM:
            sev = IssueSeverity.MEDIUM
        else:
            continue
        issues.append(Issue(
            issue_type=IssueType.LEAKAGE_CANDIDATE,
            severity=sev,
            affected_column=col_name,
            description=(
                f"Column '{col_name}' has {corr:.3f} absolute correlation with target "
                "— possible data leakage"
            ),
            evidence={"correlation": corr},
        ))
    return issues


def _imbalance_issues(profile: DataProfile) -> List[Issue]:
    """
    Flag severe class imbalance in classification targets.

    Uses the minority class frequency (lowest value in class_distribution) as
    the imbalance signal.  If the minority class makes up less than _IMBALANCE_HIGH
    of all samples, the Planner should consider SMOTE or class-weight adjustments.
    """
    if profile.target_type == TargetType.REGRESSION or not profile.class_distribution:
        return []
    min_freq = min(profile.class_distribution.values())
    if min_freq <= _IMBALANCE_HIGH:
        sev = IssueSeverity.HIGH
    elif min_freq <= _IMBALANCE_MEDIUM:
        sev = IssueSeverity.MEDIUM
    else:
        return []
    minority_class = min(profile.class_distribution, key=profile.class_distribution.get)
    return [Issue(
        issue_type=IssueType.CLASS_IMBALANCE,
        severity=sev,
        affected_column=profile.target_column,
        description=f"Class imbalance: minority class '{minority_class}' = {min_freq:.1%} of samples",
        evidence={
            "class_distribution": profile.class_distribution,
            "minority_class": minority_class,
            "minority_rate": min_freq,
        },
    )]


def detect_issues(profile: DataProfile, df: pd.DataFrame) -> List[Issue]:
    """
    Run all detectors and return a unified, severity-sorted list of issues.

    Parameters
    ----------
    profile : DataProfile produced by generate_profile().
    df      : the same raw DataFrame that was profiled (needed for outlier / leakage checks).

    Returns
    -------
    List[Issue] sorted HIGH → MEDIUM → LOW.
    """
    issues: List[Issue] = []
    issues.extend(_missing_issues(profile))
    issues.extend(_outlier_issues(profile, df))
    issues.extend(_duplicate_issues(profile))
    issues.extend(_noisy_category_issues(profile))
    issues.extend(_leakage_issues(profile, df))
    issues.extend(_imbalance_issues(profile))
    issues.sort(key=lambda i: _SEVERITY_ORDER[i.severity])
    return issues
