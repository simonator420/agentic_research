import numpy as np
import pandas as pd
import pytest

from src.agents.issue_detector import detect_issues
from src.agents.profiler import generate_profile
from src.models.schemas import IssueSeverity, IssueType


def test_detects_high_missingness():
    df = pd.DataFrame({
        "a": [None] * 40 + list(range(60)),
        "target": [0, 1] * 50,
    })
    profile = generate_profile(df, "target")
    issues = detect_issues(profile, df)
    types = [i.issue_type for i in issues]
    assert IssueType.HIGH_MISSINGNESS in types
    missing = next(i for i in issues if i.issue_type == IssueType.HIGH_MISSINGNESS)
    assert missing.severity == IssueSeverity.HIGH
    assert missing.affected_column == "a"


def test_no_issues_clean_data():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [50000, 60000, 70000, 80000, 90000],
        "target": [0, 1, 0, 1, 0],
    })
    profile = generate_profile(df, "target")
    issues = detect_issues(profile, df)
    assert len(issues) == 0


def test_detects_class_imbalance():
    target = [0] * 95 + [1] * 5
    df = pd.DataFrame({
        "x": range(100),
        "target": target,
    })
    profile = generate_profile(df, "target")
    issues = detect_issues(profile, df)
    types = [i.issue_type for i in issues]
    assert IssueType.CLASS_IMBALANCE in types
    imb = next(i for i in issues if i.issue_type == IssueType.CLASS_IMBALANCE)
    assert imb.severity == IssueSeverity.HIGH


def test_detects_duplicates():
    df = pd.DataFrame({
        "a": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
        "target": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10,
    })
    profile = generate_profile(df, "target")
    issues = detect_issues(profile, df)
    types = [i.issue_type for i in issues]
    assert IssueType.DUPLICATES in types


def test_detects_outliers():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 200).tolist() + [1000.0, -1000.0, 999.0, -999.0, 998.0, -998.0]
    df = pd.DataFrame({
        "x": values,
        "target": [0, 1] * 103,
    })
    profile = generate_profile(df, "target")
    issues = detect_issues(profile, df)
    types = [i.issue_type for i in issues]
    assert IssueType.OUTLIERS in types


def test_detects_high_cardinality():
    df = pd.DataFrame({
        "id_col": [f"id_{i}" for i in range(100)],
        "target": [0, 1] * 50,
    })
    profile = generate_profile(df, "target")
    issues = detect_issues(profile, df)
    types = [i.issue_type for i in issues]
    assert IssueType.NOISY_CATEGORIES in types


def test_issues_sorted_by_severity():
    target = [0] * 95 + [1] * 5
    values = [None] * 40 + list(range(60))
    df = pd.DataFrame({"a": values, "target": target})
    profile = generate_profile(df, "target")
    issues = detect_issues(profile, df)
    severities = [i.severity for i in issues]
    order = {"high": 0, "medium": 1, "low": 2}
    assert severities == sorted(severities, key=lambda s: order[s.value])


def test_leakage_detection_regression():
    rng = np.random.default_rng(1)
    n = 100
    target = rng.uniform(0, 100, n)
    df = pd.DataFrame({
        "leak_col": target * 1.001,  # near-perfect correlation
        "noise": rng.normal(0, 10, n),
        "target": target,
    })
    profile = generate_profile(df, "target")
    issues = detect_issues(profile, df)
    types = [i.issue_type for i in issues]
    assert IssueType.LEAKAGE_CANDIDATE in types
    leak = next(i for i in issues if i.issue_type == IssueType.LEAKAGE_CANDIDATE)
    assert leak.affected_column == "leak_col"
