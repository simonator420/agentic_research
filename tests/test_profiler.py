import numpy as np
import pandas as pd
import pytest

from src.agents.profiler import generate_profile
from src.models.schemas import TargetType


@pytest.fixture
def binary_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        "income": [30000, 50000, 70000, 90000, 110000, 130000, 150000, 170000, 190000, 210000],
        "city": ["NY", "LA", "NY", "SF", "LA", "NY", "SF", "LA", "NY", "SF"],
        "target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    })


@pytest.fixture
def regression_df():
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
        "cat": rng.choice(["A", "B", "C"], n),
        "price": rng.uniform(100, 1000, n),
    })


def test_target_type_binary(binary_df):
    profile = generate_profile(binary_df, "target")
    assert profile.target_type == TargetType.BINARY


def test_target_type_regression(regression_df):
    profile = generate_profile(regression_df, "price")
    assert profile.target_type == TargetType.REGRESSION


def test_class_distribution_binary(binary_df):
    profile = generate_profile(binary_df, "target")
    assert profile.class_distribution is not None
    assert abs(sum(profile.class_distribution.values()) - 1.0) < 1e-6


def test_class_distribution_none_for_regression(regression_df):
    profile = generate_profile(regression_df, "price")
    assert profile.class_distribution is None


def test_missing_rate_computed():
    df = pd.DataFrame({
        "a": [1.0, None, 3.0, 4.0, None],
        "target": [0, 1, 0, 1, 0],
    })
    profile = generate_profile(df, "target")
    assert abs(profile.columns["a"].missing_rate - 0.4) < 1e-6


def test_numeric_stats_populated(binary_df):
    profile = generate_profile(binary_df, "target")
    age_col = profile.columns["age"]
    assert age_col.dtype == "numeric"
    assert age_col.mean is not None
    assert age_col.q1 is not None
    assert age_col.q3 is not None


def test_categorical_top_categories(binary_df):
    profile = generate_profile(binary_df, "target")
    city_col = profile.columns["city"]
    assert city_col.dtype == "categorical"
    assert city_col.top_categories is not None
    assert len(city_col.top_categories) == 3


def test_duplicate_count():
    df = pd.DataFrame({
        "a": [1, 1, 2, 3],
        "target": [0, 0, 1, 1],
    })
    profile = generate_profile(df, "target")
    assert profile.n_duplicates == 1


def test_missing_target_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column"):
        generate_profile(df, "nonexistent")


def test_multiclass_target():
    df = pd.DataFrame({
        "x": range(30),
        "target": list(range(10)) * 3,
    })
    profile = generate_profile(df, "target")
    assert profile.target_type == TargetType.MULTICLASS
