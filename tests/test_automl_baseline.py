"""
Tests for the FLAML AutoML baseline.

Runs a very short time budget (5 s) to keep CI fast while still exercising
the full baseline interface: fit → evaluate → return BaselineResult.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.schemas import BaselineResult


@pytest.fixture
def binary_df():
    rng = np.random.default_rng(42)
    n = 120
    return pd.DataFrame({
        "age":    rng.integers(20, 70, n).astype(float),
        "income": rng.uniform(20000, 120000, n),
        "city":   rng.choice(["NY", "LA", "SF"], n),
        "target": rng.integers(0, 2, n),
    })


@pytest.fixture
def regression_df():
    rng = np.random.default_rng(7)
    n = 120
    return pd.DataFrame({
        "x1":    rng.normal(0, 1, n),
        "x2":    rng.normal(5, 2, n),
        "cat":   rng.choice(["A", "B", "C"], n),
        "price": rng.uniform(100, 1000, n),
    })


def test_flaml_returns_baseline_result(tmp_path, binary_df):
    from src.baselines.automl_baseline import run_flaml_baseline
    result = run_flaml_baseline(binary_df, "target", time_budget_s=5, cv=3)
    assert isinstance(result, BaselineResult)


def test_flaml_method_label(binary_df):
    from src.baselines.automl_baseline import run_flaml_baseline
    result = run_flaml_baseline(binary_df, "target", time_budget_s=5, cv=3)
    assert result.method == "flaml_automl"


def test_flaml_score_is_finite(binary_df):
    from src.baselines.automl_baseline import run_flaml_baseline
    result = run_flaml_baseline(binary_df, "target", time_budget_s=5, cv=3)
    assert np.isfinite(result.score)


def test_flaml_pipeline_can_predict(binary_df):
    from src.baselines.automl_baseline import run_flaml_baseline
    result = run_flaml_baseline(binary_df, "target", time_budget_s=5, cv=3)
    X = binary_df.drop(columns=["target"])
    from src.baselines.automl_baseline import _encode_for_flaml
    preds = result.best_pipeline.predict(_encode_for_flaml(X))
    assert preds.shape == (len(X),)


def test_flaml_n_configs_positive(binary_df):
    from src.baselines.automl_baseline import run_flaml_baseline
    result = run_flaml_baseline(binary_df, "target", time_budget_s=5, cv=3)
    assert result.n_configs_evaluated >= 1


def test_flaml_regression(regression_df):
    from src.baselines.automl_baseline import run_flaml_baseline
    result = run_flaml_baseline(regression_df, "price", time_budget_s=5, cv=3)
    assert isinstance(result, BaselineResult)
    assert np.isfinite(result.score)
