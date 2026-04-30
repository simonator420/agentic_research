import numpy as np
import pandas as pd
import pytest

from src.baselines.rule_based import run_rule_based, _build_plan
from src.baselines.search_based import run_search_based
from src.agents.profiler import generate_profile
from src.agents.issue_detector import detect_issues
from src.models.schemas import BaselineResult, TargetType


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _binary_df(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age":    rng.integers(20, 70, n).astype(float),
        "income": rng.uniform(20_000, 120_000, n),
        "city":   rng.choice(["NY", "LA", "SF"], n),
        "target": rng.integers(0, 2, n),
    })


def _regression_df(n=200, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, (n, 4))
    y = X[:, 0] * 3.0 + rng.normal(0, 1, n)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y
    return df


def _missingness_df(n=200, seed=2) -> pd.DataFrame:
    """Dataset with HIGH missingness to exercise the knn imputation branch."""
    rng = np.random.default_rng(seed)
    df = _binary_df(n, seed)
    mask = rng.random(n) < 0.40
    df.loc[mask, "income"] = np.nan
    return df


def _outlier_df(n=200, seed=3) -> pd.DataFrame:
    """Dataset with extreme outliers to exercise the winsorize+robust branch.

    Sets 15 rows (7.5%) to an extreme value — above the HIGH threshold of 5%.
    """
    rng = np.random.default_rng(seed)
    df = _binary_df(n, seed)
    df.loc[:14, "income"] = 1e9   # 15 rows → 7.5% outlier rate
    return df


# ---------------------------------------------------------------------------
# _build_plan (rule_based internals)
# ---------------------------------------------------------------------------

class TestBuildPlan:
    def test_default_binary_uses_logistic_median_onehot(self):
        df = _binary_df()
        profile = generate_profile(df, "target")
        issues = detect_issues(profile, df)
        plan = _build_plan(profile, issues)
        assert plan.model == "logistic_regression"
        assert plan.encoding == "onehot"

    def test_regression_uses_ridge(self):
        df = _regression_df()
        profile = generate_profile(df, "target")
        issues = detect_issues(profile, df)
        plan = _build_plan(profile, issues)
        assert plan.model == "ridge"

    def test_high_missingness_triggers_knn(self):
        df = _missingness_df()
        profile = generate_profile(df, "target")
        issues = detect_issues(profile, df)
        plan = _build_plan(profile, issues)
        assert plan.imputation == "knn"

    def test_no_high_missingness_uses_median(self):
        df = _binary_df()
        profile = generate_profile(df, "target")
        issues = detect_issues(profile, df)
        plan = _build_plan(profile, issues)
        assert plan.imputation == "median"

    def test_high_outliers_trigger_winsorize_and_robust(self):
        df = _outlier_df()
        profile = generate_profile(df, "target")
        issues = detect_issues(profile, df)
        plan = _build_plan(profile, issues)
        assert plan.outlier_handling == "winsorize"
        assert plan.scaling == "robust"

    def test_plan_id_starts_with_rule_based(self):
        df = _binary_df()
        profile = generate_profile(df, "target")
        issues = detect_issues(profile, df)
        plan = _build_plan(profile, issues)
        assert plan.plan_id.startswith("rule_based_")


# ---------------------------------------------------------------------------
# run_rule_based
# ---------------------------------------------------------------------------

class TestRunRuleBased:
    def test_returns_baseline_result(self):
        result = run_rule_based(_binary_df(), "target", cv=3)
        assert isinstance(result, BaselineResult)

    def test_method_name(self):
        result = run_rule_based(_binary_df(), "target", cv=3)
        assert result.method == "rule_based"

    def test_n_configs_evaluated_is_one(self):
        result = run_rule_based(_binary_df(), "target", cv=3)
        assert result.n_configs_evaluated == 1

    def test_score_is_finite(self):
        result = run_rule_based(_binary_df(), "target", cv=3)
        assert np.isfinite(result.score)

    def test_metric_values_nonempty(self):
        result = run_rule_based(_binary_df(), "target", cv=3)
        assert len(result.metric_values) > 0

    def test_cv_std_nonnegative(self):
        result = run_rule_based(_binary_df(), "target", cv=3)
        assert result.cv_std >= 0.0

    def test_runtime_positive(self):
        result = run_rule_based(_binary_df(), "target", cv=3)
        assert result.runtime_secs > 0.0

    def test_pipeline_can_predict(self):
        df = _binary_df()
        result = run_rule_based(df, "target", cv=3)
        X = df.drop(columns=["target"])
        preds = result.best_pipeline.predict(X)
        assert len(preds) == len(df)

    def test_regression_produces_result(self):
        result = run_rule_based(_regression_df(), "target", cv=3)
        assert isinstance(result, BaselineResult)
        assert result.best_plan.model == "ridge"

    def test_missingness_df(self):
        result = run_rule_based(_missingness_df(), "target", cv=3)
        assert np.isfinite(result.score)

    def test_outlier_df(self):
        result = run_rule_based(_outlier_df(), "target", cv=3)
        assert result.best_plan.outlier_handling == "winsorize"


# ---------------------------------------------------------------------------
# run_search_based
# ---------------------------------------------------------------------------

optuna = pytest.importorskip("optuna", reason="optuna not installed")


class TestRunSearchBased:
    def test_returns_baseline_result(self):
        result = run_search_based(_binary_df(), "target", n_trials=5, cv=3)
        assert isinstance(result, BaselineResult)

    def test_method_name(self):
        result = run_search_based(_binary_df(), "target", n_trials=5, cv=3)
        assert result.method == "search_based"

    def test_n_configs_evaluated_matches_n_trials(self):
        n = 8
        result = run_search_based(_binary_df(), "target", n_trials=n, cv=3)
        assert result.n_configs_evaluated == n

    def test_score_is_finite(self):
        result = run_search_based(_binary_df(), "target", n_trials=5, cv=3)
        assert np.isfinite(result.score)

    def test_metric_values_nonempty(self):
        result = run_search_based(_binary_df(), "target", n_trials=5, cv=3)
        assert len(result.metric_values) > 0

    def test_cv_std_nonnegative(self):
        result = run_search_based(_binary_df(), "target", n_trials=5, cv=3)
        assert result.cv_std >= 0.0

    def test_runtime_positive(self):
        result = run_search_based(_binary_df(), "target", n_trials=5, cv=3)
        assert result.runtime_secs > 0.0

    def test_pipeline_can_predict(self):
        df = _binary_df()
        result = run_search_based(df, "target", n_trials=5, cv=3)
        X = df.drop(columns=["target"])
        preds = result.best_pipeline.predict(X)
        assert len(preds) == len(df)

    def test_regression_produces_result(self):
        result = run_search_based(_regression_df(), "target", n_trials=5, cv=3)
        assert isinstance(result, BaselineResult)
        assert result.best_plan.model in [
            "linear_regression", "ridge", "random_forest",
            "gradient_boosting", "xgboost", "lightgbm",
        ]

    def test_plan_id_starts_with_search(self):
        result = run_search_based(_binary_df(), "target", n_trials=5, cv=3)
        assert result.best_plan.plan_id.startswith("search_")

    def test_best_plan_model_is_classifier_for_classification(self):
        from src.baselines.search_based import _CLASSIFIERS
        result = run_search_based(_binary_df(), "target", n_trials=5, cv=3)
        assert result.best_plan.model in _CLASSIFIERS

    def test_best_plan_model_is_regressor_for_regression(self):
        from src.baselines.search_based import _REGRESSORS
        result = run_search_based(_regression_df(), "target", n_trials=5, cv=3)
        assert result.best_plan.model in _REGRESSORS
