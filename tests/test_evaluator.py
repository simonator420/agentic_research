import numpy as np
import pandas as pd
import pytest

from src.agents.evaluator import evaluate_plans, select_best, _composite_score
from src.agents.profiler import generate_profile
from src.models.schemas import ActionPlan, EvaluationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_plan(plan_id="p1", model="logistic_regression", encoding="onehot",
               scaling="standard", imputation="median") -> ActionPlan:
    return ActionPlan(
        plan_id=plan_id,
        imputation=imputation,
        outlier_handling="none",
        encoding=encoding,
        scaling=scaling,
        model=model,
        imbalance_strategy="none",
    )


@pytest.fixture
def binary_Xy():
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({
        "age":    rng.integers(20, 70, n).astype(float),
        "income": rng.uniform(20000, 120000, n),
        "city":   rng.choice(["NY", "LA", "SF"], n),
        "target": rng.integers(0, 2, n),
    })
    profile = generate_profile(df, "target")
    X = df.drop(columns=["target"])
    y = df["target"]
    return profile, X, y


@pytest.fixture
def multiclass_Xy():
    rng = np.random.default_rng(1)
    n = 150
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
        "target": rng.integers(0, 4, n),
    })
    profile = generate_profile(df, "target")
    X = df.drop(columns=["target"])
    y = df["target"]
    return profile, X, y


@pytest.fixture
def regression_Xy():
    rng = np.random.default_rng(2)
    n = 150
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
        "cat": rng.choice(["A", "B", "C"], n),
        "price": rng.uniform(100, 1000, n),
    })
    profile = generate_profile(df, "price")
    X = df.drop(columns=["price"])
    y = df["price"]
    return profile, X, y


# ---------------------------------------------------------------------------
# evaluate_plans — result structure
# ---------------------------------------------------------------------------

def test_returns_one_result_per_plan(binary_Xy):
    profile, X, y = binary_Xy
    plans = [_make_plan("p1"), _make_plan("p2", model="random_forest")]
    results = evaluate_plans(plans, profile, X, y, cv=3)
    assert len(results) == 2


def test_result_has_plan_id(binary_Xy):
    profile, X, y = binary_Xy
    plans = [_make_plan("my_plan")]
    results = evaluate_plans(plans, profile, X, y, cv=3)
    assert results[0].plan_id == "my_plan"


def test_binary_metrics_present(binary_Xy):
    profile, X, y = binary_Xy
    results = evaluate_plans([_make_plan()], profile, X, y, cv=3)
    mv = results[0].metric_values
    for key in ("f1_macro", "f1_weighted", "accuracy", "balanced_accuracy",
                "precision_macro", "recall_macro", "mcc", "auc"):
        assert key in mv, f"missing metric: {key}"


def test_multiclass_metrics_present(multiclass_Xy):
    profile, X, y = multiclass_Xy
    results = evaluate_plans([_make_plan()], profile, X, y, cv=3)
    assert "f1_macro" in results[0].metric_values


def test_regression_metrics_present(regression_Xy):
    profile, X, y = regression_Xy
    plan = _make_plan(model="ridge", encoding="onehot")
    results = evaluate_plans([plan], profile, X, y, cv=3)
    assert "rmse" in results[0].metric_values
    assert "r2" in results[0].metric_values


def test_score_is_finite(binary_Xy):
    profile, X, y = binary_Xy
    results = evaluate_plans([_make_plan()], profile, X, y, cv=3)
    assert np.isfinite(results[0].score)


def test_cv_std_is_non_negative(binary_Xy):
    profile, X, y = binary_Xy
    results = evaluate_plans([_make_plan()], profile, X, y, cv=3)
    assert results[0].cv_std >= 0.0


def test_runtime_is_positive(binary_Xy):
    profile, X, y = binary_Xy
    results = evaluate_plans([_make_plan()], profile, X, y, cv=3)
    assert results[0].runtime_secs > 0.0


# ---------------------------------------------------------------------------
# evaluate_plans — failure handling
# ---------------------------------------------------------------------------

def test_failed_plan_gets_neg_inf_score(binary_Xy):
    """A plan with an invalid model name must not crash — score should be -inf."""
    profile, X, y = binary_Xy
    bad_plan = _make_plan(model="nonexistent_model")
    results = evaluate_plans([bad_plan], profile, X, y, cv=3)
    assert results[0].score == float("-inf")


def test_failed_plan_does_not_affect_valid_plan(binary_Xy):
    """One failing plan must not prevent the valid plan from being evaluated."""
    profile, X, y = binary_Xy
    plans = [_make_plan("bad", model="nonexistent"), _make_plan("good")]
    results = evaluate_plans(plans, profile, X, y, cv=3)
    good = next(r for r in results if r.plan_id == "good")
    assert np.isfinite(good.score)


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------

def test_select_best_returns_highest_score():
    results = [
        EvaluationResult("p1", score=0.80, metric_values={}, cv_std=0.02, runtime_secs=1.0),
        EvaluationResult("p2", score=0.91, metric_values={}, cv_std=0.01, runtime_secs=1.0),
        EvaluationResult("p3", score=0.75, metric_values={}, cv_std=0.03, runtime_secs=1.0),
    ]
    best = select_best(results)
    assert best.plan_id == "p2"


def test_select_best_ignores_failed_plans():
    results = [
        EvaluationResult("p1", score=float("-inf"), metric_values={}, cv_std=0.0, runtime_secs=0.0),
        EvaluationResult("p2", score=0.85, metric_values={}, cv_std=0.02, runtime_secs=1.0),
    ]
    best = select_best(results)
    assert best.plan_id == "p2"


def test_select_best_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        select_best([])


def test_select_best_all_failed_raises():
    results = [
        EvaluationResult("p1", score=float("-inf"), metric_values={}, cv_std=0.0, runtime_secs=0.0),
    ]
    with pytest.raises(ValueError, match="All evaluated plans failed"):
        select_best(results)


# ---------------------------------------------------------------------------
# composite score
# ---------------------------------------------------------------------------

def test_composite_score_decreases_with_higher_std():
    s1 = _composite_score(primary_metric=0.9, cv_std=0.01, n_steps=3)
    s2 = _composite_score(primary_metric=0.9, cv_std=0.10, n_steps=3)
    assert s1 > s2


def test_composite_score_decreases_with_more_steps():
    s1 = _composite_score(primary_metric=0.9, cv_std=0.05, n_steps=2)
    s2 = _composite_score(primary_metric=0.9, cv_std=0.05, n_steps=8)
    assert s1 > s2


def test_full_pipeline_binary(binary_Xy):
    """End-to-end: evaluate multiple plans and select the best."""
    profile, X, y = binary_Xy
    plans = [
        _make_plan("p1", model="logistic_regression"),
        _make_plan("p2", model="random_forest"),
    ]
    results = evaluate_plans(plans, profile, X, y, cv=3)
    best = select_best(results)
    assert best.plan_id in {"p1", "p2"}
    assert np.isfinite(best.score)


def test_full_pipeline_regression(regression_Xy):
    """End-to-end: evaluate a regression plan and select the best."""
    profile, X, y = regression_Xy
    plans = [
        _make_plan("r1", model="ridge", encoding="onehot"),
        _make_plan("r2", model="random_forest", encoding="onehot"),
    ]
    results = evaluate_plans(plans, profile, X, y, cv=3)
    best = select_best(results)
    assert best.plan_id in {"r1", "r2"}
