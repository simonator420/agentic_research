"""
automl_baseline.py — FLAML AutoML baseline for comparison with the agentic system.

Uses FLAML (Fast and Lightweight AutoML) to search for the best ML pipeline
within a fixed time budget.  FLAML performs automated model selection and
hyperparameter tuning without any domain knowledge, LLM reasoning, or memory
— it represents a state-of-the-art automated baseline against which the
agentic system's performance can be directly compared.

Unlike the rule-based and search-based baselines, FLAML operates over a much
wider model and hyperparameter space, making it a strong competitor.  Any
advantage the agentic system achieves over FLAML is attributable to the
Planner's sports-domain reasoning, memory, and iterative feedback loop.

Public API
----------
run_flaml_baseline(df, target, time_budget_s, cv, test_size, random_state)
    -> BaselineResult
"""

import time
import uuid
from typing import Optional

import numpy as np
import pandas as pd

from src.agents.evaluator import _composite_score, _evaluate_single
from src.agents.profiler import generate_profile
from src.data.loader import split_data
from src.models.schemas import ActionPlan, BaselineResult, TargetType


def run_flaml_baseline(
    df: pd.DataFrame,
    target: str,
    time_budget_s: int = 60,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    metric: Optional[str] = None,
) -> BaselineResult:
    """
    Run FLAML AutoML on a dataset and return a BaselineResult.

    FLAML is given a fixed time budget (default 60 s) and allowed to search
    over its full model and hyperparameter space.  The best model found is
    then evaluated with the same cross-validation procedure used by the
    agentic system so results are directly comparable.

    Parameters
    ----------
    df            : raw input DataFrame including the target column.
    target        : name of the column to predict.
    time_budget_s : wall-clock seconds allowed for the AutoML search.
    cv            : number of cross-validation folds for final evaluation.
    test_size     : fraction of data held out as a test set.
    random_state  : random seed for reproducibility.
    metric        : FLAML metric override (auto-selected by task type if None).

    Returns
    -------
    BaselineResult comparable to RunResult from run_agentic_pipeline().

    Raises
    ------
    ImportError if the `flaml` package is not installed.
    """
    try:
        from flaml import AutoML
    except ImportError as e:
        raise ImportError(
            "FLAML is required for this baseline. "
            "Install it with:  pip install flaml"
        ) from e

    t_start = time.perf_counter()

    X_train, X_test, y_train, y_test = split_data(
        df, target, test_size=test_size, random_state=random_state
    )

    profile = generate_profile(
        pd.concat([X_train, y_train], axis=1),
        target=target,
        run_clustering=False,
    )

    # Select FLAML task and metric based on target type
    if profile.target_type == TargetType.REGRESSION:
        flaml_task = "regression"
        flaml_metric = metric or "r2"
    else:
        flaml_task = "classification"
        flaml_metric = metric or "f1"

    # Encode categoricals — FLAML handles most preprocessing internally but
    # pandas categorical / object columns must be encoded for consistency.
    X_train_enc = _encode_for_flaml(X_train)
    X_test_enc  = _encode_for_flaml(X_test)

    automl = AutoML()
    automl.fit(
        X_train_enc,
        y_train,
        task=flaml_task,
        metric=flaml_metric,
        time_budget=time_budget_s,
        n_splits=cv,
        seed=random_state,
        verbose=0,
    )

    n_configs = len(automl.config_history) if hasattr(automl, "config_history") else 1

    # Evaluate the best model with the same CV procedure as the agentic system
    metric_values, cv_std, runtime_cv = _evaluate_single(
        automl,
        X_train_enc,
        y_train,
        profile,
        n_splits=cv,
    )

    primary = _primary_metric(metric_values, profile.target_type)
    n_steps = 2  # rough proxy — FLAML's internal pipeline has a preprocessor + model
    score = _composite_score(primary, cv_std, n_steps)

    # Build a synthetic ActionPlan capturing FLAML's best configuration
    best_estimator_name = getattr(automl, "best_estimator", "unknown")
    best_config = getattr(automl, "best_config", {}) or {}
    model_key = _map_flaml_estimator(best_estimator_name)

    plan = ActionPlan(
        plan_id=f"flaml_{uuid.uuid4().hex[:8]}",
        imputation="median",       # FLAML handles internally; median as placeholder
        outlier_handling="none",
        encoding="onehot",
        scaling="standard",
        model=model_key,
        imbalance_strategy="none",
        model_params={
            "__flaml_estimator": best_estimator_name,
            "__flaml_config": str(best_config),
            "__explanation": (
                f"FLAML AutoML selected {best_estimator_name} after evaluating "
                f"{n_configs} configuration(s) in {time_budget_s}s."
            ),
        },
    )

    total_runtime = time.perf_counter() - t_start

    return BaselineResult(
        method="flaml_automl",
        best_pipeline=automl,
        best_plan=plan,
        score=score,
        metric_values=metric_values,
        cv_std=cv_std,
        runtime_secs=total_runtime,
        n_configs_evaluated=n_configs,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_for_flaml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode object / category columns as integers so FLAML can accept them.
    FLAML has its own internal preprocessing, but it requires numeric input.
    Uses pandas factorize() — simple and deterministic.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype) == "category":
            out[col] = pd.factorize(out[col])[0].astype(float)
    return out


def _primary_metric(metric_values: dict, target_type: TargetType) -> float:
    """Extract the primary metric (same selection logic as the evaluator)."""
    if target_type == TargetType.REGRESSION:
        return metric_values.get("r2", float("-inf"))
    return metric_values.get("f1_macro", float("-inf"))


def _map_flaml_estimator(name: str) -> str:
    """Map a FLAML estimator name to the closest ActionPlan model string."""
    _MAP = {
        "lgbm":               "lightgbm",
        "lgbm_spark":         "lightgbm",
        "xgboost":            "xgboost",
        "xgb_limitdepth":     "xgboost",
        "rf":                 "random_forest",
        "extra_tree":         "random_forest",
        "lrl1":               "logistic_regression",
        "lrl2":               "logistic_regression",
        "lr":                 "linear_regression",
        "catboost":           "gradient_boosting",
        "gradient_boosting":  "gradient_boosting",
    }
    return _MAP.get(name.lower(), name)
