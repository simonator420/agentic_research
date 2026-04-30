"""
search_based.py — Search-based pipeline baseline using Optuna.

Explores the same configuration space as the Planner Agent using random
search (TPE sampler with random warm-up), but without any LLM reasoning,
feedback loop, or memory. Each trial samples a configuration independently
and evaluates it via cross-validation — there is no awareness of which
configurations have already been tried or why they failed.

This baseline isolates the contribution of agentic reasoning: if the agentic
system outperforms this baseline, the improvement is attributable to the
Planner's feedback-driven strategy rather than to the pipeline construction
machinery (Executor + Evaluator), which is identical in both systems.

Search space
------------
Identical to the set of valid ActionPlan values so the comparison is fair:
  imputation        : median | mean | knn | iterative
  outlier_handling  : winsorize | none
  encoding          : onehot | target | ordinal
  scaling           : standard | robust | minmax | none
  model             : task-dependent (classification or regression models)
  imbalance_strategy: class_weight | smote | none  (classification only)

Public API
----------
run_search_based(df, target, n_trials, cv, test_size, random_state) -> BaselineResult
"""

import time
import uuid

import pandas as pd

from src.agents.evaluator import _evaluate_single, _composite_score, _count_steps
from src.agents.executor import build_pipeline
from src.agents.issue_detector import detect_issues
from src.agents.profiler import generate_profile
from src.data.loader import split_data
from src.models.schemas import ActionPlan, BaselineResult, TargetType


# Classification and regression model lists mirror the Executor's supported models exactly.
_CLASSIFIERS = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost", "lightgbm"]
_REGRESSORS  = ["linear_regression", "ridge", "random_forest", "gradient_boosting", "xgboost", "lightgbm"]


def _sample_plan(trial, target_type: TargetType) -> ActionPlan:
    """
    Sample one ActionPlan from the search space using an Optuna trial object.

    Model choices are conditioned on the task type so the sampler never
    suggests a classifier for a regression task or vice versa.
    """
    models = _CLASSIFIERS if target_type != TargetType.REGRESSION else _REGRESSORS

    # Imbalance strategies only apply to classification tasks
    if target_type != TargetType.REGRESSION:
        imbalance = trial.suggest_categorical("imbalance_strategy", ["class_weight", "smote", "none"])
    else:
        imbalance = "none"

    return ActionPlan(
        plan_id=f"search_{trial.number}_{uuid.uuid4().hex[:4]}",
        imputation=trial.suggest_categorical("imputation", ["median", "mean", "knn", "iterative"]),
        outlier_handling=trial.suggest_categorical("outlier_handling", ["winsorize", "none"]),
        encoding=trial.suggest_categorical("encoding", ["onehot", "target", "ordinal"]),
        scaling=trial.suggest_categorical("scaling", ["standard", "robust", "minmax", "none"]),
        model=trial.suggest_categorical("model", models),
        imbalance_strategy=imbalance,
    )


def run_search_based(
    df: pd.DataFrame,
    target: str,
    n_trials: int = 50,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BaselineResult:
    """
    Run Optuna random search over the pipeline configuration space.

    Optuna's TPE sampler is used after an initial random warm-up phase.
    Unlike the agentic system, there is no feedback between trials — each
    configuration is evaluated independently with no memory of prior results.

    Parameters
    ----------
    df           : raw input DataFrame including the target column.
    target       : name of the column to predict.
    n_trials     : number of configurations to evaluate (default 50).
    cv           : number of cross-validation folds (must match the agentic system).
    test_size    : fraction of data held out as a test set.
    random_state : random seed for the Optuna sampler and train/test split.

    Returns
    -------
    BaselineResult with the best pipeline found and its cross-validated metrics.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna not installed — run: pip install optuna")

    t_start = time.perf_counter()

    X_train, X_test, y_train, y_test = split_data(
        df, target, test_size=test_size, random_state=random_state
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    profile = generate_profile(train_df, target)
    detect_issues(profile, train_df)   # detect_issues is called for consistency; result unused

    best_score = float("-inf")
    best_plan = None
    best_metrics = {}
    best_cv_std = 0.0

    def objective(trial):
        nonlocal best_score, best_plan, best_metrics, best_cv_std

        plan = _sample_plan(trial, profile.target_type)

        try:
            pipeline = build_pipeline(plan, profile, X_train)
            metric_values, cv_std, _ = _evaluate_single(pipeline, X_train, y_train, profile, cv)
            primary = next(iter(metric_values.values()))
            n_steps = _count_steps(pipeline)
            score = _composite_score(primary, cv_std, n_steps)
        except Exception:
            # Return a large negative value so Optuna deprioritises this region
            return float("-inf")

        if score > best_score:
            best_score = score
            best_plan = plan
            best_metrics = metric_values
            best_cv_std = cv_std

        return score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Fit the winning configuration on the full training set
    best_pipeline = build_pipeline(best_plan, profile, X_train)
    best_pipeline.fit(X_train, y_train)

    runtime = time.perf_counter() - t_start

    return BaselineResult(
        method="search_based",
        best_pipeline=best_pipeline,
        best_plan=best_plan,
        score=best_score,
        metric_values=best_metrics,
        cv_std=best_cv_std,
        runtime_secs=runtime,
        n_configs_evaluated=n_trials,
    )
