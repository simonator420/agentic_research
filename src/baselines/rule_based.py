"""
rule_based.py — Rule-based static pipeline baseline.

Constructs a single pipeline using a fixed set of hand-crafted heuristics,
with no iteration, no LLM, and no search. The rules are data-aware in that
they inspect the detected issues, but the decision logic is entirely
deterministic — there is no learning or feedback across attempts.

This baseline represents the kind of pipeline a practitioner might build
manually in a first pass, applying common best practices without any
automated optimisation.

Decision rules
--------------
imputation  : "knn"    if any column has HIGH missingness, else "median"
outliers    : "winsorize" + "robust" scaling if HIGH outliers detected,
              else "none" + "standard" scaling
encoding    : "onehot" always (safe default for unknown cardinality)
imbalance   : "class_weight" if HIGH class imbalance detected, else "none"
model       : LogisticRegression (binary / multiclass), Ridge (regression)

Public API
----------
run_rule_based(df, target, cv, test_size, random_state) -> BaselineResult
"""

import time
import uuid

import pandas as pd

from src.agents.evaluator import evaluate_plans, select_best
from src.agents.executor import build_pipeline
from src.agents.issue_detector import detect_issues
from src.agents.profiler import generate_profile
from src.data.loader import split_data
from src.models.schemas import ActionPlan, BaselineResult, IssueType, IssueSeverity, TargetType


def _build_plan(profile, issues) -> ActionPlan:
    """
    Translate detected issues into a single fixed ActionPlan using heuristic rules.

    Each rule targets a specific issue type and severity — no search or iteration
    is performed; the mapping is applied once and deterministically.
    """
    high_issues = {i.issue_type for i in issues if i.severity == IssueSeverity.HIGH}
    medium_issues = {i.issue_type for i in issues if i.severity == IssueSeverity.MEDIUM}
    all_flagged = high_issues | medium_issues

    # --- Imputation ---
    # KNN imputation handles correlated missingness better than univariate methods,
    # so it is preferred when missingness is severe.
    imputation = "knn" if IssueType.HIGH_MISSINGNESS in high_issues else "median"

    # --- Outlier handling and scaling ---
    # RobustScaler (IQR-based) is paired with Winsorizer when outliers are present
    # so that both clipping and scaling are outlier-resistant.
    if IssueType.OUTLIERS in high_issues:
        outlier_handling = "winsorize"
        scaling = "robust"
    else:
        outlier_handling = "none"
        scaling = "standard"

    # --- Class imbalance ---
    # class_weight="balanced" is always available without additional dependencies;
    # SMOTE is reserved for the search-based baseline where it can be evaluated
    # against alternatives.
    if IssueType.CLASS_IMBALANCE in all_flagged:
        imbalance_strategy = "class_weight"
    else:
        imbalance_strategy = "none"

    # --- Model ---
    # Conservative defaults: linear models generalise well with limited tuning
    # and serve as a reliable lower bound for comparison.
    if profile.target_type == TargetType.REGRESSION:
        model = "ridge"
    else:
        model = "logistic_regression"

    return ActionPlan(
        plan_id=f"rule_based_{uuid.uuid4().hex[:6]}",
        imputation=imputation,
        outlier_handling=outlier_handling,
        encoding="onehot",      # safe default — handles unknown categories at test time
        scaling=scaling,
        model=model,
        imbalance_strategy=imbalance_strategy,
    )


def run_rule_based(
    df: pd.DataFrame,
    target: str,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BaselineResult:
    """
    Build and evaluate a single rule-derived pipeline.

    Parameters
    ----------
    df           : raw input DataFrame including the target column.
    target       : name of the column to predict.
    cv           : number of cross-validation folds (must match the agentic system).
    test_size    : fraction of data held out as a test set.
    random_state : random seed for reproducible splits.

    Returns
    -------
    BaselineResult with the fitted pipeline and cross-validated metrics.
    """
    t_start = time.perf_counter()

    X_train, X_test, y_train, y_test = split_data(
        df, target, test_size=test_size, random_state=random_state
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    profile = generate_profile(train_df, target)
    issues = detect_issues(profile, train_df)

    plan = _build_plan(profile, issues)

    # evaluate_plans wraps the build + CV loop with consistent error handling
    results = evaluate_plans([plan], profile, X_train, y_train, cv=cv)
    best = select_best(results)

    # Fit on full training set — mirrors the Orchestrator's final step
    pipeline = build_pipeline(plan, profile, X_train)
    pipeline.fit(X_train, y_train)

    runtime = time.perf_counter() - t_start

    return BaselineResult(
        method="rule_based",
        best_pipeline=pipeline,
        best_plan=plan,
        score=best.score,
        metric_values=best.metric_values,
        cv_std=best.cv_std,
        runtime_secs=runtime,
        n_configs_evaluated=1,
    )
