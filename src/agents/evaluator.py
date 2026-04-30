"""
evaluator.py — Evaluator Agent (Step 5 of the agentic loop).

Receives a list of candidate pipelines (one per ActionPlan), runs cross-validated
evaluation on each, and returns a ranked list of EvaluationResults.

The Orchestrator uses these results in two ways:
  1. Select the best pipeline for the current iteration.
  2. Decide whether to continue iterating or stop (convergence check).

Metrics are chosen automatically based on the task type inferred from the DataProfile:
  Binary classification  : F1 (macro), AUC-ROC
  Multiclass classification: Macro F1
  Regression             : RMSE, R²

Composite score
---------------
A single scalar is computed from the task-appropriate primary metric minus a
stability penalty (cv_std) and a complexity penalty (number of pipeline steps).
This lets the Orchestrator rank plans on a single axis rather than juggling
multiple objectives.

  score = primary_metric - 0.5 * cv_std - 0.01 * n_steps

Public API
----------
evaluate_plans(plans, profile, X, y, cv) -> List[EvaluationResult]
select_best(results)                      -> EvaluationResult
"""

import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from src.agents.executor import build_pipeline
from src.models.schemas import ActionPlan, DataProfile, EvaluationResult, TargetType

# Weight on cross-validation standard deviation in the composite score.
# Higher values penalise unstable pipelines more strongly.
_CV_STD_PENALTY = 0.5

# Weight on pipeline step count — discourages overly complex preprocessing chains.
_COMPLEXITY_PENALTY = 0.01


def _count_steps(pipeline: Pipeline) -> int:
    """Count the total number of named steps in the pipeline (preprocessor + model)."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return len(pipeline.steps)
    # Count each transformer inside the ColumnTransformer as one step
    n_transformers = len(getattr(preprocessor, "transformers", []))
    return n_transformers + 1  # transformers + model


def _composite_score(primary_metric: float, cv_std: float, n_steps: int) -> float:
    """
    Combine predictive performance, stability, and complexity into one scalar.

    All three components are in the same unit (metric points), so the
    penalties are interpretable: e.g. 0.5 * 0.10 = 0.05 point deducted
    for a pipeline whose F1 varies by 0.10 across folds.
    """
    return primary_metric - _CV_STD_PENALTY * cv_std - _COMPLEXITY_PENALTY * n_steps


def _cv_splits(target_type: TargetType, y: pd.Series, n_splits: int):
    """
    Return the appropriate cross-validation splitter for the task type.

    StratifiedKFold is used for classification to maintain class ratios in
    every fold — critical when classes are imbalanced.
    KFold is used for regression where stratification is not meaningful.
    """
    if target_type == TargetType.REGRESSION:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def _evaluate_single(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    profile: DataProfile,
    n_splits: int,
) -> Tuple[dict, float, float]:
    """
    Run cross-validated evaluation for one pipeline.

    Returns (metric_values, cv_std, runtime_secs).
    cv_std is the standard deviation of the primary metric across folds,
    not across all metrics — it captures fold-to-fold instability directly.
    """
    splitter = _cv_splits(profile.target_type, y, n_splits)
    fold_primaries = []
    fold_metrics: dict = {}

    t_start = time.perf_counter()

    for train_idx, val_idx in splitter.split(X, y if profile.target_type != TargetType.REGRESSION else None):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Clone to prevent fold-to-fold state leakage
        from sklearn.base import clone
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_train, y_train)
        y_pred = fold_pipe.predict(X_val)

        if profile.target_type == TargetType.BINARY:
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            fold_primaries.append(f1)

            try:
                y_prob = fold_pipe.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_prob)
                ll  = log_loss(y_val, y_prob)
            except (AttributeError, ValueError):
                auc = float("nan")
                ll  = float("nan")

            # Sensitivity = recall of positive class; specificity = recall of negative class
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=sorted(y_val.unique())).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

            try:
                brier = brier_score_loss(y_val, y_prob) if not np.isnan(auc) else float("nan")
            except Exception:
                brier = float("nan")

            fold_metrics.setdefault("f1_macro", []).append(f1)
            fold_metrics.setdefault("f1_weighted", []).append(
                f1_score(y_val, y_pred, average="weighted", zero_division=0))
            fold_metrics.setdefault("precision_macro", []).append(
                precision_score(y_val, y_pred, average="macro", zero_division=0))
            fold_metrics.setdefault("recall_macro", []).append(
                recall_score(y_val, y_pred, average="macro", zero_division=0))
            fold_metrics.setdefault("sensitivity", []).append(sensitivity)
            fold_metrics.setdefault("specificity", []).append(specificity)
            fold_metrics.setdefault("accuracy", []).append(
                accuracy_score(y_val, y_pred))
            fold_metrics.setdefault("balanced_accuracy", []).append(
                balanced_accuracy_score(y_val, y_pred))
            fold_metrics.setdefault("mcc", []).append(
                matthews_corrcoef(y_val, y_pred))
            fold_metrics.setdefault("cohen_kappa", []).append(
                cohen_kappa_score(y_val, y_pred))
            fold_metrics.setdefault("auc", []).append(auc)
            fold_metrics.setdefault("log_loss", []).append(ll)
            fold_metrics.setdefault("brier_score", []).append(brier)

        elif profile.target_type == TargetType.MULTICLASS:
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            fold_primaries.append(f1)

            try:
                y_prob = fold_pipe.predict_proba(X_val)
                auc = roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro")
                ll  = log_loss(y_val, y_prob)
                brier = float(np.mean([
                    brier_score_loss((y_val == cls).astype(int), y_prob[:, i])
                    for i, cls in enumerate(fold_pipe.classes_)
                ]))
            except (AttributeError, ValueError):
                auc   = float("nan")
                ll    = float("nan")
                brier = float("nan")

            fold_metrics.setdefault("f1_macro", []).append(f1)
            fold_metrics.setdefault("f1_weighted", []).append(
                f1_score(y_val, y_pred, average="weighted", zero_division=0))
            fold_metrics.setdefault("precision_macro", []).append(
                precision_score(y_val, y_pred, average="macro", zero_division=0))
            fold_metrics.setdefault("recall_macro", []).append(
                recall_score(y_val, y_pred, average="macro", zero_division=0))
            fold_metrics.setdefault("accuracy", []).append(
                accuracy_score(y_val, y_pred))
            fold_metrics.setdefault("balanced_accuracy", []).append(
                balanced_accuracy_score(y_val, y_pred))
            fold_metrics.setdefault("mcc", []).append(
                matthews_corrcoef(y_val, y_pred))
            fold_metrics.setdefault("cohen_kappa", []).append(
                cohen_kappa_score(y_val, y_pred))
            fold_metrics.setdefault("auc_ovr", []).append(auc)
            fold_metrics.setdefault("log_loss", []).append(ll)
            fold_metrics.setdefault("brier_score", []).append(brier)

        else:  # REGRESSION
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2   = float(r2_score(y_val, y_pred))
            # Negate RMSE so that higher = better (consistent with other metrics)
            fold_primaries.append(-rmse)
            fold_metrics.setdefault("rmse", []).append(rmse)
            fold_metrics.setdefault("mae", []).append(
                float(mean_absolute_error(y_val, y_pred)))
            fold_metrics.setdefault("median_ae", []).append(
                float(median_absolute_error(y_val, y_pred)))
            fold_metrics.setdefault("r2", []).append(r2)
            fold_metrics.setdefault("explained_variance", []).append(
                float(explained_variance_score(y_val, y_pred)))
            # MAPE — guard against zero targets
            with np.errstate(divide="ignore", invalid="ignore"):
                mape = float(np.nanmean(
                    np.abs((y_val - y_pred) / y_val.replace(0, np.nan))) * 100)
            fold_metrics.setdefault("mape", []).append(mape)

    runtime_secs = time.perf_counter() - t_start
    cv_std = float(np.std(fold_primaries))

    # Average each metric across folds; skip NaN folds (e.g. AUC on degenerate data)
    averaged = {
        k: float(np.nanmean(v)) for k, v in fold_metrics.items()
    }

    return averaged, cv_std, runtime_secs


def evaluate_plans(
    plans: List[ActionPlan],
    profile: DataProfile,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> List[EvaluationResult]:
    """
    Build and cross-validate each ActionPlan, returning a list of EvaluationResults.

    Plans that fail to build or evaluate (e.g. due to incompatible hyperparameters)
    are recorded with score = -inf so they sink to the bottom of the ranking without
    crashing the entire iteration.

    Parameters
    ----------
    plans   : candidate ActionPlans from the Planner Agent.
    profile : DataProfile used to determine column types and task type.
    X       : feature matrix (target column already removed).
    y       : target series.
    cv      : number of cross-validation folds (default 5).

    Returns
    -------
    List[EvaluationResult] in the same order as the input plans.
    """
    results = []

    for plan in plans:
        try:
            pipeline = build_pipeline(plan, profile, X)
            metric_values, cv_std, runtime_secs = _evaluate_single(pipeline, X, y, profile, cv)

            # Primary metric is the first key in metric_values (f1, f1_macro, or -rmse)
            primary = next(iter(metric_values.values()))
            n_steps = _count_steps(pipeline)
            score = _composite_score(primary, cv_std, n_steps)

            results.append(EvaluationResult(
                plan_id=plan.plan_id,
                score=score,
                metric_values=metric_values,
                cv_std=cv_std,
                runtime_secs=runtime_secs,
            ))

        except Exception as exc:
            # Record failure with score = -inf so ranking is unaffected
            results.append(EvaluationResult(
                plan_id=plan.plan_id,
                score=float("-inf"),
                metric_values={"error": str(exc)},
                cv_std=0.0,
                runtime_secs=0.0,
            ))

    return results


def select_best(results: List[EvaluationResult]) -> EvaluationResult:
    """
    Return the EvaluationResult with the highest composite score.

    Raises ValueError if the list is empty or all plans failed (score == -inf).
    """
    if not results:
        raise ValueError("Cannot select best from an empty results list.")

    valid = [r for r in results if r.score != float("-inf")]
    if not valid:
        raise ValueError("All evaluated plans failed — no valid result to select.")

    return max(valid, key=lambda r: r.score)
