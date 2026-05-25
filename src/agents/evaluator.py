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

Visualisation and reporting (sports analytics focus)
-----------------------------------------------------
Two additional functions are provided for non-technical sports users:
  generate_visualisations() — produces a standard set of plots (missingness heatmap,
    feature importance, confusion matrix or actual-vs-predicted, and a PCA projection
    of any discovered clusters) and saves them to a configurable output directory.
  build_user_report()       — assembles a plain-language markdown report describing
    the dataset, issues, selected preprocessing strategy, evaluation results, and
    cluster patterns in terms a coach or analyst can understand.

Public API
----------
evaluate_plans(plans, profile, X, y, cv)                    -> List[EvaluationResult]
select_best(results)                                         -> EvaluationResult
generate_visualisations(profile, best_result, best_pipeline,
                        X, y, output_dir)                   -> List[str]
build_user_report(profile, issues, best_plan,
                  best_result, visualisation_paths)          -> str
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
from src.models.schemas import (
    ActionPlan,
    DataProfile,
    EvaluationResult,
    Issue,
    IssueSeverity,
    TargetType,
)

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


def generate_visualisations(
    profile: DataProfile,
    best_result: EvaluationResult,
    best_pipeline: "Pipeline",
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str = "figures",
) -> List[str]:
    """
    Generate a standard visualisation pack for non-technical sports users.

    Produces up to five plots depending on what information is available:
      1. Missing data bar chart     — always generated if any column has NaN.
      2. Feature importance chart   — when the model exposes feature_importances_.
      3. Confusion matrix           — for binary and multiclass classification.
      4. Actual vs predicted plot   — for regression.
      5. PCA cluster projection     — when the DataProfile includes cluster results.

    Each figure is saved as a PNG to output_dir and the list of saved paths
    is returned.  The function is designed to be robust: any individual plot that
    fails is silently skipped so one bad visualisation does not block the others.

    Parameters
    ----------
    profile         : DataProfile (including cluster patterns if available).
    best_result     : EvaluationResult for the selected best plan.
    best_pipeline   : sklearn Pipeline fitted on the full training set.
    X               : feature matrix (target column removed).
    y               : target series.
    output_dir      : directory where PNG files are saved (created if necessary).

    Returns
    -------
    List of file paths for successfully saved figures.
    """
    import os
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe for notebooks and scripts
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    # --- 1. Missingness bar chart ---
    try:
        missing_items = [
            (c, col.missing_rate)
            for c, col in profile.columns.items()
            if col.missing_rate > 0
        ]
        if missing_items:
            cols_m, rates_m = zip(*missing_items)
            fig, ax = plt.subplots(figsize=(10, max(3, len(cols_m) * 0.35)))
            ax.barh(list(cols_m), list(rates_m), color="#4a9fd4")
            ax.axvline(x=0.30, color="crimson", linestyle="--", alpha=0.8,
                       label="High-missingness threshold (30%)")
            ax.set_xlabel("Missing value rate")
            ax.set_title("Missing Data by Column")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8)
            plt.tight_layout()
            path = os.path.join(output_dir, "missingness.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    except Exception:
        pass

    # --- 2. Feature importance ---
    try:
        model_step = best_pipeline.named_steps.get("model")
        if model_step is not None and hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
            preprocessor = best_pipeline.named_steps.get("preprocessor")
            try:
                feat_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feat_names = [f"feature_{i}" for i in range(len(importances))]

            if len(feat_names) == len(importances):
                fi = pd.Series(importances, index=feat_names).nlargest(15).sort_values()
                fig, ax = plt.subplots(figsize=(10, 6))
                fi.plot.barh(ax=ax, color="#4a9fd4")
                ax.set_title("Top Features by Importance (plain-language labels)")
                ax.set_xlabel("Feature importance score")
                plt.tight_layout()
                path = os.path.join(output_dir, "feature_importance.png")
                fig.savefig(path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
    except Exception:
        pass

    # --- 3. Confusion matrix (classification) or actual vs predicted (regression) ---
    try:
        from sklearn.model_selection import cross_val_predict
        if profile.target_type in (TargetType.BINARY, TargetType.MULTICLASS):
            from sklearn.metrics import ConfusionMatrixDisplay
            y_pred = cross_val_predict(best_pipeline, X, y, cv=3)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax, colorbar=False)
            ax.set_title("Confusion Matrix (3-fold CV predictions)")
            plt.tight_layout()
            path = os.path.join(output_dir, "confusion_matrix.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
        else:
            y_pred = cross_val_predict(best_pipeline, X, y, cv=3)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y, y_pred, alpha=0.5, color="#4a9fd4", s=20)
            lo = min(float(y.min()), float(y_pred.min()))
            hi = max(float(y.max()), float(y_pred.max()))
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted (3-fold CV)")
            ax.legend()
            plt.tight_layout()
            path = os.path.join(output_dir, "actual_vs_predicted.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    except Exception:
        pass

    # --- 4. PCA cluster projection ---
    try:
        clusters = profile.clusters
        if clusters is not None and clusters.n_clusters > 1 and clusters.cluster_summaries:
            from sklearn.decomposition import PCA

            valid_X = X.loc[clusters.valid_indices, clusters.numeric_columns]
            valid_X = valid_X.fillna(valid_X.mean())
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(valid_X)
            labels_arr = np.array(clusters.labels)

            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=labels_arr, cmap="tab10", alpha=0.7, s=30,
            )
            plt.colorbar(sc, ax=ax, label="Cluster ID")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)")
            ax.set_title(f"Natural Groupings in the Data — {clusters.n_clusters} Clusters (PCA projection)")

            # Label cluster centroids
            for cid in range(clusters.n_clusters):
                mask = labels_arr == cid
                cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
                ax.annotate(
                    f"C{cid}", (cx, cy),
                    fontsize=10, fontweight="bold", ha="center",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7},
                )
            plt.tight_layout()
            path = os.path.join(output_dir, "cluster_projection.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    except Exception:
        pass

    return saved


def build_user_report(
    profile: DataProfile,
    issues: List[Issue],
    best_plan: ActionPlan,
    best_result: EvaluationResult,
    visualisation_paths: List[str],
) -> str:
    """
    Assemble a plain-language markdown report for non-technical sports users.

    The report explains what the data looks like, what problems were found, what
    the system decided to do and why (in plain English), how well the model
    performed, and what the discovered cluster patterns mean.  It is designed
    to be read by a coach, performance analyst, or scout who does not have a
    machine learning background.

    Parameters
    ----------
    profile              : DataProfile including optional cluster patterns.
    issues               : Severity-sorted list of detected data quality issues.
    best_plan            : The ActionPlan selected by the Evaluator.
    best_result          : EvaluationResult for the best plan.
    visualisation_paths  : Paths to generated PNG files (for the file listing).

    Returns
    -------
    Markdown-formatted string suitable for display in a Jupyter notebook cell
    or saving as a .md file.
    """
    import os

    lines = [
        "# Agentic Sports Analytics — Results Report",
        "",
        "> *Generated automatically by the Agentic Sports Analytics System.*",
        "> *This report is designed for coaches, analysts, and scouts — no machine learning background required.*",
        "",
    ]

    # --- Dataset overview ---
    lines += [
        "## 1. Dataset Overview",
        "",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Rows | {profile.n_rows:,} |",
        f"| Columns | {profile.n_cols} |",
        f"| Target column | `{profile.target_column}` |",
        f"| Task type | {profile.target_type.value.capitalize()} |",
        f"| Duplicate rows | {profile.n_duplicates} |",
        "",
    ]
    if profile.class_distribution:
        dist_str = " · ".join(
            f"**{k}**: {v:.1%}" for k, v in sorted(profile.class_distribution.items())
        )
        lines += [f"**Class distribution:** {dist_str}", ""]

    # --- Data quality ---
    high = [i for i in issues if i.severity == IssueSeverity.HIGH]
    medium = [i for i in issues if i.severity == IssueSeverity.MEDIUM]
    lines += [
        "## 2. Data Quality",
        "",
        f"The system found **{len(issues)} data quality issues** "
        f"({len(high)} high-severity, {len(medium)} medium-severity).",
        "",
    ]
    if high:
        lines.append("**High-severity issues (require attention):**")
        for iss in high:
            lines.append(f"- {iss.description}")
        lines.append("")
    if medium:
        lines.append("**Medium-severity issues (handled automatically):**")
        for iss in medium:
            lines.append(f"- {iss.description}")
        lines.append("")

    # --- Exploratory cluster patterns ---
    if profile.clusters and profile.clusters.cluster_summaries:
        cl = profile.clusters
        lines += [
            "## 3. Natural Groupings in the Data (Exploratory Analysis)",
            "",
            f"The system identified **{cl.n_clusters} natural groups** in the data "
            f"using unsupervised clustering "
            f"(silhouette score: {cl.silhouette_score:.3f} — higher is better separation; "
            f"Davies–Bouldin index: {cl.davies_bouldin_index:.3f} — lower is better).",
            "",
            "**What each group looks like:**",
            "",
        ]
        for cid, summary in cl.cluster_summaries.items():
            lines.append(f"- {summary}")
        lines += [
            "",
            "> These groupings were discovered automatically from the data. "
            "They may correspond to player archetypes, match intensity levels, "
            "injury-risk profiles, or other domain-relevant categories.",
            "",
        ]

    # --- What the system did ---
    explanation = best_plan.model_params.get("__explanation", "")
    lines += [
        "## 4. Preprocessing and Modelling Strategy",
        "",
        "The system tested multiple approaches and selected the following configuration:",
        "",
        f"| Step | Choice |",
        f"|------|--------|",
        f"| Model | {best_plan.model.replace('_', ' ').title()} |",
        f"| Missing value handling | {best_plan.imputation.replace('_', ' ').title()} |",
        f"| Outlier treatment | {best_plan.outlier_handling.replace('_', ' ').title()} |",
        f"| Category encoding | {best_plan.encoding.replace('_', ' ').title()} |",
        f"| Feature scaling | {best_plan.scaling.replace('_', ' ').title()} |",
        f"| Class imbalance | {best_plan.imbalance_strategy.replace('_', ' ').title()} |",
        "",
    ]
    if explanation:
        lines += [f"> **Why this approach?** {explanation}", ""]

    # --- Performance ---
    lines += [
        "## 5. Model Performance",
        "",
        "Results are based on 5-fold cross-validation on the training set.",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in best_result.metric_values.items():
        if isinstance(v, float) and v == v:  # skip NaN
            lines.append(f"| {k} | {v:.4f} |")
    lines += [
        f"| CV stability (std) | {best_result.cv_std:.4f} *(lower = more consistent)* |",
        f"| Composite score | {best_result.score:.4f} |",
        "",
    ]

    # --- Visualisations ---
    if visualisation_paths:
        lines += [
            "## 6. Visualisations",
            "",
            "The following figures have been saved:",
            "",
        ]
        for p in visualisation_paths:
            lines.append(f"- `{os.path.basename(p)}`")
        lines.append("")

    lines += [
        "---",
        "*End of report.*",
    ]
    return "\n".join(lines)
