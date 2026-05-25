"""
profiler.py — Profiler Agent (Step 1 of the agentic loop).

Analyses a raw DataFrame and produces a DataProfile: a structured summary of
column types, missing value rates, numeric statistics, categorical distributions,
duplicate count, inferred ML task type, and exploratory cluster patterns.

The DataProfile is the primary input to every subsequent agent:
  - Issue Detector uses it to assign severity scores without re-scanning the data.
  - Planner embeds it verbatim into the LLM prompt as dataset context.
  - dataset_fingerprint() converts it to a vector for cross-run memory retrieval.

Exploratory clustering (sports analytics focus)
-----------------------------------------------
In addition to the standard structural profile, the Profiler runs unsupervised
k-means clustering on the numeric features to surface natural groupings — e.g.
player archetypes, match intensity tiers, or injury-risk profiles — before any
predictive modelling.  The cluster result is attached to the DataProfile so that
downstream agents (Planner, Evaluator) can reference it and the final user report
can present it in plain language to non-technical sports users.

Public API
----------
generate_profile(df, target, run_clustering=True) -> DataProfile
discover_clusters(df, method="auto", max_k=8, random_state=42)  -> Optional[ClusterResult]
summarize_patterns(df, clusters)                                  -> ClusterResult
"""

from typing import Dict, Optional

import pandas as pd

from src.agents.sports_vocabulary import detect_sports_context
from src.models.schemas import ClusterResult, ColumnProfile, DataProfile, TargetType

# Up to this many sample values are stored per column for the LLM prompt.
_MAX_SAMPLE_VALUES = 5

# Numeric target columns with more unique values than this are treated as regression,
# not multiclass. Chosen conservatively — most classification tasks have ≤ 20 classes.
_MULTICLASS_THRESHOLD = 20

# Number of most-frequent categories stored in ColumnProfile.top_categories.
_TOP_CATEGORIES = 10


def _infer_dtype(series: pd.Series) -> str:
    """
    Map a pandas Series to one of four logical types.

    The order of checks matters: bool must come before numeric because
    pandas considers bool a subtype of int.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _profile_column(series: pd.Series) -> ColumnProfile:
    """
    Compute per-column statistics and return a ColumnProfile.

    Numeric stats (mean, std, min, max, q1, q3) are populated only for numeric
    columns. q1 / q3 are used downstream by the Issue Detector for IQR-based
    outlier detection, so they must be present before detect_issues() is called.
    """
    dtype = _infer_dtype(series)
    non_null = series.dropna()
    missing_rate = float(series.isna().mean())
    n_unique = int(series.nunique())
    sample_values = non_null.unique()[:_MAX_SAMPLE_VALUES].tolist()

    col = ColumnProfile(
        name=str(series.name),
        dtype=dtype,
        missing_rate=missing_rate,
        n_unique=n_unique,
        sample_values=sample_values,
    )

    if dtype == "numeric" and len(non_null) > 0:
        col.mean = float(non_null.mean())
        col.std = float(non_null.std()) if len(non_null) > 1 else 0.0
        col.min = float(non_null.min())
        col.max = float(non_null.max())
        col.q1 = float(non_null.quantile(0.25))
        col.q3 = float(non_null.quantile(0.75))

    elif dtype == "categorical":
        counts = non_null.value_counts()
        col.top_categories = {str(k): int(v) for k, v in counts.head(_TOP_CATEGORIES).items()}

    return col


def _infer_target_type(series: pd.Series) -> TargetType:
    """
    Decide the ML task type from the target column alone — no user input required.

    Rules:
      - numeric AND more than _MULTICLASS_THRESHOLD unique values → REGRESSION
      - exactly 2 unique values → BINARY
      - everything else → MULTICLASS
    """
    n_unique = series.nunique()
    if pd.api.types.is_numeric_dtype(series) and n_unique > _MULTICLASS_THRESHOLD:
        return TargetType.REGRESSION
    if n_unique <= 2:
        return TargetType.BINARY
    return TargetType.MULTICLASS


def discover_clusters(
    df: pd.DataFrame,
    method: str = "auto",
    max_k: int = 8,
    random_state: int = 42,
) -> Optional[ClusterResult]:
    """
    Run unsupervised clustering on the numeric features of a DataFrame.

    Uses k-means with the number of clusters chosen by the highest silhouette score
    across k ∈ [2, min(max_k, n_rows//5)].  Returns None when the DataFrame has
    fewer than 20 complete numeric rows or fewer than 2 numeric columns — these
    constraints prevent meaningless clustering on degenerate inputs.

    Parameters
    ----------
    df           : DataFrame to cluster (target column should already be excluded).
    method       : "auto" selects k-means; reserved for future extension to
                   "hierarchical" or "dbscan".
    max_k        : maximum number of clusters to consider.
    random_state : random seed for reproducibility.

    Returns
    -------
    ClusterResult with labels and silhouette / Davies–Bouldin quality scores,
    or None if clustering is not feasible on this data.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import davies_bouldin_score, silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    numeric_df = df.select_dtypes(include="number").dropna()

    if len(numeric_df) < 20 or len(numeric_df.columns) < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    upper_k = min(max_k + 1, len(numeric_df) // 5 + 1)
    if upper_k <= 2:
        return None

    best_k, best_sil, best_labels = 2, -1.0, None
    for k in range(2, upper_k):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, labels

    if best_labels is None:
        return None

    db = float(davies_bouldin_score(X_scaled, best_labels))

    return ClusterResult(
        n_clusters=best_k,
        labels=best_labels.tolist(),
        valid_indices=numeric_df.index.tolist(),
        numeric_columns=list(numeric_df.columns),
        silhouette_score=best_sil,
        davies_bouldin_index=db,
        cluster_summaries={},
        method="kmeans",
    )


def summarize_patterns(df: pd.DataFrame, clusters: ClusterResult) -> ClusterResult:
    """
    Enrich a ClusterResult with plain-language cluster summaries.

    For each cluster, finds the top 3 numeric features that deviate most from
    the overall mean, then writes a short human-readable label — e.g.
    "Cluster 2 (47 obs): high sprint_distance, low minutes_played, high fouls_committed".
    This lets non-technical sports users interpret the groupings without needing
    to read raw cluster centroids.

    Parameters
    ----------
    df       : the same DataFrame that was passed to discover_clusters()
               (target column excluded).
    clusters : ClusterResult whose cluster_summaries dict is to be populated.

    Returns
    -------
    A new ClusterResult with cluster_summaries filled in.
    """
    valid_df = df.loc[clusters.valid_indices, clusters.numeric_columns]
    overall_mean = valid_df.mean()
    labels_series = pd.Series(clusters.labels, index=clusters.valid_indices)

    summaries: Dict[int, str] = {}
    for cid in range(clusters.n_clusters):
        mask = labels_series == cid
        cluster_mean = valid_df.loc[mask].mean()
        diff = (cluster_mean - overall_mean).abs()
        top_feats = diff.nlargest(3)

        parts = []
        for feat in top_feats.index:
            direction = "high" if cluster_mean[feat] > overall_mean[feat] else "low"
            parts.append(f"{direction} {feat}")

        n_obs = int(mask.sum())
        summaries[cid] = f"Cluster {cid} ({n_obs} obs): {', '.join(parts)}"

    return ClusterResult(
        n_clusters=clusters.n_clusters,
        labels=clusters.labels,
        valid_indices=clusters.valid_indices,
        numeric_columns=clusters.numeric_columns,
        silhouette_score=clusters.silhouette_score,
        davies_bouldin_index=clusters.davies_bouldin_index,
        cluster_summaries=summaries,
        method=clusters.method,
    )


def generate_profile(
    df: pd.DataFrame,
    target: str,
    run_clustering: bool = True,
) -> DataProfile:
    """
    Main entry point for the Profiler Agent.

    Scans every column, infers the task type, computes class distribution
    (classification only), counts duplicate rows, and (optionally) runs
    exploratory clustering on the numeric feature space.

    Parameters
    ----------
    df             : raw input DataFrame (train + test combined, before any split).
    target         : name of the column to predict.
    run_clustering : when True, calls discover_clusters() and summarize_patterns()
                     and attaches the result to DataProfile.clusters.  Set to False
                     for speed in unit tests or ablation runs where exploratory output
                     is not needed.

    Returns
    -------
    DataProfile — used by all downstream agents and stored in memory.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    columns: Dict[str, ColumnProfile] = {col: _profile_column(df[col]) for col in df.columns}

    target_type = _infer_target_type(df[target])

    # Class distribution is only meaningful for classification tasks.
    if target_type == TargetType.REGRESSION:
        class_dist = None
    else:
        counts = df[target].value_counts(normalize=True)
        class_dist = {str(k): float(v) for k, v in counts.items()}

    feature_df = df.drop(columns=[target])

    # Sports domain vocabulary detection — scans column names only, no data access.
    sports_ctx = detect_sports_context(feature_df)

    # Exploratory clustering on features only (exclude target to avoid leaking label info).
    clusters = None
    if run_clustering:
        raw_clusters = discover_clusters(feature_df)
        if raw_clusters is not None:
            clusters = summarize_patterns(feature_df, raw_clusters)

    return DataProfile(
        n_rows=int(len(df)),
        n_cols=int(len(df.columns)),
        target_column=target,
        target_type=target_type,
        class_distribution=class_dist,
        columns=columns,
        n_duplicates=int(df.duplicated().sum()),
        clusters=clusters,
        sports_context=sports_ctx,
    )
