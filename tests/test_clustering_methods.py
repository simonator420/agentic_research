"""
Tests for hierarchical and DBSCAN clustering in discover_clusters().
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.profiler import discover_clusters, summarize_patterns
from src.models.schemas import ClusterResult


@pytest.fixture
def clusterable_df():
    """DataFrame with clear clusters in numeric features."""
    rng = np.random.default_rng(99)
    n = 60
    # Three obvious groups
    x = np.concatenate([rng.normal(0, 0.3, n), rng.normal(5, 0.3, n), rng.normal(10, 0.3, n)])
    y = np.concatenate([rng.normal(0, 0.3, n), rng.normal(5, 0.3, n), rng.normal(10, 0.3, n)])
    return pd.DataFrame({"x": x, "y": y, "cat": ["a"] * (3 * n)})


class TestKMeans:
    def test_returns_cluster_result(self, clusterable_df):
        r = discover_clusters(clusterable_df, method="kmeans")
        assert isinstance(r, ClusterResult)

    def test_method_label(self, clusterable_df):
        r = discover_clusters(clusterable_df, method="kmeans")
        assert r.method == "kmeans"

    def test_n_clusters_reasonable(self, clusterable_df):
        r = discover_clusters(clusterable_df, method="kmeans")
        assert 2 <= r.n_clusters <= 8


class TestHierarchical:
    def test_returns_cluster_result(self, clusterable_df):
        r = discover_clusters(clusterable_df, method="hierarchical")
        assert isinstance(r, ClusterResult)

    def test_method_label(self, clusterable_df):
        r = discover_clusters(clusterable_df, method="hierarchical")
        assert r.method == "hierarchical"

    def test_n_clusters_reasonable(self, clusterable_df):
        r = discover_clusters(clusterable_df, method="hierarchical")
        assert 2 <= r.n_clusters <= 8


class TestDBSCAN:
    def test_returns_cluster_result_or_none(self, clusterable_df):
        r = discover_clusters(clusterable_df, method="dbscan")
        assert r is None or isinstance(r, ClusterResult)

    def test_method_label_when_found(self, clusterable_df):
        r = discover_clusters(clusterable_df, method="dbscan")
        if r is not None:
            assert r.method == "dbscan"

    def test_no_negative_cluster_ids_in_summaries(self, clusterable_df):
        """summarize_patterns must skip noise points (label -1)."""
        r = discover_clusters(clusterable_df, method="dbscan")
        if r is not None:
            r2 = summarize_patterns(clusterable_df.select_dtypes("number"), r)
            assert all(cid >= 0 for cid in r2.cluster_summaries)


class TestAutoMethod:
    def test_returns_cluster_result(self, clusterable_df):
        r = discover_clusters(clusterable_df, method="auto")
        assert isinstance(r, ClusterResult)

    def test_selects_best_by_silhouette(self, clusterable_df):
        """auto should find a solution with silhouette > 0.5 on clearly-clustered data."""
        r = discover_clusters(clusterable_df, method="auto")
        assert r.silhouette_score > 0.5
