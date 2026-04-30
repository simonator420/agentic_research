"""
Tests for the Orchestrator.

The Planner (Claude API) is mocked throughout so tests run without network
access or an API key. All other components (Profiler, Issue Detector,
Executor, Evaluator, RunStore, VectorStore) run against real implementations
so the integration logic is exercised end-to-end.
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.schemas import ActionPlan, RunResult
from src.orchestrator import run_agentic_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
        "x1":  rng.normal(0, 1, n),
        "x2":  rng.normal(5, 2, n),
        "cat": rng.choice(["A", "B", "C"], n),
        "price": rng.uniform(100, 1000, n),
    })


def _mock_planner(plans_per_call):
    """
    Return a mock for propose_action_plans that cycles through pre-built ActionPlans.
    Each call pops the first batch, so multiple rounds get distinct plans.
    """
    call_count = {"n": 0}

    def _mock(profile, issues, history, memory=None,
               n_plans=3, model=None, api_key=None, max_retries=2):
        idx = call_count["n"]
        call_count["n"] += 1
        batch = plans_per_call[idx % len(plans_per_call)]
        return batch, f"Mock reasoning for call {idx + 1}"

    return _mock


def _make_plans(models=("logistic_regression", "random_forest", "gradient_boosting"),
                encoding="onehot") -> list:
    return [
        ActionPlan(
            plan_id=f"mock_{m}",
            imputation="median",
            outlier_handling="none",
            encoding=encoding,
            scaling="standard",
            model=m,
            imbalance_strategy="none",
        )
        for m in models
    ]


# ---------------------------------------------------------------------------
# Core return type tests
# ---------------------------------------------------------------------------

def test_returns_run_result(tmp_path, binary_df):
    mock = _mock_planner([_make_plans()])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            max_rounds=1, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )
    assert isinstance(result, RunResult)


def test_best_pipeline_is_sklearn_pipeline(tmp_path, binary_df):
    mock = _mock_planner([_make_plans()])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            max_rounds=1, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )
    assert isinstance(result.best_pipeline, Pipeline)


def test_best_pipeline_can_predict(tmp_path, binary_df):
    mock = _mock_planner([_make_plans()])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            max_rounds=1, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )
    X = binary_df.drop(columns=["target"])
    preds = result.best_pipeline.predict(X)
    assert preds.shape == (len(X),)


def test_history_contains_all_attempts(tmp_path, binary_df):
    """3 plans × 2 rounds = 6 AttemptRecords in history."""
    mock = _mock_planner([_make_plans(), _make_plans(("logistic_regression", "ridge", "random_forest"))])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            max_rounds=2, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )
    assert len(result.history) == 6


def test_run_id_is_set(tmp_path, binary_df):
    mock = _mock_planner([_make_plans()])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            run_id="my_custom_run",
            max_rounds=1, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )
    assert result.run_id == "my_custom_run"


def test_run_id_auto_generated(tmp_path, binary_df):
    mock = _mock_planner([_make_plans()])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            max_rounds=1, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )
    assert isinstance(result.run_id, str) and len(result.run_id) > 0


def test_n_iterations_matches_rounds_run(tmp_path, binary_df):
    mock = _mock_planner([_make_plans(), _make_plans()])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            max_rounds=2, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )
    assert result.n_iterations == 2


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

def test_stops_early_on_score_threshold(tmp_path, binary_df):
    """If score_threshold is set very low, run should converge in round 1."""
    mock = _mock_planner([_make_plans()])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            max_rounds=3, cv=3,
            score_threshold=0.0,   # any positive score triggers convergence
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )
    assert result.converged is True
    assert result.n_iterations == 1


def test_converged_false_when_max_rounds_exhausted(tmp_path, binary_df):
    plans_a = _make_plans(("logistic_regression", "random_forest", "gradient_boosting"))
    plans_b = _make_plans(("logistic_regression", "random_forest", "gradient_boosting"))
    mock = _mock_planner([plans_a, plans_b])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            max_rounds=2, cv=3,
            score_threshold=1.0,   # impossibly high — never converges
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )
    assert result.converged is False


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

def test_attempts_persisted_to_sqlite(tmp_path, binary_df):
    """All AttemptRecords must be saved to the SQLite store during the run."""
    from src.memory.run_store import RunStore

    mock = _mock_planner([_make_plans()])
    db_path = str(tmp_path / "runs.db")
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            max_rounds=1, cv=3,
            db_path=db_path,
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )

    with RunStore(db_path) as store:
        saved = store.load_attempts(result.run_id)
    assert len(saved) == len(result.history)


def test_best_plan_stored_in_vector_store(tmp_path, binary_df):
    """After a run, the best plan must be stored in ChromaDB."""
    from src.memory.vector_store import VectorStore

    mock = _mock_planner([_make_plans()])
    chroma_dir = str(tmp_path / "chroma")
    with patch("src.orchestrator.propose_action_plans", mock):
        run_agentic_pipeline(
            binary_df, "target",
            max_rounds=1, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=chroma_dir,
            verbose=False,
        )

    vstore = VectorStore(chroma_dir)
    assert vstore.count() == 1


# ---------------------------------------------------------------------------
# Regression task test
# ---------------------------------------------------------------------------

def test_regression_pipeline(tmp_path, regression_df):
    plans = _make_plans(("linear_regression", "ridge", "random_forest"), encoding="onehot")
    mock = _mock_planner([plans])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            regression_df, "price",
            max_rounds=1, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            verbose=False,
        )
    X = regression_df.drop(columns=["price"])
    preds = result.best_pipeline.predict(X)
    assert preds.shape == (len(X),)
    assert np.issubdtype(preds.dtype, np.floating)


# ---------------------------------------------------------------------------
# Ablation: use_memory=False
# ---------------------------------------------------------------------------

def test_use_memory_false_returns_valid_result(tmp_path, binary_df):
    """use_memory=False must still produce a valid RunResult without touching ChromaDB."""
    mock = _mock_planner([_make_plans()])
    with patch("src.orchestrator.propose_action_plans", mock):
        result = run_agentic_pipeline(
            binary_df, "target",
            max_rounds=1, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            use_memory=False,
            verbose=False,
        )
    assert isinstance(result, RunResult)
    assert np.isfinite(result.best_result.score)


def test_use_memory_false_does_not_write_chroma(tmp_path, binary_df):
    """When use_memory=False, ChromaDB must remain empty after the run."""
    from src.memory.vector_store import VectorStore

    mock = _mock_planner([_make_plans()])
    chroma_dir = str(tmp_path / "chroma")
    with patch("src.orchestrator.propose_action_plans", mock):
        run_agentic_pipeline(
            binary_df, "target",
            max_rounds=1, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=chroma_dir,
            use_memory=False,
            verbose=False,
        )
    vstore = VectorStore(chroma_dir)
    assert vstore.count() == 0


def test_use_memory_false_passes_empty_memory_to_planner(tmp_path, binary_df):
    """The Planner must receive an empty memory list when use_memory=False."""
    received_memory = []

    def _capturing_mock(profile, issues, history, memory=None,
                        n_plans=3, model=None, api_key=None, max_retries=2):
        received_memory.append(memory)
        return _make_plans(), "mock reasoning"

    with patch("src.orchestrator.propose_action_plans", _capturing_mock):
        run_agentic_pipeline(
            binary_df, "target",
            max_rounds=1, cv=3,
            db_path=str(tmp_path / "runs.db"),
            chroma_dir=str(tmp_path / "chroma"),
            use_memory=False,
            verbose=False,
        )
    assert all(m == [] for m in received_memory)
