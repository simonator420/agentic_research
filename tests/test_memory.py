"""
Tests for the memory module.

RunStore  : tested against a real SQLite database in a temp directory.
VectorStore: tested against a real ChromaDB instance in a temp directory.
Both stores are exercised without mocking — their correctness depends on
actual I/O behaviour that mocks cannot verify.
"""

import pytest

from src.memory.run_store import RunStore
from src.memory.vector_store import VectorStore
from src.models.schemas import ActionPlan, AttemptRecord, EvaluationResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_plan(plan_id: str = "p1", model: str = "logistic_regression") -> ActionPlan:
    return ActionPlan(
        plan_id=plan_id,
        imputation="median",
        outlier_handling="none",
        encoding="onehot",
        scaling="standard",
        model=model,
        imbalance_strategy="none",
        model_params={"C": 1.0},
    )


def _make_record(iteration: int = 1, plan_id: str = "p1", score: float = 0.85) -> AttemptRecord:
    plan = _make_plan(plan_id=plan_id)
    result = EvaluationResult(
        plan_id=plan_id,
        score=score,
        metric_values={"f1": 0.83, "auc": 0.88},
        cv_std=0.03,
        runtime_secs=2.5,
    )
    return AttemptRecord(iteration=iteration, plan=plan, result=result)


def _make_fingerprint(seed: int = 0) -> list:
    """Return a deterministic 8-dimensional fingerprint vector."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, 8).tolist()


# ---------------------------------------------------------------------------
# RunStore tests
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_runs.db")
    with RunStore(db_path) as s:
        yield s


def test_run_store_save_and_load(store):
    record = _make_record(iteration=1, plan_id="p1", score=0.80)
    store.save_attempt("run_abc", record)

    loaded = store.load_attempts("run_abc")
    assert len(loaded) == 1
    assert loaded[0].iteration == 1
    assert loaded[0].plan.plan_id == "p1"
    assert loaded[0].result.score == pytest.approx(0.80)


def test_run_store_preserves_model_params(store):
    """model_params (dict) must survive JSON round-trip."""
    record = _make_record()
    store.save_attempt("run_params", record)
    loaded = store.load_attempts("run_params")
    assert loaded[0].plan.model_params == {"C": 1.0}


def test_run_store_preserves_metric_values(store):
    """metric_values (dict) must survive JSON round-trip."""
    record = _make_record()
    store.save_attempt("run_metrics", record)
    loaded = store.load_attempts("run_metrics")
    assert loaded[0].result.metric_values == {"f1": 0.83, "auc": 0.88}


def test_run_store_multiple_iterations(store):
    """All iterations for a run must be returned in order."""
    for i in range(1, 4):
        store.save_attempt("run_multi", _make_record(iteration=i, plan_id=f"p{i}"))

    loaded = store.load_attempts("run_multi")
    assert len(loaded) == 3
    assert [r.iteration for r in loaded] == [1, 2, 3]


def test_run_store_isolates_runs(store):
    """Records from different run_ids must not bleed into each other."""
    store.save_attempt("run_A", _make_record(plan_id="pA"))
    store.save_attempt("run_B", _make_record(plan_id="pB"))

    loaded_A = store.load_attempts("run_A")
    loaded_B = store.load_attempts("run_B")

    assert len(loaded_A) == 1 and loaded_A[0].plan.plan_id == "pA"
    assert len(loaded_B) == 1 and loaded_B[0].plan.plan_id == "pB"


def test_run_store_empty_run(store):
    """Loading a run that has no records must return an empty list."""
    loaded = store.load_attempts("nonexistent_run")
    assert loaded == []


def test_run_store_list_runs(store):
    store.save_attempt("run_1", _make_record())
    store.save_attempt("run_2", _make_record())
    runs = store.list_runs()
    assert set(runs) == {"run_1", "run_2"}


def test_run_store_context_manager(tmp_path):
    """RunStore used as context manager must not raise on exit."""
    db_path = str(tmp_path / "ctx.db")
    with RunStore(db_path) as s:
        s.save_attempt("run_ctx", _make_record())
    # Re-open to verify data was persisted
    with RunStore(db_path) as s:
        loaded = s.load_attempts("run_ctx")
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

@pytest.fixture
def vstore(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma"))


def test_vector_store_empty_returns_empty(vstore):
    """Querying an empty store must return an empty list, not raise."""
    result = vstore.retrieve_similar(_make_fingerprint(), top_k=3)
    assert result == []


def test_vector_store_count_zero_on_init(vstore):
    assert vstore.count() == 0


def test_vector_store_store_and_retrieve(vstore):
    plan = _make_plan("stored_plan")
    fp = _make_fingerprint(seed=0)
    vstore.store_success(fp, plan, dataset_name="test_dataset")

    results = vstore.retrieve_similar(fp, top_k=1)
    assert len(results) == 1
    assert results[0].model == "logistic_regression"
    assert results[0].imputation == "median"


def test_vector_store_preserves_model_params(vstore):
    """model_params must survive the JSON encoding round-trip in metadata."""
    plan = _make_plan()
    plan.model_params = {"n_estimators": 200, "max_depth": 5}
    vstore.store_success(_make_fingerprint(), plan)

    retrieved = vstore.retrieve_similar(_make_fingerprint(), top_k=1)
    assert retrieved[0].model_params == {"n_estimators": 200, "max_depth": 5}


def test_vector_store_count_increments(vstore):
    vstore.store_success(_make_fingerprint(0), _make_plan("p1"))
    vstore.store_success(_make_fingerprint(1), _make_plan("p2"))
    assert vstore.count() == 2


def test_vector_store_top_k_cap(vstore):
    """retrieve_similar must never return more than top_k results."""
    for i in range(5):
        vstore.store_success(_make_fingerprint(i), _make_plan(f"p{i}"))

    results = vstore.retrieve_similar(_make_fingerprint(0), top_k=2)
    assert len(results) <= 2


def test_vector_store_similar_fingerprint_retrieved_first(vstore):
    """
    A fingerprint identical to the stored one should be the nearest neighbour.
    Uses two clearly distinct fingerprints to avoid ambiguity.
    """
    fp_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fp_b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    vstore.store_success(fp_a, _make_plan("plan_a"))
    vstore.store_success(fp_b, _make_plan("plan_b"))

    # Query with fp_a — plan_a should be first
    results = vstore.retrieve_similar(fp_a, top_k=2)
    assert results[0].plan_id == "plan_a"
