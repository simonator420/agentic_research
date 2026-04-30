"""
Tests for the Planner Agent.

The Claude API call is mocked throughout — tests verify prompt formatting,
JSON parsing, error handling, and ActionPlan construction without making
real network requests.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.planner import (
    _format_history,
    _format_issues,
    _format_profile,
    _format_memory,
    _parse_plans,
    propose_action_plans,
)
from src.agents.profiler import generate_profile
from src.agents.issue_detector import detect_issues
from src.models.schemas import (
    ActionPlan,
    AttemptRecord,
    EvaluationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_df():
    rng = np.random.default_rng(0)
    n = 200
    target = ([0] * 180) + ([1] * 20)   # imbalanced
    return pd.DataFrame({
        "age":    rng.integers(20, 70, n).astype(float),
        "income": np.where(rng.random(n) < 0.12, np.nan, rng.uniform(20000, 200000, n)),
        "city":   rng.choice(["NY", "LA", "SF"], n),
        "target": target,
    })


@pytest.fixture
def profile(binary_df):
    return generate_profile(binary_df, "target")


@pytest.fixture
def issues(profile, binary_df):
    return detect_issues(profile, binary_df)


@pytest.fixture
def sample_history():
    plan = ActionPlan(
        plan_id="plan_abc",
        imputation="median",
        outlier_handling="none",
        encoding="onehot",
        scaling="standard",
        model="logistic_regression",
        imbalance_strategy="none",
    )
    result = EvaluationResult(
        plan_id="plan_abc",
        score=0.72,
        metric_values={"f1": 0.70, "auc": 0.75},
        cv_std=0.04,
        runtime_secs=1.2,
    )
    return [AttemptRecord(iteration=1, plan=plan, result=result)]


def _valid_api_response(n_plans: int = 3) -> dict:
    """Build a minimal valid JSON response mimicking Claude's output."""
    models = ["logistic_regression", "random_forest", "gradient_boosting"]
    plans = [
        {
            "plan_id": f"plan_{i}",
            "imputation": "median",
            "outlier_handling": "none",
            "encoding": "onehot",
            "scaling": "standard",
            "model": models[i % len(models)],
            "imbalance_strategy": "class_weight",
            "model_params": {},
        }
        for i in range(n_plans)
    ]
    return {"plans": plans, "reasoning": "Test reasoning text."}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def test_format_profile_contains_key_info(profile):
    text = _format_profile(profile)
    assert "target" in text
    assert "binary" in text
    assert "age" in text
    assert "income" in text
    assert "city" in text


def test_format_issues_contains_severity(issues):
    text = _format_issues(issues)
    # Issues exist — text should mention at least one severity level
    assert any(level in text for level in ["HIGH", "MEDIUM", "LOW"])


def test_format_issues_empty():
    text = _format_issues([])
    assert "none" in text.lower()


def test_format_history_empty():
    text = _format_history([])
    assert "first iteration" in text.lower()


def test_format_history_shows_config(sample_history):
    text = _format_history(sample_history)
    assert "logistic_regression" in text
    assert "0.72" in text or "0.7200" in text


def test_format_memory_empty():
    text = _format_memory([])
    assert text == ""


def test_format_memory_shows_config():
    plan = ActionPlan(
        plan_id="mem_1",
        imputation="knn",
        outlier_handling="winsorize",
        encoding="target",
        scaling="robust",
        model="random_forest",
        imbalance_strategy="smote",
    )
    text = _format_memory([plan])
    assert "knn" in text
    assert "random_forest" in text


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def test_parse_plans_returns_action_plans():
    raw = json.dumps(_valid_api_response(3))
    plans, reasoning = _parse_plans(raw, n_plans=3)
    assert len(plans) == 3
    assert all(isinstance(p, ActionPlan) for p in plans)
    assert reasoning == "Test reasoning text."


def test_parse_plans_assigns_unique_ids():
    raw = json.dumps(_valid_api_response(3))
    plans, _ = _parse_plans(raw, n_plans=3)
    ids = [p.plan_id for p in plans]
    assert len(set(ids)) == 3, "Each plan must have a unique plan_id"


def test_parse_plans_respects_n_plans_cap():
    """If Claude returns more plans than requested, only n_plans are used."""
    raw = json.dumps(_valid_api_response(5))
    plans, _ = _parse_plans(raw, n_plans=2)
    assert len(plans) == 2


def test_parse_plans_invalid_json_raises():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_plans("not json at all", n_plans=3)


def test_parse_plans_missing_field_raises():
    incomplete = {"plans": [{"imputation": "median"}], "reasoning": ""}
    with pytest.raises(ValueError, match="missing required fields"):
        _parse_plans(json.dumps(incomplete), n_plans=1)


def test_parse_plans_empty_list_raises():
    empty = {"plans": [], "reasoning": "nothing"}
    with pytest.raises(ValueError, match="zero ActionPlans"):
        _parse_plans(json.dumps(empty), n_plans=3)


# ---------------------------------------------------------------------------
# propose_action_plans — mocked API
# ---------------------------------------------------------------------------

def _make_mock_client(response_json: dict):
    """Create a mock anthropic.Anthropic client that returns a preset response."""
    mock_content = MagicMock()
    mock_content.text = json.dumps(response_json)

    mock_response = MagicMock()
    mock_response.content = [mock_content]

    mock_messages = MagicMock()
    mock_messages.create.return_value = mock_response

    mock_client = MagicMock()
    mock_client.messages = mock_messages
    return mock_client


@patch("src.agents.planner.anthropic.Anthropic")
def test_propose_returns_correct_number_of_plans(mock_anthropic, profile, issues):
    mock_anthropic.return_value = _make_mock_client(_valid_api_response(3))
    plans, reasoning = propose_action_plans(profile, issues, history=[], n_plans=3)
    assert len(plans) == 3
    assert isinstance(reasoning, str)


@patch("src.agents.planner.anthropic.Anthropic")
def test_propose_passes_profile_to_api(mock_anthropic, profile, issues):
    """Verify that the API is actually called (not short-circuited)."""
    mock_client = _make_mock_client(_valid_api_response(3))
    mock_anthropic.return_value = mock_client
    propose_action_plans(profile, issues, history=[], n_plans=3)
    assert mock_client.messages.create.called


@patch("src.agents.planner.anthropic.Anthropic")
def test_propose_with_history(mock_anthropic, profile, issues, sample_history):
    mock_anthropic.return_value = _make_mock_client(_valid_api_response(3))
    plans, _ = propose_action_plans(profile, issues, history=sample_history, n_plans=3)
    assert len(plans) == 3


@patch("src.agents.planner.anthropic.Anthropic")
def test_propose_with_memory(mock_anthropic, profile, issues):
    memory_plan = ActionPlan(
        plan_id="mem_1",
        imputation="knn",
        outlier_handling="winsorize",
        encoding="target",
        scaling="robust",
        model="random_forest",
        imbalance_strategy="smote",
    )
    mock_anthropic.return_value = _make_mock_client(_valid_api_response(3))
    plans, _ = propose_action_plans(profile, issues, history=[], memory=[memory_plan], n_plans=3)
    assert len(plans) == 3


@patch("src.agents.planner.anthropic.Anthropic")
def test_propose_retries_on_bad_json(mock_anthropic, profile, issues):
    """On first call returns bad JSON; second call returns valid JSON — should succeed."""
    bad_content = MagicMock()
    bad_content.text = "THIS IS NOT JSON"

    good_content = MagicMock()
    good_content.text = json.dumps(_valid_api_response(3))

    bad_response = MagicMock()
    bad_response.content = [bad_content]

    good_response = MagicMock()
    good_response.content = [good_content]

    mock_messages = MagicMock()
    mock_messages.create.side_effect = [bad_response, good_response]

    mock_client = MagicMock()
    mock_client.messages = mock_messages
    mock_anthropic.return_value = mock_client

    plans, _ = propose_action_plans(profile, issues, history=[], n_plans=3, max_retries=1)
    assert len(plans) == 3


@patch("src.agents.planner.anthropic.Anthropic")
def test_propose_raises_after_all_retries_fail(mock_anthropic, profile, issues):
    """All retries return bad JSON → RuntimeError should be raised."""
    bad_content = MagicMock()
    bad_content.text = "NOT JSON"

    bad_response = MagicMock()
    bad_response.content = [bad_content]

    mock_messages = MagicMock()
    mock_messages.create.return_value = bad_response

    mock_client = MagicMock()
    mock_client.messages = mock_messages
    mock_anthropic.return_value = mock_client

    with pytest.raises(RuntimeError, match="failed to return valid ActionPlans"):
        propose_action_plans(profile, issues, history=[], n_plans=3, max_retries=1)
