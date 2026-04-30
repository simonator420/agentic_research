import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.agents.executor import Winsorizer, build_pipeline
from src.agents.profiler import generate_profile
from src.models.schemas import ActionPlan


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_plan(**overrides) -> ActionPlan:
    """Return a baseline ActionPlan with safe defaults; override specific fields."""
    defaults = dict(
        plan_id="test-plan",
        imputation="median",
        outlier_handling="none",
        encoding="onehot",
        scaling="standard",
        model="logistic_regression",
        imbalance_strategy="none",
    )
    defaults.update(overrides)
    return ActionPlan(**defaults)


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "age":    rng.integers(20, 70, n).astype(float),
        "income": rng.uniform(20000, 200000, n),
        "city":   rng.choice(["NY", "LA", "SF"], n),
        "target": rng.integers(0, 2, n),
    })
    return df


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame({
        "x1":    rng.normal(0, 1, n),
        "x2":    rng.normal(5, 2, n),
        "cat":   rng.choice(["A", "B", "C"], n),
        "price": rng.uniform(100, 1000, n),
    })
    return df


# ---------------------------------------------------------------------------
# Winsorizer tests
# ---------------------------------------------------------------------------

def test_winsorizer_clips_extremes():
    """Values beyond 1.5*IQR fence should be clipped to the fence boundary."""
    X = np.array([[1], [2], [3], [4], [1000]])  # 1000 is a clear outlier
    w = Winsorizer()
    w.fit(X)
    Xt = w.transform(X)
    assert Xt[-1, 0] < 1000, "Outlier was not clipped"


def test_winsorizer_preserves_inliers():
    """Values within the fence should remain unchanged."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    w = Winsorizer()
    w.fit(X)
    Xt = w.transform(X)
    np.testing.assert_array_almost_equal(Xt, X)


def test_winsorizer_uses_training_bounds():
    """Bounds from fit() must be applied to transform() — not recomputed."""
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    X_test = np.array([[100.0]])
    w = Winsorizer()
    w.fit(X_train)
    Xt = w.transform(X_test)
    assert Xt[0, 0] == w.upper_[0]


# ---------------------------------------------------------------------------
# build_pipeline — structure tests
# ---------------------------------------------------------------------------

def test_returns_sklearn_pipeline(binary_data):
    profile = generate_profile(binary_data, "target")
    X = binary_data.drop(columns=["target"])
    pipe = build_pipeline(_make_plan(), profile, X)
    assert isinstance(pipe, Pipeline)


def test_pipeline_has_preprocessor_and_model(binary_data):
    profile = generate_profile(binary_data, "target")
    X = binary_data.drop(columns=["target"])
    pipe = build_pipeline(_make_plan(), profile, X)
    step_names = [name for name, _ in pipe.steps]
    assert "preprocessor" in step_names
    assert "model" in step_names


def test_pipeline_fit_predict_binary(binary_data):
    """Pipeline must fit and produce predictions with correct shape."""
    profile = generate_profile(binary_data, "target")
    X = binary_data.drop(columns=["target"])
    y = binary_data["target"]
    pipe = build_pipeline(_make_plan(), profile, X)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_pipeline_fit_predict_regression(regression_data):
    """Regression pipeline must produce continuous predictions."""
    profile = generate_profile(regression_data, "price")
    X = regression_data.drop(columns=["price"])
    y = regression_data["price"]
    plan = _make_plan(model="ridge", encoding="onehot", scaling="robust")
    pipe = build_pipeline(plan, profile, X)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)
    assert preds.dtype == float or np.issubdtype(preds.dtype, np.floating)


# ---------------------------------------------------------------------------
# build_pipeline — strategy combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("imputation", ["median", "mean", "knn"])
def test_imputation_strategies(binary_data, imputation):
    """All supported imputation strategies should produce a working pipeline."""
    profile = generate_profile(binary_data, "target")
    X = binary_data.drop(columns=["target"])
    y = binary_data["target"]
    pipe = build_pipeline(_make_plan(imputation=imputation), profile, X)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


@pytest.mark.parametrize("scaling", ["standard", "robust", "minmax", "none"])
def test_scaling_strategies(binary_data, scaling):
    profile = generate_profile(binary_data, "target")
    X = binary_data.drop(columns=["target"])
    y = binary_data["target"]
    pipe = build_pipeline(_make_plan(scaling=scaling), profile, X)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


@pytest.mark.parametrize("encoding", ["onehot", "ordinal"])
def test_encoding_strategies(binary_data, encoding):
    profile = generate_profile(binary_data, "target")
    X = binary_data.drop(columns=["target"])
    y = binary_data["target"]
    pipe = build_pipeline(_make_plan(encoding=encoding), profile, X)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


def test_winsorize_in_pipeline(binary_data):
    """Winsorizer should be inserted into the numeric sub-pipeline when requested."""
    profile = generate_profile(binary_data, "target")
    X = binary_data.drop(columns=["target"])
    y = binary_data["target"]
    plan = _make_plan(outlier_handling="winsorize")
    pipe = build_pipeline(plan, profile, X)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


@pytest.mark.parametrize("model", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_classifier_models(binary_data, model):
    profile = generate_profile(binary_data, "target")
    X = binary_data.drop(columns=["target"])
    y = binary_data["target"]
    pipe = build_pipeline(_make_plan(model=model), profile, X)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


@pytest.mark.parametrize("model", ["linear_regression", "ridge", "random_forest", "gradient_boosting"])
def test_regressor_models(regression_data, model):
    profile = generate_profile(regression_data, "price")
    X = regression_data.drop(columns=["price"])
    y = regression_data["price"]
    pipe = build_pipeline(_make_plan(model=model, encoding="onehot"), profile, X)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


def test_class_weight_strategy(binary_data):
    """class_weight='balanced' should be injected into the model when requested."""
    profile = generate_profile(binary_data, "target")
    X = binary_data.drop(columns=["target"])
    y = binary_data["target"]
    plan = _make_plan(imbalance_strategy="class_weight")
    pipe = build_pipeline(plan, profile, X)
    pipe.fit(X, y)
    assert pipe.named_steps["model"].class_weight == "balanced"


def test_missing_values_handled(binary_data):
    """Pipeline must handle NaN values in both numeric and categorical columns."""
    df = binary_data.copy()
    df.loc[[0, 5, 10], "age"] = np.nan
    df.loc[[2, 7], "city"] = np.nan
    profile = generate_profile(df, "target")
    X = df.drop(columns=["target"])
    y = df["target"]
    pipe = build_pipeline(_make_plan(), profile, X)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_model_raises():
    df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
    profile = generate_profile(df, "target")
    X = df.drop(columns=["target"])
    plan = _make_plan(model="nonexistent_model")
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline(plan, profile, X)


def test_unknown_imputation_raises():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
    profile = generate_profile(df, "target")
    X = df.drop(columns=["target"])
    plan = _make_plan(imputation="magic")
    with pytest.raises(ValueError, match="Unknown imputation"):
        build_pipeline(plan, profile, X)
