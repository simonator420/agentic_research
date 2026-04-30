"""
executor.py — Executor Agent (Step 4 of the agentic loop).

Takes an ActionPlan produced by the Planner Agent and assembles it into a
runnable scikit-learn Pipeline with a ColumnTransformer.

Each string field in ActionPlan maps deterministically to one sklearn component:

  imputation        → SimpleImputer / KNNImputer / IterativeImputer
  outlier_handling  → Winsorizer (custom, IQR-based) or passthrough
  encoding          → OneHotEncoder / TargetEncoder / OrdinalEncoder
  scaling           → StandardScaler / RobustScaler / MinMaxScaler or passthrough
  model             → classifier or regressor chosen by target type
  imbalance_strategy→ class_weight param on model, or SMOTE via imblearn Pipeline

This template-based design is deterministic and requires no additional LLM calls.
The predefined transformation library covers the vast majority of real-world tabular
preprocessing needs while keeping pipeline construction fast and reproducible.

Public API
----------
build_pipeline(plan, profile, X) -> sklearn Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 — must import before IterativeImputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from src.models.schemas import ActionPlan, DataProfile, TargetType


# ---------------------------------------------------------------------------
# Custom transformer
# ---------------------------------------------------------------------------

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Clip feature values to the Tukey IQR fence: [Q1 - factor*IQR, Q3 + factor*IQR].

    Bounds are computed on the training set during fit() so the same limits are
    applied to the test set — this prevents test-time information leakage that
    would occur if bounds were recomputed on the full dataset.

    Parameters
    ----------
    factor : multiplier for the IQR (default 1.5, the standard Tukey fence).
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_, self.upper_)


# ---------------------------------------------------------------------------
# Component factories — each raises ValueError for unknown strategy strings
# so bad Planner output fails loudly rather than silently producing garbage.
# ---------------------------------------------------------------------------

def _get_imputer(strategy: str) -> BaseEstimator:
    """Return a fitted-on-transform imputer for numeric columns."""
    if strategy == "median":
        return SimpleImputer(strategy="median")
    if strategy == "mean":
        return SimpleImputer(strategy="mean")
    if strategy == "knn":
        # n_neighbors=5 is a reasonable default; the Planner can override via model_params
        return KNNImputer(n_neighbors=5)
    if strategy == "iterative":
        # MICE-style multivariate imputation; max_iter capped for runtime predictability
        return IterativeImputer(random_state=42, max_iter=10)
    raise ValueError(f"Unknown imputation strategy: '{strategy}'")


def _get_scaler(scaling: str) -> BaseEstimator:
    """Return a scaler for numeric columns. Only called when scaling != 'none'."""
    if scaling == "standard":
        return StandardScaler()
    if scaling == "robust":
        # RobustScaler uses median and IQR — less sensitive to outliers than StandardScaler
        return RobustScaler()
    if scaling == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaling strategy: '{scaling}'")


def _get_encoder(encoding: str) -> BaseEstimator:
    """
    Return an encoder for categorical columns.

    handle_unknown="ignore" on OneHotEncoder prevents errors when the test set
    contains categories not seen during training (common in real-world data).
    TargetEncoder requires y at fit time; sklearn's Pipeline passes y automatically.
    OrdinalEncoder assigns -1 to unseen categories instead of raising an error.
    """
    if encoding == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if encoding == "target":
        return TargetEncoder()
    if encoding == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"Unknown encoding strategy: '{encoding}'")


def _get_model(plan: ActionPlan, target_type: TargetType) -> BaseEstimator:
    """
    Instantiate the sklearn estimator specified in ActionPlan.model.

    The correct classifier vs regressor variant is selected automatically from
    target_type so the Planner only needs to name the model family.
    class_weight="balanced" is injected for models that support it when
    imbalance_strategy == "class_weight".
    """
    is_regression = target_type == TargetType.REGRESSION
    use_class_weight = plan.imbalance_strategy == "class_weight"
    params = dict(plan.model_params)

    if plan.model == "logistic_regression":
        if use_class_weight:
            params["class_weight"] = "balanced"
        return LogisticRegression(random_state=42, max_iter=1000, **params)

    if plan.model == "random_forest":
        if use_class_weight and not is_regression:
            params["class_weight"] = "balanced"
        cls = RandomForestRegressor if is_regression else RandomForestClassifier
        return cls(random_state=42, n_estimators=100, **params)

    if plan.model == "gradient_boosting":
        cls = GradientBoostingRegressor if is_regression else GradientBoostingClassifier
        return cls(random_state=42, **params)

    if plan.model == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError:
            raise ImportError("xgboost not installed — run: pip install xgboost")
        cls = XGBRegressor if is_regression else XGBClassifier
        # verbosity=0 suppresses xgboost's default console output
        return cls(random_state=42, verbosity=0, **params)

    if plan.model == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError:
            raise ImportError("lightgbm not installed — run: pip install lightgbm")
        if use_class_weight and not is_regression:
            params["class_weight"] = "balanced"
        cls = LGBMRegressor if is_regression else LGBMClassifier
        # verbose=-1 suppresses lightgbm's default console output
        return cls(random_state=42, verbose=-1, **params)

    if plan.model == "linear_regression":
        return LinearRegression(**params)

    if plan.model == "ridge":
        return Ridge(**params)

    raise ValueError(f"Unknown model: '{plan.model}'")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_pipeline(plan: ActionPlan, profile: DataProfile, X: pd.DataFrame) -> Pipeline:
    """
    Assemble a runnable sklearn Pipeline from an ActionPlan.

    Column groups (numeric / categorical) are derived by intersecting the
    DataProfile column types with the columns actually present in X.
    This is necessary because X has the target column removed, and any
    columns the user dropped beforehand should also be ignored gracefully.

    Datetime and boolean columns are silently dropped via remainder="drop"
    in ColumnTransformer — the Planner is expected to avoid relying on them.

    When imbalance_strategy == "smote", an imblearn Pipeline is returned
    instead of a plain sklearn Pipeline so that SMOTE is applied after
    preprocessing but before the model — the correct position in the fit flow.

    Parameters
    ----------
    plan    : ActionPlan with the chosen strategies and model.
    profile : DataProfile that identifies which columns are numeric / categorical.
    X       : feature matrix (target column already removed by split_data()).

    Returns
    -------
    sklearn.pipeline.Pipeline (or imblearn.pipeline.Pipeline for SMOTE).
    """
    available = set(X.columns)

    numeric_cols = [
        c for c, p in profile.columns.items()
        if p.dtype == "numeric" and c in available
    ]
    categorical_cols = [
        c for c, p in profile.columns.items()
        if p.dtype == "categorical" and c in available
    ]

    # --- Numeric sub-pipeline ---
    numeric_steps = [("imputer", _get_imputer(plan.imputation))]
    if plan.outlier_handling == "winsorize":
        # Winsorizer is inserted after imputation so it never sees NaN values
        numeric_steps.append(("outlier", Winsorizer()))
    if plan.scaling != "none":
        numeric_steps.append(("scaler", _get_scaler(plan.scaling)))

    # --- Categorical sub-pipeline ---
    # most_frequent imputation is always applied before encoding to guarantee
    # no NaN reaches the encoder (encoders generally do not handle NaN).
    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", _get_encoder(plan.encoding)),
    ]

    # Build ColumnTransformer from whichever column groups are present
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline(cat_steps), categorical_cols))

    if not transformers:
        raise ValueError(
            "No numeric or categorical columns found in X — cannot build a pipeline."
        )

    # remainder="drop" silently ignores datetime / boolean / unknown columns
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = _get_model(plan, profile.target_type)

    if plan.imbalance_strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError:
            raise ImportError("imbalanced-learn not installed — run: pip install imbalanced-learn")
        # SMOTE is placed between preprocessor and model so it only sees encoded,
        # imputed data — raw categorical strings would break SMOTE's KNN distance.
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
