"""
schemas.py — shared data structures used across all agents.

DataProfile  : output of the Profiler Agent, describes a dataset's structure and statistics.
Issue        : output of the Issue Detector Agent, describes a single detected problem.
TargetType   : inferred task type (binary classification / multiclass / regression).
IssueSeverity: HIGH / MEDIUM / LOW, used by the Planner to prioritise fixes.
IssueType    : category of a detected problem.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TargetType(str, Enum):
    """Inferred ML task type based on the target column's unique value count and dtype."""
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class IssueSeverity(str, Enum):
    """How urgently the Planner should address a detected issue."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IssueType(str, Enum):
    """Categories of data quality problems the Issue Detector can flag."""
    HIGH_MISSINGNESS = "high_missingness"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    NOISY_CATEGORIES = "noisy_categories"   # very high cardinality — likely ID or free-text
    LEAKAGE_CANDIDATE = "leakage_candidate" # feature correlates suspiciously well with target
    CLASS_IMBALANCE = "class_imbalance"


@dataclass
class ColumnProfile:
    """
    Statistics for a single DataFrame column.

    Numeric columns populate: mean, std, min, max, q1, q3.
    Categorical columns populate: top_categories (top-10 value counts).
    Boolean / datetime columns store only the shared fields.
    """
    name: str
    dtype: str          # "numeric" | "categorical" | "datetime" | "boolean"
    missing_rate: float # fraction of NaN values in [0, 1]
    n_unique: int
    sample_values: List[Any]

    # Numeric statistics (None for non-numeric columns)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q1: Optional[float] = None  # 25th percentile — used for IQR outlier detection
    q3: Optional[float] = None  # 75th percentile — used for IQR outlier detection

    # Categorical statistics (None for non-categorical columns)
    top_categories: Optional[Dict[str, int]] = None  # {value: count}, top 10


@dataclass
class DataProfile:
    """
    Full structural description of a dataset, produced by the Profiler Agent.

    Passed to the Issue Detector and then (together with detected issues) to
    the Planner Agent as the primary context for decision-making.
    """
    n_rows: int
    n_cols: int
    target_column: str
    target_type: TargetType
    class_distribution: Optional[Dict[str, float]]  # {class_label: frequency}; None for regression
    columns: Dict[str, ColumnProfile]               # keyed by column name
    n_duplicates: int
    fingerprint: List[float] = field(default_factory=list)  # filled by dataset_fingerprint()


@dataclass
class Issue:
    """
    A single data quality problem detected by the Issue Detector Agent.

    The Planner Agent receives the full list of issues, sorted by severity,
    and uses them to decide which preprocessing steps to include in an ActionPlan.
    """
    issue_type: IssueType
    severity: IssueSeverity
    affected_column: Optional[str]  # None for dataset-level issues (e.g. duplicates)
    description: str                # human-readable summary for the LLM prompt
    evidence: Dict[str, Any]        # raw numbers supporting the finding
