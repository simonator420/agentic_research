"""
schemas.py — shared data structures used across all agents.

DataProfile          : output of the Profiler Agent, describes a dataset's structure and statistics.
Issue                : output of the Issue Detector Agent, describes a single detected problem.
ClarificationQuestion: plain-language question the Issue Detector surfaces for the user
                       when a data quality issue requires domain knowledge to resolve.
ClusterResult        : output of the Profiler's exploratory clustering step.
ActionPlan           : one candidate pipeline configuration proposed by the Planner Agent.
EvaluationResult     : output of the Evaluator Agent for one ActionPlan.
AttemptRecord        : pairs an ActionPlan with its EvaluationResult for a single iteration;
                       the full list of AttemptRecords is the within-run memory passed back
                       to the Planner at each iteration so it never repeats a tried strategy.
RunResult            : final output of the Orchestrator — best pipeline, plan, full history,
                       clarification questions asked, and the plain-language user report.
BaselineResult       : output of a baseline method (rule-based or search-based), in the same
                       format as RunResult so results can be compared directly.
LLMCallRecord        : token usage and cost for a single LLM API call.
TaskSpecification    : structured output of the Goal Interpreter — mode, task type, target column.
ExploratoryResult    : output of the exploratory pipeline (no ML training).
TargetType           : inferred task type (binary classification / multiclass / regression).
IssueSeverity        : HIGH / MEDIUM / LOW, used by the Planner to prioritise fixes.
IssueType            : category of a detected problem.
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
class LLMCallRecord:
    """
    Records a single LLM API call — for token usage and cost tracking.

    Used to quantify the LLM overhead of the agentic system relative to baselines
    and to report research metrics (n_llm_calls, input_tokens, estimated_cost).

    Attributes
    ----------
    purpose              : human-readable label, e.g. "plan_proposal" | "goal_interpretation".
    model                : Claude model ID used for the call.
    input_tokens         : non-cached input tokens billed at standard input rate.
    output_tokens        : output tokens.
    cache_read_tokens    : tokens served from prompt cache (billed at cache read rate).
    cache_write_tokens   : tokens written to prompt cache (billed at cache write rate).
    estimated_cost_usd   : total estimated cost at standard Anthropic list pricing.
    """
    purpose: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    estimated_cost_usd: float


@dataclass
class TaskSpecification:
    """
    Structured interpretation of a user's natural-language goal.

    Output of the Goal Interpreter Agent.  Determines whether the user wants a
    predictive model (requires a target column) or an exploratory analysis (no
    target column — profiling, clustering, and pattern discovery only).

    Attributes
    ----------
    mode             : "predictive" | "exploratory"
    task_type        : "classification" | "regression" | "exploratory"
    goal             : the original goal_text string.
    target_column    : best-matching column to predict; None when mode=="exploratory".
    requested_outputs: list drawn from ["model", "report", "visualisations", "clusters"].
    confidence       : "high" | "medium" | "low" — how clearly the goal maps to a column.
    explanation      : one plain-language sentence for a non-technical user.
    alternatives     : other plausible target columns the user might prefer.
    """
    mode: str
    task_type: str
    goal: str
    target_column: Optional[str]
    requested_outputs: List[str]
    confidence: str
    explanation: str
    alternatives: List[str] = field(default_factory=list)


@dataclass
class ClarificationQuestion:
    """
    A plain-language question surfaced by the Issue Detector when a data quality issue
    cannot be resolved without domain knowledge (e.g. whether a column represents a
    pre-event or post-event measurement in a sports context).

    The system batches all questions and interrupts the user at most once per run.
    The user's answer is stored in the `answer` field and passed to the Planner Agent
    so it can make an informed preprocessing decision.
    """
    question_id: str
    question: str                # plain-language question for the non-technical user
    affected_column: Optional[str]
    issue_type: str              # IssueType.value string
    answer: Optional[str] = None # filled in after the user responds; None = not yet asked


@dataclass
class ClusterResult:
    """
    Output of the Profiler Agent's exploratory clustering step.

    Clustering is run on numeric features only, using k-means with the number of
    clusters selected by silhouette score. Each cluster is summarised as a
    plain-language description (e.g. "high minutes_played, low injury_count")
    so that non-technical sports users can interpret the natural groupings.

    Attributes
    ----------
    n_clusters         : number of clusters selected by silhouette optimisation.
    labels             : cluster assignment for each row in valid_indices.
    valid_indices      : original DataFrame indices that were clustered
                         (rows with NaN in any numeric column are excluded).
    numeric_columns    : columns used for clustering.
    silhouette_score   : average silhouette coefficient (higher = better separation).
    davies_bouldin_index: Davies–Bouldin index (lower = better separation).
    cluster_summaries  : {cluster_id: plain-language description}.
    method             : clustering algorithm used ("kmeans" | "hierarchical" | "dbscan").
    """
    n_clusters: int
    labels: List[int]
    valid_indices: List[int]
    numeric_columns: List[str]
    silhouette_score: float
    davies_bouldin_index: float
    cluster_summaries: Dict[int, str]
    method: str


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
    target_column: Optional[str]                    # None in exploratory mode (no prediction target)
    target_type: TargetType
    class_distribution: Optional[Dict[str, float]]  # {class_label: frequency}; None for regression
    columns: Dict[str, ColumnProfile]               # keyed by column name
    n_duplicates: int
    fingerprint: List[float] = field(default_factory=list)  # filled by dataset_fingerprint()
    clusters: Optional[ClusterResult] = None        # filled by discover_clusters() + summarize_patterns()
    sports_context: Optional[Any] = None            # SportsContext from sports_vocabulary.detect_sports_context()


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


@dataclass
class ExploratoryResult:
    """
    Output of run_exploratory_pipeline() — exploratory analysis without ML training.

    Contains profiling results, cluster patterns, data quality issues, and a plain-language
    report.  There is no trained pipeline — the user asked for exploration, not prediction.

    Attributes
    ----------
    run_id                  : unique identifier for this run.
    profile                 : full DataProfile (all columns, target_column=None).
    issues                  : data quality issues detected by the Issue Detector.
    clarification_questions : questions surfaced during analysis (with any user answers).
    user_report             : plain-language markdown report for non-technical users.
    visualisation_paths     : paths to generated PNG files.
    llm_calls               : token usage for any LLM calls made during the run.
    runtime_secs            : total wall-clock time.
    """
    run_id: str
    profile: DataProfile
    issues: List["Issue"]
    clarification_questions: List[ClarificationQuestion]
    user_report: str
    visualisation_paths: List[str] = field(default_factory=list)
    llm_calls: List[LLMCallRecord] = field(default_factory=list)
    runtime_secs: float = 0.0


@dataclass
class ActionPlan:
    """
    One candidate pipeline configuration returned by the Planner Agent (via Claude API).

    The Planner produces a list of ActionPlans per iteration; the Executor builds
    a scikit-learn Pipeline from each one, and the Evaluator picks the best.

    Each field maps directly to a specific sklearn component — no further LLM calls
    are needed after the Planner returns this structure.

    Valid values per field
    ----------------------
    imputation        : "median" | "mean" | "knn" | "iterative"
    outlier_handling  : "winsorize" | "none"
    encoding          : "onehot" | "target" | "ordinal"
    scaling           : "standard" | "robust" | "minmax" | "none"
    model             : "logistic_regression" | "random_forest" | "gradient_boosting"
                        | "xgboost" | "lightgbm" | "linear_regression" | "ridge"
    imbalance_strategy: "class_weight" | "smote" | "none"
    model_params      : optional dict passed directly to the sklearn estimator constructor
    """
    plan_id: str
    imputation: str
    outlier_handling: str
    encoding: str
    scaling: str
    model: str
    imbalance_strategy: str
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Cross-validated evaluation result for one ActionPlan, produced by the Evaluator Agent.

    score        : composite ranking score (higher = better); used to select the best plan
                   and as the primary convergence signal for the feedback loop.
    metric_values: task-appropriate metrics — F1/AUC for classification, RMSE/R² for regression.
    cv_std       : standard deviation of the primary metric across CV folds; lower = more stable
                   pipeline (a direct indicator of generalisation consistency).
    runtime_secs : wall-clock time for the full CV evaluation; used to compare search efficiency
                   across pipeline configurations.
    """
    plan_id: str
    score: float
    metric_values: Dict[str, float]  # e.g. {"f1": 0.85, "auc": 0.91} or {"rmse": 12.3, "r2": 0.88}
    cv_std: float
    runtime_secs: float


@dataclass
class AttemptRecord:
    """
    One entry in the within-run memory: an ActionPlan paired with its EvaluationResult.

    The Orchestrator accumulates these across iterations and passes the full list to
    propose_action_plans() so the Planner can see which strategies were already tried
    and what scores they achieved — enabling informed refinement rather than random search.
    """
    iteration: int
    plan: ActionPlan
    result: EvaluationResult


@dataclass
class RunResult:
    """
    Final output of the Orchestrator after a completed run.

    best_pipeline : sklearn Pipeline fitted on the full training set, ready for prediction.
    best_plan     : the ActionPlan that produced the best cross-validated score.
    best_result   : the corresponding EvaluationResult (score, metrics, cv_std).
    history       : all AttemptRecords from every iteration, in chronological order.
    n_iterations  : number of Planner→Evaluate rounds actually executed.
    converged     : True if the run stopped because the score threshold was reached;
                    False if it stopped because max_rounds was exhausted or a plateau
                    was detected.
    run_id        : unique identifier for this run, used to query the SQLite store.
    """
    best_pipeline: Any          # sklearn Pipeline — typed as Any to avoid importing sklearn here
    best_plan: ActionPlan
    best_result: EvaluationResult
    history: List[AttemptRecord]
    n_iterations: int
    converged: bool
    run_id: str
    clarification_questions: List[ClarificationQuestion] = field(default_factory=list)
    user_report: str = ""       # plain-language markdown report for non-technical sports users
    llm_calls: List[LLMCallRecord] = field(default_factory=list)  # token usage across all LLM calls


@dataclass
class BaselineResult:
    """
    Output of a baseline method, structured to match RunResult for direct comparison.

    method               : human-readable name, e.g. "rule_based" or "search_based".
    best_pipeline        : sklearn Pipeline fitted on the full training set.
    best_plan            : the ActionPlan configuration that was selected or found best.
    score                : composite evaluation score (same formula as the agentic system).
    metric_values        : task-appropriate metrics (F1/AUC for classification, RMSE/R² for regression).
    cv_std               : standard deviation of the primary metric across CV folds.
    runtime_secs         : total wall-clock time for the baseline run.
    n_configs_evaluated  : 1 for rule-based (single fixed pipeline); n_trials for search-based.
    """
    method: str
    best_pipeline: Any
    best_plan: ActionPlan
    score: float
    metric_values: Dict[str, float]
    cv_std: float
    runtime_secs: float
    n_configs_evaluated: int
