"""
planner.py — Planner Agent (Step 3 of the agentic loop).

The only agent that calls an external LLM (Claude via the Anthropic API).
Receives the dataset profile, detected issues, and the full history of previous
attempts, then proposes a fresh set of ActionPlan candidates for the Executor.

Rather than searching the configuration space blindly, the Planner reasons
about what has been tried, what failed, and why — then proposes strategies
that are diverse, issue-aware, and informed by prior evaluation results.

Prompt caching strategy
-----------------------
The system prompt is long but fully static → cached with cache_control.
The profile + issues block is fixed within a single run → also cached.
The history block grows each iteration but is short → not cached.
This minimises token cost across multi-iteration runs on the same dataset.

Public API
----------
propose_action_plans(profile, issues, history, memory, n_plans, model, api_key)
    -> (List[ActionPlan], reasoning: str)
"""

import json
import os
import uuid
from typing import List, Optional, Tuple

import anthropic

from src.models.schemas import (
    ActionPlan,
    AttemptRecord,
    ClarificationQuestion,
    ClusterResult,
    DataProfile,
    Issue,
    TargetType,
)

# ---------------------------------------------------------------------------
# System prompt — cached; describes role, constraints, and JSON schema
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert AI assistant for tabular sports analytics, working on behalf \
of coaches, performance analysts, scouts, recruitment staff, and federation analysts. \
These users have deep domain knowledge of sport but limited machine learning expertise. \
Your decisions must therefore be technically sound, practically grounded in sports \
data realities, and accompanied by plain-language explanations that a non-technical \
user can act on.

Your task: given a sports dataset profile, a list of detected data quality issues, \
the history of previously attempted pipeline configurations with their cross-validated \
evaluation scores, any answers provided by the user to clarification questions, and \
any exploratory cluster patterns discovered in the data, propose EXACTLY {n_plans} \
new and distinct ActionPlan configurations for the Executor Agent to build and evaluate.

━━━ SPORTS DATA CONTEXT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sports datasets typically contain a mix of match statistics (possession, shots, \
expected goals, pass accuracy), player profiles (age, position, physical attributes, \
market value), event-level data (pass maps, shot locations, sprint counts), workload \
and wellness indicators (training load, RPE, heart rate, sleep quality), injury and \
recovery records (injury type, return-to-play duration, re-injury history), and \
scouting or recruitment data (player ratings, contract details, transfer history).

Common data quality challenges specific to sports datasets:

  DATA LEAKAGE: post-event statistics (final score, total possession, total shots) \
  used as features when the target is the match outcome. If the user has flagged a \
  leakage-candidate column and confirmed it is a post-event measurement, treat it \
  as excluded from the model in your plans.

  CLASS IMBALANCE: injury prediction, red-card prediction, and similar rare-event \
  tasks often have very few positive cases (2-5% of observations). In these tasks, \
  standard accuracy is misleading — prefer SMOTE or class_weight and prioritise \
  recall or F1 over raw accuracy.

  NOISY CATEGORIES: player names with diacritics rendered differently across data \
  providers (Müller / Muller / Mueller), team name variants (Man Utd / Manchester \
  United / MUFC), and competition-specific codes that carry no model-useful signal. \
  If the user has confirmed a column is an identifier, exclude it by recommending \
  ordinal encoding with a note, or treat it as a high-cardinality issue.

  TEMPORAL LEAKAGE: using cumulative season statistics to predict results within \
  the same season without proper time-based splitting can inflate performance estimates. \
  Prefer robust scaling and consider whether temporal ordering matters.

  HETEROGENEOUS UNITS: speed recorded in m/s vs km/h, distance in metres vs yards, \
  different GPS tracking systems. Standard scaling is usually the safest choice \
  when unit consistency across sources is uncertain.

━━━ DECISION GUIDANCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each ActionPlan, follow this reasoning order:

  1. ADDRESS HIGH-SEVERITY ISSUES FIRST.
       - High missingness (> 30%) → prefer "knn" or "iterative" imputation.
       - Severe class imbalance (< 5% minority) → "smote" or "class_weight".
       - Many outliers → "winsorize" + "robust" scaling.
       - Confirmed leakage column → note in reasoning but you cannot drop columns;
         instead, prefer models robust to noisy/redundant features (e.g. lightgbm
         with regularisation) and flag the concern in your plain_language_explanation.

  2. MATCH THE MODEL TO THE DATA STRUCTURE.
       Tree-based models (random_forest, xgboost, lightgbm) handle mixed feature \
       types, interactions, and missing data well in sports contexts. Logistic \
       regression and ridge are better starting points when interpretability is \
       the primary goal for a coach audience. Gradient boosting is often the \
       strongest single model on structured sports data.

  3. ACCOUNT FOR CLUSTER STRUCTURE.
       If cluster patterns are provided, consider whether the discovered groupings \
       suggest that a single global model may struggle (e.g. very distinct player \
       archetypes may respond differently to the same features). In that case, prefer \
       models that handle non-linear interactions (gradient_boosting, xgboost).

  4. USE MEMORY AS A WARM START, NOT A FIXED RECIPE.
       If cross-run memory contains configurations from structurally similar past \
       datasets, use at least one as a starting point but adapt it to the specific \
       issues detected in this dataset.

  5. NEVER REPEAT A TRIED COMBINATION.
       Do not propose any (model, imputation, encoding, scaling) combination that \
       already appears in the attempt history.

  6. VARY THE MODEL FAMILY ACROSS PLANS.
       Propose diverse candidates so the Evaluator can compare different approaches.

━━━ AVAILABLE OPTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

imputation        : "median" | "mean" | "knn" | "iterative"
outlier_handling  : "winsorize" | "none"
encoding          : "onehot" | "target" | "ordinal"
scaling           : "standard" | "robust" | "minmax" | "none"
model
  classification  : "logistic_regression" | "random_forest" | "gradient_boosting"
                    | "xgboost" | "lightgbm"
  regression      : "linear_regression" | "ridge" | "random_forest"
                    | "gradient_boosting" | "xgboost" | "lightgbm"
imbalance_strategy: "class_weight" | "smote" | "none"
model_params      : optional dict of sklearn constructor kwargs,
                    e.g. {{"n_estimators": 200, "max_depth": 5}}

━━━ RESPONSE FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output ONLY valid JSON — no markdown, no prose outside the JSON block.

{{
  "plans": [
    {{
      "plan_id": "plan_<uuid>",
      "imputation": "...",
      "outlier_handling": "...",
      "encoding": "...",
      "scaling": "...",
      "model": "...",
      "imbalance_strategy": "...",
      "model_params": {{}},
      "plain_language_explanation": "One sentence explaining this plan to a non-technical \
sports analyst — e.g. 'This approach handles the extreme sprint values by clipping them \
and uses a gradient boosting model, which tends to perform well on mixed player statistics.'"
    }}
  ],
  "reasoning": "Two to three sentences explaining the overall strategy: which high-severity \
issues are being prioritised in this round, what is being varied across the plans, and why \
these particular model families were chosen for this type of sports data."
}}
"""


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def _format_profile(profile: DataProfile) -> str:
    """
    Render a DataProfile as a compact, human-readable text block for the LLM prompt.
    Keeps token count low while preserving all decision-relevant information.
    """
    lines = [
        "═══ DATASET PROFILE ═══════════════════════════════════════════════",
        f"Rows: {profile.n_rows:,}  |  Cols: {profile.n_cols}  |  Duplicates: {profile.n_duplicates}",
        f"Target: '{profile.target_column}'  |  Task: {profile.target_type.value}",
    ]

    if profile.class_distribution:
        dist = "  ".join(f"{k}={v:.1%}" for k, v in profile.class_distribution.items())
        lines.append(f"Class distribution: {dist}")

    lines.append("")
    lines.append("COLUMNS")
    lines.append("-------")

    for col_name, col in profile.columns.items():
        if col.dtype == "numeric":
            lines.append(
                f"  {col_name:<20} [numeric]    missing={col.missing_rate:.1%}"
                f"  mean={col.mean:.2g}  std={col.std:.2g}"
                f"  range=[{col.min:.2g}, {col.max:.2g}]"
                f"  q1={col.q1:.2g}  q3={col.q3:.2g}"
            )
        elif col.dtype == "categorical":
            top = ", ".join(f"{k}={v}" for k, v in list((col.top_categories or {}).items())[:5])
            lines.append(
                f"  {col_name:<20} [categor]    missing={col.missing_rate:.1%}"
                f"  unique={col.n_unique}  top: {top}"
            )
        else:
            lines.append(
                f"  {col_name:<20} [{col.dtype:<8}]  missing={col.missing_rate:.1%}"
                f"  unique={col.n_unique}"
            )

    return "\n".join(lines)


def _format_issues(issues: List[Issue]) -> str:
    """Render detected issues as a severity-sorted bullet list."""
    if not issues:
        return "═══ DETECTED ISSUES ════════════════════════════════════════════════\n  (none)"

    lines = ["═══ DETECTED ISSUES (sorted HIGH → LOW) ═══════════════════════════"]
    for issue in issues:
        col = f"  col='{issue.affected_column}'" if issue.affected_column else ""
        lines.append(f"  [{issue.severity.value.upper():<6}] {issue.issue_type.value:<22}{col}  {issue.description}")
    return "\n".join(lines)


def _format_history(history: List[AttemptRecord]) -> str:
    """
    Render the attempt history as a compact table.
    This is the most important context for the Planner's refinement step —
    it shows exactly which configurations were tried and how well they scored.
    """
    if not history:
        return "═══ ATTEMPT HISTORY ════════════════════════════════════════════════\n  (first iteration — no prior attempts)"

    lines = ["═══ ATTEMPT HISTORY ════════════════════════════════════════════════"]
    lines.append(f"  {'Iter':<5} {'Score':<7} {'CV_std':<8} {'Metrics':<30} {'Config (imp/out/enc/scl/model/imbal)'}")
    lines.append("  " + "-" * 100)

    for rec in history:
        p = rec.plan
        r = rec.result
        metrics_str = "  ".join(f"{k}={v:.3f}" for k, v in r.metric_values.items())
        config = f"{p.imputation}/{p.outlier_handling}/{p.encoding}/{p.scaling}/{p.model}/{p.imbalance_strategy}"
        lines.append(
            f"  {rec.iteration:<5} {r.score:<7.4f} {r.cv_std:<8.4f} {metrics_str:<30} {config}"
        )

    return "\n".join(lines)


def _format_memory(memory: List[ActionPlan]) -> str:
    """
    Render cross-run memory entries (successful configs from similar past datasets).
    Included only when memory is non-empty; signals the Planner to warm-start.
    """
    if not memory:
        return ""

    lines = ["═══ CROSS-RUN MEMORY (similar past datasets) ══════════════════════"]
    lines.append("  Use at least one of these as a warm-start candidate:")
    for i, plan in enumerate(memory, 1):
        config = (
            f"{plan.imputation}/{plan.outlier_handling}/{plan.encoding}"
            f"/{plan.scaling}/{plan.model}/{plan.imbalance_strategy}"
        )
        lines.append(f"  [{i}] {config}  params={plan.model_params}")
    return "\n".join(lines)


def _format_clusters(clusters: Optional[ClusterResult]) -> str:
    """
    Render exploratory cluster patterns discovered by the Profiler Agent.
    Provides the Planner with structural context about natural groupings in the data
    so it can select models and preprocessing strategies accordingly.
    """
    if clusters is None or not clusters.cluster_summaries:
        return ""

    lines = ["═══ EXPLORATORY CLUSTER PATTERNS ══════════════════════════════════"]
    lines.append(
        f"  {clusters.n_clusters} natural groups found via k-means "
        f"(silhouette={clusters.silhouette_score:.3f}, "
        f"Davies-Bouldin={clusters.davies_bouldin_index:.3f})"
    )
    lines.append(f"  Features used: {', '.join(clusters.numeric_columns[:8])}"
                 + (" ..." if len(clusters.numeric_columns) > 8 else ""))
    lines.append("")
    for cid, summary in clusters.cluster_summaries.items():
        lines.append(f"  {summary}")
    lines.append(
        "\n  NOTE: if clusters are very distinct, consider models that handle "
        "non-linear interactions well (gradient_boosting, xgboost, lightgbm)."
    )
    return "\n".join(lines)


def _format_sports_context(profile: DataProfile) -> str:
    """
    Render the sports vocabulary breakdown attached to a DataProfile.

    Shows which columns were detected as post-match stats, identity columns,
    playing-time indicators, etc., so the Planner can factor in domain-specific
    risks (leakage, identifier columns) even before correlation-based signals appear.
    """
    ctx = getattr(profile, "sports_context", None)
    if ctx is None or not ctx.is_sports:
        return ""

    lines = ["═══ SPORTS VOCABULARY DETECTION ═══════════════════════════════════"]
    lines.append(
        f"  domain={ctx.detected_domain}  confidence={ctx.confidence:.0%}"
    )
    if ctx.matched_terms:
        lines.append(f"  Matched sports terms: {', '.join(ctx.matched_terms)}")

    if ctx.post_match_cols:
        lines.append(
            f"  POST-MATCH STATS (potential leakage if used to predict same event): "
            f"{', '.join(ctx.post_match_cols)}"
        )
    if ctx.identity_cols:
        lines.append(
            f"  IDENTITY COLUMNS (should not be encoded as features): "
            f"{', '.join(ctx.identity_cols)}"
        )
    if ctx.playing_time_cols:
        lines.append(
            f"  PLAYING-TIME INDICATORS (missing may mean 'did not play'): "
            f"{', '.join(ctx.playing_time_cols)}"
        )
    if ctx.injury_cols:
        lines.append(
            f"  INJURY / WELLNESS COLUMNS (missing may mean 'no injury'): "
            f"{', '.join(ctx.injury_cols)}"
        )
    if ctx.physical_cols:
        lines.append(
            f"  PHYSICAL ATTRIBUTES (safe model features): "
            f"{', '.join(ctx.physical_cols)}"
        )
    if ctx.workload_cols:
        lines.append(
            f"  WORKLOAD METRICS: {', '.join(ctx.workload_cols)}"
        )

    return "\n".join(lines)


def _format_clarifications(questions: List[ClarificationQuestion]) -> str:
    """
    Render user answers to clarification questions.
    Only questions with non-None answers are included — unanswered questions
    are ignored so the Planner is not confused by incomplete information.
    """
    answered = [q for q in questions if q.answer is not None]
    if not answered:
        return ""

    lines = ["═══ USER CLARIFICATION ANSWERS ════════════════════════════════════"]
    for q in answered:
        col = f"  col='{q.affected_column}'" if q.affected_column else ""
        lines.append(f"  [{q.issue_type.upper()}{col}]")
        lines.append(f"    Q: {q.question}")
        lines.append(f"    A: {q.answer}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_plans(raw_json: str, n_plans: int) -> Tuple[List[ActionPlan], str]:
    """
    Parse Claude's JSON response into a list of ActionPlan objects.

    Assigns a fresh UUID to each plan_id to guarantee uniqueness even if
    Claude reuses the same plan_id string across iterations.
    The optional plain_language_explanation field is stored in model_params
    under the key "__explanation" so it travels with the plan for the user report.
    Returns (plans, reasoning_text).
    Raises ValueError on malformed JSON or missing required fields.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner returned invalid JSON: {e}\n\nRaw response:\n{raw_json}") from e

    raw_plans = data.get("plans", [])
    reasoning = data.get("reasoning", "")

    required_fields = {"imputation", "outlier_handling", "encoding", "scaling", "model", "imbalance_strategy"}

    plans = []
    for i, raw in enumerate(raw_plans[:n_plans]):
        missing = required_fields - set(raw.keys())
        if missing:
            raise ValueError(f"Plan {i} is missing required fields: {missing}")
        params = dict(raw.get("model_params", {}))
        # Store the plain-language explanation alongside the plan (excluded from sklearn params).
        explanation = raw.get("plain_language_explanation", "")
        if explanation:
            params["__explanation"] = explanation
        plans.append(ActionPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            imputation=raw["imputation"],
            outlier_handling=raw["outlier_handling"],
            encoding=raw["encoding"],
            scaling=raw["scaling"],
            model=raw["model"],
            imbalance_strategy=raw["imbalance_strategy"],
            model_params=params,
        ))

    if not plans:
        raise ValueError("Planner returned zero ActionPlans.")

    return plans, reasoning


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def propose_action_plans(
    profile: DataProfile,
    issues: List[Issue],
    history: List[AttemptRecord],
    memory: Optional[List[ActionPlan]] = None,
    clarification_questions: Optional[List[ClarificationQuestion]] = None,
    n_plans: int = 3,
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
    max_retries: int = 2,
) -> Tuple[List[ActionPlan], str]:
    """
    Call the Claude API and return a list of candidate ActionPlans.

    The prompt is assembled from five blocks with different cache lifetimes:
      1. System prompt               — static, cached (longest TTL)
      2. Profile + issues block      — fixed within a run, cached
      3. Cluster patterns block      — fixed within a run, cached with profile
      4. History + memory block      — grows each iteration, not cached
      5. Clarification answers block — appended if the user answered questions

    Parameters
    ----------
    profile                 : DataProfile from the Profiler Agent (may include clusters).
    issues                  : Issue list from the Issue Detector Agent (severity-sorted).
    history                 : All AttemptRecords from previous iterations of the current run.
    memory                  : ActionPlans retrieved from ChromaDB for similar past datasets.
    clarification_questions : Questions whose answers should be passed to the Planner.
    n_plans                 : Number of candidate plans to request (default 3).
    model                   : Claude model ID to use.
    api_key                 : Anthropic API key; falls back to ANTHROPIC_API_KEY env var.
    max_retries             : Number of times to retry on JSON parse failure before raising.

    Returns
    -------
    (plans, reasoning) — list of ActionPlans and Claude's explanation string.
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    # Fill the n_plans placeholder in the system prompt
    system_prompt = _SYSTEM_PROMPT.format(n_plans=n_plans)

    # Build the static-within-run profile+issues+clusters+sports block (cacheable)
    profile_parts = [_format_profile(profile), _format_issues(issues)]
    sports_text = _format_sports_context(profile)
    if sports_text:
        profile_parts.append(sports_text)
    cluster_text = _format_clusters(profile.clusters)
    if cluster_text:
        profile_parts.append(cluster_text)
    profile_block = "\n\n".join(profile_parts)

    # Build the dynamic history block (changes every iteration — not cached)
    history_block = _format_history(history)

    # Optionally append cross-run memory
    memory_text = _format_memory(memory or [])
    if memory_text:
        history_block = history_block + "\n\n" + memory_text

    # Optionally append user clarification answers
    clarif_text = _format_clarifications(clarification_questions or [])
    if clarif_text:
        history_block = history_block + "\n\n" + clarif_text

    history_block += f"\n\nPlease propose {n_plans} new ActionPlans."

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    # Cache the system prompt — it never changes across all API calls
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": profile_block,
                            # Cache the profile+issues block — fixed within a single run
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": history_block,
                            # History is not cached — it changes on every iteration
                        },
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences if Claude wraps the JSON (defensive)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            plans, reasoning = _parse_plans(raw, n_plans)
            return plans, reasoning
        except ValueError as e:
            last_error = e
            # On parse failure, retry — Claude occasionally wraps output unexpectedly

    raise RuntimeError(
        f"Planner failed to return valid ActionPlans after {max_retries + 1} attempts.\n"
        f"Last error: {last_error}"
    )
