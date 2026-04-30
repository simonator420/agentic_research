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
    DataProfile,
    Issue,
    TargetType,
)

# ---------------------------------------------------------------------------
# System prompt — cached; describes role, constraints, and JSON schema
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert ML pipeline optimiser for tabular datasets.

Your task: given a dataset profile, a list of detected data quality issues, \
and a history of previously attempted pipeline configurations with their \
cross-validated evaluation scores, propose EXACTLY {n_plans} new and distinct \
ActionPlan configurations for the Executor Agent to build and evaluate.

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

━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Address detected HIGH-severity issues first (e.g. high missingness → prefer
   "knn" or "iterative" imputation; class imbalance → "smote" or "class_weight";
   many outliers → "winsorize" + "robust" scaling).
2. Never repeat a (model, imputation, encoding, scaling) combination that already
   appears in the attempt history.
3. Propose diverse plans — vary the model family across candidates.
4. If memory contains successful configurations from similar past datasets,
   use them as a warm start for at least one plan.
5. Output ONLY valid JSON — no markdown, no prose outside the JSON block.

━━━ RESPONSE FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
      "model_params": {{}}
    }}
  ],
  "reasoning": "One short paragraph explaining the rationale behind each plan."
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


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_plans(raw_json: str, n_plans: int) -> Tuple[List[ActionPlan], str]:
    """
    Parse Claude's JSON response into a list of ActionPlan objects.

    Assigns a fresh UUID to each plan_id to guarantee uniqueness even if
    Claude reuses the same plan_id string across iterations.
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
        plans.append(ActionPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            imputation=raw["imputation"],
            outlier_handling=raw["outlier_handling"],
            encoding=raw["encoding"],
            scaling=raw["scaling"],
            model=raw["model"],
            imbalance_strategy=raw["imbalance_strategy"],
            model_params=raw.get("model_params", {}),
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
    n_plans: int = 3,
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
    max_retries: int = 2,
) -> Tuple[List[ActionPlan], str]:
    """
    Call the Claude API and return a list of candidate ActionPlans.

    The prompt is assembled from four blocks with different cache lifetimes:
      1. System prompt          — static, cached (longest TTL)
      2. Profile + issues block — fixed within a run, cached
      3. History block          — grows each iteration, not cached
      4. Memory block           — present only when cross-run memory is available

    Parameters
    ----------
    profile     : DataProfile from the Profiler Agent.
    issues      : Issue list from the Issue Detector Agent (severity-sorted).
    history     : All AttemptRecords from previous iterations of the current run.
    memory      : ActionPlans retrieved from ChromaDB for similar past datasets (optional).
    n_plans     : Number of candidate plans to request (default 3).
    model       : Claude model ID to use.
    api_key     : Anthropic API key; falls back to ANTHROPIC_API_KEY env var if None.
    max_retries : Number of times to retry on JSON parse failure before raising.

    Returns
    -------
    (plans, reasoning) — list of ActionPlans and Claude's explanation string.
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    # Fill the n_plans placeholder in the system prompt
    system_prompt = _SYSTEM_PROMPT.format(n_plans=n_plans)

    # Build the static-within-run profile+issues block (cacheable)
    profile_block = "\n\n".join([
        _format_profile(profile),
        _format_issues(issues),
    ])

    # Build the dynamic history block (changes every iteration — not cached)
    history_block = _format_history(history)

    # Optionally append cross-run memory
    memory_text = _format_memory(memory or [])
    if memory_text:
        history_block = history_block + "\n\n" + memory_text

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
