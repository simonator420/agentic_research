"""
goal_interpreter.py — maps a natural-language goal to a structured TaskSpecification.

Provides four levels of goal understanding, from fast rule-based to full LLM:

  infer_operating_mode(goal_text)
      Rule-based keyword scan → "predictive" | "exploratory".

  suggest_target_column(goal_text, df)
      Rule-based column-name matching → best candidate column or None.

  parse_user_goal(goal_text, df, api_key, model)
      Full Claude API call → TaskSpecification with confidence and alternatives.

  build_task_specification(goal_text, df, api_key, model)
      Main entry point: attempts LLM interpretation if api_key is available,
      otherwise falls back to rule-based inference.

Public API
----------
build_task_specification(goal_text, df, api_key, model) -> TaskSpecification
"""

import json
import os
import re
from typing import List, Optional

import pandas as pd

from src.models.schemas import LLMCallRecord, TaskSpecification

# ---------------------------------------------------------------------------
# Pricing table (USD per 1 M tokens) — used for cost estimation
# ---------------------------------------------------------------------------
_PRICING: dict = {
    "claude-sonnet-4-6":  {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "claude-opus-4-7":    {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00, "cache_write": 1.00, "cache_read": 0.08},
}
_DEFAULT_PRICING = _PRICING["claude-sonnet-4-6"]


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write: int,
    cache_read: int,
) -> float:
    p = _PRICING.get(model, _DEFAULT_PRICING)
    return (
        input_tokens  * p["input"]       / 1_000_000
        + output_tokens * p["output"]      / 1_000_000
        + cache_write   * p["cache_write"] / 1_000_000
        + cache_read    * p["cache_read"]  / 1_000_000
    )


# ---------------------------------------------------------------------------
# Rule-based helpers (no LLM — fast fallback when no API key)
# ---------------------------------------------------------------------------

_EXPLORATORY_PATTERNS = [
    r"\bexplor",
    r"\bdiscover",
    r"\bfind groups?\b",
    r"\bcluster",
    r"\bsegment",
    r"\bnatural groups?\b",
    r"\barchetypes?\b",
    r"\bprofile\b",
    r"\bpatterns?\b",
    r"\btrends?\b",
    r"\bunderstand\b",
    r"\bvisual",
    r"\bdescribe\b",
    r"\bsummar",
    r"\bwhat kind",
    r"\bwhich type",
    r"\bwhat types?\b",
    r"\banalyse\b",
    r"\banalyze\b",
    r"\bgroup\b",
]

_PREDICTIVE_PATTERNS = [
    r"\bpredict\b",
    r"\bforecast\b",
    r"\bclassif",
    r"\bdetect\b",
    r"\bbuild a model\b",
    r"\btrain\b",
    r"\bestimate\b",
    r"\bat risk\b",
    r"\blikely to\b",
    r"\bprobabilit",
    r"\bchance of\b",
    r"\bwill\s+\w+\s+(injur|hurt|score|win|lose)",
    r"\bidentif",
    r"\bwho (is|are) (most )?likely",
    r"\bwhich players?\b.*\bwill\b",
    r"\bwhich matches?\b.*\bwill\b",
]


def infer_operating_mode(goal_text: str) -> str:
    """
    Rule-based keyword scan to decide predictive vs exploratory mode.

    Returns "exploratory" only when exploratory signals are found AND no
    strong predictive signal is present.  Defaults to "predictive" when
    in doubt — safer fallback for an ML pipeline system.
    """
    text = goal_text.lower()
    has_exploratory = any(re.search(p, text) for p in _EXPLORATORY_PATTERNS)
    has_predictive  = any(re.search(p, text) for p in _PREDICTIVE_PATTERNS)

    if has_exploratory and not has_predictive:
        return "exploratory"
    return "predictive"


def _col_tokens(col: str) -> set:
    """Split a column name into normalised word tokens.

    'player_positions' → {'player', 'position'}  (strips trailing 's')
    'nation_position'  → {'nation', 'position'}
    """
    import difflib  # stdlib — already used elsewhere
    words = re.split(r"[_\-\s]+", col.lower())
    # Light stemming: strip trailing 's' to unify "positions" → "position"
    return {w.rstrip("s") for w in words if len(w) >= 2}


def _score_column(col: str, goal_tokens: set, goal_text: str) -> float:
    """
    Score a column name against the goal on three axes (all 0-1, higher = better):

    1. Token recall   — fraction of goal tokens found in column tokens.
                        'player_positions' vs goal tokens {'player','position'} → 2/2 = 1.0
                        'nation_position'  vs goal tokens {'player','position'} → 1/2 = 0.5
    2. Token precision— fraction of column tokens found in goal tokens (rewards specificity).
    3. Fuzzy string   — difflib ratio between normalised column name and goal text.

    Combined as a weighted sum: 0.6·recall + 0.2·precision + 0.2·fuzzy.
    """
    import difflib
    col_tokens = _col_tokens(col)
    col_norm   = re.sub(r"[_\-]+", " ", col.lower())

    if not goal_tokens or not col_tokens:
        return 0.0

    recall    = len(goal_tokens & col_tokens) / len(goal_tokens)
    precision = len(goal_tokens & col_tokens) / len(col_tokens)
    fuzzy     = difflib.SequenceMatcher(None, col_norm, goal_text.lower()).ratio()

    return 0.6 * recall + 0.2 * precision + 0.2 * fuzzy


def suggest_target_column(goal_text: str, df: pd.DataFrame) -> Optional[str]:
    """
    Pick the most plausible target column using a scored word-overlap approach.

    Strategy
    --------
    1. Extract the "target phrase" from the goal — the noun phrase after verbs
       like 'predict', 'classify', 'estimate', etc.
    2. Tokenise both the phrase and every column name; score each column by
       token recall (how many goal words appear in the column name), precision,
       and fuzzy string similarity.  This correctly ranks 'player_positions' above
       'nation_position' for the goal "predict player position".
    3. Fall back to sports-domain keyword groups if no scored match clears the
       minimum threshold.

    Returns the best-scoring column, or None if nothing clears 0.3.
    """
    # --- Step 1: extract target phrase from goal ---
    # Strip leading verb: "predict X" → "X", "classify X" → "X", etc.
    target_phrase = re.sub(
        r"^\s*(predict|forecast|classify|estimate|detect|identify|find|build.{0,15}model.{0,15}for)\s+",
        "",
        goal_text.lower(),
        flags=re.IGNORECASE,
    ).strip()
    # Strip trailing clauses ("based on …", "using …")
    target_phrase = re.sub(r"\s+(based on|using|from|with|by|in|for)\s+.*$", "", target_phrase).strip()

    goal_tokens = _col_tokens(target_phrase) or _col_tokens(goal_text)

    # --- Step 2: score all columns ---
    if goal_tokens:
        scored = [
            (col, _score_column(col, goal_tokens, target_phrase))
            for col in df.columns
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_col, best_score = scored[0]
        if best_score >= 0.3:
            return best_col

    # --- Step 3: domain-keyword fallback ---
    text = goal_text.lower()
    _KEYWORD_GROUPS = [
        (["injur", "hurt", "fitness", "availability"], ["injur", "hurt", "fit", "availab"]),
        (["win", "lose", "result", "outcome"],         ["win", "lose", "result", "outcome"]),
        (["goal", "score", "point"],                   ["goal", "score", "point"]),
        (["market value", "transfer", "value"],        ["value", "transfer", "market"]),
        (["play time", "minutes", "appearances"],      ["minutes", "played", "apps", "appearances"]),
        (["position", "role", "pos"],                  ["position", "pos", "role"]),
    ]
    for goal_kws, col_kws in _KEYWORD_GROUPS:
        if any(kw in text for kw in goal_kws):
            candidates = [
                col for col in df.columns
                if any(kw in col.lower() for kw in col_kws)
            ]
            if candidates:
                # Among keyword matches, rank by column name score so e.g.
                # 'player_positions' beats 'nation_position'
                candidates.sort(
                    key=lambda c: _score_column(c, goal_tokens, target_phrase),
                    reverse=True,
                )
                return candidates[0]

    return None


# ---------------------------------------------------------------------------
# LLM-based interpretation
# ---------------------------------------------------------------------------

def parse_user_goal(
    goal_text: str,
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
    _call_log: Optional[List] = None,
) -> TaskSpecification:
    """
    Call Claude to map a natural-language goal to a full TaskSpecification.

    Builds a compact column summary and asks Claude whether the goal is
    predictive or exploratory, what the target column should be, and how
    confident it is.

    Parameters
    ----------
    goal_text : user's free-text goal.
    df        : the uploaded DataFrame — column names and sample values are sent.
    api_key   : Anthropic API key; falls back to ANTHROPIC_API_KEY env var.
    model     : Claude model ID to use.
    _call_log : optional list to append the resulting LLMCallRecord to.

    Returns
    -------
    TaskSpecification with mode, task_type, target_column, confidence, explanation.
    """
    import anthropic

    # Build column summary — show enough samples to distinguish similar columns.
    # Scoring hint: pre-rank columns by word-overlap with the goal so Claude sees
    # the best candidates prominently (they appear first after sorting).
    goal_tokens = _col_tokens(goal_text)
    col_scores = [
        (col, _score_column(col, goal_tokens, goal_text))
        for col in df.columns
    ]
    # Sort: high-scoring columns first so they're salient in the prompt
    col_scores_sorted = sorted(col_scores, key=lambda x: x[1], reverse=True)

    col_lines = []
    for col, score in col_scores_sorted:
        is_numeric  = pd.api.types.is_numeric_dtype(df[col])
        dtype_label = "numeric" if is_numeric else "categorical"
        n_unique    = int(df[col].nunique())
        # Show up to 10 sample values so Claude can distinguish e.g.
        # player_positions ["GK","CB","ST",...] from nation_position ["SUB","RES",...]
        sample = df[col].dropna().unique()[:10].tolist()
        score_hint = f" ★relevance={score:.2f}" if score >= 0.3 else ""
        col_lines.append(
            f"  {col!r:<35} [{dtype_label}, {n_unique} unique{score_hint}]\n"
            f"    sample values: {sample}"
        )
    columns_text = "\n".join(col_lines)

    prompt = f"""\
You are helping a non-technical sports user set up a machine learning analysis.

Dataset: {len(df):,} rows, {len(df.columns)} columns (sorted by relevance to goal):
{columns_text}

User's goal: "{goal_text}"

Instructions:
1. MODE: is this goal PREDICTIVE (train a model to predict a column) or EXPLORATORY \
(discover patterns, clusters, natural groups — no prediction target needed)?
2. If predictive: choose the single best target column.
   CRITICAL name-matching rule — when the goal says "predict X Y":
   • Prefer the column whose name contains ALL words from "X Y" over one that \
contains only some.
   • "predict player position" → prefer 'player_positions' (contains both "player" \
and "position") over 'nation_position' (contains only "position").
   • Columns marked ★relevance≥0.3 above are pre-scored as likely matches — \
prefer these unless the sample values clearly show they are wrong.
3. Explain your reasoning in one sentence suitable for a non-technical sports user.

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{
  "mode": "predictive" | "exploratory",
  "task_type": "classification" | "regression" | "exploratory",
  "target_column": "<exact column name from the list, or null if exploratory>",
  "confidence": "high" | "medium" | "low",
  "explanation": "<one sentence for a non-technical sports user>",
  "requested_outputs": ["model", "report", "visualisations"],
  "alternatives": ["<one or two other plausible column names>"]
}}

Rules:
- target_column MUST be an exact column name copied from the list above, or null.
- confidence = "high" when the name match is unambiguous; "medium" when plausible \
but not certain; "low" when guessing.
- For exploratory goals (clustering, finding patterns, understanding data): \
mode="exploratory", target_column=null, task_type="exploratory".
- Include "model" in requested_outputs only when mode="predictive".
"""

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    # Track LLM cost
    usage = response.usage
    input_tok  = getattr(usage, "input_tokens", 0)
    output_tok = getattr(usage, "output_tokens", 0)
    cache_r    = getattr(usage, "cache_read_input_tokens", 0)
    cache_w    = getattr(usage, "cache_creation_input_tokens", 0)
    cost = _estimate_cost(model, input_tok, output_tok, cache_w, cache_r)
    record = LLMCallRecord(
        purpose="goal_interpretation",
        model=model,
        input_tokens=input_tok,
        output_tokens=output_tok,
        cache_read_tokens=cache_r,
        cache_write_tokens=cache_w,
        estimated_cost_usd=cost,
    )
    if _call_log is not None:
        _call_log.append(record)

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    target = data.get("target_column")
    if target is not None and target not in df.columns:
        matches = [c for c in df.columns if c.lower() == str(target).lower()]
        target = matches[0] if matches else None

    alternatives = [a for a in data.get("alternatives", []) if a in df.columns and a != target]

    mode = data.get("mode", "predictive")
    outputs = data.get("requested_outputs", ["report", "visualisations"])
    if mode == "predictive" and "model" not in outputs:
        outputs = ["model"] + outputs
    if "clusters" not in outputs:
        outputs = outputs + ["clusters"]

    return TaskSpecification(
        mode=mode,
        task_type=data.get("task_type", "classification"),
        goal=goal_text,
        target_column=target,
        requested_outputs=outputs,
        confidence=data.get("confidence", "low"),
        explanation=data.get("explanation", ""),
        alternatives=alternatives,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _infer_task_type(col: Optional[str], df: pd.DataFrame) -> str:
    """
    Infer whether a target column warrants 'regression' or 'classification'.

    Heuristic (mirrors the Profiler's TargetType logic):
      - Non-numeric columns                       → classification
      - Numeric with ≤ 15 distinct values         → classification (likely encoded labels)
      - Numeric with > 15 distinct values         → regression (continuous / ordinal score)

    The threshold of 15 is intentionally conservative so that a 0-10 injury score
    stays as classification while a 50-99 player rating or market value becomes
    regression.
    """
    if col is None or col not in df.columns:
        return "classification"
    series = df[col].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    return "regression" if series.nunique() > 15 else "classification"


def build_task_specification(
    goal_text: str,
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
    _call_log: Optional[List] = None,
) -> TaskSpecification:
    """
    Build a TaskSpecification from a natural-language goal.

    If an API key is available, calls Claude for a precise interpretation.
    Otherwise, falls back to rule-based keyword matching.

    Parameters
    ----------
    goal_text : user's free-text goal description.
    df        : the uploaded DataFrame.
    api_key   : Anthropic API key; falls back to ANTHROPIC_API_KEY env var.
    model     : Claude model ID.
    _call_log : optional list to append LLMCallRecord entries to.

    Returns
    -------
    TaskSpecification
    """
    # Goal interpretation is always rule-based.
    # The LLM (Claude) is reserved for the Planner, which is where the research
    # contribution lies.  Rule-based interpretation is fast, free, and works well
    # given the improved word-overlap scoring in suggest_target_column().
    mode       = infer_operating_mode(goal_text)
    target_col = suggest_target_column(goal_text, df) if mode == "predictive" else None
    outputs    = ["report", "visualisations", "clusters"]
    if mode == "predictive":
        outputs = ["model"] + outputs

    if mode == "exploratory":
        task_type = "exploratory"
    else:
        task_type = _infer_task_type(target_col, df)

    confidence = "medium" if target_col is not None else "low"
    explanation = (
        "Goal interpreted automatically. "
        "Please verify the mode and target column below."
    )

    return TaskSpecification(
        mode=mode,
        task_type=task_type,
        goal=goal_text,
        target_column=target_col,
        requested_outputs=outputs,
        confidence=confidence,
        explanation=explanation,
        alternatives=[],
    )
