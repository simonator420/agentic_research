"""
goal_resolver.py — thin compatibility shim.

Formerly contained resolve_goal() / GoalResolution.  That logic has been
consolidated into goal_interpreter.py which returns the richer TaskSpecification.

This module re-exports GoalResolution and resolve_goal() built on top of
build_task_specification() so that any existing code importing from here
continues to work without modification.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.agents.goal_interpreter import build_task_specification


@dataclass
class GoalResolution:
    """
    Legacy return type kept for backward compatibility.
    New code should use TaskSpecification from goal_interpreter instead.
    """
    goal_type:     str
    target_column: Optional[str]
    confidence:    str
    explanation:   str
    task_type:     str
    alternatives:  List[str] = field(default_factory=list)


def resolve_goal(
    df: pd.DataFrame,
    goal_text: str,
    api_key: Optional[str] = None,   # kept for backward compat — no longer used
    model: str = "claude-sonnet-4-6", # kept for backward compat — no longer used
) -> GoalResolution:
    """
    Map a free-text goal to a GoalResolution.

    Delegates to build_task_specification() (always rule-based) and converts
    the richer TaskSpecification back to the legacy GoalResolution shape.
    """
    spec = build_task_specification(goal_text, df)
    return GoalResolution(
        goal_type=spec.mode,
        target_column=spec.target_column,
        confidence=spec.confidence,
        explanation=spec.explanation,
        task_type=spec.task_type,
        alternatives=spec.alternatives,
    )
