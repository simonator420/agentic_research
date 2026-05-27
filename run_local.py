#!/usr/bin/env python3
"""
Command-line runner for the Agentic Sports Analytics pipeline.

Examples:
    python run_local.py data/titanic.csv --goal "Predict whether a passenger survived" --target Survived
    python run_local.py data/titanic.csv --target Survived --baseline rule
    python run_local.py data/titanic.csv --mode exploratory --goal "Find natural groups in the data"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _load_env() -> None:
    """Load .env when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the project locally from a terminal, without Streamlit.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/titanic.csv",
        help="Path to a CSV, Parquet, or Excel dataset. Default: data/titanic.csv",
    )
    parser.add_argument(
        "--goal",
        help="Plain-language analysis goal, e.g. 'Predict player injury risk'.",
    )
    parser.add_argument(
        "--target",
        help="Target column for predictive runs. Recommended for CLI usage.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "predictive", "exploratory"),
        default="auto",
        help="Run mode. 'auto' uses the goal interpreter. Default: auto",
    )
    parser.add_argument(
        "--baseline",
        choices=("agentic", "rule", "search", "flaml"),
        default="agentic",
        help="Runner type. Baselines do not require an Anthropic API key. Default: agentic",
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--plans-per-round", type=int, default=3)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.90)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials for --baseline search.")
    parser.add_argument("--time-budget", type=int, default=60, help="Seconds for --baseline flaml.")
    parser.add_argument("--api-key", help="Anthropic API key. Defaults to ANTHROPIC_API_KEY.")
    parser.add_argument("--figures-dir", default="figures", help="Where visualisations are saved.")
    parser.add_argument("--report-out", help="Optional path to write the markdown report.")
    parser.add_argument("--no-memory", action="store_true", help="Disable ChromaDB cross-run memory.")
    parser.add_argument("--no-report", action="store_true", help="Skip report and visualisation generation.")
    parser.add_argument(
        "--no-clarifications",
        action="store_true",
        help="Do not ask interactive clarification questions in the terminal.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging.")
    return parser


def _require_target(target: str | None, columns: list[str]) -> str:
    if not target:
        raise SystemExit(
            "Predictive CLI runs need --target unless you use --goal with the agentic runner "
            "and an Anthropic API key."
        )
    if target not in columns:
        available = ", ".join(columns)
        raise SystemExit(f"Target column '{target}' not found. Available columns: {available}")
    return target


def _require_api_key(api_key: str | None) -> str:
    if not api_key:
        raise SystemExit(
            "The full agentic predictive pipeline needs an Anthropic API key. "
            "Set ANTHROPIC_API_KEY, pass --api-key, or use --baseline rule/search/flaml."
        )
    return api_key


def _print_metrics(metrics: dict[str, Any]) -> None:
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")


def _print_plan(plan: Any) -> None:
    print("Best plan:")
    print(f"  model: {plan.model}")
    print(f"  imputation: {plan.imputation}")
    print(f"  outlier_handling: {plan.outlier_handling}")
    print(f"  encoding: {plan.encoding}")
    print(f"  scaling: {plan.scaling}")
    print(f"  imbalance_strategy: {plan.imbalance_strategy}")
    explanation = plan.model_params.get("__explanation") if getattr(plan, "model_params", None) else None
    if explanation:
        print(f"  explanation: {explanation}")


def _write_report(report: str, report_out: str | None) -> None:
    if not report_out or not report:
        return
    path = Path(report_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"Report written to: {path}")


def _print_agentic_result(result: Any, report_out: str | None) -> None:
    from src.models.schemas import ExploratoryResult

    print("\nDone.")
    print(f"Run ID: {result.run_id}")

    if isinstance(result, ExploratoryResult):
        print(f"Rows analysed: {result.profile.n_rows:,}")
        print(f"Columns: {result.profile.n_cols}")
        print(f"Issues: {len(result.issues)}")
        if result.profile.clusters:
            print(f"Clusters: {result.profile.clusters.n_clusters}")
        if result.visualisation_paths:
            print("Visualisations:")
            for path in result.visualisation_paths:
                print(f"  {path}")
        _write_report(result.user_report, report_out)
        return

    print(f"Iterations: {result.n_iterations}")
    print(f"Converged: {result.converged}")
    print(f"Best score: {result.best_result.score:.4f}")
    print(f"CV std: {result.best_result.cv_std:.4f}")
    print("Metrics:")
    _print_metrics(result.best_result.metric_values)
    _print_plan(result.best_plan)
    _write_report(result.user_report, report_out)


def _print_baseline_result(result: Any, report_out: str | None) -> None:
    print("\nDone.")
    print(f"Baseline: {result.method}")
    print(f"Configurations evaluated: {result.n_configs_evaluated}")
    print(f"Best score: {result.score:.4f}")
    print(f"CV std: {result.cv_std:.4f}")
    print(f"Runtime: {result.runtime_secs:.1f}s")
    print("Metrics:")
    _print_metrics(result.metric_values)
    _print_plan(result.best_plan)
    if report_out:
        text = (
            f"# Local Baseline Result\n\n"
            f"- Method: {result.method}\n"
            f"- Score: {result.score:.4f}\n"
            f"- CV std: {result.cv_std:.4f}\n"
            f"- Configurations evaluated: {result.n_configs_evaluated}\n"
        )
        _write_report(text, report_out)


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = _build_parser().parse_args(argv)

    from src.data.loader import load_data

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    df = load_data(str(dataset_path))
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    verbose = not args.quiet
    ask_clarifications = not args.no_clarifications

    print(f"Loaded {dataset_path}: {len(df):,} rows x {len(df.columns)} columns")

    if args.baseline != "agentic":
        target = _require_target(args.target, df.columns.tolist())
        if args.baseline == "rule":
            from src.baselines.rule_based import run_rule_based

            result = run_rule_based(df, target=target, cv=args.cv, test_size=args.test_size)
        elif args.baseline == "search":
            from src.baselines.search_based import run_search_based

            result = run_search_based(
                df,
                target=target,
                n_trials=args.n_trials,
                cv=args.cv,
                test_size=args.test_size,
            )
        else:
            from src.baselines.automl_baseline import run_flaml_baseline

            result = run_flaml_baseline(
                df,
                target=target,
                time_budget_s=args.time_budget,
                cv=args.cv,
                test_size=args.test_size,
            )
        _print_baseline_result(result, args.report_out)
        return 0

    if args.mode == "exploratory":
        from src.orchestrator import run_exploratory_pipeline

        result = run_exploratory_pipeline(
            df=df,
            goal_text=args.goal,
            figures_dir=args.figures_dir,
            verbose=verbose,
            ask_clarifications=ask_clarifications,
        )
        _print_agentic_result(result, args.report_out)
        return 0

    if args.goal and args.mode == "auto":
        if not api_key:
            from src.agents.goal_interpreter import build_task_specification
            from src.orchestrator import run_exploratory_pipeline

            spec = build_task_specification(args.goal, df)
            if spec.mode == "exploratory":
                result = run_exploratory_pipeline(
                    df=df,
                    goal_text=args.goal,
                    figures_dir=args.figures_dir,
                    verbose=verbose,
                    ask_clarifications=ask_clarifications,
                )
                _print_agentic_result(result, args.report_out)
                return 0
            _require_api_key(api_key)

        from src.orchestrator import run_pipeline_from_goal

        result = run_pipeline_from_goal(
            df=df,
            goal_text=args.goal,
            target=args.target,
            max_rounds=args.max_rounds,
            n_plans_per_round=args.plans_per_round,
            cv=args.cv,
            score_threshold=args.score_threshold,
            api_key=api_key,
            test_size=args.test_size,
            verbose=verbose,
            use_memory=not args.no_memory,
            ask_clarifications=ask_clarifications,
            figures_dir=args.figures_dir,
            generate_report=not args.no_report,
        )
        _print_agentic_result(result, args.report_out)
        return 0

    target = _require_target(args.target, df.columns.tolist())
    api_key = _require_api_key(api_key)

    from src.orchestrator import run_agentic_pipeline

    result = run_agentic_pipeline(
        df=df,
        target=target,
        goal_text=args.goal,
        max_rounds=args.max_rounds,
        n_plans_per_round=args.plans_per_round,
        cv=args.cv,
        score_threshold=args.score_threshold,
        api_key=api_key,
        test_size=args.test_size,
        verbose=verbose,
        use_memory=not args.no_memory,
        ask_clarifications=ask_clarifications,
        figures_dir=args.figures_dir,
        generate_report=not args.no_report,
    )
    _print_agentic_result(result, args.report_out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
