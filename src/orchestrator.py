"""
orchestrator.py — central coordinator of the agentic pipeline.

Ties all agents and memory components together into a single callable function.
The user only needs to provide a DataFrame and the name of the target column.

Agentic loop (per iteration)
-----------------------------
  1. Planner  — proposes n_plans_per_round candidate ActionPlans, informed by
                the detected issues, full iteration history, and cross-run memory.
  2. Executor — builds a scikit-learn Pipeline for each ActionPlan (inside Evaluator).
  3. Evaluator— runs 5-fold cross-validation on every pipeline, computes a
                composite score, and selects the best candidate for this round.
  4. Memory   — the best AttemptRecord is appended to the within-run history
                (RAM + SQLite) so the next Planner call avoids repetition.

Stopping criteria (checked after every round)
----------------------------------------------
  - score_threshold reached  → converged = True, stop immediately.
  - No improvement for 2 consecutive rounds (plateau)  → stop early.
  - max_rounds exhausted     → stop, converged = False.

After the loop
--------------
  - The best pipeline is re-fitted on the full training set.
  - The best ActionPlan is stored in ChromaDB, indexed by the dataset fingerprint,
    so future runs on similar datasets can warm-start from this configuration.

Public API
----------
run_pipeline_from_goal(df, goal_text, ...)  -> RunResult | ExploratoryResult
run_agentic_pipeline(df, target, ...)       -> RunResult
run_exploratory_pipeline(df, ...)           -> ExploratoryResult
"""

import uuid
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd

# Paths to persistent storage, anchored to the project root regardless of
# which directory the notebook or script is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH     = str(_PROJECT_ROOT / "storage" / "runs.db")
_DEFAULT_CHROMA_DIR  = str(_PROJECT_ROOT / "storage" / "chroma_db")

from src.agents.evaluator import (
    build_exploratory_report,
    build_user_report,
    evaluate_plans,
    generate_exploratory_visualisations,
    generate_visualisations,
    select_best,
)
from src.agents.data_preparer import DataPrepConfig, prepare_dataset
from src.agents.executor import build_pipeline
from src.agents.issue_detector import detect_issues, request_user_clarification
from src.agents.planner import propose_action_plans
from src.agents.profiler import generate_profile
from src.data.loader import dataset_fingerprint, split_data
from src.memory.run_store import RunStore
from src.memory.vector_store import VectorStore
from src.agents.goal_interpreter import build_task_specification
from src.models.schemas import (
    AttemptRecord,
    ClarificationQuestion,
    ExploratoryResult,
    LLMCallRecord,
    RunResult,
    TaskSpecification,
)


def run_pipeline_from_goal(
    df: pd.DataFrame,
    goal_text: str,
    target: Optional[str] = None,
    run_id: Optional[str] = None,
    max_rounds: int = 3,
    n_plans_per_round: int = 3,
    cv: int = 5,
    score_threshold: float = 0.90,
    min_improvement: float = 0.005,
    db_path: str = _DEFAULT_DB_PATH,
    chroma_dir: str = _DEFAULT_CHROMA_DIR,
    claude_model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
    use_memory: bool = True,
    ask_clarifications: bool = True,
    prefilled_questions: Optional[List[ClarificationQuestion]] = None,
    figures_dir: str = "figures",
    generate_report: bool = True,
) -> Union[RunResult, ExploratoryResult]:
    """
    Main user-facing entry point: interprets a natural-language goal and routes
    to the appropriate pipeline branch automatically.

    Workflow
    --------
    goal_text → TaskSpecification (via Goal Interpreter)
        ↓
    if exploratory → run_exploratory_pipeline()
    if predictive  → resolve/confirm target → run_agentic_pipeline()

    Parameters
    ----------
    df          : raw input DataFrame.
    goal_text   : natural-language description of what the user wants to achieve.
    target      : optional explicit target column.  When provided, overrides the
                  Goal Interpreter's suggestion (useful for programmatic calls
                  or re-runs where the user has already confirmed the column).
    All other parameters are forwarded to run_agentic_pipeline() or
    run_exploratory_pipeline() as appropriate.

    Returns
    -------
    RunResult for predictive goals, ExploratoryResult for exploratory goals.
    Raises ValueError when a predictive goal cannot be resolved to a target column.
    """
    _log = print if verbose else (lambda *a, **k: None)

    # Interpret goal → TaskSpecification
    llm_calls: List[LLMCallRecord] = []
    task_spec: TaskSpecification = build_task_specification(
        goal_text, df, api_key=api_key, _call_log=llm_calls,
    )
    _log(f"Goal    : {goal_text}")
    _log(f"Mode    : {task_spec.mode}  |  confidence={task_spec.confidence}")

    if task_spec.mode == "exploratory":
        _log("Routing to exploratory pipeline.")
        return run_exploratory_pipeline(
            df=df,
            goal_text=goal_text,
            run_id=run_id,
            figures_dir=figures_dir,
            verbose=verbose,
            ask_clarifications=ask_clarifications,
            prefilled_questions=prefilled_questions,
        )

    # Predictive path — resolve target column
    resolved_target = target or task_spec.target_column
    if resolved_target is None:
        raise ValueError(
            f"Goal '{goal_text}' was interpreted as predictive "
            f"(confidence={task_spec.confidence}), but no target column could be "
            f"identified.  Please pass target=<column_name> explicitly."
        )
    _log(f"Target  : {resolved_target}  (source: {'explicit' if target else 'goal interpreter'})")

    return run_agentic_pipeline(
        df=df,
        target=resolved_target,
        run_id=run_id,
        max_rounds=max_rounds,
        n_plans_per_round=n_plans_per_round,
        cv=cv,
        score_threshold=score_threshold,
        min_improvement=min_improvement,
        db_path=db_path,
        chroma_dir=chroma_dir,
        claude_model=claude_model,
        api_key=api_key,
        test_size=test_size,
        random_state=random_state,
        verbose=verbose,
        use_memory=use_memory,
        ask_clarifications=ask_clarifications,
        prefilled_questions=prefilled_questions,
        goal_text=goal_text,
        figures_dir=figures_dir,
        generate_report=generate_report,
    )


def run_exploratory_pipeline(
    df: pd.DataFrame,
    goal_text: Optional[str] = None,
    run_id: Optional[str] = None,
    figures_dir: str = "figures",
    verbose: bool = True,
    ask_clarifications: bool = False,
    prefilled_questions: Optional[List[ClarificationQuestion]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ExploratoryResult:
    """
    Run an exploratory analysis — no target column, no ML training.

    Profiles the full dataset (all columns treated as features), discovers natural
    clusters, generates visualisations, and produces a plain-language report.
    Suitable when the user's goal is to understand patterns or groupings rather
    than to build a predictive model.

    Parameters
    ----------
    df                 : raw input DataFrame (all columns are features).
    goal_text          : user's natural-language goal — included in the report.
    run_id             : optional identifier; auto-generated if None.
    figures_dir        : directory where PNG visualisations are saved.
    verbose            : print progress to stdout.
    ask_clarifications : if True, interactively ask clarification questions via stdin.
    prefilled_questions: pre-answered ClarificationQuestion list from a UI (skips stdin).

    Returns
    -------
    ExploratoryResult with profile, issues, clusters, report, and visualisation paths.
    """
    import time
    t_start = time.perf_counter()

    run_id = run_id or str(uuid.uuid4())
    _log = print if verbose else (lambda *a, **k: None)
    _cb = progress_callback or (lambda p, m: None)

    _log(f"Run ID : {run_id}  [EXPLORATORY MODE]")
    if goal_text:
        _log(f"Goal    : {goal_text}")
    _log(f"Dataset : {len(df):,} rows × {len(df.columns)} cols")

    # Profile all columns (no target → exploratory mode)
    _cb(0.05, "Profiling dataset…")
    profile = generate_profile(df, target=None, run_clustering=True)
    _cb(0.40, "Detecting data quality issues…")
    issues  = detect_issues(profile, df)
    _cb(0.45, "Issues detected")

    _log(f"Issues  : {len(issues)} detected "
         f"({sum(i.severity.value == 'high' for i in issues)} HIGH, "
         f"{sum(i.severity.value == 'medium' for i in issues)} MEDIUM)")
    if profile.sports_context and profile.sports_context.is_sports:
        sc = profile.sports_context
        _log(f"Sports  : sports dataset detected (confidence={sc.confidence:.0%})  "
             f"matched terms: {', '.join(sc.matched_terms[:6])}")
    if profile.clusters:
        _log(f"Clusters: {profile.clusters.n_clusters} natural groups found "
             f"(silhouette={profile.clusters.silhouette_score:.3f})")

    # Clarification questions
    if prefilled_questions is not None:
        clarification_questions = prefilled_questions
    else:
        clarification_questions = request_user_clarification(issues, profile)

    if prefilled_questions is None and ask_clarifications and clarification_questions:
        _log(f"\n{'─' * 50}")
        _log(f"The system has {len(clarification_questions)} clarification question(s):")
        for q in clarification_questions:
            _log(f"\n  [{q.issue_type.upper()}] {q.question}")
            try:
                answer = input("  Your answer: ").strip()
                q.answer = answer if answer else None
            except (EOFError, OSError):
                pass
        _log(f"{'─' * 50}")

    # Visualisations
    _log("Generating visualisations...")
    _cb(0.55, "Generating visualisations…")
    vis_paths: list = []
    try:
        Path(figures_dir).mkdir(parents=True, exist_ok=True)
        vis_paths = generate_exploratory_visualisations(
            profile=profile,
            X=df,
            output_dir=figures_dir,
        )
        if vis_paths:
            _log(f"Visualisations saved: {vis_paths}")
    except Exception as exc:
        _log(f"Visualisation generation skipped: {exc}")

    # Plain-language report
    _log("Building exploratory report...")
    _cb(0.85, "Building report…")
    user_report = ""
    try:
        user_report = build_exploratory_report(
            profile=profile,
            issues=issues,
            goal_text=goal_text or "",
            visualisation_paths=vis_paths,
        )
    except Exception as exc:
        _log(f"Report generation skipped: {exc}")

    _cb(1.0, "Done!")
    runtime = time.perf_counter() - t_start
    _log(f"\nDone. Exploratory analysis completed in {runtime:.1f}s.")

    return ExploratoryResult(
        run_id=run_id,
        profile=profile,
        issues=issues,
        clarification_questions=clarification_questions,
        user_report=user_report,
        visualisation_paths=vis_paths,
        llm_calls=[],
        runtime_secs=runtime,
    )


def run_agentic_pipeline(
    df: pd.DataFrame,
    target: str,
    run_id: Optional[str] = None,
    max_rounds: int = 3,
    n_plans_per_round: int = 3,
    cv: int = 5,
    score_threshold: float = 0.90,
    min_improvement: float = 0.005,
    db_path: str = _DEFAULT_DB_PATH,
    chroma_dir: str = _DEFAULT_CHROMA_DIR,
    claude_model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
    use_memory: bool = True,
    ask_clarifications: bool = True,
    prefilled_questions: Optional[List[ClarificationQuestion]] = None,
    goal_text: Optional[str] = None,
    figures_dir: str = "figures",
    generate_report: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> RunResult:
    """
    Run the full agentic pipeline optimisation loop on a tabular dataset.

    Parameters
    ----------
    df                 : raw input DataFrame including the target column.
    target             : name of the column to predict.
    run_id             : optional identifier for this run; auto-generated if None.
    max_rounds         : maximum number of Planner → Evaluate iterations.
    n_plans_per_round  : number of candidate ActionPlans requested per round.
    cv                 : number of cross-validation folds for evaluation.
    score_threshold    : composite score at which the loop stops early (converged).
    min_improvement    : minimum score gain required to reset the plateau counter.
    db_path            : path to the SQLite database for within-run persistence.
    chroma_dir         : directory for ChromaDB cross-run memory storage.
    claude_model       : Claude model ID passed to the Planner Agent.
    api_key            : Anthropic API key; falls back to ANTHROPIC_API_KEY env var.
    test_size          : fraction of data held out as a test set.
    random_state       : random seed for reproducible splits.
    verbose            : print round-by-round progress to stdout.
    use_memory         : when False, cross-run ChromaDB memory is disabled — no warm-start
                         retrieval and no storing of the best plan after the run. Equivalent
                         to ablation variant "agentic system without memory".
    ask_clarifications : when True, prints clarification questions for the user and reads
                         answers from stdin before the first Planner call.  Set to False in
                         ablation runs or automated batch processing.
    prefilled_questions: pre-answered ClarificationQuestion list from an external UI (e.g.
                         Streamlit). When provided, internal question generation and the
                         input() loop are both skipped.
    goal_text          : natural-language description of the user's analytical goal.
                         Passed to the Planner so it can tailor plans and explanations.
    figures_dir        : directory where visualisation PNGs are saved.
    generate_report    : when True, builds a plain-language markdown report after the loop.

    Returns
    -------
    RunResult with the best fitted pipeline, plan, evaluation result, iteration history,
    clarification questions asked, and the plain-language user report.
    """
    run_id = run_id or str(uuid.uuid4())
    _log = print if verbose else (lambda *a, **k: None)
    _cb = progress_callback or (lambda p, m: None)

    # ------------------------------------------------------------------
    # 1. Split — test set is held out entirely; all optimisation happens
    #    on X_train via cross-validation to prevent test set leakage.
    # ------------------------------------------------------------------
    _cb(0.02, "Splitting dataset…")
    X_train, X_test, y_train, y_test = split_data(
        df, target, test_size=test_size, random_state=random_state
    )
    _log(f"Run ID : {run_id}")
    if goal_text:
        _log(f"Goal    : {goal_text}")
    _log(f"Dataset : {len(df):,} rows × {len(df.columns)} cols  |  target='{target}'")
    _log(f"Split   : {len(X_train):,} train / {len(X_test):,} test")

    # ------------------------------------------------------------------
    # 2. Prepare the full dataset — drop structurally useless columns
    #    (100%-missing, zero-variance) and duplicate rows BEFORE split.
    #    This runs at the DataFrame level so the sklearn pipeline never
    #    encounters all-NaN features, which avoids imputer warnings and
    #    downstream linear-model NaN explosions.
    #    The test set is re-derived after preparation so both splits share
    #    the same column set.
    # ------------------------------------------------------------------
    _cb(0.05, "Preparing dataset…")
    df_prepared, prep_report = prepare_dataset(df, target=target)
    if prep_report.n_cols_dropped or prep_report.dropped_duplicate_rows:
        _log(f"\nData prep: {prep_report.summary()}")
    # Re-split on the cleaned DataFrame so train/test share identical columns
    X_train, X_test, y_train, y_test = split_data(
        df_prepared, target, test_size=test_size, random_state=random_state
    )

    # ------------------------------------------------------------------
    # 3. Profile the training set (including exploratory clustering) and
    #    detect data quality issues.  Profiling is done on train_df so
    #    that class distribution and target statistics reflect only
    #    training data; clustering runs on features only (target excluded).
    # ------------------------------------------------------------------
    _cb(0.08, "Profiling training data…")
    train_df = pd.concat([X_train, y_train], axis=1)
    profile = generate_profile(train_df, target, run_clustering=True)
    _cb(0.14, "Detecting data quality issues…")
    issues = detect_issues(profile, train_df)

    _log(f"\nTask    : {profile.target_type.value}")
    if profile.sports_context and profile.sports_context.is_sports:
        sc = profile.sports_context
        _log(
            f"Sports  : sports dataset detected (confidence={sc.confidence:.0%})  "
            f"post-match={len(sc.post_match_cols)}  identity={len(sc.identity_cols)}  "
            f"playing-time={len(sc.playing_time_cols)}  injury={len(sc.injury_cols)}"
        )
    if profile.clusters:
        _log(f"Clusters: {profile.clusters.n_clusters} natural groups found "
             f"(silhouette={profile.clusters.silhouette_score:.3f})")
    _log(f"Issues  : {len(issues)} detected "
         f"({sum(i.severity.value == 'high' for i in issues)} HIGH, "
         f"{sum(i.severity.value == 'medium' for i in issues)} MEDIUM, "
         f"{sum(i.severity.value == 'low' for i in issues)} LOW)")

    # ------------------------------------------------------------------
    # 2b. Optionally ask the user clarification questions for domain-
    #     specific issues that cannot be resolved automatically.
    #     All questions are batched and the user is interrupted once.
    #     When prefilled_questions is provided (e.g. from Streamlit UI),
    #     skip generation and the input() loop entirely.
    # ------------------------------------------------------------------
    if prefilled_questions is not None:
        clarification_questions = prefilled_questions
        _log(f"Clarifications: {len(clarification_questions)} pre-filled from UI "
             f"({sum(q.answer is not None for q in clarification_questions)} answered)")
    else:
        clarification_questions = request_user_clarification(issues, profile)

    if prefilled_questions is None and ask_clarifications and clarification_questions:
        _log(f"\n{'─' * 50}")
        _log(f"The system has {len(clarification_questions)} clarification question(s) "
             f"before proceeding.  Please answer each one (press Enter to skip):")
        for q in clarification_questions:
            _log(f"\n  [{q.issue_type.upper()}] {q.question}")
            try:
                answer = input("  Your answer: ").strip()
                q.answer = answer if answer else None
            except (EOFError, OSError):
                # Non-interactive environment — skip clarification
                pass
        _log(f"{'─' * 50}")
    elif prefilled_questions is None and clarification_questions:
        _log(f"Clarification questions generated ({len(clarification_questions)}) "
             f"but ask_clarifications=False — proceeding autonomously.")

    # ------------------------------------------------------------------
    # 3. Compute the dataset fingerprint and optionally query cross-run
    #    memory for similar past configurations to warm-start the Planner.
    #    When use_memory=False the fingerprint is still computed (it is
    #    stored in the profile for potential downstream use) but ChromaDB
    #    is not queried or written — this isolates the memory contribution.
    # ------------------------------------------------------------------
    fp = dataset_fingerprint(profile)
    profile.fingerprint = fp

    _cb(0.18, "Querying memory for similar past runs…")
    if use_memory:
        vstore = VectorStore(chroma_dir)
        memory = vstore.retrieve_similar(fp, top_k=3)
        _log(f"Memory  : {len(memory)} similar past configuration(s) retrieved")
    else:
        vstore = None
        memory = []
        _log("Memory  : disabled (ablation mode)")

    # ------------------------------------------------------------------
    # 4. Agentic optimisation loop
    # ------------------------------------------------------------------
    history = []
    llm_calls: List[LLMCallRecord] = []
    overall_best_result = None
    overall_best_plan = None
    no_improvement_count = 0
    converged = False
    round_num = 0

    # Progress budget: 20% setup → 80% loop → 20% finalise
    _LOOP_START = 0.20
    _LOOP_END   = 0.80

    with RunStore(db_path) as run_store:
        for round_num in range(1, max_rounds + 1):
            _log(f"\n{'─' * 50}")
            _log(f"Round {round_num}/{max_rounds}")
            _log(f"{'─' * 50}")

            _round_base = _LOOP_START + (round_num - 1) / max_rounds * (_LOOP_END - _LOOP_START)
            _round_size = (_LOOP_END - _LOOP_START) / max_rounds
            _cb(_round_base, f"Round {round_num}/{max_rounds} — asking AI planner…")

            # --- Planner: propose candidate configurations ---
            plans, reasoning = propose_action_plans(
                profile=profile,
                issues=issues,
                history=history,
                memory=memory,
                clarification_questions=clarification_questions,
                goal_text=goal_text,
                n_plans=n_plans_per_round,
                model=claude_model,
                api_key=api_key,
                _call_log=llm_calls,
            )
            _log(f"Planner : {len(plans)} plans proposed")
            _log(f"Reasoning: {reasoning}")
            _cb(_round_base + _round_size * 0.35,
                f"Round {round_num}/{max_rounds} — evaluating {len(plans)} pipeline(s) with {cv}-fold CV…")

            # --- Evaluator: cross-validate every candidate ---
            results = evaluate_plans(plans, profile, X_train, y_train, cv=cv)
            round_best_result = select_best(results)
            round_best_plan = next(p for p in plans if p.plan_id == round_best_result.plan_id)

            _log(f"Best this round : score={round_best_result.score:.4f} "
                 f"| cv_std={round_best_result.cv_std:.4f} "
                 f"| {round_best_result.metric_values}")

            # --- Persist every attempt to SQLite and RAM history ---
            for plan, result in zip(plans, results):
                record = AttemptRecord(iteration=round_num, plan=plan, result=result)
                history.append(record)
                run_store.save_attempt(run_id, record)

            # --- Update overall best ---
            if (overall_best_result is None
                    or round_best_result.score > overall_best_result.score + min_improvement):
                overall_best_result = round_best_result
                overall_best_plan = round_best_plan
                no_improvement_count = 0
                _log("Overall best updated.")
            else:
                no_improvement_count += 1
                _log(f"No significant improvement "
                     f"(plateau counter: {no_improvement_count}/2)")

            _cb(_round_base + _round_size * 0.95,
                f"Round {round_num}/{max_rounds} — best score so far: {overall_best_result.score:.4f}")

            # --- Convergence checks ---
            if overall_best_result.score >= score_threshold:
                _log(f"\nScore threshold {score_threshold} reached — converged.")
                converged = True
                break

            if no_improvement_count >= 2:
                _log("\nPlateau detected (2 consecutive rounds without improvement) — stopping.")
                break

    # ------------------------------------------------------------------
    # 5. Fit the best pipeline on the full training set.
    #    CV evaluation uses k-fold splits of X_train; the final model
    #    should use all available training data for maximum capacity.
    # ------------------------------------------------------------------
    _log(f"\n{'─' * 50}")
    _log("Fitting best pipeline on full training set...")
    _cb(0.82, "Fitting best pipeline on full training set…")
    best_pipeline = build_pipeline(overall_best_plan, profile, X_train)
    best_pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Store the best configuration in ChromaDB for future warm-starts
    #    (skipped when use_memory=False so ablation runs stay isolated).
    # ------------------------------------------------------------------
    if use_memory and vstore is not None:
        vstore.store_success(fp, overall_best_plan, dataset_name=target)
        _log("Best plan stored in cross-run memory.")
    else:
        _log("Cross-run memory write skipped (ablation mode).")

    # ------------------------------------------------------------------
    # 7. Generate visualisations and build a plain-language user report
    #    for non-technical sports users.
    # ------------------------------------------------------------------
    vis_paths: list = []
    user_report = ""

    if generate_report:
        _log("Generating visualisations and user report...")
        _cb(0.87, "Generating visualisations…")
        try:
            vis_paths = generate_visualisations(
                profile=profile,
                best_result=overall_best_result,
                best_pipeline=best_pipeline,
                X=X_train,
                y=y_train,
                output_dir=figures_dir,
            )
            if vis_paths:
                _log(f"Visualisations saved: {vis_paths}")
        except Exception as exc:
            _log(f"Visualisation generation skipped: {exc}")

        _cb(0.93, "Building report…")
        try:
            user_report = build_user_report(
                profile=profile,
                issues=issues,
                best_plan=overall_best_plan,
                best_result=overall_best_result,
                visualisation_paths=vis_paths,
            )
        except Exception as exc:
            _log(f"User report generation skipped: {exc}")

    _cb(1.0, "Done!")
    total_cost = sum(r.estimated_cost_usd for r in llm_calls)
    _log(f"\nDone. Best score: {overall_best_result.score:.4f} "
         f"in {round_num} round(s). Converged: {converged}")
    _log(f"LLM calls: {len(llm_calls)}  |  "
         f"total tokens: {sum(r.input_tokens + r.output_tokens for r in llm_calls):,}  |  "
         f"estimated cost: ${total_cost:.4f}")

    return RunResult(
        best_pipeline=best_pipeline,
        best_plan=overall_best_plan,
        best_result=overall_best_result,
        history=history,
        n_iterations=round_num,
        converged=converged,
        run_id=run_id,
        clarification_questions=clarification_questions,
        user_report=user_report,
        llm_calls=llm_calls,
    )
