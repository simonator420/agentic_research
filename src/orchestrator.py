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
run_agentic_pipeline(df, target, ...) -> RunResult
"""

import uuid
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Paths to persistent storage, anchored to the project root regardless of
# which directory the notebook or script is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH     = str(_PROJECT_ROOT / "storage" / "runs.db")
_DEFAULT_CHROMA_DIR  = str(_PROJECT_ROOT / "storage" / "chroma_db")

from src.agents.evaluator import build_user_report, evaluate_plans, generate_visualisations, select_best
from src.agents.executor import build_pipeline
from src.agents.issue_detector import detect_issues, request_user_clarification
from src.agents.planner import propose_action_plans
from src.agents.profiler import generate_profile
from src.data.loader import dataset_fingerprint, split_data
from src.memory.run_store import RunStore
from src.memory.vector_store import VectorStore
from src.models.schemas import AttemptRecord, ClarificationQuestion, RunResult


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
    figures_dir: str = "figures",
    generate_report: bool = True,
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
    figures_dir        : directory where visualisation PNGs are saved.
    generate_report    : when True, builds a plain-language markdown report after the loop.

    Returns
    -------
    RunResult with the best fitted pipeline, plan, evaluation result, iteration history,
    clarification questions asked, and the plain-language user report.
    """
    run_id = run_id or str(uuid.uuid4())
    _log = print if verbose else (lambda *a, **k: None)

    # ------------------------------------------------------------------
    # 1. Split — test set is held out entirely; all optimisation happens
    #    on X_train via cross-validation to prevent test set leakage.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(
        df, target, test_size=test_size, random_state=random_state
    )
    _log(f"Run ID : {run_id}")
    _log(f"Dataset : {len(df):,} rows × {len(df.columns)} cols  |  target='{target}'")
    _log(f"Split   : {len(X_train):,} train / {len(X_test):,} test")

    # ------------------------------------------------------------------
    # 2. Profile the training set (including exploratory clustering) and
    #    detect data quality issues.  Profiling is done on train_df so
    #    that class distribution and target statistics reflect only
    #    training data; clustering runs on features only (target excluded).
    # ------------------------------------------------------------------
    train_df = pd.concat([X_train, y_train], axis=1)
    profile = generate_profile(train_df, target, run_clustering=True)
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
    overall_best_result = None
    overall_best_plan = None
    no_improvement_count = 0
    converged = False
    round_num = 0

    with RunStore(db_path) as run_store:
        for round_num in range(1, max_rounds + 1):
            _log(f"\n{'─' * 50}")
            _log(f"Round {round_num}/{max_rounds}")
            _log(f"{'─' * 50}")

            # --- Planner: propose candidate configurations ---
            plans, reasoning = propose_action_plans(
                profile=profile,
                issues=issues,
                history=history,
                memory=memory,
                clarification_questions=clarification_questions,
                n_plans=n_plans_per_round,
                model=claude_model,
                api_key=api_key,
            )
            _log(f"Planner : {len(plans)} plans proposed")
            _log(f"Reasoning: {reasoning}")

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

    _log(f"\nDone. Best score: {overall_best_result.score:.4f} "
         f"in {round_num} round(s). Converged: {converged}")

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
    )
