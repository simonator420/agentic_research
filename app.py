"""
app.py — Streamlit interface for the Sports Analytics AI Pipeline.

Run with:
    streamlit run app.py
"""

import os
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Suppress known-benign sklearn / LightGBM warnings ────────────────────────
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.utils\.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.metrics\.pairwise")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.linear_model\._base")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*",
                        category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*",
                        category=UserWarning)
warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*",
                        category=UserWarning)
warnings.filterwarnings("ignore", message=".*least populated class.*",
                        category=UserWarning)
warnings.filterwarnings("ignore", message=".*Skipping features without any observed.*",
                        category=UserWarning)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sports Analytics AI Pipeline",
    page_icon=None,
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Layout ────────────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1000px;
}

/* ─── Page title ─────────────────────────────────────────────────────── */
h1 {
    font-weight: 800 !important;
    font-size: 2rem !important;
    letter-spacing: -0.03em !important;
    color: #111827 !important;
    margin-bottom: 0.25rem !important;
}

/* ─── Section step labels (st.subheader → h3) ────────────────────────── */
h3 {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: #9CA3AF !important;
    margin-top: 2.5rem !important;
    margin-bottom: 0.6rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid #F3F4F6 !important;
}

/* ─── Metric cards ───────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FAFAFA;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #6B7280 !important;
}

/* ─── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #F9FAFB;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] h2 {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #6B7280 !important;
}

/* ─── Expanders ──────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
[data-testid="stExpanderToggleIcon"] { color: #9CA3AF !important; }

/* ─── Buttons ────────────────────────────────────────────────────────── */
button[kind="primary"] {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}
button[kind="primary"]:hover {
    background-color: #1E40AF !important;
    border-color: #1E40AF !important;
}
button[kind="secondary"] {
    font-weight: 500 !important;
}

/* ─── Progress bar ───────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div {
    background-color: #1D4ED8 !important;
}

/* ─── Dataframe ──────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB;
    border-radius: 6px;
}

/* ─── Tabs ───────────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    color: #6B7280 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1D4ED8 !important;
    border-bottom-color: #1D4ED8 !important;
}

/* ─── Info / success / warning boxes ────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-size: 0.875rem !important;
}

/* ─── Divider ────────────────────────────────────────────────────────── */
hr { border-color: #F3F4F6 !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    _raw_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Required for the AI planner. Get yours at console.anthropic.com",
    )
    api_key = _raw_key.strip() if _raw_key else ""
    if api_key and not api_key.isascii():
        st.error(
            "The API key contains non-ASCII characters. "
            "Please re-enter it — it should start with **sk-ant-**"
        )
        api_key = ""
    elif api_key and not api_key.startswith("sk-ant-"):
        st.warning("Key format unexpected (expected **sk-ant-...**). Double-check it.")
    elif api_key:
        st.success("API key saved")
    st.divider()
    max_rounds = st.slider("Max optimisation rounds", 1, 5, 3)
    n_plans    = st.slider("Plans per round", 1, 5, 3)
    cv_folds   = st.slider("Cross-validation folds", 3, 10, 5)
    use_memory = st.checkbox(
        "Use cross-run memory",
        value=True,
        help="Warm-starts from similar past datasets stored in ChromaDB",
    )
    st.divider()
    st.caption("Sports Analytics AI Pipeline\nBryant University MSDS 2026")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Sports Analytics AI Pipeline")
st.caption(
    "Upload a sports dataset and describe what you want to find out. "
    "The system will analyse your data, ask any necessary questions, "
    "and build the best possible model for your goal."
)

# ── Session state ─────────────────────────────────────────────────────────────
_STATE_KEYS = ("df", "df_prepared", "goal_text", "task_spec", "target",
               "preflighted", "profile", "issues", "questions",
               "prep_report", "result")
for _k in _STATE_KEYS:
    if _k not in st.session_state:
        st.session_state[_k] = None


def _reset_from(stage: str) -> None:
    """Clear all state from a given stage onward."""
    stages = ["task_spec", "target", "preflighted", "df_prepared",
              "profile", "issues", "questions", "prep_report", "result"]
    idx = stages.index(stage) if stage in stages else 0
    for k in stages[idx:]:
        st.session_state[k] = None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Upload dataset
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Step 1 — Upload dataset")

demo_dir   = Path(__file__).parent / "data"
demo_files = sorted(demo_dir.glob("*.csv"))
demo_names = ["— upload my own —"] + [f.name for f in demo_files]

col_up, col_demo = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
with col_demo:
    demo_choice = st.selectbox("Or use a demo dataset", demo_names)

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif demo_choice != "— upload my own —":
    df = pd.read_csv(demo_dir / demo_choice)

if df is None:
    st.info("Upload a CSV file or choose a demo dataset to get started.")
    st.stop()

if st.session_state.df is not None and not df.equals(st.session_state.df):
    _reset_from("task_spec")
st.session_state.df = df

st.success(f"Loaded **{len(df):,} rows × {len(df.columns)} columns**")
with st.expander("Preview — first 10 rows"):
    st.dataframe(df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Describe your goal
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Describe your goal")
st.caption(
    "Tell the system what you want to find out — in plain language. "
    "No technical knowledge needed."
)

goal_examples = [
    "Predict which players are at risk of injury in the next 30 days",
    "Find out which factors most influence a player's market value",
    "Identify which matches my team is likely to win",
    "Find natural groups or archetypes among players in this dataset",
    "Predict whether a shot will result in a goal",
]
with st.expander("Goal examples"):
    for ex in goal_examples:
        st.write(f"- {ex}")

goal_text = st.text_area(
    "Your goal",
    placeholder="e.g. Predict which players are at risk of injury in the next 30 days",
    height=80,
    label_visibility="collapsed",
)

if goal_text != st.session_state.goal_text:
    _reset_from("task_spec")
    st.session_state.goal_text = goal_text

if not goal_text or not goal_text.strip():
    st.info("Describe your goal above to continue.")
    st.stop()

# ── Goal interpretation ───────────────────────────────────────────────────────
task_spec = st.session_state.task_spec

if task_spec is None:
    if st.button("Interpret goal", type="secondary"):
        from src.agents.goal_interpreter import build_task_specification
        with st.spinner("Interpreting your goal…"):
            try:
                task_spec = build_task_specification(goal_text, df)
                st.session_state.task_spec = task_spec
                st.session_state.target = task_spec.target_column
                _reset_from("preflighted")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not interpret goal: {exc}")
    if st.session_state.task_spec is None:
        st.stop()

if st.session_state.task_spec is not None:
    task_spec = st.session_state.task_spec
    _conf_label = {"high": "High", "medium": "Medium", "low": "Low"}.get(
        task_spec.confidence, task_spec.confidence.title()
    )
    _mode_label = task_spec.mode.title()
    # task_type is re-derived from the confirmed target column so it stays accurate
    # when the user picks a different column from the dropdown below.
    _effective_target = st.session_state.target or task_spec.target_column
    if task_spec.mode == "predictive" and _effective_target:
        from src.agents.goal_interpreter import _infer_task_type
        _display_task_type = _infer_task_type(_effective_target, df)
    else:
        _display_task_type = task_spec.task_type
    st.info(
        f"**{task_spec.explanation}**  \n"
        f"Mode: **{_mode_label}**  |  "
        + (f"Target: `{_effective_target}` ({_display_task_type})  |  "
           if _effective_target else "")
        + f"Confidence: {_conf_label}"
    )
    if task_spec.alternatives:
        st.caption(f"Alternative targets: {', '.join(f'`{a}`' for a in task_spec.alternatives)}")

# Target column: only required for predictive mode
target = None
if task_spec is not None and task_spec.mode == "predictive":
    current_target = st.session_state.target or df.columns[0]
    col_list = df.columns.tolist()
    default_idx = col_list.index(current_target) if current_target in col_list else 0

    confirmed_target = st.selectbox(
        "Confirm or change target column",
        col_list,
        index=default_idx,
        help="The system identified this column based on your goal. Change it if needed.",
    )
    if confirmed_target != st.session_state.target:
        st.session_state.target = confirmed_target
        _reset_from("preflighted")
    target = st.session_state.target
elif task_spec is not None and task_spec.mode == "exploratory":
    st.success(
        "Exploratory mode — no target column required. "
        "The system will profile all columns and discover natural groupings."
    )

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Analyse dataset
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Step 3 — Analyse dataset")

if st.button("Analyse dataset", type="secondary",
             help="Profile the data and detect quality issues"):
    from src.agents.data_preparer import prepare_dataset
    from src.agents.issue_detector import detect_issues, request_user_clarification
    from src.agents.profiler import generate_profile

    with st.spinner("Preparing and profiling dataset…"):
        try:
            df_clean, prep_report = prepare_dataset(df, target=target)
            profile   = generate_profile(df_clean, target, run_clustering=True)
            issues    = detect_issues(profile, df_clean)
            questions = request_user_clarification(issues, profile)
            st.session_state.update(
                preflighted=True,
                df_prepared=df_clean,
                profile=profile,
                issues=issues,
                questions=questions,
                prep_report=prep_report,
                result=None,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

if not st.session_state.preflighted:
    st.stop()

profile     = st.session_state.profile
issues      = st.session_state.issues
questions   = st.session_state.questions
prep_report = st.session_state.get("prep_report")

# Data preparation summary
if prep_report and (prep_report.n_cols_dropped or prep_report.dropped_duplicate_rows
                    or prep_report.rare_classes):
    with st.expander("Data preparation", expanded=prep_report.n_cols_dropped > 0):
        st.caption(
            f"**Before:** {prep_report.rows_before:,} rows × {prep_report.cols_before} cols"
            f"  →  "
            f"**After:** {prep_report.rows_after:,} rows × {prep_report.cols_after} cols"
        )
        for line in prep_report.summary().splitlines():
            st.write(line)

# Sports domain badge
sc = getattr(profile, "sports_context", None)
if sc and sc.detected_domain == "sports":
    terms = ", ".join(sc.matched_terms[:8])
    st.success(
        f"**Sports dataset detected** — confidence {sc.confidence:.0%}  |  "
        f"matched terms: {terms}"
    )
elif sc and sc.detected_domain == "possible_sports":
    terms = ", ".join(sc.matched_terms[:5]) if sc.matched_terms else "none"
    st.warning(
        f"**Possibly sports-related** — confidence {sc.confidence:.0%}  |  "
        f"matched: {terms}"
    )
else:
    st.info("No sports vocabulary detected — running in general tabular ML mode.")

# Clusters
if profile.clusters:
    with st.expander(
        f"Natural groupings — {profile.clusters.n_clusters} clusters found "
        f"(silhouette = {profile.clusters.silhouette_score:.3f})"
    ):
        for summary in profile.clusters.cluster_summaries.values():
            st.write(f"- {summary}")

# Issues
_sev_label = {"high": "High", "medium": "Medium", "low": "Low"}
with st.expander(f"Data quality — {len(issues)} issue(s) detected"):
    if not issues:
        st.write("No issues detected.")
    for iss in issues:
        sev_text = _sev_label.get(iss.severity.value, iss.severity.value.title())
        col_lbl  = f" (`{iss.affected_column}`)" if iss.affected_column else ""
        label    = iss.issue_type.value.replace("_", " ").title()
        st.write(f"**[{sev_text}]** {label}{col_lbl} — {iss.description}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Clarification questions
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Step 4 — Clarification questions")

answer_map: dict = {}
if questions:
    st.caption(
        "Answer these to help the AI make better decisions. "
        "Leave blank to let the system decide automatically."
    )
    for i, q in enumerate(questions, 1):
        st.markdown(f"**Q{i}.** {q.question}")
        answer_map[q.question_id] = st.text_input(
            f"Answer {i}",
            key=f"ans_{q.question_id}",
            label_visibility="collapsed",
            placeholder="Type your answer here (optional)…",
        )
        st.divider()
else:
    st.success("No clarification needed — the system can proceed autonomously.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Step 5 — Run pipeline")

_mode = task_spec.mode if task_spec is not None else "predictive"

if _mode == "predictive" and not api_key:
    st.warning("Enter your Anthropic API key in the sidebar before running.")
    st.stop()

_btn_label = "Run Agentic Pipeline" if _mode == "predictive" else "Run Exploratory Analysis"
if st.button(_btn_label, type="primary"):
    from src.orchestrator import run_agentic_pipeline, run_exploratory_pipeline

    for q in (questions or []):
        ans = answer_map.get(q.question_id, "").strip()
        q.answer = ans or None

    figures_dir = str(Path(__file__).parent / "figures")
    Path(figures_dir).mkdir(exist_ok=True)

    _df_prepared = st.session_state.get("df_prepared")
    _df_for_pipeline = df if _df_prepared is None else _df_prepared

    _prog_bar = st.progress(0, text="Starting…")
    _prog_msg = st.empty()

    def _on_progress(fraction: float, message: str) -> None:
        pct = min(int(fraction * 100), 100)
        _prog_bar.progress(pct, text=f"{pct}%  —  {message}")
        _prog_msg.caption(message)

    if _mode == "exploratory":
        status = st.status("Running exploratory analysis…", expanded=True)
        with status:
            try:
                result = run_exploratory_pipeline(
                    df=_df_for_pipeline,
                    goal_text=goal_text,
                    figures_dir=figures_dir,
                    verbose=False,
                    ask_clarifications=False,
                    prefilled_questions=questions or [],
                    progress_callback=_on_progress,
                )
                st.session_state.result = result
                _prog_bar.progress(100, text="100%  —  Complete")
                _prog_msg.empty()
                status.update(label="Exploratory analysis complete", state="complete")
            except Exception as exc:
                status.update(label="Analysis failed", state="error")
                _prog_bar.empty()
                _prog_msg.empty()
                _msg = str(exc)
                if "credit balance is too low" in _msg or "insufficient_balance" in _msg:
                    st.error(
                        "**Insufficient Anthropic API credits.**\n\n"
                        "Top up at [console.anthropic.com/settings/billing]"
                        "(https://console.anthropic.com/settings/billing), "
                        "or clear the API key field to run without LLM assistance."
                    )
                else:
                    st.error(_msg)
                    st.exception(exc)
    else:
        status = st.status(
            f"Running {max_rounds} optimisation round(s)…",
            expanded=True,
        )
        with status:
            try:
                result = run_agentic_pipeline(
                    df=_df_for_pipeline,
                    target=target,
                    goal_text=goal_text,
                    max_rounds=max_rounds,
                    n_plans_per_round=n_plans,
                    cv=cv_folds,
                    api_key=api_key,
                    use_memory=use_memory,
                    ask_clarifications=False,
                    prefilled_questions=questions or [],
                    generate_report=True,
                    figures_dir=figures_dir,
                    verbose=False,
                    progress_callback=_on_progress,
                )
                st.session_state.result = result
                _prog_bar.progress(100, text="100%  —  Complete")
                _prog_msg.empty()
                status.update(label="Pipeline complete", state="complete")
            except Exception as exc:
                status.update(label="Pipeline failed", state="error")
                _prog_bar.empty()
                _prog_msg.empty()
                _msg = str(exc)
                if "credit balance is too low" in _msg or "insufficient_balance" in _msg:
                    st.error(
                        "**Insufficient Anthropic API credits.**\n\n"
                        "Top up at [console.anthropic.com/settings/billing]"
                        "(https://console.anthropic.com/settings/billing), "
                        "or clear the API key field to run in rule-based mode."
                    )
                elif "invalid_api_key" in _msg or "authentication" in _msg.lower():
                    st.error("**Invalid API key.** Check Settings — it should start with `sk-ant-`.")
                elif "rate_limit" in _msg.lower():
                    st.error("**Rate limit hit.** Wait a moment and try again.")
                else:
                    st.error(_msg)
                    st.exception(exc)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Results
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.result:
    st.stop()

result = st.session_state.result
st.divider()
st.subheader("Step 6 — Results")

from src.models.schemas import ExploratoryResult as _ExploratoryResult

# ── Visualisation helper ──────────────────────────────────────────────────────
_VIS_META = {
    "missingness":          (
        "Missing Data Heatmap",
        "Shows which columns have missing values and how many rows are affected. "
        "Darker cells indicate more missing data.",
    ),
    "feature_importance":   (
        "Feature Importance",
        "The most influential features for prediction, ranked by the model's internal "
        "importance score. Longer bars indicate stronger predictors.",
    ),
    "actual_vs_predicted":  (
        "Actual vs Predicted",
        "Each point represents one sample. Points on the diagonal are perfectly predicted. "
        "Spread around the line reflects prediction error.",
    ),
    "per_fold_performance": (
        "Per-Fold Cross-Validation Performance",
        "Score for each CV fold. Similar bars mean the model generalises consistently; "
        "large variation signals instability.",
    ),
    "correlation_matrix":   (
        "Feature Correlation Matrix",
        "How strongly pairs of features move together. "
        "Warm colours indicate positive correlation; cool colours indicate negative.",
    ),
    "target_distribution":  (
        "Target Distribution",
        "How the target variable is distributed across its values. "
        "Highly imbalanced distributions can make modelling harder.",
    ),
    "cluster_profiles":     (
        "Cluster Profiles",
        "Average feature values per discovered group, showing what makes each cluster distinctive.",
    ),
    "pca_clusters":         (
        "PCA Cluster View",
        "Discovered groups projected onto the two principal components. "
        "Well-separated regions indicate distinct natural groupings.",
    ),
}


def _render_visualisations(figures_dir: str) -> None:
    """Display all PNGs in figures_dir as a labelled, full-width gallery."""
    fig_files = sorted(Path(figures_dir).glob("*.png"))
    if not fig_files:
        st.info("No visualisations were generated for this run.")
        return
    for fig_path in fig_files:
        stem = fig_path.stem
        title, description = _VIS_META.get(
            stem, (stem.replace("_", " ").title(), "")
        )
        st.markdown(f"#### {title}")
        if description:
            st.caption(description)
        st.image(str(fig_path), use_container_width=True)
        st.divider()


if isinstance(result, _ExploratoryResult):
    # ── Exploratory results ───────────────────────────────────────────────────
    info_cols = st.columns(3)
    info_cols[0].metric("Rows analysed", f"{result.profile.n_rows:,}")
    info_cols[1].metric("Columns", result.profile.n_cols)
    if result.profile.clusters:
        info_cols[2].metric(
            "Natural clusters",
            result.profile.clusters.n_clusters,
            help=f"Silhouette score: {result.profile.clusters.silhouette_score:.3f}",
        )

    tab_report, tab_vis = st.tabs(["Report", "Visualisations"])

    with tab_report:
        if result.user_report:
            st.markdown(result.user_report)
        else:
            st.info("No report generated.")

    with tab_vis:
        _render_visualisations(str(Path(__file__).parent / "figures"))

else:
    # ── Predictive (RunResult) results ────────────────────────────────────────
    metrics = result.best_result.metric_values
    # Determine primary metric label for the composite score header
    _first_metric = next(iter(metrics), "score")
    _score_label  = {
        "r2":       "Best Score  (R²-based)",
        "f1_macro": "Best Score  (F1-based)",
        "rmse":     "Best Score  (RMSE-based)",
    }.get(_first_metric, "Best Score")

    metric_cols = st.columns(len(metrics) + 2)
    metric_cols[0].metric(
        _score_label,
        f"{result.best_result.score:.4f}",
        help="Composite score = primary metric − 0.5×CV_std − complexity penalty",
    )
    metric_cols[1].metric(
        "CV Std",
        f"{result.best_result.cv_std:.4f}",
        help="Standard deviation of the primary metric across folds — lower is more stable",
    )
    for i, (k, v) in enumerate(metrics.items()):
        try:
            metric_cols[i + 2].metric(k.upper().replace("_", " "), f"{float(v):.4f}")
        except (TypeError, ValueError):
            metric_cols[i + 2].metric(k.upper().replace("_", " "), str(v))

    converged_label = (
        "Converged to score threshold"
        if result.converged
        else f"Stopped after {result.n_iterations} round(s)"
    )
    st.caption(converged_label)

    if result.llm_calls:
        total_cost = sum(r.estimated_cost_usd for r in result.llm_calls)
        total_toks = sum(r.input_tokens + r.output_tokens for r in result.llm_calls)
        st.caption(
            f"LLM usage: {len(result.llm_calls)} calls  |  "
            f"{total_toks:,} tokens  |  estimated cost: ${total_cost:.4f}"
        )

    # Best plan
    plan        = result.best_plan
    explanation = plan.model_params.get("__explanation", "")
    with st.expander("Best pipeline configuration", expanded=True):
        cfg_df = pd.DataFrame.from_dict(
            {
                "Model":              plan.model,
                "Imputation":         plan.imputation,
                "Outlier handling":   plan.outlier_handling,
                "Encoding":           plan.encoding,
                "Scaling":            plan.scaling,
                "Imbalance strategy": plan.imbalance_strategy,
            },
            orient="index",
            columns=["Value"],
        )
        st.table(cfg_df)
        if explanation:
            st.info(explanation)

    # Tabs
    tab_report, tab_vis, tab_hist = st.tabs(["Report", "Visualisations", "Iteration history"])

    with tab_report:
        if result.user_report:
            st.markdown(result.user_report)
        else:
            st.info("No report generated.")

    with tab_vis:
        _render_visualisations(str(Path(__file__).parent / "figures"))

    with tab_hist:
        rows = []
        for rec in result.history:
            p, r = rec.plan, rec.result
            extra = {}
            for k, v in r.metric_values.items():
                label = k.upper().replace("_", " ")
                try:
                    extra[label] = round(float(v), 4)
                except (TypeError, ValueError):
                    extra[label] = str(v)
            rows.append({
                "Round":       rec.iteration,
                "Score":       round(r.score, 4),
                "CV Std":      round(r.cv_std, 4),
                "Model":       p.model,
                "Imputation":  p.imputation,
                "Encoding":    p.encoding,
                "Scaling":     p.scaling,
                **extra,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
