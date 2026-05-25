"""
app.py — Streamlit interface for the Sports Analytics AI Pipeline.

Run with:
    streamlit run app.py
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sports Analytics AI Pipeline",
    page_icon="⚽",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Get yours at console.anthropic.com",
    )
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

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("⚽ Sports Analytics AI Pipeline")
st.caption(
    "Upload a sports dataset, pick a target column, and let the agentic system "
    "optimise a machine learning pipeline for you."
)

# ── Session state ─────────────────────────────────────────────────────────────
for _k in ("preflighted", "profile", "issues", "questions", "result", "df", "target"):
    if _k not in st.session_state:
        st.session_state[_k] = None

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("1. Upload dataset")

demo_dir = Path(__file__).parent / "data"
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
    st.info("Upload a CSV file or select a demo dataset to get started.")
    st.stop()

# Reset pre-flight if data changes
if st.session_state.df is not None and not df.equals(st.session_state.df):
    for _k in ("preflighted", "profile", "issues", "questions", "result"):
        st.session_state[_k] = None
st.session_state.df = df

st.success(f"Loaded **{len(df):,} rows × {len(df.columns)} columns**")
with st.expander("Preview (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Target column
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("2. Select target column")
target = st.selectbox("Column to predict", df.columns.tolist())
if target != st.session_state.target:
    for _k in ("preflighted", "profile", "issues", "questions", "result"):
        st.session_state[_k] = None
st.session_state.target = target

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Pre-flight analysis
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("3. Analyse dataset")

if st.button("Analyse", type="secondary", help="Profile the data and detect quality issues"):
    from src.agents.issue_detector import detect_issues, request_user_clarification
    from src.agents.profiler import generate_profile

    with st.spinner("Profiling dataset and running exploratory clustering…"):
        try:
            profile  = generate_profile(df, target, run_clustering=True)
            issues   = detect_issues(profile, df)
            questions = request_user_clarification(issues, profile)
            st.session_state.update(
                preflighted=True,
                profile=profile,
                issues=issues,
                questions=questions,
                result=None,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

if not st.session_state.preflighted:
    st.stop()

profile   = st.session_state.profile
issues    = st.session_state.issues
questions = st.session_state.questions

# ── Sports domain badge ───────────────────────────────────────────────────────
sc = getattr(profile, "sports_context", None)
if sc and sc.detected_domain == "sports":
    terms = ", ".join(sc.matched_terms[:8])
    st.success(
        f"🏈 **Sports dataset detected** — confidence {sc.confidence:.0%}  |  "
        f"terms: {terms}"
    )
elif sc and sc.detected_domain == "possible_sports":
    terms = ", ".join(sc.matched_terms[:5]) if sc.matched_terms else "none"
    st.warning(
        f"❓ **Possibly sports-related** — confidence {sc.confidence:.0%}  |  "
        f"matched: {terms}. Confirm in the clarification questions below."
    )
else:
    st.info("📊 No sports vocabulary detected — running in general tabular ML mode.")

# ── Cluster patterns ──────────────────────────────────────────────────────────
if profile.clusters:
    with st.expander(
        f"🔍 Exploratory clusters — **{profile.clusters.n_clusters}** natural groups "
        f"(silhouette = {profile.clusters.silhouette_score:.3f})"
    ):
        for summary in profile.clusters.cluster_summaries.values():
            st.write(f"• {summary}")

# ── Data quality issues ───────────────────────────────────────────────────────
_sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
with st.expander(f"⚠️ Data quality — **{len(issues)} issues** detected"):
    if not issues:
        st.write("No issues detected.")
    for iss in issues:
        icon = _sev_icon[iss.severity.value]
        col_lbl = f" (`{iss.affected_column}`)" if iss.affected_column else ""
        label   = iss.issue_type.value.replace("_", " ").title()
        st.write(f"{icon} **{label}**{col_lbl} — {iss.description}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Clarification questions (optional)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("4. Clarification questions")

answer_map: dict[str, str] = {}
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
# STEP 5 — Run
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("5. Run pipeline")

if not api_key:
    st.warning("Enter your Anthropic API key in the sidebar before running.")
    st.stop()

if st.button("🚀 Run Agentic Pipeline", type="primary"):
    from src.orchestrator import run_agentic_pipeline

    # Apply answers to pre-flight questions
    for q in (questions or []):
        ans = answer_map.get(q.question_id, "").strip()
        q.answer = ans or None

    figures_dir = str(Path(__file__).parent / "figures")
    Path(figures_dir).mkdir(exist_ok=True)

    status = st.status(
        f"Running {max_rounds} optimisation round(s) — this may take a few minutes…",
        expanded=True,
    )
    with status:
        try:
            result = run_agentic_pipeline(
                df=df,
                target=target,
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
            )
            st.session_state.result = result
            status.update(label="✅ Pipeline complete!", state="complete")
        except Exception as exc:
            status.update(label="❌ Pipeline failed", state="error")
            st.error(str(exc))
            st.exception(exc)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Results
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.result:
    st.stop()

result = st.session_state.result
st.divider()
st.subheader("Results")

# Score metrics row
metrics = result.best_result.metric_values
metric_cols = st.columns(len(metrics) + 2)
metric_cols[0].metric("Best Score", f"{result.best_result.score:.4f}")
metric_cols[1].metric(
    "CV Std", f"{result.best_result.cv_std:.4f}",
    help="Standard deviation across folds — lower is more stable",
)
for i, (k, v) in enumerate(metrics.items()):
    metric_cols[i + 2].metric(k.upper(), f"{v:.4f}")

converged_label = (
    "✅ Converged to score threshold"
    if result.converged
    else f"⏹ Stopped after {result.n_iterations} round(s)"
)
st.caption(converged_label)

# Best plan card
plan = result.best_plan
explanation = plan.model_params.get("__explanation", "")
with st.expander("🔧 Best pipeline configuration", expanded=True):
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
        st.info(f"💬 {explanation}")

# Tabs
tab_report, tab_vis, tab_hist = st.tabs(["📄 Report", "📊 Visualisations", "📈 Iteration history"])

with tab_report:
    if result.user_report:
        st.markdown(result.user_report)
    else:
        st.info("No report generated.")

with tab_vis:
    figures_dir = str(Path(__file__).parent / "figures")
    fig_files = sorted(Path(figures_dir).glob("*.png"))
    if fig_files:
        vis_cols = st.columns(2)
        for i, fig_path in enumerate(fig_files):
            vis_cols[i % 2].image(
                str(fig_path),
                caption=fig_path.stem.replace("_", " ").title(),
                use_container_width=True,
            )
    else:
        st.info("No visualisations were generated for this run.")

with tab_hist:
    rows = []
    for rec in result.history:
        p, r = rec.plan, rec.result
        rows.append({
            "Round": rec.iteration,
            "Score": round(r.score, 4),
            "CV Std": round(r.cv_std, 4),
            "Model": p.model,
            "Imputation": p.imputation,
            "Encoding": p.encoding,
            "Scaling": p.scaling,
            **{k.upper(): round(v, 4) for k, v in r.metric_values.items()},
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
