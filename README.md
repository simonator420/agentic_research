# Agentic AI for Sports Analytics

An autonomous machine learning pipeline optimisation system designed for
**non-technical sports users** — coaches, performance analysts, scouts, and
federation staff.

The system profiles a sports dataset, detects data quality issues specific to
sports data (post-match leakage, playing-time missingness, identity columns),
asks the user plain-language clarification questions, and then iteratively
proposes and evaluates preprocessing + model configurations using
[Claude](https://www.anthropic.com/claude) as the reasoning agent.

---

## What it does

1. **Profiles the dataset** — column types, missing values, outliers, class imbalance, duplicates
2. **Detects sports domain** — recognises column names from StatsBomb, Wyscout, Opta, NBA API, Sofascore (e.g. `xG`, `MP`, `playerID`, `teamName`, `minutes_played`)
3. **Runs exploratory clustering** — surfaces natural groupings (player archetypes, match intensity tiers) before any predictive modelling
4. **Asks clarification questions** — e.g. "Is `goals_scored` recorded before or after the match you want to predict?"
5. **Optimises a scikit-learn pipeline** — the Claude-powered Planner proposes configurations; the Evaluator tests them via cross-validation; the loop refines until convergence
6. **Generates a plain-language report** — with visualisations (feature importance, confusion matrix, cluster projection) written for a non-technical sports audience

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

Create a `.env` file in the project root (already listed in `.gitignore` — it will never be committed):

```
ANTHROPIC_API_KEY=sk-ant-...
```

Or export it as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The browser opens at `http://localhost:8501`. Upload any CSV, pick the target column, answer the clarification questions, and click **Run Agentic Pipeline**.

### 4. Or run locally from the terminal

You can run the same pipeline without Streamlit:

```bash
python run_local.py data/titanic.csv \
  --goal "Predict whether a passenger survived" \
  --target Survived
```

The full agentic predictive pipeline requires `ANTHROPIC_API_KEY`. For a local
no-LLM smoke test, use one of the baselines:

```bash
python run_local.py data/titanic.csv --target Survived --baseline rule
```

Exploratory analysis does not require a target column:

```bash
python run_local.py data/titanic.csv \
  --mode exploratory \
  --goal "Find natural groups in the data"
```

### 5. Or use the notebook

```bash
jupyter notebook experiments.ipynb
```

Run cells top to bottom. The demo uses `data/titanic.csv`. For real sports data, change `DATA_PATH` and `TARGET` in cell 4.

---

## Recommended sports datasets

| Dataset | Link | Suggested target |
|---------|------|-----------------|
| StatsBomb Open Data | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) | shot outcome |
| NBA shot logs | [Kaggle](https://www.kaggle.com/dansbecker/nba-shot-logs) | shot_made_flag |
| FIFA player attributes | [Kaggle](https://www.kaggle.com/stefanoleone992/fifa-22-complete-player-dataset) | overall_rating |
| Injury/wellness data | [Kaggle](https://www.kaggle.com/search?q=sports+injury+dataset) | injured |

---

## Project structure

```
agentic_research/
│
├── app.py                         ← Streamlit web interface (run this)
├── experiments.ipynb              ← Jupyter notebook (full pipeline + ablation study)
├── requirements.txt
├── .env                           ← API key (NOT committed — create this yourself)
│
├── data/
│   └── titanic.csv                ← demo dataset (60 KB)
│
├── storage/                       ← auto-created at runtime
│   ├── runs.db                    ← SQLite: within-run attempt history
│   └── chroma_db/                 ← ChromaDB: cross-run memory
│
├── figures/                       ← auto-created: PNGs of charts and visualisations
│
├── src/
│   ├── models/
│   │   └── schemas.py             ← shared dataclasses (ActionPlan, RunResult, …)
│   │
│   ├── data/
│   │   └── loader.py              ← load_data(), split_data(), dataset_fingerprint()
│   │
│   ├── agents/
│   │   ├── profiler.py            ← Step 1: DataProfile + exploratory clustering
│   │   ├── sports_vocabulary.py   ← sports domain detection (synonym groups, abbreviation expansion)
│   │   ├── issue_detector.py      ← Step 2: detect quality issues + clarification questions
│   │   ├── planner.py             ← Step 3: Claude API → propose ActionPlans
│   │   ├── executor.py            ← Step 4: build sklearn Pipeline from ActionPlan
│   │   └── evaluator.py           ← Step 5: cross-validate, visualise, build user report
│   │
│   ├── memory/
│   │   ├── run_store.py           ← SQLite store (within-run history)
│   │   └── vector_store.py        ← ChromaDB store (cross-run warm-start memory)
│   │
│   ├── baselines/
│   │   ├── rule_based.py          ← static heuristic pipeline (no LLM)
│   │   └── search_based.py        ← Optuna TPE random search (no LLM)
│   │
│   └── orchestrator.py            ← ties everything together; run_agentic_pipeline()
│
└── tests/                         ← 147 pytest tests (all pass without an API key)
    ├── test_profiler.py
    ├── test_issue_detector.py
    ├── test_planner.py
    ├── test_executor.py
    ├── test_evaluator.py
    ├── test_memory.py
    ├── test_orchestrator.py
    ├── test_baselines.py
    └── test_loader.py
```

---

## How the agentic loop works

```
Dataset (CSV)
  │
  ▼
Profiler ──────────────────────────────────────────────────────────────────┐
  │  DataProfile + sports domain detection + cluster patterns              │
  ▼                                                                         │
Issue Detector                                                              │
  │  List[Issue]  +  clarification questions for the user                  │
  ▼                                                                         │
User answers clarification questions  (Streamlit UI or terminal)           │
  │                                                                         │
  ▼                                                                         │
┌─────────────────────────────────────────────────────────────────────┐    │
│  Agentic loop  (max_rounds iterations)                              │    │
│                                                                     │    │
│  Planner (Claude API) ◄── attempt history + user answers           │    │
│      │                ◄── cross-run memory (ChromaDB)              │    │
│      │  List[ActionPlan]  with plain-language explanations         │    │
│      ▼                                                              │    │
│  Executor  →  scikit-learn Pipeline (one per plan)                  │    │
│      ▼                                                              │    │
│  Evaluator →  cross-validation → composite score                   │    │
│      │                                                              │    │
│      └── best plan saved to SQLite + fed back to Planner ──────────┘    │
└─────────────────────────────────────────────────────────────────────┘    │
  │                                                                         │
  ▼                                                                         │
Best pipeline re-fitted on full training set                               │
Best plan stored in ChromaDB (warm-start for next run) ◄───────────────────┘
  │
  ▼
RunResult  +  plain-language report  +  visualisations
```

**Stopping criteria:**
- `score_threshold` reached → converged, stop immediately
- No improvement for 2 consecutive rounds (plateau) → stop early
- `max_rounds` exhausted → stop

---

## Sports domain detection

The `sports_vocabulary` module normalises column names and matches them against
synonym groups organised by domain:

| Category | Examples detected |
|----------|------------------|
| Performance / post-match | `goals_scored`, `xG`, `xga`, `possession`, `PTS`, `REB` |
| Playing time | `minutes_played`, `MP`, `appearances`, `starts` |
| Injury / wellness | `injury_status`, `return_to_play`, `wellness_score`, `hrv` |
| Identity columns | `player_id`, `teamName`, `match_id`, `playerID` |
| Physical attributes | `height`, `weight`, `sprint_speed`, `vo2max` |
| Workload metrics | `training_load`, `ACWR`, `session_rpe`, `HSR` |

Abbreviation expansion handles provider-specific shorthand:
`xG → expected goals`, `MP → minutes played`, `PTS → points`, `AST → assists`

Confidence thresholds:
- ≥ 5% of columns matched → **sports** (apply full sports logic)
- 2–5% → **possible_sports** (ask user to confirm)
- < 2% → **general_tabular** (run without sports-specific warnings)

---

## Ablation study

Five variants run on the same dataset for comparison:

| # | Variant | Code |
|---|---------|------|
| 1 | Rule-based baseline | `run_rule_based(df, target)` |
| 2 | Search-based baseline (Optuna, 50 trials) | `run_search_based(df, target, n_trials=50)` |
| 3 | Agentic — no memory | `run_agentic_pipeline(..., use_memory=False)` |
| 4 | Agentic — no feedback loop | `run_agentic_pipeline(..., max_rounds=1)` |
| 5 | Warm-start experiment | Two sequential runs; compare `n_iterations` |
| 6 | Full agentic system | `run_agentic_pipeline(...)` |

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_rounds` | `3` | Maximum Planner → Evaluate iterations |
| `n_plans_per_round` | `3` | Candidate pipelines per round |
| `cv` | `5` | Cross-validation folds |
| `score_threshold` | `0.90` | Early-stop threshold |
| `use_memory` | `True` | Disable ChromaDB for ablation |
| `ask_clarifications` | `True` | Collect user answers via terminal |
| `prefilled_questions` | `None` | Pre-answered questions from Streamlit UI |

---

## Composite score formula

Used by all variants for direct comparison:

```
score = primary_metric − 0.5 × cv_std − 0.01 × n_pipeline_steps
```

- `primary_metric`: F1 (classification) or −RMSE (regression) — higher is always better
- `cv_std`: standard deviation across folds (penalises instability)
- `n_pipeline_steps`: number of Pipeline steps (penalises unnecessary complexity)

---

## Running the tests

```bash
pytest                              # all 147 tests
pytest tests/test_orchestrator.py -v
pytest tests/test_baselines.py -v
```

All tests pass without an API key — the Planner (Claude) is mocked in the relevant tests.
