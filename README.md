# Agentic AutoML — Project Guide

Autonomous ML pipeline optimisation system using Claude as the Planner Agent.
The system profiles a tabular dataset, detects data quality issues, and iteratively
proposes and evaluates preprocessing + model configurations until a strong pipeline
is found.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Anthropic API key

Create a `.env` file in the project root (already listed in `.gitignore`):

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Add your dataset

Place any CSV file in the `data/` folder. A sample `titanic.csv` is included.

### 4. Open the notebook and run

```bash
jupyter notebook experiments.ipynb
```

Run cells top to bottom. Change `DATA_PATH` and `TARGET` in cell 4 to point at
your dataset.

---

## Project structure

```
agentic_research/
│
├── experiments.ipynb          ← main entry point (run this)
├── requirements.txt
├── .env                       ← API key (not committed)
│
├── data/                      ← put your CSV datasets here
│   └── titanic.csv
│
├── storage/                   ← auto-created at runtime
│   ├── runs.db                ← SQLite: within-run attempt history
│   └── chroma_db/             ← ChromaDB: cross-run memory
│
├── src/
│   ├── models/
│   │   └── schemas.py         ← all shared dataclasses (ActionPlan, RunResult, …)
│   │
│   ├── data/
│   │   └── loader.py          ← load_data(), split_data(), dataset_fingerprint()
│   │
│   ├── agents/
│   │   ├── profiler.py        ← Step 1: generate DataProfile from a DataFrame
│   │   ├── issue_detector.py  ← Step 2: detect data quality issues (missingness, outliers, …)
│   │   ├── planner.py         ← Step 3: call Claude API → propose ActionPlans
│   │   ├── executor.py        ← Step 4: build sklearn Pipeline from an ActionPlan
│   │   └── evaluator.py       ← Step 5: cross-validate pipelines, pick the best
│   │
│   ├── memory/
│   │   ├── run_store.py       ← SQLite store — saves every attempt within a run
│   │   └── vector_store.py    ← ChromaDB store — saves best plans across runs
│   │
│   ├── baselines/
│   │   ├── rule_based.py      ← static heuristic pipeline (no LLM, no search)
│   │   └── search_based.py    ← Optuna TPE random search (no LLM, no feedback)
│   │
│   └── orchestrator.py        ← ties everything together; call run_agentic_pipeline()
│
└── tests/                     ← pytest unit tests for every module
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
Dataset
  │
  ▼
Profiler ──────────────────────────────────────────────────────────────┐
  │  DataProfile (shape, dtypes, stats, class distribution)            │
  ▼                                                                     │
Issue Detector                                                          │
  │  List[Issue] (missingness / outliers / imbalance / leakage / …)    │
  ▼                                                                     │
┌─────────────────────────────────────────────────────────────────┐    │
│  Agentic loop  (max_rounds iterations)                          │    │
│                                                                 │    │
│  Planner (Claude API) ◄── history of past attempts             │    │
│      │                ◄── cross-run memory (ChromaDB)          │    │
│      │  List[ActionPlan]                                        │    │
│      ▼                                                          │    │
│  Executor  →  sklearn Pipeline (per plan)                       │    │
│      ▼                                                          │    │
│  Evaluator →  cross-validation → composite score               │    │
│      │                                                          │    │
│      └── best plan saved to SQLite + fed back to Planner ──────┘    │
└─────────────────────────────────────────────────────────────────┘    │
  │                                                                     │
  ▼                                                                     │
Best pipeline re-fitted on full training set                           │
Best plan stored in ChromaDB (warm-start for next run) ◄──────────────┘
  │
  ▼
RunResult  (best_pipeline, best_plan, metric_values, history, …)
```

**Stopping criteria** (checked after every round):
- `score_threshold` reached → `converged = True`, stop immediately
- No improvement for 2 consecutive rounds (plateau) → stop early
- `max_rounds` exhausted → stop, `converged = False`

---

## Notebook sections

| Cell | Section | What it does |
|------|---------|--------------|
| 1 | Setup | Adds project root to `sys.path`, loads `.env` |
| 2 | Imports | Loads all public functions |
| 4 | Load dataset | Reads CSV, shows shape and first rows |
| 6 | Profile & issues | Prints detected data quality issues |
| 8 | Run agentic pipeline | Runs the full optimisation loop |
| 10 | Results | Prints best plan, score, metrics |
| 12 | Iteration history | DataFrame of all tried configurations |
| 14 | Predict | Runs best pipeline on the test split |
| 16 | Inspect SQLite | Lists all stored runs and their scores |
| 18–25 | **Ablation study** | Runs all comparison variants (see below) |

---

## Ablation study (notebook section 8)

Five variants are run on the same dataset and compared in a summary table:

| # | Variant | How it's run |
|---|---------|--------------|
| 1 | Rule-based baseline | `run_rule_based(df, TARGET)` |
| 2 | Search-based baseline (Optuna) | `run_search_based(df, TARGET, n_trials=50)` |
| 3 | Agentic — no memory | `run_agentic_pipeline(..., use_memory=False)` |
| 4 | Agentic — no feedback loop | `run_agentic_pipeline(..., max_rounds=1)` |
| 5 | Warm-start experiment | Run full pipeline twice in sequence; compare `n_iterations` |
| 6 | Full agentic system | `run_agentic_pipeline(...)` (default settings) |

The final cell produces a summary DataFrame sorted by composite score.

---

## Key parameters of `run_agentic_pipeline`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_rounds` | `3` | Maximum Planner → Evaluate iterations |
| `n_plans_per_round` | `3` | Candidate pipelines proposed per round |
| `cv` | `5` | Cross-validation folds |
| `score_threshold` | `0.90` | Early-stop when this composite score is reached |
| `use_memory` | `True` | Set `False` to disable ChromaDB (ablation variant 3) |
| `verbose` | `True` | Print round-by-round progress |
| `random_state` | `42` | Seed for reproducible train/test splits |

---

## Composite score formula

All variants use the same formula so results are directly comparable:

```
score = primary_metric − 0.5 × cv_std − 0.01 × n_pipeline_steps
```

- `primary_metric`: F1 (binary/multiclass) or −RMSE (regression) — higher is always better
- `cv_std`: standard deviation across CV folds (penalises instability)
- `n_pipeline_steps`: number of steps in the sklearn Pipeline (penalises unnecessary complexity)

---

## Running the tests

```bash
# Run all tests
pytest

# Run a specific module
pytest tests/test_orchestrator.py -v

# Run just baseline tests
pytest tests/test_baselines.py -v
```

All 147 tests should pass without an API key — the Planner (Claude) is mocked
in the orchestrator and planner tests.

---

## Storage files

| File | Purpose | Created by |
|------|---------|------------|
| `storage/runs.db` | SQLite database; every `AttemptRecord` from every run | Orchestrator (auto) |
| `storage/chroma_db/` | ChromaDB vector store; best plans indexed by dataset fingerprint | Orchestrator (auto) |

Both paths are fixed relative to the project root regardless of where you launch
the notebook from. To start fresh, delete these files — they will be recreated
on the next run.
