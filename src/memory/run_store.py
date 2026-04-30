"""
run_store.py — SQLite-backed within-run memory.

Persists every AttemptRecord to disk during a run so that:
  - The Orchestrator can resume after a crash without losing iteration history.
  - Completed runs are available for post-hoc analysis and result logging.

Within-run history also lives in RAM as List[AttemptRecord] while the
Orchestrator is running — the SQLite store is the durable backup, not the
primary data structure used during the feedback loop.

Each run is identified by a run_id (UUID string). Multiple runs can coexist
in the same database file, which makes batch experiments easy to manage.

Public API
----------
RunStore(db_path)
    .save_attempt(run_id, record)  -> None
    .load_attempts(run_id)         -> List[AttemptRecord]
    .list_runs()                   -> List[str]
    .close()                       -> None
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.models.schemas import ActionPlan, AttemptRecord, EvaluationResult


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS attempts (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT    NOT NULL,
    iteration          INTEGER NOT NULL,
    plan_id            TEXT    NOT NULL,
    -- ActionPlan fields stored as individual columns for easy SQL querying
    imputation         TEXT    NOT NULL,
    outlier_handling   TEXT    NOT NULL,
    encoding           TEXT    NOT NULL,
    scaling            TEXT    NOT NULL,
    model              TEXT    NOT NULL,
    imbalance_strategy TEXT    NOT NULL,
    model_params       TEXT    NOT NULL,  -- JSON blob
    -- EvaluationResult fields
    score              REAL    NOT NULL,
    metric_values      TEXT    NOT NULL,  -- JSON blob
    cv_std             REAL    NOT NULL,
    runtime_secs       REAL    NOT NULL,
    created_at         TEXT    NOT NULL   -- ISO-8601 UTC timestamp
);
"""

_INSERT = """
INSERT INTO attempts (
    run_id, iteration, plan_id,
    imputation, outlier_handling, encoding, scaling, model,
    imbalance_strategy, model_params,
    score, metric_values, cv_std, runtime_secs, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class RunStore:
    """
    SQLite store for within-run attempt history.

    Designed as a context manager so the connection is always closed cleanly:

        with RunStore("runs.db") as store:
            store.save_attempt(run_id, record)
            history = store.load_attempts(run_id)
    """

    def __init__(self, db_path: str = "runs.db"):
        self.db_path = str(Path(db_path))
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        # Improve write throughput for batch experiment logging
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def save_attempt(self, run_id: str, record: AttemptRecord) -> None:
        """
        Persist one AttemptRecord to the database.

        Called by the Orchestrator immediately after each evaluation round
        so that partial results survive a crash or keyboard interrupt.
        """
        p = record.plan
        r = record.result
        self._conn.execute(_INSERT, (
            run_id,
            record.iteration,
            p.plan_id,
            p.imputation,
            p.outlier_handling,
            p.encoding,
            p.scaling,
            p.model,
            p.imbalance_strategy,
            json.dumps(p.model_params),
            r.score,
            json.dumps(r.metric_values),
            r.cv_std,
            r.runtime_secs,
            datetime.now(timezone.utc).isoformat(),
        ))
        self._conn.commit()

    def load_attempts(self, run_id: str) -> List[AttemptRecord]:
        """
        Retrieve all AttemptRecords for a given run, ordered by iteration.

        Used to resume a crashed run or to replay history for analysis.
        """
        cursor = self._conn.execute(
            "SELECT * FROM attempts WHERE run_id = ? ORDER BY iteration",
            (run_id,),
        )
        records = []
        for row in cursor.fetchall():
            plan = ActionPlan(
                plan_id=row["plan_id"],
                imputation=row["imputation"],
                outlier_handling=row["outlier_handling"],
                encoding=row["encoding"],
                scaling=row["scaling"],
                model=row["model"],
                imbalance_strategy=row["imbalance_strategy"],
                model_params=json.loads(row["model_params"]),
            )
            result = EvaluationResult(
                plan_id=row["plan_id"],
                score=row["score"],
                metric_values=json.loads(row["metric_values"]),
                cv_std=row["cv_std"],
                runtime_secs=row["runtime_secs"],
            )
            records.append(AttemptRecord(
                iteration=row["iteration"],
                plan=plan,
                result=result,
            ))
        return records

    def list_runs(self) -> List[str]:
        """Return the distinct run_ids stored in the database."""
        cursor = self._conn.execute(
            "SELECT DISTINCT run_id FROM attempts ORDER BY run_id"
        )
        return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
