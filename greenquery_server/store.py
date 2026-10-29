"""SQLite persistence for jobs and benchmark records (archive + live)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from greenquery_server import settings

# Archived snapshot file -> dashboard dataset label (mirrors dashboard/src/data/benchmarks.ts).
ARCHIVE_FILES = {
    "animals_query_metrics.json": "Animals",
    "artwork_query_metrics.json": "Artwork",
    "medical_query_metrics.json": "Medical",
    "medical_online_query_metrics.json": "Medical",
    "movie_query_metrics.json": "Movie",
    "movie_local_query_metrics.json": "Movie",
    "rotowire_query_metrics.json": "RotoWire",
    "rotowire_online_query_metrics.json": "RotoWire",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status TEXT NOT NULL,
    step TEXT,
    spec TEXT NOT NULL,
    cost_estimate_usd REAL,
    validation_status TEXT,
    record_count INTEGER DEFAULT 0,
    error TEXT
);
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    job_id TEXT,
    dataset TEXT NOT NULL,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS records_source ON records(source);
CREATE TABLE IF NOT EXISTS challenge_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    name TEXT NOT NULL,
    query TEXT NOT NULL,
    system TEXT NOT NULL,
    model TEXT NOT NULL,
    energy_wh REAL NOT NULL,
    quality REAL NOT NULL,
    mode TEXT NOT NULL,
    job_id TEXT
);
CREATE INDEX IF NOT EXISTS challenge_query ON challenge_entries(query);
"""


class Store:
    def __init__(self, path: Path = settings.DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
            if "slurm_job_id" not in columns:  # migrate databases created before SLURM support
                self._conn.execute("ALTER TABLE jobs ADD COLUMN slurm_job_id TEXT")
            self._conn.commit()

    # ---------------------------------------------------------------- archive
    def import_archive(self, archive_dir: Path = settings.ARCHIVE_DIR) -> int:
        """Load the committed snapshot once; later starts keep the existing rows."""
        with self._lock:
            if self._conn.execute("SELECT 1 FROM records WHERE source='archive' LIMIT 1").fetchone():
                return 0
            count = 0
            now = time.time()
            for name, dataset in ARCHIVE_FILES.items():
                path = archive_dir / name
                if not path.exists():
                    continue
                for record in json.loads(path.read_text()):
                    self._conn.execute(
                        "INSERT INTO records(source, job_id, dataset, created_at, payload) VALUES (?,?,?,?,?)",
                        ("archive", None, dataset, now, json.dumps(record)),
                    )
                    count += 1
            self._conn.commit()
            return count

    # ---------------------------------------------------------------- records
    def add_records(self, job_id: str, dataset: str, records: Iterable[Dict[str, Any]]) -> int:
        with self._lock:
            now = time.time()
            rows = [("live", job_id, dataset, now, json.dumps(record)) for record in records]
            self._conn.executemany(
                "INSERT INTO records(source, job_id, dataset, created_at, payload) VALUES (?,?,?,?,?)", rows
            )
            self._conn.commit()
            return len(rows)

    def records(self, source: str = "all", job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT id, source, job_id, dataset, created_at, payload FROM records"
        clauses, params = [], []
        if source != "all":
            clauses.append("source = ?")
            params.append(source)
        if job_id:
            clauses.append("job_id = ?")
            params.append(job_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY id", params).fetchall()
        return [
            {**json.loads(row["payload"]), "record_id": row["id"], "source": row["source"],
             "job_id": row["job_id"], "dataset": row["dataset"], "created_at": row["created_at"]}
            for row in rows
        ]

    # ------------------------------------------------------------------- jobs
    # ------------------------------------------------------------ challenge
    def add_challenge_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT INTO challenge_entries (created_at, name, query, system, model, energy_wh, quality, mode, job_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), entry["name"], entry["query"], entry["system"], entry["model"], entry["energy_wh"],
                 entry["quality"], entry["mode"], entry.get("job_id")),
            )
        return {**entry, "id": cursor.lastrowid}

    def challenge_board(self, query: str, min_quality: float, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM challenge_entries WHERE query = ? AND quality >= ? ORDER BY energy_wh ASC LIMIT ?",
                (query, min_quality, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear_challenge(self) -> int:
        with self._lock, self._conn:
            return self._conn.execute("DELETE FROM challenge_entries").rowcount

    def create_job(self, job_id: str, spec: Dict[str, Any], cost_estimate: Optional[float]) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs(id, created_at, updated_at, status, step, spec, cost_estimate_usd) VALUES (?,?,?,?,?,?,?)",
                (job_id, now, now, "queued", None, json.dumps(spec), cost_estimate),
            )
            self._conn.commit()

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id))
            self._conn.commit()

    def job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_dict(row) if row else None

    def jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._job_dict(row) for row in rows]

    def mark_interrupted(self) -> List[Dict[str, Any]]:
        """Fail local jobs left running by a previous server; return SLURM jobs to resume watching."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='failed', error='Server restarted while the job was running', updated_at=? "
                "WHERE status IN ('queued', 'running') AND slurm_job_id IS NULL",
                (time.time(),),
            )
            self._conn.commit()
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') AND slurm_job_id IS NOT NULL"
            ).fetchall()
        return [self._job_dict(row) for row in rows]

    @staticmethod
    def _job_dict(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["spec"] = json.loads(data["spec"])
        return data
