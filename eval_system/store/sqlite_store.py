"""
SQLite 存储实现。

用 JSON 列存放 list/dict 字段，换 Postgres 时可平滑升级为 jsonb。
所有写操作串行化（SQLite 单写者），骨架规模足够；大规模时换 Postgres。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Optional

from ..models import TestCase, TaskType, CaseResult, Run
from .base import Store


class SQLiteStore(Store):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        # check_same_thread=False：允许被 asyncio 线程池调用；用锁保证串行写。
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    input TEXT NOT NULL,
                    expected_answer TEXT,
                    rubric TEXT,
                    expected_tools TEXT,
                    tags TEXT,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_cases_dataset ON cases(dataset);

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    models TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    finished_at REAL,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);

                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    trajectory TEXT NOT NULL,
                    score REAL NOT NULL,
                    passed INTEGER NOT NULL,
                    scorer TEXT NOT NULL,
                    reason TEXT,
                    latency_ms REAL,
                    tokens INTEGER,
                    cost REAL,
                    error TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);

                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_configs (
                    name TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,        -- mock | openai
                    model TEXT,                -- 真实模型名（openai 用）
                    base_url TEXT,             -- API 端点（openai 用）
                    api_key_env TEXT,          -- 存放 key 的环境变量名（openai 用）
                    price_per_1k REAL DEFAULT 0,
                    quality REAL DEFAULT 0.8,  -- mock 用
                    is_judge INTEGER DEFAULT 0,-- 是否可作裁判
                    created_at REAL NOT NULL
                );
                """
            )
            self._conn.commit()

    # ---- 用例 ----
    def add_cases(self, cases: list[TestCase]) -> None:
        with self._lock:
            self._conn.executemany(
                """INSERT OR REPLACE INTO cases
                   (id, dataset, task_type, input, expected_answer, rubric,
                    expected_tools, tags, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        c.id, c.dataset, c.task_type.value, c.input,
                        c.expected_answer, c.rubric,
                        json.dumps(c.expected_tools), json.dumps(c.tags),
                        json.dumps(c.metadata),
                    )
                    for c in cases
                ],
            )
            self._conn.commit()

    def get_cases(self, dataset: str) -> list[TestCase]:
        rows = self._conn.execute(
            "SELECT * FROM cases WHERE dataset=?", (dataset,)
        ).fetchall()
        return [self._row_to_case(r) for r in rows]

    def delete_case(self, case_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM cases WHERE id=?", (case_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_datasets(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT dataset, COUNT(*) AS n FROM cases GROUP BY dataset"
        ).fetchall()
        return [{"name": r["dataset"], "case_count": r["n"]} for r in rows]

    @staticmethod
    def _row_to_case(r: sqlite3.Row) -> TestCase:
        return TestCase(
            id=r["id"], dataset=r["dataset"],
            task_type=TaskType(r["task_type"]), input=r["input"],
            expected_answer=r["expected_answer"], rubric=r["rubric"],
            expected_tools=json.loads(r["expected_tools"] or "[]"),
            tags=json.loads(r["tags"] or "[]"),
            metadata=json.loads(r["metadata"] or "{}"),
        )

    # ---- 运行 ----
    def create_run(self, run: Run) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO runs (id, dataset, models, status, created_at, finished_at, metadata)
                   VALUES (?,?,?,?,?,?,?)""",
                (run.id, run.dataset, json.dumps(run.models), run.status,
                 run.created_at, run.finished_at, json.dumps(run.metadata)),
            )
            self._conn.commit()

    def finish_run(self, run_id: str, status: str) -> None:
        import time
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status=?, finished_at=? WHERE id=?",
                (status, time.time(), run_id),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> Optional[Run]:
        r = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._row_to_run(r) if r else None

    def list_runs(self, limit: int = 50) -> list[Run]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    @staticmethod
    def _row_to_run(r: sqlite3.Row) -> Run:
        return Run(
            id=r["id"], dataset=r["dataset"], models=json.loads(r["models"]),
            status=r["status"], created_at=r["created_at"],
            finished_at=r["finished_at"], metadata=json.loads(r["metadata"] or "{}"),
        )

    # ---- 结果 ----
    def add_result(self, result: CaseResult) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO results
                   (id, run_id, case_id, model, dataset, task_type, trajectory,
                    score, passed, scorer, reason, latency_ms, tokens, cost, error, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    result.id, result.run_id, result.case_id, result.model,
                    result.dataset, result.task_type, json.dumps(result.trajectory),
                    result.score, int(result.passed), result.scorer, result.reason,
                    result.latency_ms, result.tokens, result.cost, result.error,
                    result.created_at,
                ),
            )
            self._conn.commit()

    def get_results(self, run_id: str) -> list[CaseResult]:
        rows = self._conn.execute(
            "SELECT * FROM results WHERE run_id=?", (run_id,)
        ).fetchall()
        return [
            CaseResult(
                id=r["id"], run_id=r["run_id"], case_id=r["case_id"],
                model=r["model"], dataset=r["dataset"], task_type=r["task_type"],
                trajectory=json.loads(r["trajectory"]), score=r["score"],
                passed=bool(r["passed"]), scorer=r["scorer"], reason=r["reason"],
                latency_ms=r["latency_ms"], tokens=r["tokens"], cost=r["cost"],
                error=r["error"], created_at=r["created_at"],
            )
            for r in rows
        ]

    # ---- 缓存 ----
    def cache_get(self, key: str) -> Optional[dict]:
        r = self._conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else None

    def cache_set(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?,?)",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    # ---- 设置（键值）----
    def get_setting(self, key: str) -> Optional[str]:
        r = self._conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
        return r["value"] if r else None

    def get_all_settings(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)",
                (key, str(value)),
            )
            self._conn.commit()

    # ---- 模型接入配置 ----
    def list_model_configs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM model_configs ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def add_model_config(self, cfg: dict) -> None:
        import time
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO model_configs
                   (name, kind, model, base_url, api_key_env, price_per_1k,
                    quality, is_judge, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    cfg["name"], cfg["kind"], cfg.get("model"),
                    cfg.get("base_url"), cfg.get("api_key_env"),
                    float(cfg.get("price_per_1k", 0)),
                    float(cfg.get("quality", 0.8)),
                    int(cfg.get("is_judge", 0)),
                    time.time(),
                ),
            )
            self._conn.commit()

    def delete_model_config(self, name: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM model_configs WHERE name=?", (name,)
            )
            self._conn.commit()
            return cur.rowcount > 0
