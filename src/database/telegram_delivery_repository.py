"""
Durable состояние Telegram-доставки (таблица telegram_deliveries) на sqlite3.

Одна строка на (resource_url, analysis_input_hash) = одна версия анализа тендера: UNIQUE не даёт
отправить одну и ту же версию дважды, а изменившийся тендер (новый input_hash нового deep result)
получает новую строку. Статусы: pending (попытка начата), sent, failed. Строка sent никогда не
переводится обратно (begin_attempt для неё не вызывается).

Из других таблиц только читается список deep_completed (tender_pipeline_state).
"""

import logging
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.database import pipeline_state_repository
from src.database.tender_repository import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_SENT = "sent"
STATUS_FAILED = "failed"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS telegram_deliveries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_url        TEXT NOT NULL,
    analysis_input_hash TEXT NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('pending', 'sent', 'failed')),
    telegram_message_id INTEGER,
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    last_error          TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    sent_at             TEXT,
    UNIQUE (resource_url, analysis_input_hash),
    FOREIGN KEY (resource_url) REFERENCES announcements (resource_url)
)
"""

BEGIN_ATTEMPT = """
INSERT INTO telegram_deliveries (
    resource_url, analysis_input_hash, status, attempt_count, created_at, updated_at
) VALUES (?, ?, 'pending', 1, ?, ?)
ON CONFLICT(resource_url, analysis_input_hash) DO UPDATE SET
    status = 'pending', attempt_count = attempt_count + 1, updated_at = excluded.updated_at
WHERE status != 'sent'
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect(db_path=None):
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            yield conn


def init_db(db_path=None) -> Path:
    """Все таблицы пайплайна (announcements ... tender_pipeline_state) + telegram_deliveries."""
    path = pipeline_state_repository.init_db(db_path)
    with _connect(path) as conn:
        conn.execute(CREATE_TABLE)
    return path


def list_deep_completed(db_path=None) -> list[dict]:
    """[{resource_url, input_hash}] тендеров в состоянии deep_completed, старые первыми."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT resource_url, input_hash FROM tender_pipeline_state WHERE state = ? ORDER BY updated_at, resource_url",
            (pipeline_state_repository.STATE_DEEP_COMPLETED,),
        ).fetchall()
    return [{"resource_url": url, "input_hash": input_hash} for url, input_hash in rows]


def get_delivery(resource_url: str, analysis_input_hash: str, db_path=None) -> dict | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM telegram_deliveries WHERE resource_url = ? AND analysis_input_hash = ?",
            (resource_url, analysis_input_hash),
        ).fetchone()
    return dict(row) if row else None


def get_delivery_by_id(delivery_id: int, db_path=None) -> dict | None:
    """Строка доставки по telegram_deliveries.id (используется callback worker'ом)."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM telegram_deliveries WHERE id = ?", (delivery_id,)).fetchone()
    return dict(row) if row else None


def begin_attempt(resource_url: str, analysis_input_hash: str, db_path=None) -> bool:
    """pending + attempt_count+1. False, если версия уже sent (повторная отправка запрещена)."""
    now = _utc_now()
    with _connect(db_path) as conn:
        changed = conn.execute(BEGIN_ATTEMPT, (resource_url, analysis_input_hash, now, now)).rowcount
    return changed > 0


def mark_sent(resource_url: str, analysis_input_hash: str, message_id: int | None, db_path=None) -> None:
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE telegram_deliveries SET status = 'sent', telegram_message_id = ?, last_error = NULL, "
            "sent_at = ?, updated_at = ? WHERE resource_url = ? AND analysis_input_hash = ?",
            (message_id, now, now, resource_url, analysis_input_hash),
        )


def mark_failed(resource_url: str, analysis_input_hash: str, error: str, db_path=None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE telegram_deliveries SET status = 'failed', last_error = ?, updated_at = ? "
            "WHERE resource_url = ? AND analysis_input_hash = ? AND status != 'sent'",
            (error, _utc_now(), resource_url, analysis_input_hash),
        )
