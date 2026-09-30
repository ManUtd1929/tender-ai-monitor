"""
Текущее состояние AI-пайплайна по каждому объявлению (таблица tender_pipeline_state) на sqlite3.

Одна строка на resource_url = ГДЕ тендер находится сейчас. Это состояние, а не история: история
расходов — ai_usage_events (append-only), результаты анализа — tender_triage /
tender_deep_analysis. Здесь только то, чему в них нет места: результат Commercial Gate, отложенные
состояния (бюджет / размер input), ошибки, escalation-кандидат и raw-ответ модели Deep.

reason_code — машинная причина (значения src.ai.escalation / commercial_gate / OpenAI error kind),
message — человекочитаемое пояснение. input_hash/model/prompt_version описывают ТУ стадию, на
которой состояние получено: по ним пайплайн узнаёт, что блокирующее условие изменилось
(изменился контекст, промпт или модель) и повторять вызов можно.
"""

import json
import logging
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.database import analysis_repository
from src.database.tender_repository import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

STATE_PENDING_TRIAGE = "pending_triage"
STATE_TRIAGE_DEFERRED_BUDGET = "triage_deferred_budget"
STATE_TRIAGE_ERROR = "triage_error"
STATE_TRIAGE_COMPLETED = "triage_completed"
STATE_STOPPED_NOT_RELEVANT = "stopped_not_relevant"
STATE_COMMERCIAL_GATE_PASS = "commercial_gate_pass"
STATE_COMMERCIAL_GATE_SKIP = "commercial_gate_skip"
STATE_DEEP_DEFERRED_BUDGET = "deep_deferred_budget"
STATE_DEEP_DEFERRED_INPUT = "deep_deferred_input"
STATE_DEEP_COMPLETED = "deep_completed"
STATE_DEEP_ERROR = "deep_error"
STATE_ESCALATION_CANDIDATE = "escalation_candidate"

STATES = (
    STATE_PENDING_TRIAGE, STATE_TRIAGE_DEFERRED_BUDGET, STATE_TRIAGE_ERROR, STATE_TRIAGE_COMPLETED,
    STATE_STOPPED_NOT_RELEVANT, STATE_COMMERCIAL_GATE_PASS, STATE_COMMERCIAL_GATE_SKIP,
    STATE_DEEP_DEFERRED_BUDGET, STATE_DEEP_DEFERRED_INPUT, STATE_DEEP_COMPLETED,
    STATE_DEEP_ERROR, STATE_ESCALATION_CANDIDATE,
)

CREATE_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS tender_pipeline_state (
    resource_url        TEXT PRIMARY KEY,
    state               TEXT NOT NULL,
    reason_code         TEXT,
    message             TEXT,
    input_hash          TEXT,
    model               TEXT,
    prompt_version      TEXT,
    error_kind          TEXT,
    escalation_reason   TEXT,
    details_json        TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    FOREIGN KEY (resource_url) REFERENCES announcements (resource_url)
)
"""

UPSERT_STATE = """
INSERT INTO tender_pipeline_state (
    resource_url, state, reason_code, message, input_hash, model, prompt_version,
    error_kind, escalation_reason, details_json, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(resource_url) DO UPDATE SET
    state = excluded.state, reason_code = excluded.reason_code, message = excluded.message,
    input_hash = excluded.input_hash, model = excluded.model, prompt_version = excluded.prompt_version,
    error_kind = excluded.error_kind, escalation_reason = excluded.escalation_reason,
    details_json = excluded.details_json, updated_at = excluded.updated_at
"""


def _resolve_path(db_path) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


@contextmanager
def _connect(db_path=None):
    with closing(sqlite3.connect(_resolve_path(db_path))) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            yield conn


def init_db(db_path=None) -> Path:
    """Таблицы announcements/enrichment/documents/анализа (analysis_repository) + tender_pipeline_state."""
    path = analysis_repository.init_db(db_path)
    with _connect(path) as conn:
        conn.execute(CREATE_STATE_TABLE)
    return path


def save_state(
    resource_url: str,
    state: str,
    *,
    reason_code: str | None = None,
    message: str | None = None,
    input_hash: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    error_kind: str | None = None,
    escalation_reason: str | None = None,
    details: dict | None = None,
    db_path=None,
) -> None:
    """Перезаписывает состояние тендера. ValueError — неизвестное состояние или нет объявления."""
    if state not in STATES:
        raise ValueError(f"Неизвестное состояние пайплайна: {state!r}")
    details_json = json.dumps(details or {}, sort_keys=True, ensure_ascii=False, default=str)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        if conn.execute("SELECT 1 FROM announcements WHERE resource_url = ?", (resource_url,)).fetchone() is None:
            raise ValueError(f"Объявление не найдено в announcements: {resource_url}")
        conn.execute(UPSERT_STATE, (
            resource_url, state, reason_code, message, input_hash, model, prompt_version,
            error_kind, escalation_reason, details_json, now,
        ))
    logger.info("Pipeline state: %s -> %s (%s)", resource_url, state, reason_code)


def get_state(resource_url: str, db_path=None) -> dict | None:
    """dict с колонками + распарсенным details_json под ключом "details"."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tender_pipeline_state WHERE resource_url = ?", (resource_url,)
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["details"] = json.loads(data["details_json"])
    return data


def count_by_state(db_path=None) -> dict:
    with _connect(db_path) as conn:
        return dict(conn.execute("SELECT state, COUNT(*) FROM tender_pipeline_state GROUP BY state").fetchall())
