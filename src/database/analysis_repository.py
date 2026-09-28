"""
Production-хранилище результатов AI relevance-анализа (таблицы tender_triage и
tender_deep_analysis) на стандартном sqlite3.

Общий у модулей только файл БД; announcements, announcement_enrichment,
document_downloads/document_extractions здесь только читаются (проверка resource_url
и построение tender_context), не изменяются. init_db строится поверх
document_repository.init_db, потому что кандидатов и input_hash нельзя посчитать без
announcement + enrichment + документов.

result_json хранит полный validated JSON-результат (relevance_schema.validate_triage_result /
validate_deep_analysis_result); отдельные колонки (relevance_status, opportunity_type, ...)
дублируют его самые частые поля для быстрых запросов и для сравнения "что изменилось".

input_hash — src.ai.tender_context.compute_input_hash(tender_context); если он не совпадает
с последним сохранённым, объявление снова становится кандидатом на (пере)анализ
(get_triage_candidates / get_deep_analysis_candidates). Кандидаты вычисляются в Python:
для каждого объявления с enrichment строится tender_context и текущий input_hash — SQL
здесь только сужает множество объявлений, которые вообще стоит проверять.
"""

import json
import logging
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.ai import tender_context as tender_context_module
from src.database import document_repository, enrichment_repository
from src.database.tender_repository import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

STATUS_NEW = "new"
STATUS_UPDATED = "updated"
STATUS_EXISTING = "existing"

DEFAULT_PROVIDER = "fake"
DEFAULT_MODEL = "fake"
DEFAULT_PROMPT_VERSION = "v1"

TRIAGE_META_FIELDS = ("input_hash", "prompt_version", "provider", "model")
TRIAGE_RESULT_FIELDS = (
    "relevance_status",
    "opportunity_type",
    "category",
    "confidence",
    "reason",
    "requires_deep_analysis",
)
TRIAGE_COMPARED_FIELDS = TRIAGE_META_FIELDS + TRIAGE_RESULT_FIELDS + ("result_json",)

DEEP_COMPARED_FIELDS = TRIAGE_META_FIELDS + ("result_json",)

CREATE_TRIAGE_TABLE = """
CREATE TABLE IF NOT EXISTS tender_triage (
    resource_url TEXT PRIMARY KEY,
    input_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    relevance_status TEXT NOT NULL,
    opportunity_type TEXT NOT NULL,
    category TEXT,
    confidence TEXT NOT NULL,
    reason TEXT NOT NULL,
    requires_deep_analysis INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    analyzed_at TEXT NOT NULL,
    FOREIGN KEY (resource_url) REFERENCES announcements (resource_url)
)
"""

CREATE_DEEP_ANALYSIS_TABLE = """
CREATE TABLE IF NOT EXISTS tender_deep_analysis (
    resource_url    TEXT PRIMARY KEY,
    input_hash      TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    result_json     TEXT NOT NULL,
    analyzed_at     TEXT NOT NULL,
    FOREIGN KEY (resource_url) REFERENCES announcements (resource_url)
)
"""

SELECT_TRIAGE_COMPARED = (
    f"SELECT {', '.join(TRIAGE_COMPARED_FIELDS)} FROM tender_triage WHERE resource_url = ?"
)
INSERT_TRIAGE = f"""
INSERT INTO tender_triage (
    resource_url, {', '.join(TRIAGE_COMPARED_FIELDS)}, analyzed_at
) VALUES ({', '.join('?' * (len(TRIAGE_COMPARED_FIELDS) + 2))})
"""
UPDATE_TRIAGE = f"""
UPDATE tender_triage
SET {', '.join(f'{name} = ?' for name in TRIAGE_COMPARED_FIELDS)}, analyzed_at = ?
WHERE resource_url = ?
"""

SELECT_DEEP_COMPARED = (
    f"SELECT {', '.join(DEEP_COMPARED_FIELDS)} FROM tender_deep_analysis WHERE resource_url = ?"
)
INSERT_DEEP = f"""
INSERT INTO tender_deep_analysis (
    resource_url, {', '.join(DEEP_COMPARED_FIELDS)}, analyzed_at
) VALUES ({', '.join('?' * (len(DEEP_COMPARED_FIELDS) + 2))})
"""
UPDATE_DEEP = f"""
UPDATE tender_deep_analysis
SET {', '.join(f'{name} = ?' for name in DEEP_COMPARED_FIELDS)}, analyzed_at = ?
WHERE resource_url = ?
"""

# Кандидаты на triage: у объявления есть enrichment (документы не обязательны —
# видим tender даже если документ ещё не скачался, coverage это отразит).
SELECT_ENRICHED_RESOURCE_URLS = """
SELECT announcements.resource_url
FROM announcements
JOIN announcement_enrichment
    ON announcement_enrichment.resource_url = announcements.resource_url
ORDER BY announcements.first_seen_at ASC, announcements.id ASC
"""

# Кандидаты на deep analysis: последний triage — relevant или maybe.
SELECT_DEEP_ANALYSIS_TRIAGE_URLS = """
SELECT resource_url FROM tender_triage
WHERE relevance_status IN ('relevant', 'maybe')
ORDER BY analyzed_at ASC
"""


def _resolve_path(db_path) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect(db_path=None):
    """Соединение с foreign_keys = ON; commit/rollback и закрытие гарантированы."""
    path = _resolve_path(db_path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            yield conn


def init_db(db_path=None) -> Path:
    """
    Создаёт announcements/enrichment/document-таблицы (если их нет) и таблицы анализа.
    Модуль сам вызывает enrichment_repository.init_db: в отличие от document_repository
    (где JOIN announcement_enrichment есть только в кандидатах и ответственность за
    init_db несёт вызывающий код, см. monitor.run_monitor), analysis_repository JOIN'ит
    announcement_enrichment в собственном SELECT_ENRICHED_RESOURCE_URLS, поэтому таблица
    обязана существовать после его собственного init_db.
    """
    path = document_repository.init_db(db_path)
    enrichment_repository.init_db(path)

    with _connect(path) as conn:
        conn.execute(CREATE_TRIAGE_TABLE)
        conn.execute(CREATE_DEEP_ANALYSIS_TABLE)

    logger.info("Таблицы AI-анализа готовы: %s", path)
    return path


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _validate_limit(limit) -> None:
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError(f"limit должен быть положительным целым или None: {limit!r}")


def _validate_meta(resource_url, input_hash, provider, model, prompt_version) -> None:
    missing = [
        name for name, value in (
            ("resource_url", resource_url),
            ("input_hash", input_hash),
            ("provider", provider),
            ("model", model),
            ("prompt_version", prompt_version),
        )
        if _is_blank(value)
    ]
    if missing:
        raise ValueError(f"Отсутствуют обязательные поля: {', '.join(missing)}")


# --------------------------------------------------------------------------
# triage
# --------------------------------------------------------------------------

def save_triage(
    resource_url: str,
    input_hash: str,
    result: dict,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    db_path=None,
) -> str:
    """
    Сохраняет уже validated triage result (relevance_schema.validate_triage_result).
    Возвращает "new" / "updated" / "existing". ValueError — пустое обязательное поле
    или нет announcements.resource_url.
    """
    _validate_meta(resource_url, input_hash, provider, model, prompt_version)

    result_json = json.dumps(result, sort_keys=True, ensure_ascii=False)
    values = (
        input_hash, prompt_version, provider, model,
        result["relevance_status"], result["opportunity_type"], result["category"],
        result["confidence"], result["reason"], int(result["requires_deep_analysis"]),
        result_json,
    )
    now = _utc_now()

    with _connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM announcements WHERE resource_url = ?", (resource_url,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"Объявление не найдено в announcements: {resource_url}")

        row = conn.execute(SELECT_TRIAGE_COMPARED, (resource_url,)).fetchone()
        if row is None:
            conn.execute(INSERT_TRIAGE, (resource_url, *values, now))
            status = STATUS_NEW
        elif tuple(row) == values:
            status = STATUS_EXISTING
        else:
            conn.execute(UPDATE_TRIAGE, (*values, now, resource_url))
            status = STATUS_UPDATED

    logger.info("Triage сохранён (%s): %s -> %s", status, resource_url, result["relevance_status"])
    return status


def get_triage(resource_url: str, db_path=None) -> dict | None:
    """dict с колонками таблицы + распарсенным result_json под ключом "result"."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tender_triage WHERE resource_url = ?", (resource_url,)
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["requires_deep_analysis"] = bool(data["requires_deep_analysis"])
    data["result"] = json.loads(data["result_json"])
    return data


def count_triage(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM tender_triage").fetchone()[0]


# --------------------------------------------------------------------------
# deep analysis
# --------------------------------------------------------------------------

def save_deep_analysis(
    resource_url: str,
    input_hash: str,
    result: dict,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    db_path=None,
) -> str:
    """
    Сохраняет уже validated deep analysis result (relevance_schema.validate_deep_analysis_result).
    Возвращает "new" / "updated" / "existing". ValueError — пустое обязательное поле
    или нет announcements.resource_url.
    """
    _validate_meta(resource_url, input_hash, provider, model, prompt_version)

    result_json = json.dumps(result, sort_keys=True, ensure_ascii=False)
    values = (input_hash, prompt_version, provider, model, result_json)
    now = _utc_now()

    with _connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM announcements WHERE resource_url = ?", (resource_url,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"Объявление не найдено в announcements: {resource_url}")

        row = conn.execute(SELECT_DEEP_COMPARED, (resource_url,)).fetchone()
        if row is None:
            conn.execute(INSERT_DEEP, (resource_url, *values, now))
            status = STATUS_NEW
        elif tuple(row) == values:
            status = STATUS_EXISTING
        else:
            conn.execute(UPDATE_DEEP, (*values, now, resource_url))
            status = STATUS_UPDATED

    logger.info("Deep analysis сохранён (%s): %s", status, resource_url)
    return status


def get_deep_analysis(resource_url: str, db_path=None) -> dict | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tender_deep_analysis WHERE resource_url = ?", (resource_url,)
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["result"] = json.loads(data["result_json"])
    return data


def count_deep_analysis(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM tender_deep_analysis").fetchone()[0]


# --------------------------------------------------------------------------
# кандидаты
# --------------------------------------------------------------------------

def get_triage_candidates(db_path=None, limit: int | None = None) -> list:
    """
    Объявления с enrichment, для которых triage отсутствует или их текущий
    tender_context дал другой input_hash (документ/enrichment/announcement изменились).
    Каждый элемент: {resource_url, tender_context, input_hash}. limit применяется
    после фильтрации (сколько реально устаревших/новых кандидатов вернуть), поэтому
    объявления с этим кодом всегда просматриваются все, самые старые (first_seen_at) первыми.
    limit=None — все; limit — положительное целое, иначе ValueError.
    """
    _validate_limit(limit)

    with _connect(db_path) as conn:
        urls = [row[0] for row in conn.execute(SELECT_ENRICHED_RESOURCE_URLS).fetchall()]

    candidates = []
    for resource_url in urls:
        context = tender_context_module.build_tender_context(resource_url, db_path=db_path)
        input_hash = tender_context_module.compute_input_hash(context)
        existing = get_triage(resource_url, db_path=db_path)
        if existing is None or existing["input_hash"] != input_hash:
            candidates.append({
                "resource_url": resource_url, "tender_context": context, "input_hash": input_hash,
            })
            if limit is not None and len(candidates) >= limit:
                break
    return candidates


def count_triage_candidates(db_path=None) -> int:
    return len(get_triage_candidates(db_path=db_path))


def get_deep_analysis_candidates(db_path=None, limit: int | None = None) -> list:
    """
    Объявления, у которых последний triage relevant/maybe и (deep analysis отсутствует
    или их текущий tender_context дал другой input_hash). Каждый элемент: {resource_url,
    tender_context, input_hash, triage} (triage — result_json последнего triage).
    limit=None — все; limit — положительное целое, иначе ValueError.
    """
    _validate_limit(limit)

    with _connect(db_path) as conn:
        urls = [row[0] for row in conn.execute(SELECT_DEEP_ANALYSIS_TRIAGE_URLS).fetchall()]

    candidates = []
    for resource_url in urls:
        triage = get_triage(resource_url, db_path=db_path)
        if triage is None:
            continue
        context = tender_context_module.build_tender_context(resource_url, db_path=db_path)
        input_hash = tender_context_module.compute_input_hash(context)
        existing = get_deep_analysis(resource_url, db_path=db_path)
        if existing is None or existing["input_hash"] != input_hash:
            candidates.append({
                "resource_url": resource_url, "tender_context": context, "input_hash": input_hash,
                "triage": triage["result"],
            })
            if limit is not None and len(candidates) >= limit:
                break
    return candidates


def count_deep_analysis_candidates(db_path=None) -> int:
    return len(get_deep_analysis_candidates(db_path=db_path))
