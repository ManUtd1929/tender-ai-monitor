"""
Read-only evaluation workflow: выбрать реальные тендеры из SQLite и подготовить
компактный материал для ручной разметки (golden set) перед подключением LLM.

AI здесь ничего не решает — модуль только читает production-данные (через уже
существующие build_tender_context / build_triage_context) и формирует "болванки"
(evaluation case), в которых поле "expected" размечает человек. HTTP не используется.

Read-only гарантии:
    - кандидаты (get_evaluation_candidates) выбираются через отдельное SQLite-соединение
      в режиме "mode=ro" (см. _connect_read_only, тот же приём, что и
      document_repository._connect_read_only): записать через него нельзя, а
      несуществующий файл БД не создаётся;
    - модуль НИГДЕ не вызывает repository.init_db (CREATE TABLE) и НИГДЕ не вызывает
      save_*/write-функции; единственная запись, которую делает модуль — это экспорт
      JSON-файла evaluation dataset на диск (export_evaluation_dataset), сама БД не
      меняется;
    - построение полного tender_context для отобранных кандидатов переиспользует уже
      протестированные SELECT-only функции src.ai.tender_context /
      src.database.enrichment_repository / src.database.document_repository (они
      только читают; ничего не создают и не изменяют).

CLI (python -m src.ai.evaluation_dataset) по умолчанию печатает таблицу кандидатов;
не запускает src.monitor и не изменяет tenders.db / data/documents.
"""

import argparse
import hashlib
import json
import logging
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from src.ai import tender_context as tender_context_module
from src.database.tender_repository import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

DEFAULT_CLI_LIMIT = 20

EXPECTED_TEMPLATE_FIELDS = (
    "relevance_status",
    "opportunity_type",
    "category",
    "expected_reason",
    "notes",
)

SELECT_CANDIDATE_ROWS = """
SELECT
    announcements.resource_url AS resource_url,
    announcements.resource_type AS resource_type,
    announcements.source_section AS source_section,
    announcement_enrichment.procurement_type AS procurement_type,
    (SELECT COUNT(*) FROM announcement_cpv
        WHERE announcement_cpv.resource_url = announcements.resource_url) AS cpv_count,
    (SELECT COUNT(*) FROM announcement_documents
        WHERE announcement_documents.resource_url = announcements.resource_url) AS document_count
FROM announcements
JOIN announcement_enrichment
    ON announcement_enrichment.resource_url = announcements.resource_url
ORDER BY announcements.first_seen_at ASC, announcements.id ASC
"""


def _resolve_path(db_path) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


@contextmanager
def _connect_read_only(db_path=None):
    """Соединение mode=ro: записать через него нельзя; несуществующая БД не создаётся."""
    path = _resolve_path(db_path).resolve()
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
        yield conn


def _validate_limit(limit) -> None:
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError(f"limit должен быть положительным целым или None: {limit!r}")


# --------------------------------------------------------------------------
# candidate selection (diversity без keyword bias)
# --------------------------------------------------------------------------

def _diversity_key(row: dict) -> tuple:
    """
    Чисто структурные метаданные (никаких keyword/AI): resource_type, procurement_type,
    source_section и наличие CPV/документов. Используется только для разнообразия
    выборки, не для классификации релевантности.
    """
    return (
        row["resource_type"],
        row["procurement_type"],
        row["source_section"],
        row["cpv_count"] > 0,
        row["document_count"] > 0,
    )


def _diversified_order(rows: list) -> list:
    """
    Round-robin по группам _diversity_key в порядке первого появления группы; внутри
    группы порядок сохраняется (rows уже отсортированы по first_seen_at, id). Детерминировано
    при детерминированном входе.
    """
    groups: dict = {}
    group_order = []
    for row in rows:
        key = _diversity_key(row)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(row)

    result = []
    round_index = 0
    while len(result) < len(rows):
        progressed = False
        for key in group_order:
            group = groups[key]
            if round_index < len(group):
                result.append(group[round_index])
                progressed = True
        round_index += 1
        if not progressed:
            break
    return result


def _fetch_candidate_rows(db_path=None) -> list:
    with _connect_read_only(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(SELECT_CANDIDATE_ROWS).fetchall()
    return [dict(row) for row in rows]


def get_evaluation_candidates(db_path=None, limit: int | None = None) -> list:
    """
    Announcements, у которых уже есть enrichment (полный document extraction не
    обязателен — human evaluator должен видеть тендер, даже если документ ещё не
    обработан; document_coverage это отразит). Диверсифицирует выборку по структурным
    метаданным (см. _diversity_key), keyword-фильтров нет. Возвращает список
    tender_context (src.ai.tender_context.build_tender_context) в порядке отбора.
    limit=None — все кандидаты; limit — положительное целое, иначе ValueError.
    Read-only: список кандидатов читается через mode=ro соединение; полный
    tender_context каждого отобранного кандидата строится уже существующими
    SELECT-only функциями (см. модульный docstring).
    """
    _validate_limit(limit)

    rows = _fetch_candidate_rows(db_path)
    ordered = _diversified_order(rows)
    selected = ordered if limit is None else ordered[:limit]

    return [
        tender_context_module.build_tender_context(row["resource_url"], db_path=db_path)
        for row in selected
    ]


# --------------------------------------------------------------------------
# evaluation case format
# --------------------------------------------------------------------------

def _case_id(resource_url: str) -> str:
    """Стабильный/детерминированный идентификатор case: зависит только от resource_url
    (не от содержимого), чтобы разметка одного тендера не "переезжала" на новый case_id
    при обновлении объявления — для этого есть отдельный input_hash."""
    return "case-" + hashlib.sha256(resource_url.encode("utf-8")).hexdigest()[:16]


def build_evaluation_case(tender_context: dict) -> dict:
    """
    tender_context (build_tender_context) -> evaluation case для ручной разметки.
    "expected" — пустая болванка, которую заполняет человек; AI/heuristics её не
    трогают. Чистая функция: tender_context не изменяется.
    """
    announcement = tender_context["announcement"]
    enrichment = tender_context["enrichment"]
    resource_url = tender_context["resource_url"]

    return {
        "case_id": _case_id(resource_url),
        "resource_url": resource_url,
        "input_hash": tender_context_module.compute_input_hash(tender_context),
        "title": announcement.get("title"),
        "contracting_authority": enrichment.get("contracting_authority"),
        "procurement_type": enrichment.get("procurement_type"),
        "procedure_type": enrichment.get("procedure_type"),
        "cpv_codes": list(enrichment.get("cpv_codes") or []),
        "deadline_at": announcement.get("deadline_at"),
        "document_coverage": dict(tender_context["document_coverage"]),
        "triage_context": tender_context_module.build_triage_context(tender_context),
        "expected": {name: None for name in EXPECTED_TEMPLATE_FIELDS},
    }


def build_evaluation_dataset(db_path=None, limit: int | None = None) -> list:
    """get_evaluation_candidates + build_evaluation_case для каждого; read-only (см. выше)."""
    return [
        build_evaluation_case(tender_context)
        for tender_context in get_evaluation_candidates(db_path=db_path, limit=limit)
    ]


def export_evaluation_dataset(cases: list, path) -> Path:
    """
    Пишет cases в JSON-файл на диск (path). Это единственная запись, которую делает
    модуль, и она не затрагивает SQLite: cases уже построены в памяти.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8",
    )
    logger.info("Evaluation dataset экспортирован: %s (case: %d)", output_path, len(cases))
    return output_path


# --------------------------------------------------------------------------
# CLI (read-only по умолчанию)
# --------------------------------------------------------------------------

def _truncate(value, width: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def _format_table(cases_context: list) -> str:
    columns = (
        ("RESOURCE_URL", 45), ("TITLE", 35), ("PROC_TYPE", 12),
        ("AUTHORITY", 25), ("CPV", 12), ("DEADLINE", 20), ("COVERAGE", 10),
    )
    header = "  ".join(name.ljust(width) for name, width in columns)
    lines = [header, "-" * len(header)]

    for tender_context in cases_context:
        announcement = tender_context["announcement"]
        enrichment = tender_context["enrichment"]
        cpv = ",".join(item["code"] for item in enrichment.get("cpv_codes") or []) or "-"
        coverage = "complete" if tender_context["document_coverage"]["coverage_complete"] else "incomplete"
        values = (
            tender_context["resource_url"], announcement.get("title"),
            enrichment.get("procurement_type"), enrichment.get("contracting_authority"),
            cpv, announcement.get("deadline_at"), coverage,
        )
        lines.append("  ".join(_truncate(value, width).ljust(width) for value, (_, width) in zip(values, columns)))

    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ai.evaluation_dataset",
        description=(
            "Read-only просмотр/экспорт кандидатов для ручной разметки AI relevance "
            "(golden set перед подключением LLM). Никогда не изменяет tenders.db / "
            "data/documents и не вызывает src.monitor."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_CLI_LIMIT,
        help=f"Сколько кандидатов показать/экспортировать (по умолчанию {DEFAULT_CLI_LIMIT})",
    )
    parser.add_argument(
        "--resource-url", default=None,
        help="Показать/экспортировать один конкретный тендер вместо списка кандидатов",
    )
    parser.add_argument(
        "--export", default=None, metavar="PATH",
        help="Путь для экспорта JSON evaluation template (DB при этом не изменяется)",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Путь к SQLite БД (по умолчанию production data/tenders.db, read-only)",
    )
    return parser


def main(argv=None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.resource_url:
        tender_context = tender_context_module.build_tender_context(
            args.resource_url, db_path=args.db_path,
        )
        case = build_evaluation_case(tender_context)
        if args.export:
            path = export_evaluation_dataset([case], args.export)
            print(f"Экспортирован 1 case: {path}")
        else:
            print(json.dumps(case, ensure_ascii=False, indent=2, sort_keys=True))
        return

    candidates = get_evaluation_candidates(db_path=args.db_path, limit=args.limit)

    if args.export:
        cases = [build_evaluation_case(tender_context) for tender_context in candidates]
        path = export_evaluation_dataset(cases, args.export)
        print(f"Экспортировано {len(cases)} case(s): {path}")
        return

    if not candidates:
        print("Кандидатов не найдено (нет объявлений с enrichment).")
        return

    print(_format_table(candidates))
    print()
    print(f"Всего показано: {len(candidates)} (--limit {args.limit})")


if __name__ == "__main__":
    main()
