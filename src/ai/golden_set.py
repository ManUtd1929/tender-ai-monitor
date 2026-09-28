"""
Компактный ручной golden set (procurement-only MVP) для оценки AI relevance-анализа.

В golden set хранятся только выбранные реальные кейсы и человеческие expected labels:
case_id, resource_url, input_hash, title, expected. triage_context НЕ копируется —
при необходимости он заново строится из БД через существующий слой
(src.ai.tender_context / src.ai.evaluation_dataset.build_evaluation_case). Сохранённый
input_hash сравнивается с текущим: если тендер изменился, кейс помечается как stale,
а сохранённый hash молча НЕ заменяется (пересмотр label — решение человека).

Golden labels задаёт только человек: здесь нет AI/HTTP и автоматической классификации.

Read-only гарантии:
    - БД читается только через существующие SELECT-only функции tender_context; перед
      этим проверяется, что файл БД существует (соединение mode=ro), поэтому
      несуществующая БД не создаётся; init_db/save_* не вызываются нигде;
    - единственная запись — JSON-файл golden set (save_golden_set), и только по явному
      вызову функции. CLI ничего не пишет: ни в БД, ни в golden set.

Правила валидации expected (глобальная relevance schema не менялась — используются её
enum'ы и validate_triage_result):
    - relevance_status и opportunity_type — из src.ai.relevance_schema;
    - для MVP opportunity_type только procurement / other_service / unrelated / unclear
      (logistics и logistics_and_procurement валидны в схеме, но не в этом golden set);
    - unclear допустим только с maybe (правило схемы);
    - category: непустая строка для procurement / other_service; null допустим только для
      unrelated / unclear, где категорию определить нечем;
    - expected_reason не пустой.

CLI: python -m src.ai.golden_set --show | --validate | --hydrate | --resource-url URL
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

from src.ai import evaluation_dataset
from src.ai import relevance_schema
from src.ai import tender_context as tender_context_module

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_SET_PATH = PROJECT_ROOT / "evaluation" / "golden_set_procurement.json"

REQUIRED_CASE_FIELDS = ("case_id", "resource_url", "input_hash", "title", "expected")
EXPECTED_FIELDS = evaluation_dataset.EXPECTED_TEMPLATE_FIELDS

# Procurement-only MVP: logistics / logistics_and_procurement остаются в глобальной схеме.
GOLDEN_OPPORTUNITY_TYPES = ("procurement", "other_service", "unrelated", "unclear")
CATEGORY_OPTIONAL_TYPES = ("unrelated", "unclear")

INPUT_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")

HASH_OK = "ok"
HASH_STALE = "stale"
HASH_MISSING = "missing"
HASH_ERROR = "error"


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def _is_blank(value) -> bool:
    return not isinstance(value, str) or not value.strip()


def _validate_expected(expected) -> list:
    """Список ошибок expected (пустой — всё в порядке)."""
    if not isinstance(expected, dict):
        return [f"expected должен быть dict: {expected!r}"]

    extra = sorted(set(expected) - set(EXPECTED_FIELDS))
    missing = [name for name in EXPECTED_FIELDS if name not in expected]
    if extra or missing:
        errors = []
        if extra:
            errors.append(f"expected: неизвестные поля {extra}")
        if missing:
            errors.append(f"expected: отсутствуют поля {missing}")
        return errors

    errors = []
    status = expected["relevance_status"]
    opportunity_type = expected["opportunity_type"]
    category = expected["category"]
    notes = expected["notes"]

    if status not in relevance_schema.RELEVANCE_STATUSES:
        errors.append(
            f"expected.relevance_status: недопустимое значение {status!r} "
            f"(ожидается одно из {relevance_schema.RELEVANCE_STATUSES})"
        )
    if opportunity_type not in relevance_schema.OPPORTUNITY_TYPES:
        errors.append(
            f"expected.opportunity_type: недопустимое значение {opportunity_type!r} "
            f"(ожидается одно из {relevance_schema.OPPORTUNITY_TYPES})"
        )
    elif opportunity_type not in GOLDEN_OPPORTUNITY_TYPES:
        errors.append(
            f"expected.opportunity_type={opportunity_type!r} не допускается в procurement-only "
            f"golden set (допустимо: {GOLDEN_OPPORTUNITY_TYPES})"
        )
    if _is_blank(expected["expected_reason"]):
        errors.append("expected.expected_reason должен быть непустой строкой")
    if notes is not None and not isinstance(notes, str):
        errors.append(f"expected.notes должен быть str или null: {notes!r}")

    if category is not None and _is_blank(category):
        errors.append(f"expected.category должен быть непустой строкой или null: {category!r}")
    elif category is None and opportunity_type in GOLDEN_OPPORTUNITY_TYPES \
            and opportunity_type not in CATEGORY_OPTIONAL_TYPES:
        errors.append(
            f"expected.category не может быть null при opportunity_type={opportunity_type!r} "
            f"(null допустим только для {CATEGORY_OPTIONAL_TYPES})"
        )

    if errors:
        return errors

    # Межполевые правила схемы (например unclear только с maybe) проверяет сама схема:
    # human label оборачивается в triage result с нейтральными служебными полями.
    try:
        relevance_schema.validate_triage_result({
            "relevance_status": status,
            "opportunity_type": opportunity_type,
            "category": category,
            "confidence": "high",
            "reason": expected["expected_reason"],
            "requires_deep_analysis": status != "not_relevant",
            "evidence": [],
        })
    except ValueError as error:
        errors.append(f"expected нарушает relevance schema: {error}")
    return errors


def _validate_case(case) -> list:
    if not isinstance(case, dict):
        return [f"кейс должен быть dict: {case!r}"]

    missing = [name for name in REQUIRED_CASE_FIELDS if name not in case]
    if missing:
        return [f"отсутствуют обязательные поля {missing}"]

    errors = []
    for name in ("case_id", "resource_url", "title"):
        if _is_blank(case[name]):
            errors.append(f"{name} должен быть непустой строкой: {case[name]!r}")
    input_hash = case["input_hash"]
    if not isinstance(input_hash, str) or not INPUT_HASH_PATTERN.fullmatch(input_hash):
        errors.append(f"input_hash должен быть sha256 hex (64 символа): {input_hash!r}")
    errors.extend(_validate_expected(case["expected"]))
    return errors


def validate_golden_set(cases) -> list:
    """
    Возвращает список всех найденных ошибок (строки); пустой список — golden set
    валиден. Проверяет структуру, обязательные поля, уникальность case_id и resource_url,
    expected labels (см. модульный docstring). Ничего не изменяет.
    """
    if not isinstance(cases, list):
        return [f"golden set должен быть списком кейсов: {type(cases).__name__}"]

    errors = []
    seen_ids: dict = {}
    seen_urls: dict = {}
    for index, case in enumerate(cases):
        label = f"кейс #{index}"
        if isinstance(case, dict) and isinstance(case.get("case_id"), str):
            label += f" ({case['case_id']})"
        errors.extend(f"{label}: {problem}" for problem in _validate_case(case))

        if not isinstance(case, dict):
            continue
        for field, seen in (("case_id", seen_ids), ("resource_url", seen_urls)):
            value = case.get(field)
            if not isinstance(value, str):
                continue
            if value in seen:
                errors.append(f"{label}: дубликат {field} {value!r} (уже в кейсе #{seen[value]})")
            else:
                seen[value] = index
    return errors


# --------------------------------------------------------------------------
# file I/O
# --------------------------------------------------------------------------

def load_golden_set(path=DEFAULT_GOLDEN_SET_PATH) -> list:
    """
    Читает JSON golden set (без валидации содержимого — см. validate_golden_set).
    FileNotFoundError — файла нет (не создаётся); ValueError — не JSON или не список.
    """
    golden_path = Path(path)
    text = golden_path.read_text(encoding="utf-8")
    try:
        cases = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{golden_path}: невалидный JSON: {error}") from error
    if not isinstance(cases, list):
        raise ValueError(f"{golden_path}: ожидается JSON-список кейсов, получено {type(cases).__name__}")
    return cases


def save_golden_set(cases: list, path=DEFAULT_GOLDEN_SET_PATH) -> Path:
    """
    Явное сохранение golden set. Невалидный набор не пишется (ValueError со всеми
    ошибками). Запись атомарная (temp-файл + replace), чтобы не испортить размеченный файл.
    """
    errors = validate_golden_set(cases)
    if errors:
        raise ValueError("Golden set невалиден:\n" + "\n".join(f"  - {error}" for error in errors))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cases, ensure_ascii=False, indent=2) + "\n"

    fd, temp_name = tempfile.mkstemp(dir=output_path.parent, prefix=output_path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(temp_name, output_path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise

    logger.info("Golden set сохранён: %s (кейсов: %d)", output_path, len(cases))
    return output_path


# --------------------------------------------------------------------------
# build / hydrate (read-only по отношению к БД)
# --------------------------------------------------------------------------

def _build_current_case(resource_url: str, db_path=None) -> dict:
    """
    Текущий evaluation case (с triage_context) из БД. Сначала проверяется, что БД можно
    открыть в режиме mode=ro: иначе обычный sqlite3.connect в репозиториях создал бы
    пустой файл БД. ValueError — тендера нет в БД.
    """
    with evaluation_dataset._connect_read_only(db_path):
        pass
    tender_context = tender_context_module.build_tender_context(resource_url, db_path=db_path)
    return evaluation_dataset.build_evaluation_case(tender_context)


def build_golden_case(resource_url: str, expected: dict, db_path=None) -> dict:
    """
    Компактный golden case для resource_url: case_id / input_hash / title берутся из
    текущего состояния БД теми же функциями, что и в evaluation dataset; expected —
    человеческий label. ValueError — тендера нет в БД или expected невалиден.
    Кейс не записывается никуда: сохранять — save_golden_set.
    """
    if not isinstance(expected, dict):
        raise ValueError(f"expected должен быть dict: {expected!r}")
    unknown = sorted(set(expected) - set(EXPECTED_FIELDS))
    if unknown:
        raise ValueError(f"expected: неизвестные поля {unknown}")
    # category и notes можно не указывать — тогда null; остальные поля обязательны.
    expected = {name: expected.get(name) for name in EXPECTED_FIELDS}

    current = _build_current_case(resource_url, db_path=db_path)
    case = {
        "case_id": current["case_id"],
        "resource_url": current["resource_url"],
        "input_hash": current["input_hash"],
        "title": current["title"],
        "expected": expected,
    }

    errors = _validate_case(case)
    if errors:
        raise ValueError(f"Невалидный golden case для {resource_url}: " + "; ".join(errors))
    return case


def hydrate_golden_case(case: dict, db_path=None) -> dict:
    """
    Заново строит текущий контекст для golden case и сравнивает hash. Возвращает:
        golden_case         — переданный кейс (не изменяется; hash не заменяется);
        stored_input_hash / current_input_hash;
        hash_match          — True/False: тендер не изменился / изменился (stale);
        current_case        — текущий evaluation case (в т.ч. triage_context) для
                              повторного просмотра тендера человеком.
    ValueError — кейс невалиден или тендера больше нет в БД.
    """
    errors = _validate_case(case)
    if errors:
        raise ValueError("Невалидный golden case: " + "; ".join(errors))

    current = _build_current_case(case["resource_url"], db_path=db_path)
    return {
        "golden_case": case,
        "stored_input_hash": case["input_hash"],
        "current_input_hash": current["input_hash"],
        "hash_match": case["input_hash"] == current["input_hash"],
        "current_case": current,
    }


def check_golden_set(cases: list, db_path=None) -> list:
    """
    Hydrate для каждого кейса; ошибка одного кейса (тендера нет в БД, БД недоступна) не
    останавливает остальные (правило проекта №13). Возвращает [{"case", "status", "detail"}],
    status: ok / stale / missing / error. current_case не включается — только статус.
    """
    results = []
    for case in cases:
        try:
            hydrated = hydrate_golden_case(case, db_path=db_path)
            status = HASH_OK if hydrated["hash_match"] else HASH_STALE
            detail = None if hydrated["hash_match"] else (
                f"stored {hydrated['stored_input_hash'][:12]} != current {hydrated['current_input_hash'][:12]}"
            )
        except ValueError as error:
            status, detail = HASH_MISSING, str(error)
        except (sqlite3.Error, OSError) as error:
            status, detail = HASH_ERROR, f"{type(error).__name__}: {error}"
        logger.info("Golden case %s: %s", case["case_id"], status)
        results.append({"case": case, "status": status, "detail": detail})
    return results


# --------------------------------------------------------------------------
# CLI (ничего не пишет: ни БД, ни golden set)
# --------------------------------------------------------------------------

def _truncate(value, width: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def _format_table(results: list) -> str:
    columns = (
        ("CASE_ID", 21), ("TITLE", 40), ("RELEVANCE", 12),
        ("OPPORTUNITY", 14), ("CATEGORY", 20), ("HASH", 8),
    )
    header = "  ".join(name.ljust(width) for name, width in columns)
    lines = [header, "-" * len(header)]
    for result in results:
        case = result["case"]
        expected = case["expected"]
        values = (
            case["case_id"], case["title"], expected["relevance_status"],
            expected["opportunity_type"], expected["category"] or "-", result["status"],
        )
        lines.append("  ".join(_truncate(value, width).ljust(width) for value, (_, width) in zip(values, columns)))
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ai.golden_set",
        description=(
            "Ручной procurement-only golden set. Read-only: не изменяет ни БД, ни файл "
            "golden set, не создаёт его, не использует AI/сеть."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--show", action="store_true", help="Таблица кейсов и статус hash")
    mode.add_argument("--validate", action="store_true", help="Валидировать JSON (exit 1 при ошибках)")
    mode.add_argument(
        "--hydrate", action="store_true",
        help="Сверить input_hash всех кейсов с текущей БД (exit 1, если есть stale/missing/error)",
    )
    mode.add_argument(
        "--resource-url", default=None, metavar="URL",
        help="Показать текущий evaluation case тендера, чтобы принять решение по label",
    )
    parser.add_argument(
        "--path", default=str(DEFAULT_GOLDEN_SET_PATH),
        help=f"Файл golden set (по умолчанию {DEFAULT_GOLDEN_SET_PATH})",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Путь к SQLite БД (по умолчанию production data/tenders.db, read-only)",
    )
    return parser


def _load_valid_golden_set(path) -> list | None:
    """Кейсы валидного файла или None (проблема уже напечатана)."""
    try:
        cases = load_golden_set(path)
    except FileNotFoundError:
        print(f"Файл golden set не найден: {path} (CLI его не создаёт)")
        return None
    except ValueError as error:
        print(f"Ошибка: {error}")
        return None

    errors = validate_golden_set(cases)
    if errors:
        print(f"Golden set невалиден ({len(errors)} ошибок):")
        for error in errors:
            print(f"  - {error}")
        return None
    return cases


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.resource_url:
        try:
            current = _build_current_case(args.resource_url, db_path=args.db_path)
        except (ValueError, sqlite3.Error) as error:
            print(f"Ошибка: {error}")
            return 1
        print(json.dumps(current, ensure_ascii=False, indent=2))
        return 0

    cases = _load_valid_golden_set(args.path)
    if cases is None:
        return 1

    if args.validate:
        print(f"Golden set валиден: {len(cases)} кейс(ов)")
        return 0

    results = check_golden_set(cases, db_path=args.db_path)

    if args.show:
        print(_format_table(results))
        print()
        print(f"Всего кейсов: {len(cases)}")
        return 0

    # --hydrate: только отчёт, JSON не переписывается.
    problems = [result for result in results if result["status"] != HASH_OK]
    for result in problems:
        print(f"{result['status'].upper():8} {result['case']['case_id']}  "
              f"{result['case']['resource_url']}  {result['detail']}")
    print(f"Проверено: {len(results)}, актуальных: {len(results) - len(problems)}, "
          f"требуют внимания: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    # Заголовки тендеров бывают не на языке консоли: не падать на UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
