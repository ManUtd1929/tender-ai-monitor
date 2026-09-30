"""
Ручной evaluator OpenAI triage на human-labeled procurement golden set.

Golden set (evaluation/golden_set_procurement.json) используется ТОЛЬКО здесь, как
эталон для сравнения: expected labels в запрос к модели не попадают (модель получает
только triage_context из БД). Reason/notes с expected не сравниваются дословно —
оцениваются relevance_status, opportunity_type и category.

Безопасность:
    - без явного режима ничего не запускается (--dry-run / --case-id / --all);
    - --dry-run не вызывает API и не создаёт client;
    - БД читается только через golden_set.hydrate_golden_case (mode=ro); golden set и БД
      не изменяются; единственная запись — JSON artifact в evaluation/runs/;
    - если input_hash кейса устарел (тендер изменился), API для него не вызывается;
    - ошибка одного кейса (API, validation, stale, отсутствие в БД) не останавливает
      остальные (правило проекта №13).

CLI:
    python -m src.ai.golden_set_evaluator --dry-run
    python -m src.ai.golden_set_evaluator --case-id <CASE_ID>
    python -m src.ai.golden_set_evaluator --all
    python -m src.ai.golden_set_evaluator --all --model gpt-5.6-luna     # benchmark другой модели

Benchmark другой модели (например Luna) идёт через тот же prompt (procurement-v2), ту же
Structured Outputs схему, grounding, retry и golden set — apples-to-apples, без model-specific
правил. Artifact хранит model, prompt_version, reasoning_effort, metrics, usage и
оценку стоимости USD по src.ai.pricing (pricing_version). Критерии допуска модели к production
triage (см. production_criteria): relevance_status 24/24, core 24/24, false negatives 0,
false positives 0, API/validation errors 0; category accuracy вторична. Если модель не проходит
core — production triage остаётся на Terra до отдельного разбора. Ledger evaluator не пишет.
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.ai import golden_set
from src.ai import openai_triage
from src.ai import preflight
from src.ai import pricing
from src.ai import triage_prompt
from src.ai.budget_settings import BudgetSettings

logger = logging.getLogger(__name__)

EVALUATOR_VERSION = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "evaluation" / "runs"

CORE_FIELDS = ("relevance_status", "opportunity_type", "category")
STATUS_ORDER = ("relevant", "maybe", "not_relevant")

RECORD_SCORED = "scored"
RECORD_ERROR = "error"
RECORD_SKIPPED = "skipped"

# error.kind для кейсов, где API не вызывался.
SKIP_STALE = "stale_golden_hash"
SKIP_MISSING = "missing_tender"
SKIP_HYDRATE_ERROR = "hydrate_error"
ERROR_UNEXPECTED = "unexpected_error"

MISMATCH_FALSE_NEGATIVE = "false_negative"
MISMATCH_FALSE_POSITIVE = "false_positive"
MISMATCH_MAYBE = "maybe_unclear"
MISMATCH_LABEL = "label_mismatch"


# --------------------------------------------------------------------------
# golden set loading / hydration
# --------------------------------------------------------------------------

def load_valid_cases(path) -> list:
    """Кейсы валидного golden set. ValueError/FileNotFoundError — файла нет или он невалиден."""
    cases = golden_set.load_golden_set(path)
    errors = golden_set.validate_golden_set(cases)
    if errors:
        raise ValueError("Golden set невалиден:\n" + "\n".join(f"  - {error}" for error in errors))
    return cases


def prepare_case(case: dict, db_path=None) -> dict:
    """
    Read-only hydrate кейса. {"status": ok|stale|missing|error, "detail", "triage_context"}.
    triage_context есть только при status=ok: stale/missing кейс к модели не идёт.
    """
    try:
        hydrated = golden_set.hydrate_golden_case(case, db_path=db_path)
    except ValueError as error:
        return {"status": golden_set.HASH_MISSING, "detail": str(error), "triage_context": None}
    except (sqlite3.Error, OSError) as error:
        return {
            "status": golden_set.HASH_ERROR, "detail": f"{type(error).__name__}: {error}",
            "triage_context": None,
        }
    if not hydrated["hash_match"]:
        return {
            "status": golden_set.HASH_STALE,
            "detail": (
                f"stored {hydrated['stored_input_hash'][:12]} != "
                f"current {hydrated['current_input_hash'][:12]}"
            ),
            "triage_context": None,
        }
    return {
        "status": golden_set.HASH_OK, "detail": None,
        "triage_context": hydrated["current_case"]["triage_context"],
    }


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def expected_core(case: dict) -> dict:
    return {name: case["expected"][name] for name in CORE_FIELDS}


def compare_prediction(expected: dict, prediction: dict) -> dict:
    """
    Per-field точное сравнение (reason не сравнивается). category сравнивается всегда
    (null == null — тоже совпадение для full accuracy), но category_evaluated=True только
    если expected category != null: именно такие кейсы входят в category accuracy.
    """
    checks = {name: prediction[name] == expected[name] for name in CORE_FIELDS}
    checks["core"] = checks["relevance_status"] and checks["opportunity_type"]
    checks["full"] = checks["core"] and checks["category"]
    checks["category_evaluated"] = expected["category"] is not None
    return checks


def classify_mismatch(expected: dict, prediction: dict) -> str | None:
    """Тип ошибки scored кейса или None, если все три поля совпали."""
    expected_status = expected["relevance_status"]
    predicted_status = prediction["relevance_status"]
    if expected_status == "relevant" and predicted_status == "not_relevant":
        return MISMATCH_FALSE_NEGATIVE
    if expected_status == "not_relevant" and predicted_status == "relevant":
        return MISMATCH_FALSE_POSITIVE
    if expected_status != predicted_status and "maybe" in (expected_status, predicted_status):
        return MISMATCH_MAYBE
    if not all(prediction[name] == expected[name] for name in CORE_FIELDS):
        return MISMATCH_LABEL
    return None


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def _base_record(case: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "resource_url": case["resource_url"],
        "input_hash": case["input_hash"],
        "status": None,
        "prediction": None,
        "expected": expected_core(case),
        "checks": None,
        "mismatch": None,
        "usage": None,
        "response_id": None,
        "attempts": None,
        "error": None,
    }


def _call_analyzer(analyzer, triage_context: dict) -> dict:
    with_metadata = getattr(analyzer, "triage_with_metadata", None)
    if with_metadata is not None:
        return with_metadata(triage_context)
    return {"result": analyzer.triage(triage_context), "usage": None, "response_id": None, "attempts": None}


def evaluate_case(case: dict, analyzer, db_path=None) -> dict:
    """
    Один golden case -> record. Не бросает исключений (кроме KeyboardInterrupt/SystemExit):
    любая ошибка превращается в record со status error/skipped и error {kind, message}.
    """
    record = _base_record(case)
    prepared = prepare_case(case, db_path=db_path)
    if prepared["status"] != golden_set.HASH_OK:
        kind = {
            golden_set.HASH_STALE: SKIP_STALE,
            golden_set.HASH_MISSING: SKIP_MISSING,
        }.get(prepared["status"], SKIP_HYDRATE_ERROR)
        logger.warning("Golden case %s пропущен без вызова API: %s (%s)", case["case_id"], kind, prepared["detail"])
        record["status"] = RECORD_SKIPPED
        record["error"] = {"kind": kind, "message": prepared["detail"]}
        return record

    try:
        outcome = _call_analyzer(analyzer, prepared["triage_context"])
    except openai_triage.TriageError as error:
        logger.error("Golden case %s: %s: %s", case["case_id"], error.kind, error)
        record["status"] = RECORD_ERROR
        record["error"] = {"kind": error.kind, "message": str(error)}
        record["usage"] = error.usage
        record["response_id"] = error.response_id
        record["attempts"] = error.attempts
        return record
    except Exception as error:
        logger.exception("Golden case %s: неожиданная ошибка", case["case_id"])
        record["status"] = RECORD_ERROR
        record["error"] = {"kind": ERROR_UNEXPECTED, "message": f"{type(error).__name__}: {error}"}
        return record

    prediction = outcome["result"]
    record["status"] = RECORD_SCORED
    record["prediction"] = prediction
    record["checks"] = compare_prediction(record["expected"], prediction)
    record["mismatch"] = classify_mismatch(record["expected"], prediction)
    record["usage"] = outcome.get("usage")
    record["response_id"] = outcome.get("response_id")
    record["attempts"] = outcome.get("attempts")
    return record


def run_evaluation(cases: list, analyzer, db_path=None, progress=None) -> list:
    """evaluate_case для каждого кейса; ошибка одного не останавливает остальные."""
    records = []
    for case in cases:
        record = evaluate_case(case, analyzer, db_path=db_path)
        records.append(record)
        if progress is not None:
            progress(record)
    return records


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _accuracy(correct: int, total: int) -> dict:
    return {"correct": correct, "total": total, "accuracy": (correct / total) if total else None}


def compute_metrics(records: list) -> dict:
    scored = [record for record in records if record["status"] == RECORD_SCORED]
    category_scored = [record for record in scored if record["checks"]["category_evaluated"]]

    confusion = {
        expected: {predicted: 0 for predicted in STATUS_ORDER} for expected in STATUS_ORDER
    }
    for record in scored:
        confusion[record["expected"]["relevance_status"]][record["prediction"]["relevance_status"]] += 1

    mismatches = {
        MISMATCH_FALSE_NEGATIVE: [], MISMATCH_FALSE_POSITIVE: [], MISMATCH_MAYBE: [], MISMATCH_LABEL: [],
    }
    for record in scored:
        if record["mismatch"] is not None:
            mismatches[record["mismatch"]].append(record["case_id"])

    error_kinds: dict = {}
    for record in records:
        if record["error"] is not None:
            error_kinds[record["error"]["kind"]] = error_kinds.get(record["error"]["kind"], 0) + 1

    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0,
                    "cache_write_tokens": 0}
    for record in records:
        for name in usage_totals:
            usage_totals[name] += (record["usage"] or {}).get(name) or 0

    return {
        "total_cases": len(records),
        "scored_cases": len(scored),
        "error_cases": sum(1 for record in records if record["status"] == RECORD_ERROR),
        "skipped_cases": sum(1 for record in records if record["status"] == RECORD_SKIPPED),
        "error_kinds": error_kinds,
        "relevance_status": _accuracy(sum(r["checks"]["relevance_status"] for r in scored), len(scored)),
        "opportunity_type": _accuracy(sum(r["checks"]["opportunity_type"] for r in scored), len(scored)),
        "category": _accuracy(sum(r["checks"]["category"] for r in category_scored), len(category_scored)),
        "core": _accuracy(sum(r["checks"]["core"] for r in scored), len(scored)),
        "full": _accuracy(sum(r["checks"]["full"] for r in scored), len(scored)),
        "confusion_relevance_status": confusion,
        "mismatches": mismatches,
        "usage_totals": usage_totals,
    }


# --------------------------------------------------------------------------
# run artifact
# --------------------------------------------------------------------------

def _timestamp_now() -> datetime:
    return datetime.now(timezone.utc)


def cost_summary(model, records: list, pricing_overrides=None) -> dict:
    """
    {pricing_version, estimated_cost_usd (str Decimal | None), error}: сумма cost_from_usage по
    записям с usage (включая ошибочные, если провайдер вернул usage). Модель без цены не роняет
    evaluator: cost=None и текст ошибки; стоимость не выдумывается.
    """
    summary = {"pricing_version": pricing.PRICING_VERSION, "estimated_cost_usd": None, "error": None}
    total = Decimal(0)
    try:
        for record in records:
            if record["usage"]:
                total += pricing.cost_from_usage(model, record["usage"], pricing_overrides)
    except pricing.CostEstimationError as error:
        summary["error"] = str(error)
        return summary
    summary["estimated_cost_usd"] = str(total)
    return summary


def production_criteria(metrics: dict) -> dict:
    """
    Критерии допуска модели к production triage (документируют решение, не подсказывают модели
    ответы): {checks: {name: bool}, passed: bool}. Категория вторична и в допуск не входит.
    """
    total = metrics["total_cases"]
    checks = {
        "relevance_status_all_correct": metrics["relevance_status"]["correct"] == total == metrics["scored_cases"],
        "core_all_correct": metrics["core"]["correct"] == total == metrics["scored_cases"],
        "false_negatives_zero": not metrics["mismatches"][MISMATCH_FALSE_NEGATIVE],
        "false_positives_zero": not metrics["mismatches"][MISMATCH_FALSE_POSITIVE],
        "api_validation_errors_zero": metrics["error_cases"] == 0 and metrics["skipped_cases"] == 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def build_artifact(records: list, analyzer, mode: str, golden_path, started_at: datetime) -> dict:
    metrics = compute_metrics(records)
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "prompt_version": getattr(analyzer, "prompt_version", None),
        "provider": getattr(analyzer, "provider", None),
        "model": getattr(analyzer, "model", None),
        "reasoning_effort": getattr(analyzer, "reasoning_effort", None),
        "timestamp": started_at.isoformat(),
        "mode": mode,
        "golden_set": Path(golden_path).name,
        "metrics": metrics,
        "usage": pricing.normalize_usage(metrics["usage_totals"]),
        "cost": cost_summary(getattr(analyzer, "model", None), records),
        "production_criteria": production_criteria(metrics),
        "cases": records,
    }


def save_artifact(artifact: dict, runs_dir=DEFAULT_RUNS_DIR) -> Path:
    """
    evaluation/runs/<UTC timestamp>_<model>.json. Существующий файл не перезаписывается
    (режим "x"). API key в artifact не попадает: он нигде не хранится в records.
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(artifact["timestamp"]).strftime("%Y%m%dT%H%M%SZ")
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", artifact["model"] or "unknown-model")
    path = runs_dir / f"{stamp}_{model}.json"
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    logger.info("Run artifact сохранён: %s", path)
    return path


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def _truncate(value, width: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def _pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _percent(metric: dict) -> str:
    if metric["accuracy"] is None:
        return f"n/a ({metric['correct']}/{metric['total']})"
    return f"{metric['accuracy'] * 100:5.1f}% ({metric['correct']}/{metric['total']})"


def _label(labels: dict) -> str:
    return f"{labels['relevance_status']} / {labels['opportunity_type']} / {labels['category']}"


def format_case_detail(record: dict, case: dict) -> str:
    lines = [f"CASE {record['case_id']}", f"  title: {record['title']}", f"  status: {record['status']}"]
    expected = record["expected"]
    lines.append(f"  expected: {_label(expected)}")
    lines.append(f"  expected_reason (human rationale, не сравнивается): {case['expected']['expected_reason']}")
    if record["error"] is not None:
        lines.append(f"  ERROR [{record['error']['kind']}]: {record['error']['message']}")
    if record["prediction"] is not None:
        prediction = record["prediction"]
        lines.append("  prediction:")
        lines.append(json.dumps(prediction, ensure_ascii=False, indent=4))
        for name in CORE_FIELDS:
            note = ""
            if name == "category" and not record["checks"]["category_evaluated"]:
                note = "  (expected category null: не входит в category accuracy)"
            lines.append(
                f"  {name:18} {_pass_fail(record['checks'][name])}  "
                f"expected={expected[name]!r} predicted={prediction[name]!r}{note}"
            )
        lines.append(f"  core (status+type)  {_pass_fail(record['checks']['core'])}")
        lines.append(f"  full (status+type+category)  {_pass_fail(record['checks']['full'])}")
    lines.append(f"  usage: {record['usage']}  response_id: {record['response_id']}  attempts: {record['attempts']}")
    return "\n".join(lines)


def format_progress(record: dict) -> str:
    if record["status"] == RECORD_SCORED:
        verdict = "OK  " if record["mismatch"] is None else "FAIL"
        predicted = _label(record["prediction"])
    else:
        verdict = "ERR " if record["status"] == RECORD_ERROR else "SKIP"
        predicted = f"[{record['error']['kind']}]"
    return f"{verdict} {record['case_id']}  {_truncate(record['title'], 40):40}  -> {predicted}"


def format_confusion(confusion: dict) -> str:
    width = 14
    header = "expected \\ predicted".ljust(22) + "".join(name.rjust(width) for name in STATUS_ORDER)
    lines = [header]
    for expected in STATUS_ORDER:
        lines.append(expected.ljust(22) + "".join(str(confusion[expected][p]).rjust(width) for p in STATUS_ORDER))
    return "\n".join(lines)


def _format_failure(record: dict) -> str:
    return (
        f"  {record['case_id']}  {_truncate(record['title'], 60)}\n"
        f"      expected:   {_label(record['expected'])}\n"
        f"      prediction: {_label(record['prediction'])}  (confidence {record['prediction']['confidence']})\n"
        f"      reason:     {record['prediction']['reason']}"
    )


def format_cost_and_criteria(artifact: dict) -> str:
    cost = artifact["cost"]
    cost_text = f"${cost['estimated_cost_usd']}" if cost["estimated_cost_usd"] is not None else f"н/д ({cost['error']})"
    lines = [f"Оценка стоимости прогона (pricing {cost['pricing_version']}): {cost_text}", "Критерии допуска к production triage:"]
    for name, ok in artifact["production_criteria"]["checks"].items():
        lines.append(f"  {_pass_fail(ok)}  {name}")
    lines.append("  ИТОГ: " + ("модель проходит критерии" if artifact["production_criteria"]["passed"] else "НЕ проходит критерии"))
    return "\n".join(lines)


def format_report(records: list, metrics: dict) -> str:
    lines = [
        "=" * 72,
        f"Всего кейсов: {metrics['total_cases']}, оценено: {metrics['scored_cases']}, "
        f"ошибок API/validation: {metrics['error_cases']}, пропущено (без вызова API): {metrics['skipped_cases']}",
    ]
    if metrics["error_kinds"]:
        lines.append(f"Виды ошибок: {metrics['error_kinds']}")
    if metrics["scored_cases"] != metrics["total_cases"]:
        lines.append("ВНИМАНИЕ: accuracy ниже считается только по оценённым кейсам.")
    lines += [
        "",
        f"relevance_status accuracy:            {_percent(metrics['relevance_status'])}",
        f"opportunity_type accuracy:            {_percent(metrics['opportunity_type'])}",
        f"category accuracy (expected != null): {_percent(metrics['category'])}",
        f"core (status + type) accuracy:        {_percent(metrics['core'])}",
        f"full (status + type + category):      {_percent(metrics['full'])}",
        "",
        "Confusion matrix (relevance_status):",
        format_confusion(metrics["confusion_relevance_status"]),
        "",
        f"Usage (сумма): {metrics['usage_totals']}",
    ]

    by_id = {record["case_id"]: record for record in records}
    sections = (
        (MISMATCH_FALSE_NEGATIVE, "ОПАСНО — FALSE NEGATIVE (expected relevant, model not_relevant)"),
        (MISMATCH_FALSE_POSITIVE, "ОПАСНО — FALSE POSITIVE (expected not_relevant, model relevant)"),
        (MISMATCH_MAYBE, "MAYBE/UNCLEAR ошибки"),
        (MISMATCH_LABEL, "Прочие расхождения (status совпал; type/category — нет)"),
    )
    lines.append("")
    if not any(metrics["mismatches"].values()):
        lines.append("Failed cases: нет")
    for key, title in sections:
        case_ids = metrics["mismatches"][key]
        if case_ids:
            lines.append(f"{title}: {len(case_ids)}")
            lines.extend(_format_failure(by_id[case_id]) for case_id in case_ids)
            lines.append("")

    not_scored = [record for record in records if record["status"] != RECORD_SCORED]
    if not_scored:
        lines.append(f"Не оценены ({len(not_scored)}):")
        for record in not_scored:
            lines.append(
                f"  {record['status']:8} {record['case_id']}  [{record['error']['kind']}] "
                f"{_truncate(record['error']['message'], 120)}"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ai.golden_set_evaluator",
        description=(
            "Оценка OpenAI triage на procurement golden set. Без явного режима ничего не "
            "запускается; --dry-run не вызывает API. Не пишет в БД и не меняет golden set."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Проверить golden set/hash/конфиг, API не вызывается")
    mode.add_argument("--case-id", default=None, metavar="CASE_ID", help="Один реальный API-вызов для этого кейса")
    mode.add_argument("--all", action="store_true", help="Явно запустить все кейсы (реальные API-вызовы)")
    parser.add_argument("--path", default=str(golden_set.DEFAULT_GOLDEN_SET_PATH), help="Файл golden set")
    parser.add_argument("--db-path", default=None, help="SQLite БД (по умолчанию data/tenders.db, read-only)")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="Куда писать run artifact")
    parser.add_argument("--model", default=None, help="Переопределить OPENAI_MODEL")
    parser.add_argument(
        "--reasoning-effort", default=None, help="Переопределить OPENAI_TRIAGE_REASONING_EFFORT",
    )
    return parser


def _default_analyzer(args) -> openai_triage.OpenAITriageAnalyzer:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")  # не перезаписывает уже заданные переменные окружения
    return openai_triage.OpenAITriageAnalyzer(model=args.model, reasoning_effort=args.reasoning_effort)


def _print_cost_preflight(triage_contexts: list, analyzer) -> None:
    """Консервативная верхняя оценка стоимости полного прогона (эвристика preflight, не точный подсчёт)."""
    if not triage_contexts:
        return
    try:
        total = sum(
            (preflight.preflight_request(
                analyzer.build_request(context), analyzer.model, "triage", BudgetSettings(),
            )["estimated_cost_usd"] for context in triage_contexts),
            Decimal(0),
        )
    except pricing.CostEstimationError as error:
        print(f"Оценка стоимости недоступна: {error}")
        return
    print(
        f"Консервативная оценка стоимости прогона {len(triage_contexts)} кейс(ов) на {analyzer.model} "
        f"(верхняя граница, pricing {pricing.PRICING_VERSION}): ${total.quantize(Decimal('0.0001'))}"
    )


def _dry_run(cases: list, analyzer, args) -> int:
    print(f"Golden set: {args.path} — {len(cases)} кейс(ов), структура валидна")

    counts: dict = {}
    problems = []
    request_sizes = []
    triage_contexts = []
    for case in cases:
        prepared = prepare_case(case, db_path=args.db_path)
        counts[prepared["status"]] = counts.get(prepared["status"], 0) + 1
        if prepared["status"] != golden_set.HASH_OK:
            problems.append(f"  {prepared['status'].upper():8} {case['case_id']}  {prepared['detail']}")
        else:
            request_sizes.append(len(triage_prompt.build_user_input(prepared["triage_context"])))
            triage_contexts.append(prepared["triage_context"])
    print(f"Hydrate/hash: { {name: counts.get(name, 0) for name in ('ok', 'stale', 'missing', 'error')} }")
    for line in problems:
        print(line)
    if request_sizes:
        print(f"Размер user input: min {min(request_sizes)}, max {max(request_sizes)} символов")

    print(
        f"OpenAI config: model={analyzer.model}, reasoning_effort={analyzer.reasoning_effort}, "
        f"prompt_version={analyzer.prompt_version}, api_key={'set' if analyzer.has_api_key else 'MISSING'}"
    )
    config_ok = analyzer.has_api_key
    if not config_ok:
        print("  ПРОБЛЕМА: OPENAI_API_KEY не задан (env или .env) — реальный запуск невозможен")
    _print_cost_preflight(triage_contexts, analyzer)
    print("DRY RUN: API-вызовов не выполнено, client не создавался")
    return 0 if not problems and config_ok else 1


def main(argv=None, analyzer=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    try:
        cases = load_valid_cases(args.path)
    except FileNotFoundError:
        print(f"Файл golden set не найден: {args.path}")
        return 2
    except ValueError as error:
        print(f"Ошибка: {error}")
        return 2

    if analyzer is None:
        try:
            analyzer = _default_analyzer(args)
        except openai_triage.TriageError as error:
            print(f"Ошибка конфигурации: {error}")
            return 2

    if args.dry_run:
        return _dry_run(cases, analyzer, args)

    if args.all:
        selected, mode = cases, "all"
    else:
        selected = [case for case in cases if case["case_id"] == args.case_id]
        mode = "case"
        if not selected:
            print(f"Кейс {args.case_id!r} не найден в {args.path}")
            return 2

    try:
        ensure_client = getattr(analyzer, "ensure_client", None)
        if ensure_client is not None:
            ensure_client()
    except openai_triage.TriageError as error:
        print(f"Ошибка конфигурации (API не вызывался): {error}")
        return 2

    started_at = _timestamp_now()
    print(
        f"Запуск: {len(selected)} кейс(ов), model={analyzer.model}, "
        f"reasoning_effort={analyzer.reasoning_effort}, prompt={analyzer.prompt_version}"
    )
    progress = (lambda record: print(format_progress(record), flush=True)) if len(selected) > 1 else None
    records = run_evaluation(selected, analyzer, db_path=args.db_path, progress=progress)

    if len(selected) == 1:
        print(format_case_detail(records[0], selected[0]))
    artifact = build_artifact(records, analyzer, mode, args.path, started_at)
    print(format_report(records, artifact["metrics"]))
    print(format_cost_and_criteria(artifact))
    print(f"\nRun artifact: {save_artifact(artifact, args.runs_dir)}")

    metrics = artifact["metrics"]
    return 0 if metrics["error_cases"] == 0 and metrics["skipped_cases"] == 0 else 1


if __name__ == "__main__":
    # Заголовки тендеров бывают не на языке консоли: не падать на UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
