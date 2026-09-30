"""
Ручной human-review evaluator OpenAI deep analysis на ОДНОМ явно выбранном golden
procurement case (evaluation/golden_set_procurement.json).

У нас пока нет human-labeled deep golden set (никто не размечал ожидаемые items/lots/
barriers/etc для deep analysis) — поэтому здесь нет и не может быть "deep accuracy":
только запуск deep analyzer на выбранном тендере и вывод предсказания для ручной проверки
специалистом. triage label для контекста (analyzer.deep_analyze(..., triage_result) требует
triage_result) берётся из human "expected" golden label (relevance_status/opportunity_type/
category/expected_reason), обёрнутого в стандартную triage-схему с confidence="high" —
это НЕ настоящий AI triage output, а лучшая доступная проекция golden label.

Безопасность:
    - без явного режима ничего не запускается (--dry-run / --case-id);
    - --dry-run не вызывает API и не создаёт client;
    - НЕТ режима "--all": один случайный дорогой batch-запуск исключён явно — только
      dry-run или ровно один --case-id за вызов (правило задачи: "не легко дорогой запуск");
    - БД читается только через существующие read-only функции (golden_set.hydrate_golden_case,
      src.ai.tender_context); golden set и БД не изменяются; единственная запись — JSON
      artifact в evaluation/deep_runs/;
    - если input_hash кейса устарел (тендер изменился), API для него не вызывается;
    - если golden label не relevant+procurement и не maybe+unclear (т.е. triage не отправил
      бы этот тендер на deep analysis), API не вызывается.

CLI:
    python -m src.ai.deep_analysis_evaluator --dry-run
    python -m src.ai.deep_analysis_evaluator --case-id <CASE_ID>
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.ai import deep_prompt
from src.ai import golden_set
from src.ai import openai_deep_analysis
from src.ai import pricing
from src.ai import relevance_schema
from src.ai import tender_context as tender_context_module
from src.ai.golden_set_evaluator import cost_summary, load_valid_cases

logger = logging.getLogger(__name__)

EVALUATOR_VERSION = "2"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEEP_RUNS_DIR = PROJECT_ROOT / "evaluation" / "deep_runs"

RECORD_SCORED = "scored"
RECORD_ERROR = "error"
RECORD_SKIPPED = "skipped"

SKIP_STALE = "stale_golden_hash"
SKIP_MISSING = "missing_tender"
SKIP_HYDRATE_ERROR = "hydrate_error"
SKIP_NOT_ELIGIBLE = "not_eligible_for_deep_analysis"
ERROR_UNEXPECTED = "unexpected_error"

# Только эти комбинации golden label доходят до deep analysis в проде (см.
# src.database.analysis_repository.get_deep_analysis_candidates: последний triage
# relevant/maybe). requires_deep_analysis=False (not_relevant) значит нечего анализировать.


# --------------------------------------------------------------------------
# case preparation (read-only)
# --------------------------------------------------------------------------

def _triage_result_from_expected(expected: dict) -> dict:
    """
    Синтетический triage_result из human golden label (см. модульный docstring). Проходит
    ту же relevance_schema.validate_triage_result, что и настоящий AI triage output —
    гарантированно совместим с analyzer.deep_analyze(...) interface.
    """
    status = expected["relevance_status"]
    return relevance_schema.validate_triage_result({
        "relevance_status": status,
        "opportunity_type": expected["opportunity_type"],
        "category": expected["category"],
        "confidence": "high",
        "reason": expected["expected_reason"],
        "requires_deep_analysis": status != "not_relevant",
        "evidence": [],
    })


def prepare_case(case: dict, db_path=None) -> dict:
    """
    Read-only hydrate одного golden case + полный tender_context/deep_analysis_context.
    {"status": ok|stale|missing|error|not_eligible, "detail", "tender_context",
    "deep_analysis_context", "triage_result"} — последние три есть только при status="ok".
    """
    try:
        hydrated = golden_set.hydrate_golden_case(case, db_path=db_path)
    except ValueError as error:
        return {
            "status": golden_set.HASH_MISSING, "detail": str(error),
            "tender_context": None, "deep_analysis_context": None, "triage_result": None,
        }
    except (sqlite3.Error, OSError) as error:
        return {
            "status": golden_set.HASH_ERROR, "detail": f"{type(error).__name__}: {error}",
            "tender_context": None, "deep_analysis_context": None, "triage_result": None,
        }
    if not hydrated["hash_match"]:
        return {
            "status": golden_set.HASH_STALE,
            "detail": (
                f"stored {hydrated['stored_input_hash'][:12]} != "
                f"current {hydrated['current_input_hash'][:12]}"
            ),
            "tender_context": None, "deep_analysis_context": None, "triage_result": None,
        }

    triage_result = _triage_result_from_expected(case["expected"])
    if not triage_result["requires_deep_analysis"]:
        return {
            "status": SKIP_NOT_ELIGIBLE,
            "detail": (
                f"golden label {case['expected']['relevance_status']}/"
                f"{case['expected']['opportunity_type']} не отправляется на deep analysis"
            ),
            "tender_context": None, "deep_analysis_context": None, "triage_result": None,
        }

    tender_context = tender_context_module.build_tender_context(case["resource_url"], db_path=db_path)
    deep_analysis_context = tender_context_module.build_deep_analysis_context(tender_context)
    return {
        "status": golden_set.HASH_OK, "detail": None,
        "tender_context": tender_context, "deep_analysis_context": deep_analysis_context,
        "triage_result": triage_result,
    }


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def _call_analyzer(analyzer, tender_context: dict, deep_analysis_context: dict, triage_result: dict) -> dict:
    with_metadata = getattr(analyzer, "deep_analyze_with_metadata", None)
    if with_metadata is not None:
        return with_metadata(tender_context, deep_analysis_context, triage_result)
    return {
        "result": analyzer.deep_analyze(tender_context, deep_analysis_context, triage_result),
        "usage": None, "response_id": None, "attempts": None,
    }


def _base_record(case: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "resource_url": case["resource_url"],
        "input_hash": case["input_hash"],
        "triage_label": {
            "relevance_status": case["expected"]["relevance_status"],
            "opportunity_type": case["expected"]["opportunity_type"],
            "category": case["expected"]["category"],
        },
        "status": None,
        "document_coverage": None,
        "input_size_chars": None,
        "evidence_units": None,
        "prediction": None,
        "raw_model_output": None,
        "usage": None,
        "response_id": None,
        "attempts": None,
        "error": None,
    }


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
            SKIP_NOT_ELIGIBLE: SKIP_NOT_ELIGIBLE,
        }.get(prepared["status"], SKIP_HYDRATE_ERROR)
        logger.warning("Golden case %s пропущен без вызова API: %s (%s)", case["case_id"], kind, prepared["detail"])
        record["status"] = RECORD_SKIPPED
        record["error"] = {"kind": kind, "message": prepared["detail"]}
        return record

    tender_context = prepared["tender_context"]
    deep_analysis_context = prepared["deep_analysis_context"]
    triage_result = prepared["triage_result"]
    record["document_coverage"] = dict(tender_context["document_coverage"])
    deep_context = deep_prompt.build_deep_context(tender_context, deep_analysis_context, triage_result)
    record["input_size_chars"] = len(deep_prompt.build_user_input(deep_context))
    record["evidence_units"] = len(deep_context["evidence_catalog"])

    try:
        outcome = _call_analyzer(analyzer, tender_context, deep_analysis_context, triage_result)
    except openai_deep_analysis.DeepAnalysisError as error:
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

    record["status"] = RECORD_SCORED
    # prediction — материализованный результат (evidence.text из каталога, для ручной проверки);
    # raw_model_output — ответ модели (только evidence_ids). Сам каталог в artifact не пишется.
    record["prediction"] = outcome["result"]
    record["raw_model_output"] = outcome.get("raw_model_output")
    record["usage"] = outcome.get("usage")
    record["response_id"] = outcome.get("response_id")
    record["attempts"] = outcome.get("attempts")
    return record


# --------------------------------------------------------------------------
# run artifact
# --------------------------------------------------------------------------

def _timestamp_now() -> datetime:
    return datetime.now(timezone.utc)


def build_artifact(record: dict, analyzer, golden_path, started_at: datetime) -> dict:
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "prompt_version": getattr(analyzer, "prompt_version", None),
        "provider": getattr(analyzer, "provider", None),
        "model": getattr(analyzer, "model", None),
        "reasoning_effort": getattr(analyzer, "reasoning_effort", None),
        "timestamp": started_at.isoformat(),
        "golden_set": Path(golden_path).name,
        "usage": pricing.normalize_usage(record.get("usage")),
        "cost": cost_summary(getattr(analyzer, "model", None), [record]),
        "case": record,
    }


def save_artifact(artifact: dict, runs_dir=DEFAULT_DEEP_RUNS_DIR) -> Path:
    """
    evaluation/deep_runs/<UTC timestamp>_<case_id>_<model>.json. Существующий файл не
    перезаписывается (режим "x"). API key в artifact не попадает: он нигде не хранится
    в record (см. DeepAnalysisError._redact на стороне analyzer).
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(artifact["timestamp"]).strftime("%Y%m%dT%H%M%SZ")
    case_id = re.sub(r"[^A-Za-z0-9._-]+", "_", artifact["case"]["case_id"])
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", artifact["model"] or "unknown-model")
    path = runs_dir / f"{stamp}_{case_id}_{model}.json"
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    logger.info("Run artifact сохранён: %s", path)
    return path


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def format_case_detail(record: dict) -> str:
    lines = [
        f"CASE {record['case_id']}",
        f"  title: {record['title']}",
        f"  triage label (human golden, не настоящий AI triage): {record['triage_label']}",
        f"  status: {record['status']}",
    ]
    if record["document_coverage"] is not None:
        lines.append(f"  document_coverage: {record['document_coverage']}")
    if record["input_size_chars"] is not None:
        lines.append(f"  input size (DEEP_CONTEXT user input): {record['input_size_chars']} символов")
    if record["evidence_units"] is not None:
        lines.append(f"  evidence catalog: {record['evidence_units']} units")
    if record["error"] is not None:
        lines.append(f"  ERROR [{record['error']['kind']}]: {record['error']['message']}")
    if record["prediction"] is not None:
        prediction = record["prediction"]
        lines.append(f"  validation status: OK (прошёл relevance_schema + deep-MVP + evidence_ids каталога)")
        lines.append(f"  manual_review_required: {prediction['manual_review_required']}")
        lines.append(f"  missing_information: {prediction['missing_information']}")
        lines.append(f"  participation_barriers: {json.dumps(prediction['participation_barriers'], ensure_ascii=False)}")
        lines.append("  deep prediction:")
        lines.append(json.dumps(prediction, ensure_ascii=False, indent=4))
    lines.append(f"  usage: {record['usage']}  response_id: {record['response_id']}  attempts: {record['attempts']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ai.deep_analysis_evaluator",
        description=(
            "Human-review запуск OpenAI deep analysis на ОДНОМ golden procurement case. "
            "Без явного режима ничего не запускается; --dry-run не вызывает API. Нет "
            "режима --all — только dry-run или ровно один --case-id за вызов. Не пишет "
            "в БД и не меняет golden set."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Проверить golden set/hash/конфиг, API не вызывается")
    mode.add_argument("--case-id", default=None, metavar="CASE_ID", help="Один реальный API-вызов для этого кейса")
    parser.add_argument("--path", default=str(golden_set.DEFAULT_GOLDEN_SET_PATH), help="Файл golden set")
    parser.add_argument("--db-path", default=None, help="SQLite БД (по умолчанию data/tenders.db, read-only)")
    parser.add_argument("--runs-dir", default=str(DEFAULT_DEEP_RUNS_DIR), help="Куда писать run artifact")
    parser.add_argument("--model", default=None, help="Переопределить DEEP_PRIMARY_MODEL (приоритет над env/default)")
    parser.add_argument(
        "--reasoning-effort", default=None, help="Переопределить DEEP_PRIMARY_REASONING_EFFORT",
    )
    return parser


def _default_analyzer(args) -> openai_deep_analysis.OpenAIDeepAnalysisAnalyzer:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")  # не перезаписывает уже заданные переменные окружения
    return openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(model=args.model, reasoning_effort=args.reasoning_effort)


def _dry_run(cases: list, analyzer, args) -> int:
    print(f"Golden set: {args.path} — {len(cases)} кейс(ов), структура валидна")

    counts: dict = {}
    problems = []
    input_sizes = []
    for case in cases:
        prepared = prepare_case(case, db_path=args.db_path)
        counts[prepared["status"]] = counts.get(prepared["status"], 0) + 1
        if prepared["status"] != golden_set.HASH_OK:
            problems.append(f"  {prepared['status'].upper():8} {case['case_id']}  {prepared['detail']}")
        else:
            deep_context = deep_prompt.build_deep_context(
                prepared["tender_context"], prepared["deep_analysis_context"], prepared["triage_result"],
            )
            input_sizes.append(len(deep_prompt.build_user_input(deep_context)))
    print(f"Hydrate/hash/eligibility: {counts}")
    for line in problems:
        print(line)
    if input_sizes:
        print(f"Размер DEEP_CONTEXT user input: min {min(input_sizes)}, max {max(input_sizes)} символов")

    print(
        f"OpenAI config: model={analyzer.model}, reasoning_effort={analyzer.reasoning_effort}, "
        f"prompt_version={analyzer.prompt_version}, api_key={'set' if analyzer.has_api_key else 'MISSING'}"
    )
    config_ok = analyzer.has_api_key
    if not config_ok:
        print("  ПРОБЛЕМА: OPENAI_API_KEY не задан (env или .env) — реальный запуск невозможен")
    eligible = counts.get(golden_set.HASH_OK, 0)
    print(f"Кейсов, готовых к deep analysis (relevant+procurement / maybe+unclear, hash ok): {eligible}")
    print("DRY RUN: API-вызовов не выполнено, client не создавался")
    return 0 if config_ok else 1


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
        except openai_deep_analysis.DeepAnalysisError as error:
            print(f"Ошибка конфигурации: {error}")
            return 2

    if args.dry_run:
        return _dry_run(cases, analyzer, args)

    selected = [case for case in cases if case["case_id"] == args.case_id]
    if not selected:
        print(f"Кейс {args.case_id!r} не найден в {args.path}")
        return 2
    case = selected[0]

    try:
        ensure_client = getattr(analyzer, "ensure_client", None)
        if ensure_client is not None:
            ensure_client()
    except openai_deep_analysis.DeepAnalysisError as error:
        print(f"Ошибка конфигурации (API не вызывался): {error}")
        return 2

    started_at = _timestamp_now()
    print(
        f"Запуск: кейс {case['case_id']}, model={analyzer.model}, "
        f"reasoning_effort={analyzer.reasoning_effort}, prompt={analyzer.prompt_version}"
    )
    record = evaluate_case(case, analyzer, db_path=args.db_path)
    print(format_case_detail(record))

    artifact = build_artifact(record, analyzer, args.path, started_at)
    print(f"Usage: {artifact['usage']}  estimated_cost_usd={artifact['cost']['estimated_cost_usd']}")
    print(f"\nRun artifact: {save_artifact(artifact, args.runs_dir)}")

    return 0 if record["status"] == RECORD_SCORED else 1


if __name__ == "__main__":
    # Заголовки тендеров бывают не на языке консоли: не падать на UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
