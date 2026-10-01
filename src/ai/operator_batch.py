"""
Операторский live batch: реальный AI pipeline для ровно N явно выбранных тендеров.

    python -m src.ai.analysis_pipeline --run-batch --limit 5                         # PREVIEW: ни client, ни API, ни записи
    python -m src.ai.analysis_pipeline --run-batch --limit 5 --confirm-paid-call     # реальный запуск

--limit обязателен (операторский предохранитель, не бизнес-лимит в день). Без --confirm-paid-call — только
read-only preview: БД открыта mode=ro, резервов/записей/API-вызовов нет.

Выбор кандидатов — ОДНА функция select_batch поверх AnalysisPipeline.select_candidates (та же production-логика,
включая Operational Eligibility Gate): preview и live используют её же, поэтому показанный список = обрабатываемый.
Истёкшие (skip_expired, $0) в выбор и в limit не входят и в live не трогаются. После N тендеров — СТОП, без
продолжения по backlog.

Здесь нет AI-логики: на каждый тендер вызывается существующий process_announcement (идемпотентность, Gate,
BudgetGuard, durable резервы, Deep Admission, ledger, escalation). Нерешённый accounting -> новые платные вызовы
не начинаются, остаток отчитывается как not started. Обычные ошибки тендера batch не прерывают.
Не зависит от AI_ANALYSIS_ENABLED (ручной запуск). Terra не вызывается. Ключ API не печатается.
"""

import sqlite3
from contextlib import closing
from decimal import Decimal

from src.ai import commercial_gate, operational_eligibility, preflight, pricing
from src.ai import tender_context as tender_context_module
from src.ai.analysis_pipeline import REASON_ACCOUNTING_BLOCKED, REASON_SKIP_EXPIRED, AnalysisPipeline, _ReadOnlyStore
from src.ai.budget_guard import ANALYSIS_TRIAGE
from src.ai.one_shot import EXIT_FAILED, EXIT_OK, EXIT_USAGE, _max_event_id
from src.database import ai_usage_repository, pipeline_state_repository as state_repo, tender_repository

TRIAGE_REASONS = ("needs_triage", "retry_triage")  # остальные причины = triage уже актуален (reusable)

DEFERRED_STATES = (state_repo.STATE_TRIAGE_DEFERRED_BUDGET, state_repo.STATE_DEEP_DEFERRED_BUDGET)
DEEP_ERROR_STATES = (state_repo.STATE_DEEP_ERROR, state_repo.STATE_TRIAGE_ERROR)


def _ledger_rows_after(db_path, after_id):
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute(
            "SELECT input_tokens, cached_input_tokens, cache_write_tokens, output_tokens, estimated_cost_usd "
            "FROM ai_usage_events WHERE id > ?", (after_id,),
        ).fetchall()


def select_batch(pipeline: AnalysisPipeline, limit: int) -> dict:
    """
    Единый детерминированный выбор для preview и live: {selected: [candidate], expired: [candidate]}.
    selected — не более limit платных/рабочих кандидатов (новые раньше retry); expired в limit не входят.
    """
    candidates = pipeline.select_candidates(limit)
    selected = [c for c in candidates if c["reason"] != REASON_SKIP_EXPIRED]
    expired = [c for c in candidates if c["reason"] == REASON_SKIP_EXPIRED]
    return {"selected": selected, "expired": expired}


def _describe(candidate, ctx, eligibility, triage, settings) -> dict:
    """Строка preview: факты о кандидате и (если нужен новый triage) консервативная оценка."""
    reason = candidate["reason"]
    row = {
        "resource_url": candidate["resource_url"], "title": ctx["announcement"].get("title"),
        "operational_status": eligibility.status,
        "resolved_deadline": eligibility.resolved_deadline.isoformat() if eligibility.resolved_deadline else None,
        "analysis_status": reason, "needs_new_triage": reason in TRIAGE_REASONS,
        "estimated_triage_input_tokens": None, "estimated_triage_max_cost_usd": None, "estimate_error": None,
    }
    if row["needs_new_triage"]:
        request = triage.build_request(tender_context_module.build_triage_context(ctx))
        try:
            pre = preflight.preflight_request(request, triage.model, ANALYSIS_TRIAGE, settings)
        except pricing.CostEstimationError as error:
            row["estimate_error"] = str(error)
        else:
            row["estimated_triage_input_tokens"] = pre["estimated_input_tokens"]
            row["estimated_triage_max_cost_usd"] = pre["estimated_cost_usd"]
    return row


def _month_actual(store) -> Decimal:
    if "ai_usage_events" not in store.tables:
        return Decimal(0)
    return ai_usage_repository.current_month_cost(db_path=store.db_path)


def build_preview(selection, pipeline, store, triage, settings, db_path, limit) -> dict:
    rows = []
    for candidate in selection["selected"]:
        url = candidate["resource_url"]
        ctx = tender_context_module.build_tender_context(url, db_path=db_path)
        eligibility = operational_eligibility.evaluate_context(ctx, pipeline.clock())
        rows.append(_describe(candidate, ctx, eligibility, triage, settings))
    reservations = store.outstanding_reservations()
    reserved = sum((Decimal(r["estimated_cost_usd"]) for r in reservations), Decimal(0))
    statuses = [r["operational_status"] for r in rows]
    return {
        "limit": limit, "rows": rows,
        "selected_count": len(rows),
        "new_triage_count": sum(r["needs_new_triage"] for r in rows),
        "reusable_triage_count": sum(not r["needs_new_triage"] for r in rows),
        "expired_excluded_count": len(selection["expired"]),
        "deadline_unknown_count": statuses.count(operational_eligibility.STATUS_UNKNOWN),
        "deadline_conflict_count": statuses.count(operational_eligibility.STATUS_CONFLICT),
        "monthly_actual_spend_usd": _month_actual(store),
        "outstanding_reservations_count": len(reservations), "outstanding_reservations_usd": reserved,
        "effective_committed_spend_usd": store.month_spend(),
        "triage_max_cost_total_usd": sum(
            (r["estimated_triage_max_cost_usd"] or Decimal(0) for r in rows), Decimal(0),
        ),
        "estimate_errors": sum(r["estimate_error"] is not None for r in rows),
        "hard_limit_usd": settings.monthly_budget_usd, "soft_limit_usd": settings.soft_limit_usd,
        "triage_model": triage.model, "accounting_blocks": store.accounting_blocks(),
    }


def _print_preview(preview, out) -> None:
    out(f"BATCH PREVIEW (--limit {preview['limit']}): выбрано {preview['selected_count']} тендеров")
    for index, row in enumerate(preview["rows"], 1):
        out(f"[{index}] {row['resource_url']}")
        out(f"    title: {row['title']}")
        out(f"    deadline status: {row['operational_status']}, resolved deadline: {row['resolved_deadline'] or '-'}")
        kind = "new triage needed" if row["needs_new_triage"] else "reusable triage (без нового triage-вызова)"
        out(f"    analysis status: {row['analysis_status']} — {kind}")
        if row["needs_new_triage"]:
            if row["estimate_error"]:
                out(f"    estimated triage: оценить нельзя ({row['estimate_error']})")
            else:
                out(f"    estimated triage input tokens: {row['estimated_triage_input_tokens']}")
                out(f"    estimated triage max cost: ${row['estimated_triage_max_cost_usd']}")
    out("BATCH TOTALS")
    out(f"  selected tenders: {preview['selected_count']}")
    out(f"  new triage calls potentially required: {preview['new_triage_count']}")
    out(f"  known reusable triage: {preview['reusable_triage_count']}")
    out(f"  expired excluded (не в limit, не обрабатываются): {preview['expired_excluded_count']}")
    out(f"  deadline_unknown: {preview['deadline_unknown_count']}")
    out(f"  deadline_conflict: {preview['deadline_conflict_count']}")
    out(f"  monthly actual spend: ${preview['monthly_actual_spend_usd']}")
    out(f"  outstanding reservations: {preview['outstanding_reservations_count']} "
        f"(${preview['outstanding_reservations_usd']})")
    out(f"  effective committed spend: ${preview['effective_committed_spend_usd']}")
    out(f"  conservative triage max cost total: ${preview['triage_max_cost_total_usd']} "
        "(консервативная ВЕРХНЯЯ граница, не прогноз фактической стоимости; Deep сюда не входит — "
        "он проходит свой Admission/BudgetGuard перед каждым вызовом)")
    out(f"  hard monthly budget: ${preview['hard_limit_usd']}")
    out(f"  soft limit: ${preview['soft_limit_usd']}")
    if preview["estimate_errors"]:
        out(f"  WARNING: стоимость не оценена для {preview['estimate_errors']} тендеров")
    if preview["accounting_blocks"] or preview["outstanding_reservations_count"]:
        out("  WARNING: есть нерешённый учёт — платные вызовы будут заблокированы "
            "(python -m src.ai.accounting_reconcile --list)")


def _summarize(selected, results, selection, rows) -> dict:
    states = [r.get("state") for r in results]
    reused = sum(1 for r in results if r.get("reused", {}).get("triage"))
    return {
        "selected": len(selected), "processed": len(results), "not_started": len(selected) - len(results),
        "triage_calls": sum(r["api_calls"]["triage"] for r in results),
        "deep_calls": sum(r["api_calls"]["deep"] for r in results),
        "reused_triage": reused,
        "not_relevant": states.count(state_repo.STATE_STOPPED_NOT_RELEVANT),
        "commercial_gate_skips": states.count(state_repo.STATE_COMMERCIAL_GATE_SKIP),
        "deep_completed": states.count(state_repo.STATE_DEEP_COMPLETED),
        "deep_validation_errors": sum(states.count(s) for s in DEEP_ERROR_STATES),
        "escalation_candidates": states.count(state_repo.STATE_ESCALATION_CANDIDATE),
        "expired_excluded": len(selection["expired"]),
        "budget_deferred": sum(states.count(s) for s in DEFERRED_STATES),
        "accounting_blocked": sum(1 for r in results if r.get("reason_code") == REASON_ACCOUNTING_BLOCKED),
        "input_tokens": sum(r[0] for r in rows), "cached_tokens": sum(r[1] for r in rows),
        "cache_write_tokens": sum(r[2] for r in rows), "output_tokens": sum(r[3] for r in rows),
        "cost_usd": sum((Decimal(r[4]) for r in rows), Decimal(0)),
    }


def _print_report(summary, db_path, out) -> None:
    out("BATCH OPERATOR REPORT")
    labels = (
        ("selected tenders", "selected"), ("processed tenders", "processed"), ("not started", "not_started"),
        ("triage API calls", "triage_calls"), ("deep API calls", "deep_calls"), ("reused triage", "reused_triage"),
        ("not_relevant", "not_relevant"), ("commercial gate skips", "commercial_gate_skips"),
        ("deep_completed", "deep_completed"), ("deep/triage validation or errors", "deep_validation_errors"),
        ("escalation candidates", "escalation_candidates"), ("expired excluded", "expired_excluded"),
        ("budget deferred", "budget_deferred"), ("accounting blocked", "accounting_blocked"),
    )
    for label, key in labels:
        out(f"  {label}: {summary[key]}")
    out("  usage (THIS batch):")
    out(f"    input_tokens: {summary['input_tokens']}")
    out(f"    cached_tokens: {summary['cached_tokens']}")
    out(f"    cache_write_tokens: {summary['cache_write_tokens']}")
    out(f"    output_tokens: {summary['output_tokens']}")
    out(f"  actual estimated_cost_usd (ledger): ${summary['cost_usd']}")
    out(f"  monthly actual spend after batch: ${ai_usage_repository.current_month_cost(db_path=db_path)}")
    out(f"  outstanding reservations after batch: {len(ai_usage_repository.outstanding_reservations(db_path=db_path))}")


def run_batch(
    limit, triage, deep, settings, db_path=None, confirm=False, out=print,
    clock=operational_eligibility.utc_now, gate=commercial_gate.evaluate,
) -> int:
    """Код возврата: 0 — ok / preview, 1 — были ошибки тендеров или accounting-блок, 2 — ошибка запуска (вызовов нет)."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        out(f"ERROR: --limit обязателен и должен быть положительным целым: {limit!r}")
        return EXIT_USAGE
    db_path = tender_repository.resolve_db_path(db_path)
    try:
        store = _ReadOnlyStore(db_path)
    except sqlite3.Error as error:
        out(f"ERROR: БД недоступна: {error}")
        return EXIT_USAGE

    # preview и live делят select_batch; для preview store read-only, для live — боевой (те же данные).
    preview_pipeline = AnalysisPipeline(triage, deep, settings, db_path=db_path, store=store, clock=clock, gate=gate)
    selection = select_batch(preview_pipeline, limit)
    preview = build_preview(selection, preview_pipeline, store, triage, settings, db_path, limit)
    _print_preview(preview, out)
    if not confirm:
        out("PREVIEW ONLY: OpenAI client не создан, API-вызовов, резервов и записей нет. "
            "Для реального запуска добавьте --confirm-paid-call")
        return EXIT_OK

    selected = selection["selected"]
    out(f"LIVE BATCH: обрабатываются ровно выбранные {len(selected)} тендеров, затем СТОП")
    pipeline = AnalysisPipeline(triage, deep, settings, db_path=db_path, clock=clock, gate=gate)
    pipeline._ensure_ready()
    before = _max_event_id(db_path)
    results = []
    for index, candidate in enumerate(selected, 1):
        outcome = pipeline.process_announcement(candidate["resource_url"])
        results.append(outcome)
        out(f"  [{index}/{len(selected)}] {candidate['resource_url']} -> {outcome['state']}"
            + (f" ({outcome['reason_code']})" if outcome.get("reason_code") else ""))
        if outcome.get("reason_code") == REASON_ACCOUNTING_BLOCKED or pipeline._accounting_failure is not None:
            out(f"ACCOUNTING FAIL-CLOSED: платные вызовы остановлены, не начато: {len(selected) - index}")
            break
    summary = _summarize(selected, results, selection, _ledger_rows_after(db_path, before))
    _print_report(summary, db_path, out)
    failed = summary["accounting_blocked"] or pipeline._accounting_failure is not None or any(
        r["failed"] for r in results
    )
    return EXIT_FAILED if failed else EXIT_OK
