"""
Операторский one-shot: реальный AI pipeline ровно для ОДНОГО явно указанного resource_url.

    python -m src.ai.analysis_pipeline --run-one "<URL>"                       # preflight, без вызовов
    python -m src.ai.analysis_pipeline --run-one "<URL>" --confirm-paid-call   # реальный запуск
    python -m src.ai.analysis_pipeline --run-one "<URL>" --allow-expired --confirm-paid-call   # истёкший срок

Истёкший срок (Operational Eligibility) по умолчанию останавливает тендер до любого платного вызова даже с
--confirm-paid-call. --allow-expired обходит ТОЛЬКО этот deadline-гейт; BudgetGuard, accounting, резервы,
лимит Deep input, идемпотентность, валидация и Commercial Gate работают как обычно.

Здесь нет AI-логики: вызывается существующий AnalysisPipeline.process_announcement(url) (идемпотентность,
Commercial Gate, резервы, ledger, BudgetGuard остаются в нём). Никакого выбора backlog / process_batch.
Не зависит от AI_ANALYSIS_ENABLED (ручной запуск), но без confirm платных вызовов нет. Terra не вызывается.
Analyzers инъектируются: client создаётся ими лениво только при реальном вызове (без confirm — никогда).
"""

import sqlite3
from contextlib import closing
from decimal import Decimal

from src.ai import commercial_gate, deep_admission, deep_prompt, operational_eligibility, preflight, pricing
from src.ai import tender_context as tender_context_module
from src.ai.analysis_pipeline import AnalysisPipeline, _ReadOnlyStore, _stage_is_current
from src.ai.budget_guard import ANALYSIS_TRIAGE
from src.database import ai_usage_repository, tender_repository

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def _print_operational_eligibility(ctx, now, allow_expired, out) -> operational_eligibility.Eligibility:
    eligibility = operational_eligibility.evaluate_context(ctx, now)
    out("Operational Eligibility:")
    out(f"  status: {eligibility.status}")
    out(f"  resolved deadline: {eligibility.resolved_deadline.isoformat() if eligibility.resolved_deadline else '-'}")
    for source, raw, parsed in eligibility.deadline_sources:
        out(f"  deadline source: {source} = {raw!r} -> {parsed or 'не распознано'}")
    if not eligibility.deadline_sources:
        out("  deadline source: нет значений (срок неизвестен, пропуск запрещён)")
    out(f"  evaluation time: {eligibility.evaluated_at.isoformat()}")
    out(f"  expired: {str(eligibility.expired).lower()}")
    out(f"  operator override (--allow-expired): {str(allow_expired).lower()}")
    return eligibility


def _print_triage_preflight(url, triage, deep, settings, store, ctx, input_hash, out) -> bool:
    """Печатает preflight triage. False — оценить стоимость нельзя (вызывать нельзя)."""
    request = triage.build_request(tender_context_module.build_triage_context(ctx))
    try:
        pre = preflight.preflight_request(request, triage.model, ANALYSIS_TRIAGE, settings)
    except pricing.CostEstimationError as error:
        out(f"ERROR: стоимость triage оценить нельзя: {error}")
        return False
    actual = ai_usage_repository.current_month_cost(db_path=store.db_path) if "ai_usage_events" in store.tables else Decimal(0)
    reservations = store.outstanding_reservations()
    reserved = sum((Decimal(r["estimated_cost_usd"]) for r in reservations), Decimal(0))
    reuse = _stage_is_current(store.get_triage(url), input_hash, triage)
    out("PREFLIGHT (one-shot, один тендер)")
    out(f"  resource_url: {url}")
    out(f"  triage model: {triage.model}")
    out(f"  triage reasoning effort: {settings.triage_reasoning_effort}")
    out(f"  deep primary model: {deep.model}")
    out(f"  deep reasoning effort: {settings.deep_primary_reasoning_effort}")
    out(f"  monthly actual spend: ${actual}")
    out(f"  outstanding reservations: {len(reservations)} (${reserved})")
    out(f"  effective committed spend: ${store.month_spend()}")
    out(f"  estimated triage input tokens: {pre['estimated_input_tokens']}")
    out(f"  estimated triage max cost: ${pre['estimated_cost_usd']}")
    out(f"  monthly hard limit: ${settings.monthly_budget_usd}")
    out(f"  soft limit: ${settings.soft_limit_usd}")
    out(f"  triage result current (будет переиспользован, без вызова): {reuse}")
    return True


def _gate_with_deep_preflight(deep, settings, store, out):
    """Обёртка Commercial Gate: ничего не меняет, но печатает Gate и Deep Admission ДО Deep-вызова."""
    def gate(triage_result, ctx, min_value):
        result = commercial_gate.evaluate(triage_result, ctx, min_value)
        out(f"DEEP PREFLIGHT: Commercial Gate result: {result['gate_decision']} ({result['gate_reason']})")
        if result["gate_decision"] == commercial_gate.DEEP_CANDIDATE:
            deep_context = deep_prompt.build_deep_context(
                ctx, tender_context_module.build_deep_analysis_context(ctx), triage_result,
            )
            admission = deep_admission.evaluate_deep_admission(result, deep, deep_context, store.month_spend(), settings)
            pre = admission["preflight"]
            out(f"  Deep Admission result: {admission['deep_analysis_status']} ({admission['reason']})")
            if pre is not None:
                out(f"  estimated Deep input tokens: {pre['estimated_input_tokens']}")
                out(f"  estimated Deep max cost: ${pre['estimated_cost_usd']}")
        return result
    return gate


def _ledger_rows(db_path, url, after_id):
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute(
            "SELECT input_tokens, cached_input_tokens, cache_write_tokens, output_tokens, estimated_cost_usd "
            "FROM ai_usage_events WHERE reference = ? AND id > ?", (url, after_id),
        ).fetchall()


def _max_event_id(db_path) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute("SELECT COALESCE(MAX(id), 0) FROM ai_usage_events").fetchone()[0]


def _print_report(url, outcome, rows, db_path, out) -> None:
    from src.database import analysis_repository

    triage_row = analysis_repository.get_triage(url, db_path=db_path)
    deep_row = analysis_repository.get_deep_analysis(url, db_path=db_path)
    state = outcome["state"]
    out("OPERATOR REPORT")
    out(f"  resource_url: {url}")
    out(f"  final pipeline state: {state}" + (f" (reason: {outcome['reason_code']})" if outcome.get("reason_code") else ""))
    if triage_row is not None:
        out(f"  triage: {'reused' if outcome['reused']['triage'] else 'called'}, "
            f"relevance_status={triage_row['relevance_status']}, reason={triage_row['result'].get('reason')}")
    else:
        out("  triage: нет результата")
    gate_info = "см. состояние пайплайна"
    if state.startswith("commercial_gate") or outcome.get("reason_code") in (commercial_gate.DEEP_CANDIDATE,):
        gate_info = f"{outcome.get('reason_code')} ({outcome.get('message')})"
    elif deep_row is not None or state.startswith("deep") or state == "escalation_candidate":
        gate_info = "deep_candidate (Deep достигнут)"
    elif state == "stopped_not_relevant":
        gate_info = "не вызывался (not_relevant)"
    out(f"  Commercial Gate result: {gate_info}")
    if deep_row is not None:
        out(f"  deep: {'reused' if outcome['reused']['deep'] else 'called'}, summary={deep_row['result'].get('summary')}")
    else:
        out(f"  deep: нет результата ({outcome.get('message')})" if state != "stopped_not_relevant" else "  deep: не запускался")
    out(f"  escalation reason: {outcome.get('escalation_reason') or '-'}")
    calls = outcome["api_calls"]
    out(f"  API calls actually made: {calls['triage'] + calls['deep']} (triage {calls['triage']}, deep {calls['deep']})")
    out("  usage:")
    out(f"    input_tokens: {sum(r[0] for r in rows)}")
    out(f"    cached_tokens: {sum(r[1] for r in rows)}")
    out(f"    cache_write_tokens: {sum(r[2] for r in rows)}")
    out(f"    output_tokens: {sum(r[3] for r in rows)}")
    out(f"  actual estimated_cost_usd (ledger): ${sum((Decimal(r[4]) for r in rows), Decimal(0))}")
    reserved = ai_usage_repository.outstanding_reservations(db_path=db_path)
    out(f"  monthly actual spend after run: ${ai_usage_repository.current_month_cost(db_path=db_path)}")
    out(f"  outstanding reservations after run: {len(reserved)}")


def run_one(
    resource_url, triage, deep, settings, db_path=None, confirm=False, out=print, allow_expired=False,
    clock=operational_eligibility.utc_now,
) -> int:
    """Код возврата: 0 — ok / preflight-only, 1 — пайплайн завершился ошибкой, 2 — ошибка запуска (вызовов нет)."""
    db_path = tender_repository.resolve_db_path(db_path)  # один раз; дальше везде уже resolved Path, не None
    try:
        store = _ReadOnlyStore(db_path)
    except sqlite3.Error as error:
        out(f"ERROR: БД недоступна: {error}")
        return EXIT_USAGE
    if resource_url not in store.enriched_urls():
        out(f"ERROR: resource_url отсутствует в БД (нет объявления с enrichment): {resource_url}")
        return EXIT_USAGE

    ctx = tender_context_module.build_tender_context(resource_url, db_path=db_path)
    input_hash = tender_context_module.compute_input_hash(ctx)
    eligibility = _print_operational_eligibility(ctx, clock(), allow_expired, out)
    if not _print_triage_preflight(resource_url, triage, deep, settings, store, ctx, input_hash, out):
        return EXIT_USAGE
    if not confirm:
        out("DRY-RUN: платных вызовов нет, OpenAI client не создан. Для реального запуска добавьте --confirm-paid-call")
        return EXIT_OK

    if eligibility.expired and not allow_expired:
        out("Срок подачи истёк: НОВЫЕ платные вызовы не выполняются. Исторический платный запуск: "
            "--allow-expired --confirm-paid-call (BudgetGuard и остальные ограничения остаются).")
    pipeline = AnalysisPipeline(
        triage, deep, settings, db_path=db_path, gate=_gate_with_deep_preflight(deep, settings, store, out),
        clock=clock, allow_expired=allow_expired,
    )
    pipeline._ensure_ready()
    before = _max_event_id(db_path)
    outcome = pipeline.process_announcement(resource_url)  # ровно один тендер, без batch
    _print_report(resource_url, outcome, _ledger_rows(db_path, resource_url, before), db_path, out)
    return EXIT_FAILED if outcome["failed"] else EXIT_OK
