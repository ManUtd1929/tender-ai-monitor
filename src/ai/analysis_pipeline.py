"""
Production-оркестратор AI-анализа: только координирует уже реализованные стадии.

    announcement + enrichment + documents (tender_context)
      -> triage (Luna)          [preflight + BudgetGuard, ledger]
      -> routing по relevance_status:
            not_relevant -> СТОП (Gate и Deep не вызываются)
            relevant / maybe -> Commercial Gate (детерминированный, без LLM)
      -> Deep Admission (размер input + BudgetGuard)
      -> Deep (Luna)            [ledger]
      -> persistence

Маршрутизация maybe: Commercial Gate семантически поддерживает relevant/maybe (см. его
docstring), поэтому maybe проходит тот же Gate; relevance_status от Gate не меняется, а
причина отказа (если есть) явная. maybe никогда не превращается в not_relevant.

Владелец записи в usage ledger — ЭТОТ модуль (анализаторы в БД не пишут): ровно одна запись на
каждый фактический ответ API (usage из результата или из ошибки; дубликат response_id не пишется).
Порядок перед КАЖДЫМ платным вызовом: нет нерешённых резервов/блоков -> preflight -> BudgetGuard
(effective committed spend = ledger + нерешённые резервы) -> durable резерв в SQLite -> ТОЛЬКО ПОТОМ
вызов. После ответа usage и закрытие резерва — одна транзакция. Не удалось — резерв остаётся в БД,
платные вызовы блокируются (в т.ч. после перезапуска) до reconcile. Ответ без usage: резерв
освобождается только при заведомо неплатных ошибках (NO_BILLING_KINDS), иначе остаётся нерешённым.

Идемпотентность: платный вызов происходит, только если сохранённый результат стадии не
соответствует ТЕКУЩЕМУ (input_hash, prompt_version, provider, model). Отложенные состояния
(бюджет, размер input) и ошибки транспорта пересчитываются на следующем запуске без API-вызова, пока
блокирующее условие не изменится; детерминированные ошибки содержимого (validation/refusal/...) с тем же
(hash, prompt, model) повторно не оплачиваются.

Operational Eligibility Gate (src.ai.operational_eligibility): детерминированная проверка срока подачи ДО
любого НОВОГО платного вызова (triage и Deep). Надёжно истёкший срок -> состояние skipped_expired
(reason deadline_expired), $0: ни резерва, ни анализатора, ни client. Уже сохранённые triage/deep
результаты не удаляются и переиспользуются как раньше; пересчёт идёт каждый раз из текущих полей, поэтому
продление срока (меняет input_hash) само возвращает тендер в работу. unknown/conflict НЕ пропускаются.

Terra НЕ вызывается: причина эскалации только записывается (src.ai.escalation).
OpenAI client здесь не создаётся: analyzers инъектируются (в тестах — fakes), а production-analyzers
создают client лениво при первом реальном вызове. Dry-run не вызывает и не создаёт ничего платного.

CLI:
    python -m src.ai.analysis_pipeline --dry-run [--limit N]
    python -m src.ai.analysis_pipeline --run-one "<URL>" [--confirm-paid-call] [--allow-expired]   (см. src.ai.one_shot)
    python -m src.ai.analysis_pipeline --run-batch --limit N [--confirm-paid-call]   (см. src.ai.operator_batch)
"""

import argparse
import dataclasses
import logging
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.ai import (
    commercial_gate, deep_admission, deep_prompt, escalation, openai_triage, operational_eligibility, preflight,
    pricing,
)
from src.ai import tender_context as tender_context_module
from src.ai.budget_guard import ANALYSIS_DEEP, ANALYSIS_TRIAGE, BudgetGuard
from src.ai.budget_settings import BudgetSettings, load_budget_settings
from src.database import ai_usage_repository, analysis_repository, pipeline_state_repository as state_repo
from src.database.tender_repository import DEFAULT_DB_PATH, resolve_db_path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AI_ANALYSIS_BATCH_LIMIT_ENV = "AI_ANALYSIS_BATCH_LIMIT"

# Ошибки содержимого ответа: с тем же (hash, prompt, model) повтор дал бы тот же класс ошибки и
# снова стоил бы денег. Транспортные (timeout/rate_limit/api_error/config/auth) повторяются.
# Ошибки без ответа модели, при которых биллинга заведомо нет: резерв освобождается без usage.
# timeout / api_error / непредвиденное исключение — неоднозначны (запрос мог быть обработан) -> fail-closed.
NO_BILLING_KINDS = frozenset({
    openai_triage.KIND_CONFIG, openai_triage.KIND_AUTHENTICATION, openai_triage.KIND_RATE_LIMIT,
})

NON_RETRYABLE_ERROR_KINDS = frozenset({
    openai_triage.KIND_VALIDATION, openai_triage.KIND_INVALID_OUTPUT,
    openai_triage.KIND_REFUSAL, openai_triage.KIND_INCOMPLETE,
})

REASON_NOT_RELEVANT = "not_relevant"
REASON_PRICING_UNKNOWN = "pricing_unknown"
REASON_UNEXPECTED_ERROR = "unexpected_error"
REASON_ACCOUNTING_BLOCKED = "accounting_blocked"
# Только в outcome (в БД не сохраняется): тендер не тронут и будет обработан после reconcile.
OUTCOME_ACCOUNTING_BLOCKED = "accounting_blocked"

# Причины, по которым тендер попадает в batch (в порядке приоритета: сначала новые, потом retry).
# needs_routing — переиспользование без вызова ($0), истёкший срок ему не мешает.
PAID_WORK_REASONS = ("needs_triage", "retry_triage", "needs_deep", "retry_deep")
REASON_SKIP_EXPIRED = "skip_expired"  # причина кандидата (в БД не пишется): нужен только $0-учёт состояния
FRESH_REASONS = ("needs_triage", "needs_routing", "needs_deep")
RETRY_REASONS = ("retry_triage", "retry_deep")

FAILED_STATES = (
    state_repo.STATE_TRIAGE_ERROR, state_repo.STATE_DEEP_ERROR, state_repo.STATE_ESCALATION_CANDIDATE,
)


# --------------------------------------------------------------------------
# хранилище состояния (реальное и read-only для dry-run)
# --------------------------------------------------------------------------

class _RepositoryStore:
    def __init__(self, db_path):
        self.db_path = db_path

    def enriched_urls(self):
        return analysis_repository.list_enriched_resource_urls(db_path=self.db_path)

    def get_triage(self, url):
        return analysis_repository.get_triage(url, db_path=self.db_path)

    def get_deep(self, url):
        return analysis_repository.get_deep_analysis(url, db_path=self.db_path)

    def get_state(self, url):
        return state_repo.get_state(url, db_path=self.db_path)

    def month_spend(self):
        """effective committed spend: факт из ledger + оценки нерешённых резервов месяца."""
        return ai_usage_repository.effective_committed_spend(db_path=self.db_path)

    def accounting_blocks(self):
        return ai_usage_repository.unresolved_accounting_blocks(db_path=self.db_path)

    def outstanding_reservations(self):
        return ai_usage_repository.outstanding_reservations(db_path=self.db_path)


class _ReadOnlyStore(_RepositoryStore):
    """Dry-run: ничего не создаёт и не мигрирует; отсутствующая таблица = «данных ещё нет»."""

    def __init__(self, db_path):
        super().__init__(db_path)
        path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as conn:
            self.tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self._path = path

    def enriched_urls(self):
        needed = {"announcements", "announcement_enrichment"}
        return super().enriched_urls() if needed <= self.tables else []

    def get_triage(self, url):
        return super().get_triage(url) if "tender_triage" in self.tables else None

    def get_deep(self, url):
        return super().get_deep(url) if "tender_deep_analysis" in self.tables else None

    def get_state(self, url):
        return super().get_state(url) if "tender_pipeline_state" in self.tables else None

    def accounting_blocks(self):
        if "ai_accounting_blocks" not in self.tables:
            return []
        with closing(sqlite3.connect(f"file:{self._path.as_posix()}?mode=ro", uri=True)) as conn:
            rows = conn.execute(
                "SELECT id, created_at, reference, failure FROM ai_accounting_blocks WHERE resolved_at IS NULL"
            ).fetchall()
        return [{"id": r[0], "created_at": r[1], "reference": r[2], "failure": r[3]} for r in rows]

    def outstanding_reservations(self):
        if "ai_call_reservations" not in self.tables:
            return []
        with closing(sqlite3.connect(f"file:{self._path.as_posix()}?mode=ro", uri=True)) as conn:
            rows = conn.execute(
                "SELECT id, created_at, analysis_type, reference, estimated_cost_usd, status FROM ai_call_reservations"
                " WHERE status IN ('reserved', 'unresolved') ORDER BY id"
            ).fetchall()
        return [
            {"id": r[0], "created_at": r[1], "analysis_type": r[2], "reference": r[3],
             "estimated_cost_usd": r[4], "status": r[5]} for r in rows
        ]

    def month_spend(self):
        """effective committed spend (как в боевом store), только чтение."""
        now = datetime.now(timezone.utc)
        start, end = ai_usage_repository.month_bounds(now.year, now.month)
        total = Decimal(0)
        with closing(sqlite3.connect(f"file:{self._path.as_posix()}?mode=ro", uri=True)) as conn:
            if "ai_usage_events" in self.tables:
                rows = conn.execute(
                    "SELECT estimated_cost_usd FROM ai_usage_events WHERE created_at >= ? AND created_at < ?",
                    (start, end),
                ).fetchall()
                total += sum((Decimal(row[0]) for row in rows), Decimal(0))
            if "ai_call_reservations" in self.tables:
                rows = conn.execute(
                    "SELECT estimated_cost_usd FROM ai_call_reservations"
                    " WHERE status IN ('reserved', 'unresolved') AND created_at >= ? AND created_at < ?",
                    (start, end),
                ).fetchall()
                total += sum((Decimal(row[0]) for row in rows), Decimal(0))
        return total


# --------------------------------------------------------------------------
# чистые проверки идемпотентности
# --------------------------------------------------------------------------

def _stage_is_current(row, input_hash: str, analyzer) -> bool:
    """Сохранённый результат стадии соответствует текущему (hash, prompt_version, provider, model)."""
    return (
        row is not None
        and row["input_hash"] == input_hash
        and row["prompt_version"] == analyzer.prompt_version
        and row["provider"] == analyzer.provider
        and row["model"] == analyzer.model
    )


def _known_failure(prior, input_hash: str, analyzer, states) -> bool:
    """Та же стадия уже дала неповторяемую ошибку содержимого для ТОГО ЖЕ (hash, prompt, model)."""
    return (
        prior is not None
        and prior["state"] in states
        and prior["input_hash"] == input_hash
        and prior["model"] == analyzer.model
        and prior["prompt_version"] == analyzer.prompt_version
        and prior["error_kind"] in NON_RETRYABLE_ERROR_KINDS
    )


def _admission_details(admission: dict) -> dict:
    details = {
        "deep_analysis_status": admission["deep_analysis_status"],
        "reason": admission["reason"],
        "reason_code": admission["reason_code"],
    }
    if admission["preflight"] is not None:
        details["preflight"] = dict(admission["preflight"])
    if admission["budget"] is not None:
        details["budget"] = dataclasses.asdict(admission["budget"])
    return details


# --------------------------------------------------------------------------
# оркестратор
# --------------------------------------------------------------------------

class AnalysisPipeline:
    """
    triage_analyzer / deep_analyzer — объекты с model, provider, prompt_version, build_request(context) и
    triage_with_metadata(triage_context) / deep_analyze_with_metadata(tender_context, deep_analysis_context,
    triage_result) (OpenAITriageAnalyzer / OpenAIDeepAnalysisAnalyzer; в тестах — fakes).
    gate — commercial_gate.evaluate-совместимая функция.
    """

    def __init__(
        self, triage_analyzer, deep_analyzer, settings: BudgetSettings, db_path=None,
        gate=commercial_gate.evaluate, pricing_overrides: dict | None = None, store=None,
        clock=operational_eligibility.utc_now, allow_expired: bool = False,
    ):
        self.triage_analyzer = triage_analyzer
        self.deep_analyzer = deep_analyzer
        self.settings = settings
        self.db_path = db_path
        self.gate = gate
        self.pricing_overrides = pricing_overrides
        self.guard = BudgetGuard(settings)
        self.clock = clock  # инъектируемые часы: aware datetime; бизнес-логика не вызывает datetime.now()
        self.allow_expired = allow_expired  # операторский override ТОЛЬКО deadline-гейта (one-shot)
        self.store = store if store is not None else _RepositoryStore(db_path)
        self._ready = False
        # Резерв на случай, когда даже запись блока в БД не удалась: тогда этот экземпляр больше не платит.
        self._accounting_failure: str | None = None

    def _ensure_ready(self) -> None:
        if not self._ready:
            state_repo.init_db(self.db_path)
            ai_usage_repository.init_db(self.db_path)
            self._ready = True

    # -- состояние -----------------------------------------------------------

    def _save_state(self, url, state, input_hash, analyzer=None, **fields) -> None:
        state_repo.save_state(
            url, state, input_hash=input_hash,
            model=analyzer.model if analyzer is not None else None,
            prompt_version=analyzer.prompt_version if analyzer is not None else None,
            db_path=self.db_path, **fields,
        )

    def _finish(self, outcome: dict, state: str, reason_code=None, message=None, **extra) -> dict:
        outcome.update(state=state, reason_code=reason_code, message=message, **extra)
        outcome["failed"] = state in FAILED_STATES or state == OUTCOME_ACCOUNTING_BLOCKED
        return outcome

    # -- ledger --------------------------------------------------------------

    def _reserve(self, analysis_type, analyzer, url, input_hash, estimated_cost):
        """
        Durable pre-call reservation. (reservation_id, None) или (None, причина): без успешной записи
        резерва платный вызов НЕ начинается (нестоимость $0).
        """
        try:
            reservation_id = ai_usage_repository.reserve_call(
                analysis_type, analyzer.model, estimated_cost, reference=url,
                prompt_version=analyzer.prompt_version, input_hash=input_hash, db_path=self.db_path,
            )
        except ai_usage_repository.ReservationBlocked as error:
            return None, f"резерв не создан: {error}"
        except Exception as error:
            logger.exception("Резерв не записан, платный вызов не начат (%s, %s)", analysis_type, url)
            return None, f"резерв не записан ({type(error).__name__}: {error})"
        logger.info("Резерв #%s создан: %s %s оценка $%s", reservation_id, analysis_type, url, estimated_cost)
        return reservation_id, None

    def _settle(self, reservation_id, analysis_type, analyzer, url, usage, response_id, success, error_kind=None):
        """
        Фактический usage -> ledger и закрытие резерва одной транзакцией (settle_reservation). Возвращает
        стоимость или None (дубликат response_id — безопасно). Если не удалось, резерв ОСТАЁТСЯ в БД
        нерешённым (переживёт перезапуск) — платные вызовы остановлены до reconcile.
        """
        try:
            return ai_usage_repository.settle_reservation(
                reservation_id, analysis_type, analyzer.model, usage, reference=url, response_id=response_id,
                prompt_version=analyzer.prompt_version, success=success, error_kind=error_kind,
                pricing_overrides=self.pricing_overrides, db_path=self.db_path,
            )
        except Exception as error:
            self._accounting_unresolved(
                reservation_id, analysis_type, analyzer, url, usage, response_id, success, error_kind, error,
            )
            return None

    def _finalize_error(self, reservation_id, analysis_type, analyzer, url, error):
        """
        Итог TriageError. С usage — платный ответ: settle (успешный контент не обязателен). Без usage —
        release только если биллинг заведомо невозможен (NO_BILLING_KINDS); иначе неоднозначно ->
        резерв остаётся нерешённым (fail-closed).
        """
        if error.usage is not None:
            return self._settle(
                reservation_id, analysis_type, analyzer, url, error.usage, error.response_id, False, error.kind,
            )
        if error.kind in NO_BILLING_KINDS:
            try:
                ai_usage_repository.release_reservation(
                    reservation_id, f"нет платного ответа ({error.kind}), usage отсутствует", db_path=self.db_path,
                )
            except Exception:
                logger.critical("Резерв #%s не освобождён (%s): остаётся нерешённым", reservation_id, url, exc_info=True)
                self._accounting_failure = f"резерв #{reservation_id} не освобождён"
            return None
        logger.critical(
            "Биллинг неоднозначен (%s, usage нет): резерв #%s остаётся нерешённым; %s. Платные вызовы остановлены "
            "до решения оператора (python -m src.ai.accounting_reconcile --release).", error.kind, reservation_id, url,
        )
        self._mark_unresolved(reservation_id, f"неоднозначный биллинг: {error.kind}")
        self._accounting_failure = f"резерв #{reservation_id}: неоднозначный биллинг ({error.kind})"
        return None

    def _mark_unresolved(self, reservation_id, note):
        try:
            ai_usage_repository.mark_reservation_unresolved(reservation_id, note, db_path=self.db_path)
        except Exception:
            logger.warning("Резерв #%s не помечен unresolved (остаётся reserved — блокирует так же)", reservation_id)

    def _accounting_unresolved(
        self, reservation_id, analysis_type, analyzer, url, usage, response_id, success, error_kind, error,
    ):
        failure = f"{type(error).__name__}: {error}"
        logger.critical(
            "Ledger: НЕ удалось завершить учёт платного ответа (%s, %s, response_id=%s, резерв #%s): %s. "
            "Резерв остаётся в БД; платные AI-вызовы остановлены до reconcile.",
            analysis_type, url, response_id, reservation_id, failure,
        )
        self._accounting_failure = failure
        self._mark_unresolved(reservation_id, failure)
        try:  # best-effort: сохраняет реальный usage для reconcile; устойчивость обеспечивает сам резерв
            block_id = ai_usage_repository.open_accounting_block(
                analysis_type, analyzer.model, usage, failure, reference=url, response_id=response_id,
                prompt_version=analyzer.prompt_version, success=success, error_kind=error_kind,
                reservation_id=reservation_id, db_path=self.db_path,
            )
            logger.critical("Accounting-блок #%s открыт; снять: python -m src.ai.accounting_reconcile", block_id)
        except Exception:
            logger.critical("Accounting-блок не сохранён (резерв #%s блокирует и без него); usage=%s",
                            reservation_id, usage, exc_info=True)

    def _accounting_block_reason(self) -> str | None:
        """Причина, по которой платный вызов запрещён (нерешённый долг учёта), или None. Ошибка чтения = ошибка."""
        reservations = self.store.outstanding_reservations()
        if reservations:
            first = reservations[0]
            return (
                f"нерешённых резервов вызовов: {len(reservations)} (первый #{first['id']}, "
                f"{first['status']}, {first['reference']})"
            )
        blocks = self.store.accounting_blocks()
        if blocks:
            return f"нерешённых accounting-блоков: {len(blocks)} (первый #{blocks[0]['id']}: {blocks[0]['failure']})"
        if self._accounting_failure is not None:
            return f"ledger не принял платный usage в этом запуске: {self._accounting_failure}"
        return None

    def _blocked_outcome(self, outcome, url, reason):
        logger.error("Платный вызов не начат (accounting fail-closed): %s — %s", url, reason)
        return self._finish(outcome, OUTCOME_ACCOUNTING_BLOCKED, REASON_ACCOUNTING_BLOCKED, reason)

    # -- operational eligibility ----------------------------------------------

    def _expired_blocks_new_call(self, eligibility) -> bool:
        return eligibility.expired and not self.allow_expired

    def _skip_expired(self, url, input_hash, eligibility, outcome) -> dict:
        """Срок надёжно истёк: $0-состояние для ЭТОГО набора данных. Не необратимо: пересчитывается из источников."""
        facts = eligibility.to_dict()
        message = f"срок подачи истёк: {facts['resolved_deadline']} (проверено {facts['evaluated_at']})"
        logger.info("Тендер пропущен без AI (deadline_expired): %s — %s", url, message)
        self._save_state(
            url, state_repo.STATE_SKIPPED_EXPIRED, input_hash,
            reason_code=operational_eligibility.REASON_DEADLINE_EXPIRED, message=message,
            details={
                "resolved_deadline": facts["resolved_deadline"], "deadline_sources": facts["deadline_sources"],
                "evaluated_at": facts["evaluated_at"],
            },
        )
        return self._finish(
            outcome, state_repo.STATE_SKIPPED_EXPIRED, operational_eligibility.REASON_DEADLINE_EXPIRED, message,
        )

    # -- триаж ---------------------------------------------------------------

    def _run_triage(self, url, ctx, input_hash, outcome):
        """(triage_result, None) или (None, finished_outcome)."""
        analyzer = self.triage_analyzer
        block_reason = self._accounting_block_reason()
        if block_reason:
            return None, self._blocked_outcome(outcome, url, block_reason)
        triage_context = tender_context_module.build_triage_context(ctx)
        request = analyzer.build_request(triage_context)

        try:
            pre = preflight.preflight_request(
                request, analyzer.model, ANALYSIS_TRIAGE, self.settings, self.pricing_overrides,
            )
        except pricing.CostEstimationError as error:
            self._save_state(
                url, state_repo.STATE_TRIAGE_ERROR, input_hash, analyzer,
                reason_code=REASON_PRICING_UNKNOWN, error_kind=REASON_PRICING_UNKNOWN, message=str(error),
            )
            return None, self._finish(outcome, state_repo.STATE_TRIAGE_ERROR, REASON_PRICING_UNKNOWN, str(error))

        decision = self.guard.check(
            self.store.month_spend(), pre["estimated_cost_usd"], ANALYSIS_TRIAGE, analyzer.model,
        )
        if not decision.allowed:
            logger.info("Triage отложен (бюджет): %s — %s", url, decision.reason)
            self._save_state(
                url, state_repo.STATE_TRIAGE_DEFERRED_BUDGET, input_hash, analyzer,
                reason_code=escalation.BUDGET_DEFERRED, message=decision.reason,
                details={"budget": dataclasses.asdict(decision)},
            )
            return None, self._finish(
                outcome, state_repo.STATE_TRIAGE_DEFERRED_BUDGET, escalation.BUDGET_DEFERRED, decision.reason,
            )

        reservation_id, failure = self._reserve(
            ANALYSIS_TRIAGE, analyzer, url, input_hash, pre["estimated_cost_usd"],
        )
        if reservation_id is None:
            return None, self._blocked_outcome(outcome, url, failure)

        outcome["api_calls"]["triage"] += 1
        try:
            response = analyzer.triage_with_metadata(triage_context)
        except openai_triage.TriageError as error:
            self._finalize_error(reservation_id, ANALYSIS_TRIAGE, analyzer, url, error)
            logger.error("Triage API/validation ошибка (%s): %s — %s", error.kind, url, error)
            self._save_state(
                url, state_repo.STATE_TRIAGE_ERROR, input_hash, analyzer,
                reason_code=error.kind, error_kind=error.kind, message=str(error),
                details={"usage": error.usage, "response_id": error.response_id, "attempts": error.attempts},
            )
            return None, self._finish(outcome, state_repo.STATE_TRIAGE_ERROR, error.kind, str(error))

        self._settle(
            reservation_id, ANALYSIS_TRIAGE, analyzer, url, response["usage"], response["response_id"], True,
        )
        analysis_repository.save_triage(
            url, input_hash, response["result"], provider=analyzer.provider, model=analyzer.model,
            prompt_version=analyzer.prompt_version, db_path=self.db_path,
        )
        self._save_state(
            url, state_repo.STATE_TRIAGE_COMPLETED, input_hash, analyzer,
            reason_code=response["result"]["relevance_status"],
            details={"usage": response["usage"], "response_id": response["response_id"]},
        )
        return response["result"], None

    # -- deep ----------------------------------------------------------------

    def _run_deep(self, url, ctx, input_hash, triage_result, gate_result, outcome):
        analyzer = self.deep_analyzer
        block_reason = self._accounting_block_reason()
        if block_reason:
            return self._blocked_outcome(outcome, url, block_reason)
        gate_details = {"gate": gate_result, "min_deep_value_amd": self.settings.min_deep_value_amd}

        deep_analysis_context = tender_context_module.build_deep_analysis_context(ctx)
        deep_context = deep_prompt.build_deep_context(ctx, deep_analysis_context, triage_result)
        admission = deep_admission.evaluate_deep_admission(
            gate_result, analyzer, deep_context, self.store.month_spend(), self.settings, self.pricing_overrides,
        )
        details = {**gate_details, "admission": _admission_details(admission)}
        status = admission["deep_analysis_status"]

        if status != deep_admission.READY:
            if status == deep_admission.DEFERRED_INPUT_TOO_LARGE:
                state = state_repo.STATE_DEEP_DEFERRED_INPUT
                details["max_deep_input_tokens"] = self.settings.max_deep_input_tokens
            elif status == deep_admission.BUDGET_DEFERRED:
                state = state_repo.STATE_DEEP_DEFERRED_BUDGET
            else:  # pricing_unknown: цену модели оценить нельзя, вызывать нельзя
                state = state_repo.STATE_DEEP_ERROR
            reason_code = admission["reason_code"] or status
            logger.info("Deep не вызывается (%s): %s — %s", status, url, admission["reason"])
            self._save_state(
                url, state, input_hash, analyzer, reason_code=reason_code, message=admission["reason"],
                error_kind=status if state == state_repo.STATE_DEEP_ERROR else None, details=details,
            )
            return self._finish(outcome, state, reason_code, admission["reason"])

        reservation_id, failure = self._reserve(
            ANALYSIS_DEEP, analyzer, url, input_hash, admission["preflight"]["estimated_cost_usd"],
        )
        if reservation_id is None:
            return self._blocked_outcome(outcome, url, failure)

        outcome["api_calls"]["deep"] += 1
        try:
            response = analyzer.deep_analyze_with_metadata(ctx, deep_analysis_context, triage_result)
        except openai_triage.TriageError as error:  # DeepAnalysisError — его подкласс
            self._finalize_error(reservation_id, ANALYSIS_DEEP, analyzer, url, error)
            escalation_reason = escalation.reason_for_error_kind(error.kind)
            state = state_repo.STATE_ESCALATION_CANDIDATE if escalation_reason else state_repo.STATE_DEEP_ERROR
            logger.error("Deep ошибка (%s): %s — %s; escalation=%s", error.kind, url, error, escalation_reason)
            self._save_state(
                url, state, input_hash, analyzer, reason_code=error.kind, error_kind=error.kind,
                escalation_reason=escalation_reason, message=str(error),
                details={**details, "usage": error.usage, "response_id": error.response_id, "attempts": error.attempts,
                         **({"raw_model_output": error.raw_model_output}
                            if getattr(error, "raw_model_output", None) is not None else {})},
            )
            return self._finish(outcome, state, error.kind, str(error), escalation_reason=escalation_reason)

        cost = self._settle(
            reservation_id, ANALYSIS_DEEP, analyzer, url, response["usage"], response["response_id"], True,
        )
        analysis_repository.save_deep_analysis(
            url, input_hash, response["result"], provider=analyzer.provider, model=analyzer.model,
            prompt_version=analyzer.prompt_version, db_path=self.db_path,
        )
        self._save_state(
            url, state_repo.STATE_DEEP_COMPLETED, input_hash, analyzer, reason_code="deep_completed",
            details={
                **details, "raw_model_output": response["raw_model_output"], "usage": response["usage"],
                "response_id": response["response_id"], "attempts": response["attempts"],
                "estimated_cost_usd": cost,
            },
        )
        return self._finish(outcome, state_repo.STATE_DEEP_COMPLETED, "deep_completed")

    # -- один тендер ---------------------------------------------------------

    def process_announcement(self, resource_url: str) -> dict:
        """
        Полный проход одного объявления. Никогда не бросает: ошибка -> outcome с failed=True (и, по
        возможности, сохранённым состоянием). outcome: {resource_url, state, reason_code, message,
        api_calls{triage, deep}, reused{triage, deep}, failed, [escalation_reason, error_type]}.
        """
        self._ensure_ready()
        outcome = {
            "resource_url": resource_url, "api_calls": {"triage": 0, "deep": 0},
            "reused": {"triage": False, "deep": False},
        }
        progress = {"stage": "context", "input_hash": None}
        try:
            return self._process(resource_url, outcome, progress)
        except Exception as error:
            logger.exception("Необработанная ошибка пайплайна: %s (стадия %s)", resource_url, progress["stage"])
            state = state_repo.STATE_TRIAGE_ERROR if progress["stage"] in ("context", "triage") else state_repo.STATE_DEEP_ERROR
            try:
                self._save_state(
                    resource_url, state, progress["input_hash"], reason_code=REASON_UNEXPECTED_ERROR,
                    error_kind=type(error).__name__, message=str(error),
                    details={"stage": progress["stage"]},
                )
            except Exception:
                logger.exception("Не удалось сохранить состояние ошибки: %s", resource_url)
            return self._finish(
                outcome, state, REASON_UNEXPECTED_ERROR, str(error),
                error_type=type(error).__name__, failed_stage=progress["stage"],
            )

    def _process(self, url: str, outcome: dict, progress: dict) -> dict:
        ctx = tender_context_module.build_tender_context(url, db_path=self.db_path)
        input_hash = tender_context_module.compute_input_hash(ctx)
        progress["input_hash"] = input_hash
        prior = self.store.get_state(url)
        eligibility = operational_eligibility.evaluate_context(ctx, self.clock())
        outcome["operational_eligibility"] = eligibility.status

        # --- triage: reuse или новый вызов
        progress["stage"] = "triage"
        triage_row = self.store.get_triage(url)
        if _stage_is_current(triage_row, input_hash, self.triage_analyzer):
            logger.info("Triage актуален, пропуск вызова: %s", url)
            triage_result = triage_row["result"]
            outcome["reused"]["triage"] = True
        else:
            if _known_failure(prior, input_hash, self.triage_analyzer, (state_repo.STATE_TRIAGE_ERROR,)):
                logger.info("Triage: известная неповторяемая ошибка для тех же входных данных, пропуск: %s", url)
                return self._finish(outcome, prior["state"], prior["reason_code"], prior["message"], skipped_known_failure=True)
            if self._expired_blocks_new_call(eligibility):
                return self._skip_expired(url, input_hash, eligibility, outcome)
            triage_result, finished = self._run_triage(url, ctx, input_hash, outcome)
            if finished is not None:
                return finished

        # --- routing
        status = triage_result["relevance_status"]
        if status == "not_relevant":
            if prior is None or prior["state"] != state_repo.STATE_STOPPED_NOT_RELEVANT or prior["input_hash"] != input_hash:
                self._save_state(
                    url, state_repo.STATE_STOPPED_NOT_RELEVANT, input_hash, self.triage_analyzer,
                    reason_code=REASON_NOT_RELEVANT, message=triage_result["reason"],
                )
            return self._finish(outcome, state_repo.STATE_STOPPED_NOT_RELEVANT, REASON_NOT_RELEVANT, triage_result["reason"])

        # --- relevant / maybe: reuse Deep
        progress["stage"] = "deep"
        if _stage_is_current(self.store.get_deep(url), input_hash, self.deep_analyzer):
            logger.info("Deep актуален, пропуск вызова: %s", url)
            outcome["reused"]["deep"] = True
            if prior is None or prior["state"] != state_repo.STATE_DEEP_COMPLETED or prior["input_hash"] != input_hash:
                self._save_state(
                    url, state_repo.STATE_DEEP_COMPLETED, input_hash, self.deep_analyzer,
                    reason_code="deep_reused", details={"note": "использован ранее сохранённый deep analysis"},
                )
            return self._finish(outcome, state_repo.STATE_DEEP_COMPLETED, "deep_reused")
        if _known_failure(
            prior, input_hash, self.deep_analyzer,
            (state_repo.STATE_DEEP_ERROR, state_repo.STATE_ESCALATION_CANDIDATE),
        ):
            logger.info("Deep: известная неповторяемая ошибка для тех же входных данных, пропуск: %s", url)
            return self._finish(
                outcome, prior["state"], prior["reason_code"], prior["message"],
                escalation_reason=prior["escalation_reason"], skipped_known_failure=True,
            )

        # --- Operational Eligibility: дальше только НОВАЯ работа (Gate -> Deep); reuse выше уже обработан
        if self._expired_blocks_new_call(eligibility):
            return self._skip_expired(url, input_hash, eligibility, outcome)

        # --- Commercial Gate (relevant и maybe; relevance_status не меняется)
        gate_result = self.gate(triage_result, ctx, self.settings.min_deep_value_amd)
        logger.info("Commercial Gate: %s -> %s (%s)", url, gate_result["gate_decision"], gate_result["gate_reason"])
        if gate_result["gate_decision"] != commercial_gate.DEEP_CANDIDATE:
            self._save_state(
                url, state_repo.STATE_COMMERCIAL_GATE_SKIP, input_hash, self.deep_analyzer,
                reason_code=gate_result["gate_decision"], message=gate_result["gate_reason"],
                details={"gate": gate_result, "min_deep_value_amd": self.settings.min_deep_value_amd,
                         "relevance_status": status},
            )
            return self._finish(
                outcome, state_repo.STATE_COMMERCIAL_GATE_SKIP, gate_result["gate_decision"], gate_result["gate_reason"],
            )
        self._save_state(
            url, state_repo.STATE_COMMERCIAL_GATE_PASS, input_hash, self.deep_analyzer,
            reason_code=gate_result["gate_decision"], message=gate_result["gate_reason"],
            details={"gate": gate_result, "min_deep_value_amd": self.settings.min_deep_value_amd,
                     "relevance_status": status},
        )
        return self._run_deep(url, ctx, input_hash, triage_result, gate_result, outcome)

    # -- batch ---------------------------------------------------------------

    def assess(self, url: str) -> tuple:
        """
        (pending_reason, eligibility), только чтение. Если работа нужна, но срок надёжно истёк (и нет override),
        причина = skip_expired: тендер не тратит AI и не занимает место в операторском limit. Если состояние
        skipped_expired для текущих данных уже записано — settled (None), повторно не пишется.
        """
        ctx = tender_context_module.build_tender_context(url, db_path=self.db_path)
        eligibility = operational_eligibility.evaluate_context(ctx, self.clock())
        reason = self._pending_reason_for(url, ctx)
        if reason in PAID_WORK_REASONS and self._expired_blocks_new_call(eligibility):
            prior = self.store.get_state(url)
            settled = (
                prior is not None and prior["state"] == state_repo.STATE_SKIPPED_EXPIRED
                and prior["input_hash"] == tender_context_module.compute_input_hash(ctx)
            )
            return (None if settled else REASON_SKIP_EXPIRED), eligibility
        return reason, eligibility

    def pending_reason(self, url: str) -> str | None:
        """
        Почему тендер надо обрабатывать (needs_* новые, retry_* повтор, skip_expired — только $0-учёт
        истёкшего срока) или None, если он «settled»: для текущих (hash, prompt, model, настроек) результат /
        терминальное состояние уже есть. Чтение only; API не вызывается.
        """
        return self.assess(url)[0]

    def _pending_reason_for(self, url: str, ctx: dict) -> str | None:
        input_hash = tender_context_module.compute_input_hash(ctx)
        prior = self.store.get_state(url)
        same = prior is not None and prior["input_hash"] == input_hash
        triage_row = self.store.get_triage(url)

        if not _stage_is_current(triage_row, input_hash, self.triage_analyzer):
            if _known_failure(prior, input_hash, self.triage_analyzer, (state_repo.STATE_TRIAGE_ERROR,)):
                return None
            return "needs_triage" if prior is None else "retry_triage"

        if triage_row["relevance_status"] == "not_relevant":
            return None if same and prior["state"] == state_repo.STATE_STOPPED_NOT_RELEVANT else "needs_routing"

        if _stage_is_current(self.store.get_deep(url), input_hash, self.deep_analyzer):
            return None if same and prior["state"] == state_repo.STATE_DEEP_COMPLETED else "needs_routing"
        if _known_failure(
            prior, input_hash, self.deep_analyzer,
            (state_repo.STATE_DEEP_ERROR, state_repo.STATE_ESCALATION_CANDIDATE),
        ):
            return None
        if same and prior["state"] == state_repo.STATE_COMMERCIAL_GATE_SKIP:
            if prior["details"].get("min_deep_value_amd") == self.settings.min_deep_value_amd:
                return None
        if same and prior["state"] == state_repo.STATE_DEEP_DEFERRED_INPUT:
            if (
                prior["details"].get("max_deep_input_tokens") == self.settings.max_deep_input_tokens
                and prior["model"] == self.deep_analyzer.model
                and prior["prompt_version"] == self.deep_analyzer.prompt_version
            ):
                return None
        if prior is None or prior["state"] in (
            state_repo.STATE_TRIAGE_COMPLETED, state_repo.STATE_COMMERCIAL_GATE_PASS,
            state_repo.STATE_PENDING_TRIAGE,
        ):
            return "needs_deep"
        return "retry_deep"

    def select_candidates(self, limit: int | None = None) -> list:
        """
        [{resource_url, reason, operational_status}]: только тендеры, требующие работы; новые раньше retry,
        внутри — по first_seen_at. limit — оператор-лимит числа тендеров за запуск (не бизнес-лимит в день).
        Кандидаты skip_expired ($0) в limit не входят и идут в конце: истёкшие не вытесняют активных.
        """
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError(f"limit должен быть положительным целым или None: {limit!r}")
        fresh, retry, expired = [], [], []
        for url in self.store.enriched_urls():
            try:
                reason, eligibility = self.assess(url)
                status = eligibility.status
            except Exception:
                logger.exception("Не удалось определить состояние тендера, считаю кандидатом: %s", url)
                reason, status = "needs_triage", operational_eligibility.STATUS_UNKNOWN
            candidate = {"resource_url": url, "reason": reason, "operational_status": status}
            if reason in FRESH_REASONS:
                fresh.append(candidate)
            elif reason in RETRY_REASONS:
                retry.append(candidate)
            elif reason == REASON_SKIP_EXPIRED:
                expired.append(candidate)
        candidates = fresh + retry
        if limit is not None:
            candidates = candidates[:limit]
        return candidates + expired

    def process_batch(self, limit: int | None = None) -> dict:
        """Ошибка одного тендера не прерывает остальные (process_announcement не бросает)."""
        self._ensure_ready()
        candidates = self.select_candidates(limit)
        logger.info("AI analysis: кандидатов %d", len(candidates))
        results, state_counts = [], {}
        api_calls = {"triage": 0, "deep": 0}
        accounting_blocked = False
        ledger_before = self._ledger_max_id()
        for candidate in candidates:
            if accounting_blocked:
                logger.critical(
                    "AI batch остановлен: accounting-блок, новые платные вызовы не начаты; "
                    "остаток кандидатов перейдёт на следующий запуск после reconcile"
                )
                break
            try:
                outcome = self.process_announcement(candidate["resource_url"])
            except Exception as error:  # защитный слой: process_announcement по контракту не бросает
                logger.exception("Batch: неожиданное исключение: %s", candidate["resource_url"])
                outcome = {
                    "resource_url": candidate["resource_url"], "state": None, "failed": True,
                    "reason_code": REASON_UNEXPECTED_ERROR, "error_type": type(error).__name__,
                    "message": str(error), "api_calls": {"triage": 0, "deep": 0},
                }
            results.append(outcome)
            accounting_blocked = (
                outcome.get("reason_code") == REASON_ACCOUNTING_BLOCKED or self._accounting_failure is not None
            )
            state_counts[outcome["state"]] = state_counts.get(outcome["state"], 0) + 1
            for name in api_calls:
                api_calls[name] += outcome["api_calls"][name]
        failures = [r for r in results if r["failed"]]
        logger.info(
            "AI analysis завершён: обработано %d, ошибок %d, вызовов triage %d / deep %d",
            len(results), len(failures), api_calls["triage"], api_calls["deep"],
        )
        return {
            "candidate_count": len(candidates), "processed_count": len(results),
            "expired_candidate_count": sum(1 for c in candidates if c["reason"] == REASON_SKIP_EXPIRED),
            "failed_count": len(failures), "state_counts": state_counts, "api_calls": api_calls,
            "results": results, "failures": failures, "accounting_blocked": accounting_blocked,
            "batch_cost_usd": self._ledger_cost_since(ledger_before),
        }

    def _ledger_max_id(self) -> int | None:
        try:
            with closing(sqlite3.connect(resolve_db_path(self.db_path))) as conn:
                return conn.execute("SELECT COALESCE(MAX(id), 0) FROM ai_usage_events").fetchone()[0]
        except Exception:
            logger.warning("Не удалось прочитать ledger до batch; стоимость batch будет неизвестна", exc_info=True)
            return None

    def _ledger_cost_since(self, before_id: int | None) -> str | None:
        """Фактическая стоимость этого batch по ledger (строка Decimal) или None, если недоступна."""
        if before_id is None:
            return None
        try:
            with closing(sqlite3.connect(resolve_db_path(self.db_path))) as conn:
                rows = conn.execute(
                    "SELECT estimated_cost_usd FROM ai_usage_events WHERE id > ?", (before_id,)
                ).fetchall()
            return str(sum((Decimal(r[0]) for r in rows), Decimal(0)))
        except Exception:
            logger.warning("Не удалось прочитать стоимость batch из ledger", exc_info=True)
            return None


# --------------------------------------------------------------------------
# production-конфигурация и dry-run
# --------------------------------------------------------------------------

def build_analyzers(settings: BudgetSettings, environ=None):
    """Production-analyzers из конфига. Client НЕ создаётся (создаётся лениво при первом реальном вызове)."""
    from src.ai.openai_deep_analysis import OpenAIDeepAnalysisAnalyzer
    from src.ai.openai_triage import OpenAITriageAnalyzer

    triage = OpenAITriageAnalyzer(
        model=settings.triage_model, reasoning_effort=settings.triage_reasoning_effort, environ=environ,
    )
    deep = OpenAIDeepAnalysisAnalyzer(
        model=settings.deep_primary_model, reasoning_effort=settings.deep_primary_reasoning_effort, environ=environ,
    )
    return triage, deep


def run_ai_analysis(db_path=None, limit: int | None = None, environ=None) -> dict:
    """
    Реальный запуск AI batch. limit=None = без лимита: monitor так НЕ вызывает (см. run_monitor_ai_batch).
    Без OPENAI_API_KEY не вызывает ничего.
    """
    settings = load_budget_settings(environ)
    triage, deep = build_analyzers(settings, environ)
    if not (triage.has_api_key and deep.has_api_key):
        logger.error("AI analysis пропущен: OPENAI_API_KEY не задан")
        return {"status": "skipped_no_api_key"}
    return AnalysisPipeline(triage, deep, settings, db_path=db_path).process_batch(limit)


class BatchLimitConfigError(ValueError):
    """AI_ANALYSIS_BATCH_LIMIT отсутствует или некорректен: AI в monitor не запускается (fail-safe)."""


def load_monitor_batch_limit(environ=None) -> int:
    """
    Лимит тендеров на ОДИН запуск monitor (не дневной и не бизнес-лимит). Обязателен: нет значения,
    не целое, <= 0 -> BatchLimitConfigError. Молчаливого «без лимита» нет.
    """
    environ = os.environ if environ is None else environ
    raw = (environ.get(AI_ANALYSIS_BATCH_LIMIT_ENV) or "").strip()
    if not raw:
        raise BatchLimitConfigError(f"{AI_ANALYSIS_BATCH_LIMIT_ENV} не задан (обязателен при AI_ANALYSIS_ENABLED=true)")
    try:
        value = int(raw)
    except ValueError:
        raise BatchLimitConfigError(f"{AI_ANALYSIS_BATCH_LIMIT_ENV} должен быть целым числом: {raw!r}") from None
    if value <= 0:
        raise BatchLimitConfigError(f"{AI_ANALYSIS_BATCH_LIMIT_ENV} должен быть > 0: {raw!r}")
    return value


def summarize_monitor_batch(batch: dict, limit: int) -> dict:
    """Компактная сводка автоматического batch для monitor result/log (без секретов)."""
    results = batch["results"]
    states = [r.get("state") for r in results]
    count = states.count
    # Истёкшие обрабатываются без AI ($0) и в batch_limit не входят: считаются отдельно от AI work items.
    expired_excluded = batch.get("expired_candidate_count", count(state_repo.STATE_SKIPPED_EXPIRED))
    selected_ai = batch["candidate_count"] - expired_excluded
    processed_ai = batch["processed_count"] - count(state_repo.STATE_SKIPPED_EXPIRED)
    return {
        "status": "ok", "ai_enabled": True, "batch_limit": limit,
        "selected_ai_candidates": selected_ai, "processed_ai_candidates": processed_ai,
        "expired_excluded": expired_excluded,
        "selected": selected_ai, "processed": processed_ai,  # aliases (только AI work items, без expired)
        "triage_api_calls": batch["api_calls"]["triage"], "deep_api_calls": batch["api_calls"]["deep"],
        "deep_completed": count(state_repo.STATE_DEEP_COMPLETED),
        "not_relevant": count(state_repo.STATE_STOPPED_NOT_RELEVANT),
        "errors": len(batch["failures"]),
        "escalations": count(state_repo.STATE_ESCALATION_CANDIDATE),
        "budget_deferred": count(state_repo.STATE_TRIAGE_DEFERRED_BUDGET) + count(state_repo.STATE_DEEP_DEFERRED_BUDGET),
        "accounting_blocked": batch["accounting_blocked"],
        "batch_cost_usd": batch.get("batch_cost_usd"),
    }


def run_monitor_ai_batch(db_path=None, environ=None) -> dict:
    """
    Production-вход AI для monitor: ровно один batch не более AI_ANALYSIS_BATCH_LIMIT тендеров, затем СТОП
    (остаток backlog — следующему запуску). Без корректного лимита ничего не создаётся и не вызывается.
    """
    try:
        limit = load_monitor_batch_limit(environ)
    except BatchLimitConfigError as error:
        logger.error("AI-анализ не запущен: %s", error)
        return {"status": "config_error", "ai_enabled": True, "batch_limit": None, "error_message": str(error)}
    batch = run_ai_analysis(db_path=db_path, limit=limit, environ=environ)
    if "results" not in batch:  # например skipped_no_api_key
        return {"ai_enabled": True, "batch_limit": limit, **batch}
    summary = summarize_monitor_batch(batch, limit)
    logger.info("AI batch (monitor): %s", summary)
    return summary


def _add_triage_preflight(total: dict, pre: dict) -> None:
    total["candidate_count"] += 1
    total["estimated_input_tokens_total"] += pre["estimated_input_tokens"]
    total["estimated_output_tokens_total"] += pre["estimated_output_tokens"]
    total["estimated_ordinary_or_cache_write_cost_usd"] += pre["estimated_input_cost_usd"]
    total["estimated_output_cost_usd"] += pre["estimated_output_cost_usd"]
    total["estimated_total_cost_usd"] += pre["estimated_cost_usd"]
    total["rate_tiers"][pre["rate_tier"]] = total["rate_tiers"].get(pre["rate_tier"], 0) + 1
    total["max_output_tokens_per_candidate"] = pre["estimated_output_tokens"]


def _operational_summary(pipeline: "AnalysisPipeline", store) -> dict:
    """
    Read-only срез по ВСЕМ объявлениям с enrichment (не зависит от limit): статусы срока и разбивка тех,
    кому нужна работа, по статусу срока (needs_triage_active / skip_expired / deadline_unknown / ...).
    """
    status_counts, pending = {}, {}
    for url in store.enriched_urls():
        reason, eligibility = pipeline.assess(url)
        status_counts[eligibility.status] = status_counts.get(eligibility.status, 0) + 1
        if reason is None:
            continue
        if reason == REASON_SKIP_EXPIRED:
            key = REASON_SKIP_EXPIRED
        elif eligibility.status == operational_eligibility.STATUS_ELIGIBLE:
            key = f"{reason}_active"
        elif eligibility.status == operational_eligibility.STATUS_UNKNOWN:
            key = f"{reason}_deadline_unknown"
        else:
            key = f"{reason}_deadline_conflict"
        pending[key] = pending.get(key, 0) + 1
    return {"status_counts": status_counts, "pending_by_status": pending}


def dry_run(db_path=None, limit: int | None = None, environ=None, now=None) -> dict:
    """
    Отчёт без платных операций и без записи: analyzers создаются, но client — никогда (build_request
    client не требует); БД открывается через read-only store. Показывает кандидатов и маршрутизацию,
    вычислимую без AI (Gate / Deep Admission по уже сохранённому triage).
    """
    settings = load_budget_settings(environ)
    triage, deep = build_analyzers(settings, environ)
    store = _ReadOnlyStore(db_path)
    clock = (lambda: now) if now is not None else operational_eligibility.utc_now
    pipeline = AnalysisPipeline(triage, deep, settings, db_path=db_path, store=store, clock=clock)

    spend = store.month_spend()
    candidates = pipeline.select_candidates(limit)
    reason_counts, routing = {}, {}
    triage_preflight = {
        "candidate_count": 0, "estimated_input_tokens_total": 0, "estimated_output_tokens_total": 0,
        "estimated_ordinary_or_cache_write_cost_usd": Decimal(0), "estimated_output_cost_usd": Decimal(0),
        "estimated_total_cost_usd": Decimal(0), "input_rate_per_million": None,
        "output_rate_per_million": None, "rate_tiers": {}, "max_output_tokens_per_candidate": None,
    }

    for candidate in candidates:
        url = candidate["resource_url"]
        reason_counts[candidate["reason"]] = reason_counts.get(candidate["reason"], 0) + 1
        if candidate["reason"] == REASON_SKIP_EXPIRED:  # $0: ни preflight, ни Gate/Deep Admission
            routing[REASON_SKIP_EXPIRED] = routing.get(REASON_SKIP_EXPIRED, 0) + 1
            continue
        triage_row = store.get_triage(url)
        ctx = tender_context_module.build_tender_context(url, db_path=db_path)
        if candidate["reason"] in ("needs_triage", "retry_triage"):
            request = triage.build_request(tender_context_module.build_triage_context(ctx))
            try:
                pre = preflight.preflight_request(request, triage.model, ANALYSIS_TRIAGE, settings)
                _add_triage_preflight(triage_preflight, pre)
            except pricing.CostEstimationError:
                pass
            key = "pending_triage"
        elif triage_row["relevance_status"] == "not_relevant":
            key = "stopped_not_relevant"
        else:
            triage_result = triage_row["result"]
            gate_result = commercial_gate.evaluate(triage_result, ctx, settings.min_deep_value_amd)
            if gate_result["gate_decision"] != commercial_gate.DEEP_CANDIDATE:
                key = f"commercial_gate_skip:{gate_result['gate_decision']}"
            else:
                deep_context = deep_prompt.build_deep_context(
                    ctx, tender_context_module.build_deep_analysis_context(ctx), triage_result,
                )
                admission = deep_admission.evaluate_deep_admission(gate_result, deep, deep_context, spend, settings)
                key = f"deep:{admission['deep_analysis_status']}"
        routing[key] = routing.get(key, 0) + 1

    operational = _operational_summary(pipeline, store)
    return {
        "operational_eligibility": operational,
        "evaluated_at": clock().isoformat(),
        "db_path": str(Path(db_path) if db_path is not None else DEFAULT_DB_PATH),
        "config": {
            "triage_model": settings.triage_model, "triage_reasoning_effort": settings.triage_reasoning_effort,
            "deep_primary_model": settings.deep_primary_model,
            "deep_primary_reasoning_effort": settings.deep_primary_reasoning_effort,
            "deep_fallback_model": settings.deep_fallback_model,
            "deep_fallback_reasoning_effort": settings.deep_fallback_reasoning_effort,
            "max_deep_input_tokens": settings.max_deep_input_tokens,
            "monthly_budget_usd": settings.monthly_budget_usd, "soft_limit_usd": settings.soft_limit_usd,
            "max_single_deep_estimated_cost_usd": settings.max_single_deep_estimated_cost_usd,
            "min_deep_value_amd": settings.min_deep_value_amd,
        },
        "month_spend_usd": spend,
        "month_remaining_usd": settings.monthly_budget_usd - spend,
        "enriched_total": len(store.enriched_urls()),
        "limit": limit,
        "candidate_count": len(candidates),
        "candidate_reasons": reason_counts,
        "routing_preview": routing,
        "estimated_pending_triage_cost_usd": triage_preflight["estimated_total_cost_usd"],
        "triage_preflight": triage_preflight,
        "accounting_blocks": store.accounting_blocks(),
        "outstanding_reservations": store.outstanding_reservations(),
    }


def _print_triage_preflight(t: dict, model: str) -> None:
    count = t["candidate_count"]
    average = t["estimated_total_cost_usd"] / count if count else Decimal(0)
    print()
    print("TRIAGE PREFLIGHT ESTIMATE (верхняя граница: cache hit не предполагается, output = резерв max_output_tokens)")
    print(f"  model: {model}, rate tier: {t['rate_tiers'] or '-'}")
    print(f"  candidate_count: {count}")
    print(f"  estimated_input_tokens_total: {t['estimated_input_tokens_total']}")
    print(f"  estimated_output_tokens_total: {t['estimated_output_tokens_total']} "
          f"(резерв {t['max_output_tokens_per_candidate']} на кандидата, не прогноз фактического output)")
    print(f"  estimated_ordinary_or_cache_write_cost: ${t['estimated_ordinary_or_cache_write_cost_usd']}")
    print(f"  estimated_output_cost: ${t['estimated_output_cost_usd']}")
    print(f"  estimated_total_cost: ${t['estimated_total_cost_usd']}")
    print(f"  average_estimated_cost_per_candidate: ${average}")
    print()


def _print_dry_run(report: dict) -> None:
    print("AI analysis pipeline — DRY RUN (без OpenAI, без записи в БД)")
    print(f"БД: {report['db_path']}")
    print()
    print("Конфигурация:")
    for name, value in report["config"].items():
        print(f"  {name}: {value}")
    print()
    print(f"Расход за месяц (UTC): ${report['month_spend_usd']}, остаток: ${report['month_remaining_usd']}")
    print(f"Объявлений с enrichment: {report['enriched_total']}")
    print(f"Время оценки срока (aware): {report['evaluated_at']}")
    operational = report["operational_eligibility"]
    print("Operational Eligibility (все объявления с enrichment, независимо от limit):")
    for name, count in sorted(operational["status_counts"].items()):
        print(f"  {name}: {count}")
    print("  требуют работы, по статусу срока:")
    for name, count in sorted(operational["pending_by_status"].items()):
        print(f"    {name}: {count}")
    print(f"Кандидатов к обработке (limit={report['limit']}; skip_expired в limit не входят): {report['candidate_count']}")
    for name, count in sorted(report["candidate_reasons"].items()):
        print(f"  {name}: {count}")
    _print_triage_preflight(report["triage_preflight"], report["config"]["triage_model"])
    blocks, reservations = report["accounting_blocks"], report["outstanding_reservations"]
    print(f"Accounting-блоки (нерешённые): {len(blocks)}; нерешённые резервы вызовов: {len(reservations)}" + (
        " — ПЛАТНЫЕ ВЫЗОВЫ ЗАБЛОКИРОВАНЫ, см. python -m src.ai.accounting_reconcile --list"
        if blocks or reservations else ""
    ))
    print("Маршрутизация (что можно вычислить без AI):")
    for name, count in sorted(report["routing_preview"].items()):
        print(f"  {name}: {count}")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="AI analysis pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Отчёт без API и без записи")
    mode.add_argument("--run-one", metavar="RESOURCE_URL", help="Один явно выбранный тендер (preflight без confirm)")
    mode.add_argument("--run-batch", action="store_true",
                      help="Операторский batch ровно из --limit N тендеров (без --confirm-paid-call — только preview)")
    parser.add_argument("--confirm-paid-call", action="store_true",
                        help="Только с --run-one / --run-batch: разрешить платные вызовы")
    parser.add_argument("--allow-expired", action="store_true",
                        help="Только с --run-one: обойти ТОЛЬКО deadline-гейт (BudgetGuard/учёт/Gate остаются)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число тендеров (оператор)")
    parser.add_argument("--db-path", default=None, help="SQLite БД (для --run-one: иначе DATABASE_PATH из env/.env, иначе data/tenders.db)")
    args = parser.parse_args(argv)
    if args.confirm_paid_call and not (args.run_one or args.run_batch):
        parser.error("--confirm-paid-call допустим только с --run-one или --run-batch")
    if args.run_batch and args.limit is None:
        parser.error("--run-batch требует явный --limit N (скрытого значения по умолчанию нет)")
    if args.run_batch and args.limit <= 0:
        parser.error("--limit должен быть > 0")
    if args.allow_expired and not args.run_one:
        parser.error("--allow-expired допустим только с --run-one")
    if args.run_one and args.limit is not None:
        parser.error("--limit несовместим с --run-one (всегда ровно один тендер)")

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")  # не перезаписывает уже заданные переменные окружения
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    if args.run_batch:
        from src.ai import one_shot, operator_batch

        settings = load_budget_settings()
        triage, deep = build_analyzers(settings)
        if args.confirm_paid_call and not (triage.has_api_key and deep.has_api_key):
            print("ERROR: OPENAI_API_KEY не задан, вызовов нет")
            return one_shot.EXIT_USAGE
        return operator_batch.run_batch(
            args.limit, triage, deep, settings, db_path=resolve_db_path(args.db_path), confirm=args.confirm_paid_call,
        )
    if args.run_one:
        from src.ai import one_shot

        settings = load_budget_settings()
        triage, deep = build_analyzers(settings)
        if args.confirm_paid_call and not (triage.has_api_key and deep.has_api_key):
            print("ERROR: OPENAI_API_KEY не задан, вызовов нет")
            return one_shot.EXIT_USAGE
        return one_shot.run_one(
            args.run_one, triage, deep, settings, db_path=resolve_db_path(args.db_path),
            confirm=args.confirm_paid_call, allow_expired=args.allow_expired,
        )
    _print_dry_run(dry_run(db_path=args.db_path, limit=args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
