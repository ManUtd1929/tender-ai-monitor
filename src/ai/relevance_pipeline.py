"""
Оркестрация двухэтапного AI relevance-анализа: build context -> вызов injectable
analyzer -> validate -> сохранение в analysis_repository, с изоляцией ошибок на
уровне одного тендера (как document_pipeline.process_enriched_announcements).

Analyzer — любой объект с методами:

    analyzer.triage(triage_context: dict) -> dict          (сырой, ещё не validated triage result)
    analyzer.deep_analyze(tender_context: dict,
                           deep_analysis_context: dict,
                           triage_result: dict) -> dict     (сырой deep analysis result)

Реального AI/HTTP здесь нет и не подключается: production FakeAnalyzer этот модуль не
создаёт (правило задачи "fake analyzer только в tests"). Пайплайн не вызывается из
src.monitor — интеграция будет отдельным шагом после проверки контекста и поведения.
"""

import logging

from src.ai import relevance_schema
from src.ai import tender_context as tender_context_module
from src.database import analysis_repository

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# STAGE 1: triage
# --------------------------------------------------------------------------

def run_triage(
    analyzer,
    db_path=None,
    limit: int | None = None,
    provider: str = analysis_repository.DEFAULT_PROVIDER,
    model: str = analysis_repository.DEFAULT_MODEL,
    prompt_version: str = analysis_repository.DEFAULT_PROMPT_VERSION,
) -> dict:
    """
    Triage кандидатов (analysis_repository.get_triage_candidates): analyzer.triage(triage_context)
    -> relevance_schema.validate_triage_result -> analysis_repository.save_triage. Ошибка
    одного кандидата (analyzer, validation или repository) не останавливает остальных —
    попадает в failures. Возвращает сводку со счётчиками (new/updated/existing/relevant/
    maybe/not_relevant/failed) и списками results/failures.
    """
    logger.info("Запуск triage")
    analysis_repository.init_db(db_path=db_path)

    candidates = analysis_repository.get_triage_candidates(db_path=db_path, limit=limit)
    logger.info("Кандидатов на triage: %d", len(candidates))

    results = []
    failures = []
    storage_counts = {"new": 0, "updated": 0, "existing": 0}
    relevance_counts = {"relevant": 0, "maybe": 0, "not_relevant": 0}

    for candidate in candidates:
        resource_url = candidate["resource_url"]
        try:
            triage_context = tender_context_module.build_triage_context(candidate["tender_context"])
            raw_result = analyzer.triage(triage_context)
            validated = relevance_schema.validate_triage_result(raw_result)
            storage_status = analysis_repository.save_triage(
                resource_url, candidate["input_hash"], validated,
                provider=provider, model=model, prompt_version=prompt_version, db_path=db_path,
            )
        except Exception as error:
            logger.exception("Triage не удался: %s", resource_url)
            failures.append({
                "resource_url": resource_url,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
            continue

        storage_counts[storage_status] += 1
        relevance_counts[validated["relevance_status"]] += 1
        results.append({
            "resource_url": resource_url,
            "storage_status": storage_status,
            "result": validated,
        })

    logger.info(
        "Triage завершён: кандидатов %d, успешно %d, ошибок %d "
        "(relevant %d, maybe %d, not_relevant %d)",
        len(candidates), len(results), len(failures),
        relevance_counts["relevant"], relevance_counts["maybe"], relevance_counts["not_relevant"],
    )

    return {
        "candidate_count": len(candidates),
        "success_count": len(results),
        "failed_count": len(failures),
        "storage_new_count": storage_counts["new"],
        "storage_updated_count": storage_counts["updated"],
        "storage_existing_count": storage_counts["existing"],
        "relevant_count": relevance_counts["relevant"],
        "maybe_count": relevance_counts["maybe"],
        "not_relevant_count": relevance_counts["not_relevant"],
        "results": results,
        "failures": failures,
    }


# --------------------------------------------------------------------------
# STAGE 2: deep analysis
# --------------------------------------------------------------------------

def run_deep_analysis(
    analyzer,
    db_path=None,
    limit: int | None = None,
    provider: str = analysis_repository.DEFAULT_PROVIDER,
    model: str = analysis_repository.DEFAULT_MODEL,
    prompt_version: str = analysis_repository.DEFAULT_PROMPT_VERSION,
) -> dict:
    """
    Deep analysis кандидатов (analysis_repository.get_deep_analysis_candidates: последний
    triage relevant/maybe): analyzer.deep_analyze(tender_context, deep_analysis_context,
    triage_result) -> relevance_schema.validate_deep_analysis_result ->
    analysis_repository.save_deep_analysis. Ошибка одного кандидата не останавливает
    остальных — попадает в failures.
    """
    logger.info("Запуск deep analysis")
    analysis_repository.init_db(db_path=db_path)

    candidates = analysis_repository.get_deep_analysis_candidates(db_path=db_path, limit=limit)
    logger.info("Кандидатов на deep analysis: %d", len(candidates))

    results = []
    failures = []
    storage_counts = {"new": 0, "updated": 0, "existing": 0}

    for candidate in candidates:
        resource_url = candidate["resource_url"]
        try:
            deep_context = tender_context_module.build_deep_analysis_context(candidate["tender_context"])
            raw_result = analyzer.deep_analyze(
                candidate["tender_context"], deep_context, candidate["triage"],
            )
            validated = relevance_schema.validate_deep_analysis_result(raw_result)
            storage_status = analysis_repository.save_deep_analysis(
                resource_url, candidate["input_hash"], validated,
                provider=provider, model=model, prompt_version=prompt_version, db_path=db_path,
            )
        except Exception as error:
            logger.exception("Deep analysis не удался: %s", resource_url)
            failures.append({
                "resource_url": resource_url,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
            continue

        storage_counts[storage_status] += 1
        results.append({
            "resource_url": resource_url,
            "storage_status": storage_status,
            "result": validated,
        })

    logger.info(
        "Deep analysis завершён: кандидатов %d, успешно %d, ошибок %d",
        len(candidates), len(results), len(failures),
    )

    return {
        "candidate_count": len(candidates),
        "success_count": len(results),
        "failed_count": len(failures),
        "storage_new_count": storage_counts["new"],
        "storage_updated_count": storage_counts["updated"],
        "storage_existing_count": storage_counts["existing"],
        "results": results,
        "failures": failures,
    }


def run_relevance_pipeline(
    analyzer,
    db_path=None,
    triage_limit: int | None = None,
    deep_analysis_limit: int | None = None,
) -> dict:
    """
    Последовательно запускает run_triage, затем run_deep_analysis (её кандидаты уже
    учитывают только что сохранённый triage). Не вызывается из src.monitor — это
    отдельный запуск, вызывающий код сам решает, когда его использовать.
    """
    triage_result = run_triage(analyzer, db_path=db_path, limit=triage_limit)
    deep_analysis_result = run_deep_analysis(analyzer, db_path=db_path, limit=deep_analysis_limit)
    return {
        "triage": triage_result,
        "deep_analysis": deep_analysis_result,
    }


def main():
    print("Relevance pipeline module. Use explicit processing functions with an injected analyzer.")


if __name__ == "__main__":
    main()
