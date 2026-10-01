"""
Operational Eligibility Gate: детерминированная проверка срока подачи заявок ДО любого платного AI-вызова.

Это не relevance и не коммерческая привлекательность: только «надёжно ли известно, что срок уже прошёл».
Чистые функции, без БД, сети и AI; текущее время передаётся явно (инъекция), локальная таймзона сервера
не используется.

Источники срока (только структурированные поля, никаких дат из title/description/текста документов):
    announcement.deadline_at           — срок со страницы списка Gnumner (из tender_time_raw уже разобран скрапером);
    enrichment.dates.deadline_at_detail — срок со страницы тендера (eAuction/ARMEPS), нормализован.
Флаг deadline_at_match не источник даты: расхождение источников выявляется здесь самостоятельно.

Статусы:
    eligible  — срок известен и в будущем (или источники согласованы и он в будущем);
    expired   — ВСЕ известные сроки уже прошли (deadline <= now);
    unknown   — нет ни одного разбираемого срока (в т.ч. «Бессрочный», NULL, мусор) -> НЕ пропускаем;
    conflict  — сроки расходятся и хотя бы один ещё в будущем -> НЕ пропускаем (без выбора «раннего»).

Таймзона: наивные значения = Asia/Yerevan; значения с offset сохраняют его. Дата без времени истекает в
23:59:59 локального дня (не в 00:00). Граница: deadline <= now -> expired.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo

STATUS_ELIGIBLE = "eligible"
STATUS_EXPIRED = "expired"
STATUS_UNKNOWN = "unknown"
STATUS_CONFLICT = "conflict"

REASON_DEADLINE_EXPIRED = "deadline_expired"

SOURCE_LIST = "announcements.deadline_at"
SOURCE_DETAIL = "announcement_enrichment.deadline_at_detail"

# Только форматы, которые реально выдают скрапер и detail-страницы (см. resource_enrichment).
_DATE_ONLY_FORMATS = ("%Y-%m-%d", "%d/%m/%Y")
_NAIVE_DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M")
_ISO_WITH_OFFSET = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})$")

# Армения без перехода на летнее время с 2012 года: постоянный UTC+4. Fallback нужен там, где в системе
# нет базы tz (например Windows без пакета tzdata); результат тот же.
_FALLBACK_YEREVAN = timezone(timedelta(hours=4), "Asia/Yerevan")


def _load_local_timezone() -> tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("Asia/Yerevan")
    except Exception:
        return _FALLBACK_YEREVAN


LOCAL_TZ = _load_local_timezone()


def parse_deadline(value, local_tz: tzinfo = LOCAL_TZ) -> datetime | None:
    """Строка срока -> aware datetime; None, если значение пустое или формат не распознан (дата не угадывается)."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None

    for fmt in _DATE_ONLY_FORMATS:
        try:
            day = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return day.replace(hour=23, minute=59, second=59, tzinfo=local_tz)

    for fmt in _NAIVE_DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=local_tz)
        except ValueError:
            continue

    if _ISO_WITH_OFFSET.match(text):
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class Eligibility:
    status: str
    resolved_deadline: datetime | None
    deadline_sources: tuple  # ((source, raw_value, parsed_iso | None), ...) — только непустые значения
    evaluated_at: datetime

    @property
    def expired(self) -> bool:
        return self.status == STATUS_EXPIRED

    def to_dict(self) -> dict:
        """Компактные детерминированные факты для tender_pipeline_state.details / отчётов."""
        return {
            "status": self.status,
            "resolved_deadline": self.resolved_deadline.isoformat() if self.resolved_deadline else None,
            "deadline_sources": [
                {"source": source, "raw": raw, "parsed": parsed} for source, raw, parsed in self.deadline_sources
            ],
            "evaluated_at": self.evaluated_at.isoformat(),
        }


def evaluate(list_deadline, detail_deadline, now: datetime, local_tz: tzinfo = LOCAL_TZ) -> Eligibility:
    """
    now — обязательный aware datetime (инъекция часов). Значение со срезанной таймзоной — ошибка, а не
    молчаливая интерпретация: сравнение не должно зависеть от локальных часов сервера.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now должен быть timezone-aware datetime")

    sources, parsed_values, has_malformed = [], [], False
    for source, raw in ((SOURCE_LIST, list_deadline), (SOURCE_DETAIL, detail_deadline)):
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        parsed = parse_deadline(raw, local_tz)
        sources.append((source, raw, parsed.isoformat() if parsed is not None else None))
        if parsed is None:
            has_malformed = True
        else:
            parsed_values.append(parsed)

    def result(status, resolved=None):
        return Eligibility(status, resolved, tuple(sources), now)

    if not parsed_values:
        return result(STATUS_UNKNOWN)

    latest = max(parsed_values)  # aware-сравнение по моменту времени, не по локальному представлению
    all_expired = all(value <= now for value in parsed_values)
    if all_expired:
        # Нераспознанное второе значение: возможно, оно в будущем -> истечение не установлено надёжно.
        return result(STATUS_UNKNOWN if has_malformed else STATUS_EXPIRED, latest)

    agree = len(set(parsed_values)) == 1
    return result(STATUS_ELIGIBLE if agree else STATUS_CONFLICT, latest)


def evaluate_context(tender_context: dict, now: datetime, local_tz: tzinfo = LOCAL_TZ) -> Eligibility:
    """Срок берётся из уже собранного tender_context (тех же полей, что входят в input_hash)."""
    return evaluate(
        tender_context["announcement"].get("deadline_at"),
        (tender_context["enrichment"].get("dates") or {}).get("deadline_at_detail"),
        now, local_tz,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
