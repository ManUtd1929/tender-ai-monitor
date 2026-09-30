"""
Ручная сверка fail-closed учёта AI-расходов (см. ai_usage_repository: резервы и accounting blocks).

Платные вызовы блокируются, пока есть нерешённый резерв (reserved / unresolved) или accounting-блок.
Автоматического снятия нет; каждое действие оператора требует --note и остаётся в БД (аудит).

Поддерживаемые пути:
  --list                          показать нерешённые резервы и блоки (только чтение).
  --reconcile --note "..."        для блоков с СОХРАНЁННЫМ реальным usage: в одной транзакции записать usage
                                  в ledger, закрыть связанный резерв (settled) и блок. Если response_id уже
                                  в ledger — повторно не пишется. Ошибка = откат всего.
  --release ID --note "..."       явное решение оператора для резерва БЕЗ известного usage (например, timeout):
                                  оператор сам подтвердил (dashboard OpenAI), что биллинга не было, либо принял
                                  риск. Резерв -> released, пометка сохраняется. Токены не выдумываются: если
                                  биллинг был, но usage неизвестен, добавьте расход вручную отдельной
                                  ledger-записью до release или оставьте резерв (он остаётся в committed spend).

OpenAI не вызывается.
"""

import argparse
import sys

from src.database import ai_usage_repository


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="Сверка нерешённых резервов и accounting-блоков AI usage")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Показать нерешённые резервы и блоки")
    group.add_argument("--reconcile", action="store_true", help="Записать usage блоков в ledger и закрыть резервы")
    group.add_argument("--release", type=int, metavar="RESERVATION_ID", help="Снять резерв без usage (решение оператора)")
    parser.add_argument("--note", default=None, help="Пометка оператора (обязательна для --reconcile/--release)")
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args(argv)

    ai_usage_repository.init_db(args.db_path)
    if args.list:
        reservations = ai_usage_repository.outstanding_reservations(args.db_path)
        blocks = ai_usage_repository.unresolved_accounting_blocks(args.db_path)
        print(f"Нерешённых резервов: {len(reservations)}")
        for r in reservations:
            print(
                f"  резерв #{r['id']} {r['created_at']} {r['status']} {r['analysis_type']} {r['model']} "
                f"ref={r['reference']} оценка=${r['estimated_cost_usd']} {r['resolution'] or ''}"
            )
        print(f"Нерешённых блоков: {len(blocks)}")
        for block in blocks:
            print(
                f"  блок #{block['id']} {block['created_at']} {block['analysis_type']} {block['model']} "
                f"ref={block['reference']} response_id={block['response_id']} резерв={block['reservation_id']} "
                f"usage={block['usage']} причина: {block['failure']}"
            )
        return 0
    if not args.note or not args.note.strip():
        parser.error("--reconcile / --release требуют непустой --note")
    if args.release is not None:
        ai_usage_repository.release_reservation(args.release, f"operator: {args.note.strip()}", args.db_path)
        print(f"Резерв #{args.release} снят оператором (released).")
        return 0
    for item in ai_usage_repository.reconcile_accounting_blocks(args.note, db_path=args.db_path):
        print(f"  блок #{item['id']}: cost=${item['cost_usd']} duplicate={item['duplicate']}")
    print("Готово: usage записан в ledger, связанные резервы и блоки закрыты.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
