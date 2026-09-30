"""
Офлайн-симулятор месячной стоимости AI. НИКАКИХ API-вызовов и сети: только арифметика по
src.ai.pricing. Результат ПРИБЛИЗИТЕЛЬНЫЙ — это прогноз для планирования, а не гарантия.
Жёсткую гарантию бюджета даёт Budget Guard (src.ai.budget_guard) по фактическому ledger.

Модель расчёта:
    triage_calls = announcements_per_day * days
    deep_calls   = --deep-per-month, иначе ceil(triage_calls * relevant_rate * deep_pass_rate)
    стоимость    = calls * стоимость_одного_вызова(pricing.calculate_cost) по средним токенам

Input раскладывается на ordinary / cached / cache-write (по разным ставкам, см. src.ai.pricing);
cached и cache-write токены — часть input_tokens, по умолчанию 0. Terra-fallback не моделируется:
он не вызывается автоматически (можно посчитать отдельно через --deep-model gpt-5.6-terra).

Средние токены по умолчанию — из реальных запусков проекта (это не гарантия для будущих):
    triage: ~4100 input / ~300 output (golden set, 24 кейса: ~96 000 in / ~6 000 out суммарно);
            cached input по умолчанию 0 (консервативно, кэш только удешевит);
    deep:   75 473 input / 4 350 output (simple Deep; complex medical v4 — ~103k input).
Реальные токены Luna при reasoning medium могут отличаться — после benchmark подставьте свои
через --triage-* / --deep-* флаги.

    python -m src.ai.cost_simulator --announcements-per-day 30 --days 30 --deep-per-month 30
    python -m src.ai.cost_simulator --announcements-per-day 30 --relevant-rate 0.2 --deep-pass-rate 0.5

Код возврата: 0 — прогноз укладывается в бюджет, 1 — нет (OVER BUDGET), 2 — ошибка аргументов.
"""

import argparse
import sys
from decimal import ROUND_CEILING, Decimal

from src.ai import budget_settings, pricing

DEFAULT_TRIAGE_INPUT_TOKENS = 4100
DEFAULT_TRIAGE_OUTPUT_TOKENS = 300
DEFAULT_DEEP_INPUT_TOKENS = 75_473
DEFAULT_DEEP_OUTPUT_TOKENS = 4_350


def _breakdown(split: tuple, output_tokens: int) -> dict:
    ordinary, cached, cache_write = split
    return {
        "ordinary_input": ordinary, "cached_input": cached, "cache_write_input": cache_write, "output": output_tokens,
    }


def simulate(
    *,
    announcements_per_day: int,
    days_per_month: int = 30,
    deep_per_month: int | None = None,
    relevant_rate: Decimal | None = None,
    deep_pass_rate: Decimal | None = None,
    triage_model: str = budget_settings.DEFAULT_TRIAGE_MODEL,
    deep_model: str = budget_settings.DEFAULT_DEEP_PRIMARY_MODEL,
    triage_input_tokens: int = DEFAULT_TRIAGE_INPUT_TOKENS,
    triage_cached_input_tokens: int = 0,
    triage_cache_write_tokens: int = 0,
    triage_output_tokens: int = DEFAULT_TRIAGE_OUTPUT_TOKENS,
    deep_input_tokens: int = DEFAULT_DEEP_INPUT_TOKENS,
    deep_cached_input_tokens: int = 0,
    deep_cache_write_tokens: int = 0,
    deep_output_tokens: int = DEFAULT_DEEP_OUTPUT_TOKENS,
    monthly_budget_usd: Decimal = budget_settings.DEFAULT_MONTHLY_BUDGET_USD,
    pricing_overrides: dict | None = None,
) -> dict:
    """Детерминированный прогноз. ValueError — некорректные входные параметры."""
    if announcements_per_day < 0 or days_per_month <= 0:
        raise ValueError("announcements_per_day >= 0 и days_per_month > 0")
    triage_calls = announcements_per_day * days_per_month

    if deep_per_month is None:
        if relevant_rate is None or deep_pass_rate is None:
            raise ValueError("Укажите --deep-per-month либо оба --relevant-rate и --deep-pass-rate")
        for name, rate in (("relevant_rate", relevant_rate), ("deep_pass_rate", deep_pass_rate)):
            if not Decimal(0) <= rate <= Decimal(1):
                raise ValueError(f"{name} должен быть в диапазоне 0..1: {rate}")
        deep_calls = int((Decimal(triage_calls) * relevant_rate * deep_pass_rate).to_integral_value(rounding=ROUND_CEILING))
    else:
        if deep_per_month < 0:
            raise ValueError("deep_per_month должен быть >= 0")
        deep_calls = deep_per_month

    triage_unit = pricing.calculate_cost(
        triage_model, triage_input_tokens, triage_output_tokens, triage_cached_input_tokens, pricing_overrides,
        cache_write_tokens=triage_cache_write_tokens,
    )
    deep_unit = pricing.calculate_cost(
        deep_model, deep_input_tokens, deep_output_tokens, deep_cached_input_tokens, pricing_overrides,
        cache_write_tokens=deep_cache_write_tokens,
    )
    triage_split = pricing.split_input_tokens(triage_input_tokens, triage_cached_input_tokens, triage_cache_write_tokens)
    deep_split = pricing.split_input_tokens(deep_input_tokens, deep_cached_input_tokens, deep_cache_write_tokens)
    triage_cost = triage_unit * triage_calls
    deep_cost = deep_unit * deep_calls
    total = triage_cost + deep_cost
    return {
        "approximate": True,
        "pricing_version": pricing.PRICING_VERSION,
        "triage_model": triage_model,
        "deep_model": deep_model,
        "triage_calls_per_month": triage_calls,
        "deep_calls_per_month": deep_calls,
        "triage_tokens_per_call": _breakdown(triage_split, triage_output_tokens),
        "deep_tokens_per_call": _breakdown(deep_split, deep_output_tokens),
        "triage_cost_per_call_usd": triage_unit,
        "deep_cost_per_call_usd": deep_unit,
        "triage_cost_usd": triage_cost,
        "deep_cost_usd": deep_cost,
        "total_usd": total,
        "monthly_budget_usd": monthly_budget_usd,
        "buffer_usd": monthly_budget_usd - total,
        "within_budget": total <= monthly_budget_usd,
    }


def _money(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.01'))}"


def _unit_money(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.0001'))}"


def _tokens(breakdown: dict) -> str:
    return (
        f"ordinary input {breakdown['ordinary_input']}, cached input {breakdown['cached_input']}, "
        f"cache-write input {breakdown['cache_write_input']}, output {breakdown['output']}"
    )


def format_report(result: dict) -> str:
    lines = [
        f"Cost simulation (ПРИБЛИЗИТЕЛЬНО, pricing {result['pricing_version']}; API не вызывается)",
        f"  triage calls/month:  {result['triage_calls_per_month']}  x {_unit_money(result['triage_cost_per_call_usd'])}"
        f" ({result['triage_model']}) = projected {_money(result['triage_cost_usd'])}",
        f"    tokens/call: {_tokens(result['triage_tokens_per_call'])}",
        f"  deep calls/month:    {result['deep_calls_per_month']}  x {_unit_money(result['deep_cost_per_call_usd'])}"
        f" ({result['deep_model']}, deep primary) = projected {_money(result['deep_cost_usd'])}",
        f"    tokens/call: {_tokens(result['deep_tokens_per_call'])}",
        f"  monthly totals: triage {_money(result['triage_cost_usd'])} | deep primary {_money(result['deep_cost_usd'])}"
        f" | overall {_money(result['total_usd'])}",
        f"  TOTAL:               {_money(result['total_usd'])} of {_money(result['monthly_budget_usd'])} budget",
        f"  buffer remaining:    {_money(result['buffer_usd'])}",
        "  OK: укладывается в бюджет" if result["within_budget"] else "  OVER BUDGET: прогноз превышает месячный бюджет",
        "  Гарантию бюджета даёт Budget Guard по фактическому ledger, а не этот прогноз.",
    ]
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ai.cost_simulator",
        description="Офлайн-оценка месячной стоимости AI (без API-вызовов, приблизительно).",
    )
    parser.add_argument("--announcements-per-day", type=int, required=True)
    parser.add_argument("--days", type=int, default=30, help="дней в месяце (по умолчанию 30)")
    parser.add_argument("--deep-per-month", type=int, default=None, help="число Deep-вызовов в месяц")
    parser.add_argument("--relevant-rate", type=Decimal, default=None, help="доля relevant после triage (0..1)")
    parser.add_argument("--deep-pass-rate", type=Decimal, default=None, help="доля relevant, проходящая gate в Deep (0..1)")
    parser.add_argument("--triage-model", default=budget_settings.DEFAULT_TRIAGE_MODEL)
    parser.add_argument("--deep-model", default=budget_settings.DEFAULT_DEEP_PRIMARY_MODEL,
                        help="модель Deep primary (по умолчанию gpt-5.6-luna)")
    parser.add_argument("--triage-input-tokens", type=int, default=DEFAULT_TRIAGE_INPUT_TOKENS)
    parser.add_argument("--triage-cached-input-tokens", type=int, default=0)
    parser.add_argument("--triage-cache-write-tokens", type=int, default=0, help="часть input, записанная в кэш")
    parser.add_argument("--triage-output-tokens", type=int, default=DEFAULT_TRIAGE_OUTPUT_TOKENS)
    parser.add_argument("--deep-input-tokens", type=int, default=DEFAULT_DEEP_INPUT_TOKENS)
    parser.add_argument("--deep-cached-input-tokens", type=int, default=0)
    parser.add_argument("--deep-cache-write-tokens", type=int, default=0, help="часть input, записанная в кэш")
    parser.add_argument("--deep-output-tokens", type=int, default=DEFAULT_DEEP_OUTPUT_TOKENS)
    parser.add_argument("--budget", type=Decimal, default=budget_settings.DEFAULT_MONTHLY_BUDGET_USD, help="месячный бюджет USD")
    return parser


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        result = simulate(
            announcements_per_day=args.announcements_per_day, days_per_month=args.days,
            deep_per_month=args.deep_per_month, relevant_rate=args.relevant_rate,
            deep_pass_rate=args.deep_pass_rate, triage_model=args.triage_model, deep_model=args.deep_model,
            triage_input_tokens=args.triage_input_tokens, triage_cached_input_tokens=args.triage_cached_input_tokens,
            triage_cache_write_tokens=args.triage_cache_write_tokens,
            triage_output_tokens=args.triage_output_tokens, deep_input_tokens=args.deep_input_tokens,
            deep_cached_input_tokens=args.deep_cached_input_tokens,
            deep_cache_write_tokens=args.deep_cache_write_tokens, deep_output_tokens=args.deep_output_tokens,
            monthly_budget_usd=args.budget,
        )
    except (ValueError, pricing.CostEstimationError) as error:
        print(f"Ошибка: {error}")
        return 2
    print(format_report(result))
    return 0 if result["within_budget"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
