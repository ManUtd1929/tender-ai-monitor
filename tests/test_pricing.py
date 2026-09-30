"""
Тесты src.ai.pricing и src.ai.budget_settings: точные Decimal-примеры, ordinary/cached/cache-write/
output, long-context (порог 272000), reasoning не тарифицируется отдельно, неизвестная модель и
противоречивый usage отклоняются, precedence настроек primary/fallback моделей.

Запуск: python -m unittest tests.test_pricing -v
"""

import unittest
from decimal import Decimal
from pathlib import Path

from src.ai import budget_settings, pricing

M = 1_000_000
LUNA, TERRA = "gpt-5.6-luna", "gpt-5.6-terra"


def flat_pricing(input_, cached, cache_write, output):
    return pricing.ModelPricing(short=pricing.Rates(*(Decimal(str(v)) for v in (input_, cached, cache_write, output))))


class CostTests(unittest.TestCase):
    def test_luna_short_context_ordinary_input_only(self):
        self.assertEqual(pricing.calculate_cost(LUNA, 200_000, M), Decimal("0.04") + Decimal("1.20"))

    def test_terra_short_context_ordinary_input_only(self):
        self.assertEqual(pricing.calculate_cost(TERRA, 200_000, 0), Decimal("0.40"))
        self.assertEqual(pricing.calculate_cost(TERRA, 0, M), Decimal("12.00"))

    def test_terra_exact_real_deep_run(self):
        self.assertEqual(pricing.calculate_cost(TERRA, 75_473, 4_350), Decimal("0.203146"))

    def test_cached_input_billed_at_cached_rate(self):
        # 200k input, 100k cached: 100k*2.00 + 100k*0.20 = 0.20 + 0.02
        self.assertEqual(pricing.calculate_cost(TERRA, 200_000, 0, cached_input_tokens=100_000), Decimal("0.22"))

    def test_cache_write_billed_at_cache_write_rate(self):
        self.assertEqual(pricing.calculate_cost(LUNA, 100_000, 0, cache_write_tokens=100_000), Decimal("0.025"))
        self.assertEqual(pricing.calculate_cost(TERRA, 100_000, 0, cache_write_tokens=100_000), Decimal("0.25"))

    def test_mixed_ordinary_cached_cache_write_output(self):
        # ordinary 50k*2.00 + cached 30k*0.20 + write 20k*2.50 + out 10k*12.00, всё /1e6
        cost = pricing.calculate_cost(
            TERRA, 100_000, 10_000, cached_input_tokens=30_000, cache_write_tokens=20_000,
        )
        self.assertEqual(cost, Decimal("0.100") + Decimal("0.006") + Decimal("0.050") + Decimal("0.120"))

    def test_split_input_tokens(self):
        self.assertEqual(pricing.split_input_tokens(100, 30, 20), (50, 30, 20))
        self.assertEqual(pricing.split_input_tokens(100, None, None), (100, 0, 0))
        self.assertEqual(pricing.split_input_tokens(100, 60, 40), (0, 60, 40))

    def test_invalid_token_breakdown_rejected(self):
        for cached, write in ((150, 0), (0, 150), (60, 50)):
            with self.subTest(cached=cached, write=write), self.assertRaises(pricing.CostEstimationError):
                pricing.calculate_cost(LUNA, 100, 0, cached_input_tokens=cached, cache_write_tokens=write)

    def test_long_context_threshold_boundary(self):
        self.assertEqual(pricing.calculate_cost(LUNA, 272_000, 0), Decimal(272_000) * Decimal("0.20") / M)
        self.assertEqual(pricing.calculate_cost(LUNA, 272_001, 0), Decimal(272_001) * Decimal("0.40") / M)
        self.assertEqual(pricing.calculate_cost(TERRA, 272_000, 0), Decimal(272_000) * Decimal("2.00") / M)
        self.assertEqual(pricing.calculate_cost(TERRA, 272_001, 0), Decimal(272_001) * Decimal("4.00") / M)

    def test_luna_long_context_rates(self):
        cost = pricing.calculate_cost(LUNA, 300_000, 1_000, cached_input_tokens=100_000, cache_write_tokens=50_000)
        expected = (Decimal(150_000) * Decimal("0.40") + Decimal(100_000) * Decimal("0.04")
                    + Decimal(50_000) * Decimal("0.50") + Decimal(1_000) * Decimal("1.80")) / M
        self.assertEqual(cost, expected)

    def test_terra_long_context_rates(self):
        cost = pricing.calculate_cost(TERRA, 300_000, 1_000, cached_input_tokens=100_000, cache_write_tokens=50_000)
        expected = (Decimal(150_000) * Decimal("4.00") + Decimal(100_000) * Decimal("0.40")
                    + Decimal(50_000) * Decimal("5.00") + Decimal(1_000) * Decimal("18.00")) / M
        self.assertEqual(cost, expected)

    def test_pricing_table_values(self):
        luna, terra = pricing.MODEL_PRICING[LUNA], pricing.MODEL_PRICING[TERRA]
        rates = pricing.Rates
        self.assertEqual(luna.short, rates(Decimal("0.20"), Decimal("0.02"), Decimal("0.25"), Decimal("1.20")))
        self.assertEqual(luna.long, rates(Decimal("0.40"), Decimal("0.04"), Decimal("0.50"), Decimal("1.80")))
        self.assertEqual(terra.short, rates(Decimal("2.00"), Decimal("0.20"), Decimal("2.50"), Decimal("12.00")))
        self.assertEqual(terra.long, rates(Decimal("4.00"), Decimal("0.40"), Decimal("5.00"), Decimal("18.00")))
        self.assertEqual(luna.long_context_threshold_tokens, 272_000)
        self.assertEqual(terra.long_context_threshold_tokens, 272_000)

    def test_total_tokens_is_not_multiplied_by_one_price(self):
        usage = {"input_tokens": 1000, "output_tokens": 1000, "total_tokens": 2000, "cached_tokens": 0}
        naive = Decimal(2000) * Decimal("2.00") / M
        self.assertNotEqual(pricing.cost_from_usage(TERRA, usage), naive)
        self.assertEqual(pricing.cost_from_usage(TERRA, usage), Decimal("0.014"))

    def test_reasoning_tokens_not_double_billed(self):
        base = {"input_tokens": 1000, "output_tokens": 500, "cached_tokens": None}
        with_reasoning = dict(base, reasoning_tokens=400)
        self.assertEqual(pricing.cost_from_usage(TERRA, base), pricing.cost_from_usage(TERRA, with_reasoning))

    def test_missing_or_none_cache_fields_treated_as_zero(self):
        legacy = {"input_tokens": 10, "output_tokens": 10}
        with_none = {"input_tokens": 10, "output_tokens": 10, "cached_tokens": None, "cache_write_tokens": None}
        expected = pricing.calculate_cost(LUNA, 10, 10)
        self.assertEqual(pricing.cost_from_usage(LUNA, legacy), expected)
        self.assertEqual(pricing.cost_from_usage(LUNA, with_none), expected)

    def test_cost_from_usage_includes_cache_write(self):
        usage = {"input_tokens": 100_000, "output_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 100_000}
        self.assertEqual(pricing.cost_from_usage(LUNA, usage), Decimal("0.025"))

    def test_cost_from_usage_rejects_inconsistent_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 0, "cached_tokens": 80, "cache_write_tokens": 50}
        with self.assertRaises(pricing.CostEstimationError):
            pricing.cost_from_usage(LUNA, usage)

    def test_normalize_usage_defaults_to_zero(self):
        normalized = pricing.normalize_usage({"input_tokens": 5, "cached_tokens": None})
        self.assertEqual(normalized["cache_write_tokens"], 0)
        self.assertEqual(normalized["input_tokens"], 5)
        self.assertEqual(set(normalized), set(pricing.USAGE_FIELDS))
        self.assertEqual(pricing.normalize_usage(None)["total_tokens"], 0)

    def test_result_is_decimal_not_float(self):
        self.assertIsInstance(pricing.calculate_cost(LUNA, 1, 1), Decimal)

    def test_unknown_model_rejected(self):
        with self.assertRaises(pricing.CostEstimationError):
            pricing.calculate_cost("gpt-unknown", 1, 1)
        with self.assertRaises(pricing.CostEstimationError):
            pricing.calculate_conservative_cost("gpt-unknown", 1, 1)

    def test_unknown_model_allowed_with_override(self):
        override = {"gpt-x": flat_pricing(1, "0.5", 3, 2)}
        self.assertEqual(pricing.calculate_cost("gpt-x", M, M, overrides=override), Decimal(3))
        self.assertEqual(pricing.calculate_cost("gpt-x", M, 0, overrides=override, cache_write_tokens=M), Decimal(3))

    def test_missing_usage_and_invalid_tokens_rejected(self):
        with self.assertRaises(pricing.CostEstimationError):
            pricing.cost_from_usage(LUNA, None)
        with self.assertRaises(pricing.CostEstimationError):
            pricing.calculate_cost(LUNA, -1, 0)
        with self.assertRaises(pricing.CostEstimationError):
            pricing.calculate_cost(LUNA, 1.5, 0)
        with self.assertRaises(pricing.CostEstimationError):
            pricing.calculate_cost(LUNA, 10, 0, cache_write_tokens=-1)

    def test_model_without_long_tariff_always_uses_short_rates(self):
        override = {"gpt-x": flat_pricing(1, 1, 1, 1)}
        self.assertEqual(pricing.calculate_cost("gpt-x", 500_000, 0, overrides=override), Decimal("0.5"))


class ConservativeCostTests(unittest.TestCase):
    def test_input_priced_at_cache_write_rate_not_cached(self):
        cost = pricing.calculate_conservative_cost(LUNA, 100_000, 10_000)
        self.assertEqual(cost, (Decimal(100_000) * Decimal("0.25") + Decimal(10_000) * Decimal("1.20")) / M)
        self.assertGreater(cost, pricing.calculate_cost(LUNA, 100_000, 10_000))
        self.assertGreater(cost, pricing.calculate_cost(LUNA, 100_000, 10_000, cached_input_tokens=100_000))

    def test_triage_sized_estimate_regression(self):
        """Ставки за 1M (не за 100k), input один раз по cache-write, output-резерв триажа 8000, short tier."""
        b = pricing.conservative_cost_breakdown(LUNA, 7_109, 8_000)
        self.assertEqual(b["rate_tier"], "short")
        self.assertEqual(b["input_rate_per_million"], Decimal("0.25"))
        self.assertEqual(b["input_cost_usd"], Decimal("0.00177725"))
        self.assertEqual(b["output_cost_usd"], Decimal("0.0096"))
        self.assertEqual(b["total_cost_usd"], Decimal("0.01137725"))
        self.assertEqual(pricing.calculate_conservative_cost(LUNA, 7_109, 8_000), b["total_cost_usd"])
        # 1M output-токенов = ровно ставка за 1M; reasoning как параметр не существует (не тарифицируется)
        self.assertEqual(pricing.calculate_conservative_cost(LUNA, 0, M), Decimal("1.20"))
        self.assertEqual(pricing.calculate_conservative_cost(LUNA, 272_000, 0), Decimal("0.068"))

    def test_breakdown_marks_long_tier_only_above_threshold(self):
        self.assertEqual(pricing.conservative_cost_breakdown(LUNA, 272_000, 0)["rate_tier"], "short")
        self.assertEqual(pricing.conservative_cost_breakdown(LUNA, 272_001, 0)["rate_tier"], "long")

    def test_uses_max_of_input_and_cache_write_rate(self):
        override = {"gpt-x": flat_pricing(5, 1, 2, 0)}
        self.assertEqual(pricing.calculate_conservative_cost("gpt-x", M, 0, override), Decimal(5))

    def test_long_context_rates_by_estimated_input(self):
        self.assertEqual(pricing.calculate_conservative_cost(TERRA, 272_001, 0), Decimal(272_001) * Decimal("5.00") / M)
        self.assertEqual(pricing.calculate_conservative_cost(TERRA, 272_000, 0), Decimal(272_000) * Decimal("2.50") / M)


class BudgetSettingsTests(unittest.TestCase):
    def test_defaults(self):
        settings = budget_settings.load_budget_settings({})
        self.assertEqual(settings.monthly_budget_usd, Decimal(15))
        self.assertEqual(settings.soft_limit_usd, Decimal(12))
        self.assertEqual(settings.triage_model, "gpt-5.6-luna")
        self.assertEqual(settings.triage_reasoning_effort, "medium")
        self.assertEqual(settings.deep_primary_model, "gpt-5.6-luna")
        self.assertEqual(settings.deep_primary_reasoning_effort, "high")
        self.assertEqual(settings.deep_fallback_model, "gpt-5.6-terra")
        self.assertEqual(settings.deep_fallback_reasoning_effort, "high")
        self.assertEqual(settings.max_deep_input_tokens, 140_000)
        self.assertEqual(settings.max_single_deep_estimated_cost_usd, Decimal("0.75"))
        self.assertIsNone(settings.min_deep_value_amd)

    def test_new_variables_override_legacy_openai_model(self):
        env = {"OPENAI_MODEL": "gpt-5.6-terra", "TRIAGE_MODEL": "gpt-5.6-luna", "DEEP_MODEL": "gpt-5.6-terra"}
        self.assertEqual(budget_settings.load_budget_settings(env).triage_model, "gpt-5.6-luna")

    def test_legacy_openai_model_is_fallback_until_explicit_migration(self):
        settings = budget_settings.load_budget_settings({"OPENAI_MODEL": "gpt-5.6-terra"})
        self.assertEqual(settings.triage_model, "gpt-5.6-terra")

    def test_deep_primary_precedence(self):
        load = budget_settings.load_budget_settings
        self.assertEqual(load({}).deep_primary_model, "gpt-5.6-luna")
        self.assertEqual(load({"OPENAI_MODEL": "m-openai"}).deep_primary_model, "m-openai")
        self.assertEqual(load({"OPENAI_MODEL": "m-openai", "DEEP_MODEL": "m-legacy"}).deep_primary_model, "m-legacy")
        env = {"OPENAI_MODEL": "m-openai", "DEEP_MODEL": "m-legacy", "DEEP_PRIMARY_MODEL": "m-primary"}
        self.assertEqual(load(env).deep_primary_model, "m-primary")
        self.assertEqual(load({"DEEP_PRIMARY_MODEL": "  ", "DEEP_MODEL": "m-legacy"}).deep_primary_model, "m-legacy")

    def test_legacy_deep_model_does_not_configure_fallback(self):
        load = budget_settings.load_budget_settings
        self.assertEqual(load({"DEEP_MODEL": "m-legacy"}).deep_fallback_model, "gpt-5.6-terra")
        settings = load({"DEEP_FALLBACK_MODEL": "m-fb"})
        self.assertEqual(settings.deep_fallback_model, "m-fb")
        self.assertEqual(settings.deep_primary_model, "gpt-5.6-luna")

    def test_deep_effort_precedence_and_validation(self):
        load = budget_settings.load_budget_settings
        self.assertEqual(load({"DEEP_REASONING_EFFORT": "low"}).deep_primary_reasoning_effort, "low")
        env = {"DEEP_REASONING_EFFORT": "low", "DEEP_PRIMARY_REASONING_EFFORT": "medium"}
        self.assertEqual(load(env).deep_primary_reasoning_effort, "medium")
        self.assertEqual(load({"DEEP_FALLBACK_REASONING_EFFORT": "xhigh"}).deep_fallback_reasoning_effort, "xhigh")
        with self.assertRaises(budget_settings.BudgetConfigError):
            load({"DEEP_FALLBACK_REASONING_EFFORT": "ultra"})

    def test_env_example_matches_target_config(self):
        text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
        env = dict(line.split("=", 1) for line in text.splitlines() if "=" in line and not line.startswith("#"))
        settings = budget_settings.load_budget_settings(env)
        self.assertEqual(
            (settings.triage_model, settings.deep_primary_model, settings.deep_fallback_model),
            ("gpt-5.6-luna", "gpt-5.6-luna", "gpt-5.6-terra"),
        )
        self.assertEqual(settings.monthly_budget_usd, Decimal(15))
        self.assertEqual(settings.soft_limit_usd, Decimal(12))
        self.assertEqual(settings.max_deep_input_tokens, 140_000)
        self.assertNotIn("DEEP_MODEL", env)

    def test_legacy_effort_fallback_and_empty_is_unset(self):
        settings = budget_settings.load_budget_settings(
            {"OPENAI_TRIAGE_REASONING_EFFORT": "low", "TRIAGE_REASONING_EFFORT": "  "}
        )
        self.assertEqual(settings.triage_reasoning_effort, "low")

    def test_invalid_values_rejected(self):
        for env in (
            {"MONTHLY_AI_BUDGET_USD": "abc"}, {"MONTHLY_AI_BUDGET_USD": "0"}, {"AI_SOFT_LIMIT_USD": "20"},
            {"MAX_DEEP_INPUT_TOKENS": "-5"}, {"MIN_DEEP_VALUE_AMD": "x"},
            {"DEEP_PRIMARY_REASONING_EFFORT": "ultra"}, {"DEEP_REASONING_EFFORT": "ultra"},
        ):
            with self.subTest(env=env), self.assertRaises(budget_settings.BudgetConfigError):
                budget_settings.load_budget_settings(env)

    def test_min_deep_value_and_single_cap_none(self):
        settings = budget_settings.load_budget_settings(
            {"MIN_DEEP_VALUE_AMD": "1500000", "MAX_SINGLE_DEEP_ESTIMATED_COST_USD": "none"}
        )
        self.assertEqual(settings.min_deep_value_amd, 1_500_000)
        self.assertIsNone(settings.max_single_deep_estimated_cost_usd)


if __name__ == "__main__":
    unittest.main()
