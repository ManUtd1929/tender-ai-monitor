"""
Тесты src.ai.budget_guard (чистые решения) и src.ai.preflight (консервативная оценка, блокировка
oversized Deep ДО создания client/запроса). Сеть и БД production заблокированы safety_guards.

Запуск: python -m unittest tests.test_budget_guard -v
"""

import unittest
from decimal import Decimal
from types import SimpleNamespace

from src.ai import budget_guard, commercial_gate, deep_admission, deep_prompt, openai_deep_analysis, openai_triage
from src.ai import preflight
from src.ai.budget_settings import BudgetSettings
from tests.test_deep_prompt import make_deep_analysis_context, make_tender_context, make_triage_result

D = Decimal
SETTINGS = BudgetSettings()  # hard 15, soft 12, single deep cap 0.75, max deep input 140000


def check(spend, estimate, analysis_type="deep", settings=SETTINGS, model="gpt-5.6-terra"):
    return budget_guard.BudgetGuard(settings).check(D(spend), D(estimate), analysis_type, model)


class BudgetGuardTests(unittest.TestCase):
    def test_under_soft_limit_allowed(self):
        decision = check("3", "0.30")
        self.assertEqual(decision.decision, budget_guard.ALLOWED)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.remaining_usd, D(12))
        self.assertIsNone(decision.deep_status)

    def test_above_soft_limit_deep_is_deferred_not_called(self):
        decision = check("12.00", "0.30")
        self.assertEqual(decision.decision, budget_guard.SOFT_LIMIT_MODE)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.deep_status, "budget_deferred")

    def test_above_soft_limit_cheap_triage_still_allowed_if_budget_fits(self):
        decision = check("13", "0.01", analysis_type="triage", model="gpt-5.6-luna")
        self.assertEqual(decision.decision, budget_guard.SOFT_LIMIT_MODE)
        self.assertTrue(decision.allowed)
        self.assertIsNone(decision.deep_status)

    def test_hard_limit_blocks_everything(self):
        for analysis_type in ("triage", "deep"):
            with self.subTest(analysis_type=analysis_type):
                decision = check("15", "0.001", analysis_type=analysis_type)
                self.assertEqual(decision.decision, budget_guard.BLOCKED_HARD_LIMIT)
                self.assertFalse(decision.allowed)
        self.assertFalse(check("16.5", "0.001", analysis_type="triage").allowed)

    def test_next_call_that_would_exceed_remaining_budget_is_blocked(self):
        decision = check("14.90", "0.20", analysis_type="triage", model="gpt-5.6-luna")
        self.assertEqual(decision.decision, budget_guard.INSUFFICIENT_BUDGET_REMAINING)
        self.assertFalse(decision.allowed)

    def test_call_exactly_fitting_remaining_budget_is_allowed(self):
        self.assertTrue(check("14.90", "0.10", analysis_type="triage").allowed)

    def test_single_call_cap_blocks_expensive_deep(self):
        decision = check("1", "0.80")
        self.assertEqual(decision.decision, budget_guard.BLOCKED_SINGLE_CALL_LIMIT)
        self.assertEqual(decision.deep_status, "budget_deferred")

    def test_single_call_cap_does_not_apply_to_triage_and_can_be_disabled(self):
        self.assertTrue(check("1", "0.80", analysis_type="triage").allowed)
        no_cap = BudgetSettings(max_single_deep_estimated_cost_usd=None)
        self.assertTrue(check("1", "0.80", settings=no_cap).allowed)

    def test_settings_are_respected(self):
        tight = BudgetSettings(monthly_budget_usd=D(1), soft_limit_usd=D("0.5"))
        self.assertEqual(check("1", "0.01", settings=tight).decision, budget_guard.BLOCKED_HARD_LIMIT)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            check("1", "0.1", analysis_type="other")
        with self.assertRaises(ValueError):
            check("-1", "0.1")


def deep_analyzer(**kwargs):
    return openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(environ={}, **kwargs)


def deep_context(chunk_text=None):
    analysis = make_deep_analysis_context()
    if chunk_text is not None:
        analysis["chunks"][0]["text"] = chunk_text
    return deep_prompt.build_deep_context(make_tender_context(), analysis, make_triage_result())


class PreflightTests(unittest.TestCase):
    def test_estimate_is_conservative_relative_to_worst_observed_ratio(self):
        # худший наблюдённый реальный запуск: 759 157 символов -> 287 300 токенов (2.64 симв/токен)
        self.assertGreater(preflight.estimate_input_tokens("x" * 759_157), 287_300 * 0.99 - 1)
        # типичные запуски (3.4 симв/токен) оцениваются выше факта
        self.assertGreater(preflight.estimate_input_tokens("x" * 255_497), 75_473)

    def test_normal_deep_request_passes_and_has_conservative_cost(self):
        analyzer = deep_analyzer()
        result = preflight.preflight_request(
            analyzer.build_request(deep_context()), analyzer.model, "deep", SETTINGS,
        )
        self.assertEqual(result["status"], preflight.STATUS_OK)
        self.assertEqual(result["estimated_output_tokens"], analyzer.max_output_tokens)
        # верхняя граница output = max_output_tokens по ставке output модели (Luna 1.20 / 1M)
        self.assertEqual(analyzer.model, "gpt-5.6-luna")
        self.assertGreaterEqual(result["estimated_cost_usd"], D(analyzer.max_output_tokens) * D("1.20") / 1_000_000)

    def test_preflight_cost_is_conservative_no_cache_hit_assumed(self):
        from src.ai import pricing

        analyzer = deep_analyzer()
        result = preflight.preflight_request(analyzer.build_request(deep_context()), analyzer.model, "deep", SETTINGS)
        tokens_in, tokens_out = result["estimated_input_tokens"], result["estimated_output_tokens"]
        ordinary = pricing.calculate_cost(analyzer.model, tokens_in, tokens_out)
        all_cached = pricing.calculate_cost(analyzer.model, tokens_in, tokens_out, cached_input_tokens=tokens_in)
        all_write = pricing.calculate_cost(analyzer.model, tokens_in, tokens_out, cache_write_tokens=tokens_in)
        self.assertEqual(result["estimated_cost_usd"], all_write)
        self.assertGreater(result["estimated_cost_usd"], ordinary)
        self.assertGreater(result["estimated_cost_usd"], all_cached)

    def test_preflight_long_context_uses_long_rates(self):
        from src.ai import pricing

        request = {"input": "x" * 700_000, "max_output_tokens": 1000}  # ~291k оценённых токенов
        result = preflight.preflight_request(request, "gpt-5.6-terra", "triage", SETTINGS)
        self.assertGreater(result["estimated_input_tokens"], 272_000)
        expected = (D(result["estimated_input_tokens"]) * D("5.00") + D(1000) * D("18.00")) / 1_000_000
        self.assertEqual(result["estimated_cost_usd"], expected)

    def test_oversized_deep_input_is_flagged(self):
        analyzer = deep_analyzer()
        request = analyzer.build_request(deep_context("я" * 400_000))
        result = preflight.preflight_request(request, analyzer.model, "deep", SETTINGS)
        self.assertEqual(result["status"], preflight.STATUS_DEEP_INPUT_TOO_LARGE)
        self.assertGreater(result["estimated_input_tokens"], SETTINGS.max_deep_input_tokens)

    def test_input_limit_applies_to_deep_only(self):
        analyzer = deep_analyzer()
        request = analyzer.build_request(deep_context("я" * 400_000))
        self.assertEqual(preflight.preflight_request(request, analyzer.model, "triage", SETTINGS)["status"], "ok")

    def test_unknown_model_raises_cost_error(self):
        from src.ai import pricing

        with self.assertRaises(pricing.CostEstimationError):
            preflight.preflight_request({"input": "x"}, "gpt-unknown", "deep", SETTINGS)


def fixed_request_analyzer(chars):
    """Analyzer-заглушка: build_request даёт запрос ровно из chars символов (без OpenAI и client)."""
    return SimpleNamespace(
        model="gpt-5.6-luna",
        build_request=lambda context: {"instructions": "", "input": "я" * (chars - 1), "max_output_tokens": 16_000},
    )


class InputLimitCalibrationTests(unittest.TestCase):
    """Complex medical benchmark: 305 901 симв. DEEP_CONTEXT (факт API: 103 356 input tokens)."""

    MEDICAL_CHARS = 305_901

    def test_default_limit_and_coefficient(self):
        self.assertEqual(SETTINGS.max_deep_input_tokens, 140_000)
        self.assertEqual(preflight.CHARS_PER_TOKEN, D("2.4"))

    def test_estimator_stays_conservative_for_medical_dimensions(self):
        # 305 901 / 2.4 -> 127 459, + REQUEST_OVERHEAD_TOKENS 300 = 127 759 (>> факта 103 356)
        request = {"instructions": "", "input": "я" * (self.MEDICAL_CHARS - 1)}  # +1 символ-разделитель при склейке
        tokens, _ = preflight.estimate_request_tokens(request)
        self.assertEqual(tokens, 127_759)
        self.assertGreater(tokens, 103_356)

    def test_medical_dimensions_pass_admission_at_default_limit(self):
        result = deep_admission.evaluate_deep_admission(
            candidate_gate(), fixed_request_analyzer(self.MEDICAL_CHARS), {}, D(0), SETTINGS,
        )
        self.assertEqual(result["preflight"]["estimated_input_tokens"], 127_759)
        self.assertEqual(result["preflight"]["status"], preflight.STATUS_OK)
        self.assertEqual(result["deep_analysis_status"], "ready_for_deep")
        self.assertTrue(result["call_allowed"])

    def test_estimated_input_over_140000_is_still_blocked_without_model_call(self):
        chars = 336_100  # ceil(336100/2.4)+300 = 140 342 > 140 000
        analyzer = fixed_request_analyzer(chars)
        analyzer.ensure_client = lambda: self.fail("client не должен создаваться при блокировке")
        result = deep_admission.evaluate_deep_admission(candidate_gate(), analyzer, {}, D(0), SETTINGS)
        self.assertGreater(result["preflight"]["estimated_input_tokens"], 140_000)
        self.assertEqual(result["deep_analysis_status"], "deferred_input_too_large")
        self.assertEqual(result["reason_code"], "deep_input_limit_exceeded")
        self.assertFalse(result["call_allowed"])


class HardBudgetTests(unittest.TestCase):
    """Жёсткий лимит 15 USD не может быть намеренно превышен консервативной оценкой вызова."""

    def test_conservative_estimate_above_remaining_is_blocked(self):
        from src.ai import pricing

        estimate = pricing.calculate_conservative_cost("gpt-5.6-terra", 100_000, 16_000)
        decision = check(D("15") - estimate + D("0.0001"), estimate, model="gpt-5.6-terra")
        self.assertEqual(decision.decision, budget_guard.INSUFFICIENT_BUDGET_REMAINING)
        self.assertFalse(decision.allowed)

    def test_never_allowed_when_spend_plus_estimate_exceeds_hard_limit(self):
        for spend in ("0", "5", "11.99", "12", "14.5", "14.99", "15", "20"):
            for estimate in ("0.01", "0.5", "3", "16"):
                for analysis_type in ("triage", "deep"):
                    decision = check(spend, estimate, analysis_type=analysis_type,
                                     settings=BudgetSettings(max_single_deep_estimated_cost_usd=None))
                    if decision.allowed:
                        self.assertLessEqual(D(spend) + D(estimate), D(15), (spend, estimate, analysis_type))

    def test_soft_limit_12_defers_deep_but_not_cheap_triage(self):
        self.assertEqual(SETTINGS.soft_limit_usd, D(12))
        self.assertFalse(check("12", "0.1", analysis_type="deep").allowed)
        self.assertTrue(check("11.99", "0.1", analysis_type="deep").allowed)
        self.assertTrue(check("12", "0.001", analysis_type="triage", model="gpt-5.6-luna").allowed)

    def test_deep_admission_uses_conservative_estimate_for_budget(self):
        analyzer = deep_analyzer()
        pre = preflight.preflight_request(analyzer.build_request(deep_context()), analyzer.model, "deep", SETTINGS)
        spend = D(15) - pre["estimated_cost_usd"] + D("0.000001")
        settings = BudgetSettings(soft_limit_usd=D(15))
        result = deep_admission.evaluate_deep_admission(candidate_gate(), analyzer, deep_context(), spend, settings)
        self.assertEqual(result["deep_analysis_status"], "budget_deferred")
        self.assertEqual(result["reason_code"], "budget_deferred")
        self.assertFalse(result["call_allowed"])


class ExplodingClient:
    """Любое обращение к client — провал теста: заблокированный Deep не должен дойти до провайдера."""

    def __init__(self):
        self.created = 0
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.created += 1
        raise AssertionError("responses.create вызван для заблокированного Deep")


def candidate_gate():
    return {
        "gate_decision": commercial_gate.DEEP_CANDIDATE, "gate_reason": "test", "facts_used": {},
        "estimated_value_amd": None, "total_quantity": None, "total_lots": None, "category": "x", "confidence": "low",
    }


class DeepAdmissionTests(unittest.TestCase):
    def test_oversized_deep_blocked_before_client_creation_or_request(self):
        client = ExplodingClient()
        analyzer = deep_analyzer(client=client)
        analyzer.ensure_client = lambda: self.fail("client не должен создаваться при блокировке")
        result = deep_admission.evaluate_deep_admission(
            candidate_gate(), analyzer, deep_context("я" * 400_000), D(0), SETTINGS,
        )
        self.assertEqual(result["deep_analysis_status"], "deferred_input_too_large")
        self.assertEqual(result["reason_code"], "deep_input_limit_exceeded")
        self.assertFalse(result["call_allowed"])
        self.assertEqual(client.created, 0)

    def test_no_client_is_needed_for_admission(self):
        analyzer = deep_analyzer()
        self.assertFalse(analyzer.has_api_key)
        result = deep_admission.evaluate_deep_admission(candidate_gate(), analyzer, deep_context(), D(0), SETTINGS)
        self.assertEqual(result["deep_analysis_status"], "ready_for_deep")
        self.assertTrue(result["call_allowed"])
        self.assertIsNone(analyzer._client)

    def test_budget_deferred_is_not_not_relevant(self):
        analyzer = deep_analyzer()
        result = deep_admission.evaluate_deep_admission(candidate_gate(), analyzer, deep_context(), D("12.5"), SETTINGS)
        self.assertEqual(result["deep_analysis_status"], "budget_deferred")
        self.assertNotIn("not_relevant", result["deep_analysis_status"])
        self.assertFalse(result["call_allowed"])

    def test_hard_limit_defers_deep(self):
        analyzer = deep_analyzer()
        result = deep_admission.evaluate_deep_admission(candidate_gate(), analyzer, deep_context(), D(15), SETTINGS)
        self.assertEqual(result["deep_analysis_status"], "budget_deferred")
        self.assertEqual(result["budget"].decision, budget_guard.BLOCKED_HARD_LIMIT)

    def test_unknown_model_reports_pricing_unknown_without_call(self):
        analyzer = deep_analyzer(model="gpt-unknown")
        result = deep_admission.evaluate_deep_admission(candidate_gate(), analyzer, deep_context(), D(0), SETTINGS)
        self.assertEqual(result["deep_analysis_status"], "pricing_unknown")
        self.assertFalse(result["call_allowed"])

    def test_gate_skip_short_circuits_preflight(self):
        analyzer = deep_analyzer()
        gate = dict(candidate_gate(), gate_decision=commercial_gate.SKIP_LOW_VALUE, gate_reason="мало")
        result = deep_admission.evaluate_deep_admission(gate, analyzer, deep_context(), D(0), SETTINGS)
        self.assertEqual(result["deep_analysis_status"], "skipped_low_value")
        self.assertIsNone(result["preflight"])

    def test_triage_analyzer_accepts_luna_for_preflight(self):
        triage = openai_triage.OpenAITriageAnalyzer(model="gpt-5.6-luna", environ={})
        request = triage.build_request({"resource_url": "u", "title": "t"})
        result = preflight.preflight_request(request, triage.model, "triage", SETTINGS)
        self.assertEqual(result["status"], "ok")
        self.assertLess(result["estimated_cost_usd"], D("0.05"))


if __name__ == "__main__":
    unittest.main()
