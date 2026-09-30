"""
Тесты src.ai.cost_simulator: офлайн (ноль сетевых вызовов), детерминированность, репрезентативный
сценарий < $15, явная пометка превышения бюджета.

Запуск: python -m unittest tests.test_cost_simulator -v
"""

import io
import unittest
from contextlib import redirect_stdout
from decimal import Decimal

from src.ai import cost_simulator as simulator
from tests import safety_guards


class SimulatorTests(unittest.TestCase):
    def representative(self, **overrides):
        params = dict(announcements_per_day=30, days_per_month=30, deep_per_month=30)
        params.update(overrides)
        return simulator.simulate(**params)

    def test_representative_scenario_is_under_budget(self):
        result = self.representative()
        self.assertEqual(result["triage_calls_per_month"], 900)
        self.assertEqual(result["deep_calls_per_month"], 30)
        # Luna triage: 900 * (4100*0.20 + 300*1.20)/1e6 = 900 * 0.00118 = 1.062
        self.assertEqual(result["triage_cost_usd"], Decimal("1.062"))
        # Luna deep primary: 30 * (75473*0.20 + 4350*1.20)/1e6 = 30 * 0.0203146
        self.assertEqual(result["deep_cost_usd"], Decimal("0.609438"))
        self.assertEqual(result["total_usd"], Decimal("1.671438"))
        self.assertTrue(result["within_budget"])
        self.assertEqual(result["buffer_usd"], Decimal(15) - Decimal("1.671438"))
        self.assertTrue(result["approximate"])

    def test_over_budget_scenario_is_clearly_flagged(self):
        result = self.representative(deep_per_month=900)
        self.assertFalse(result["within_budget"])
        self.assertLess(result["buffer_usd"], 0)
        self.assertIn("OVER BUDGET", simulator.format_report(result))
        self.assertIn("OK:", simulator.format_report(self.representative()))

    def test_deterministic(self):
        self.assertEqual(self.representative(), self.representative())
        self.assertEqual(simulator.format_report(self.representative()), simulator.format_report(self.representative()))

    def test_rates_derive_deep_calls(self):
        result = simulator.simulate(
            announcements_per_day=30, days_per_month=30, relevant_rate=Decimal("0.2"), deep_pass_rate=Decimal("0.5"),
        )
        self.assertEqual(result["deep_calls_per_month"], 90)

    def test_missing_deep_volume_rejected(self):
        with self.assertRaises(ValueError):
            simulator.simulate(announcements_per_day=30)
        with self.assertRaises(ValueError):
            simulator.simulate(announcements_per_day=30, relevant_rate=Decimal("1.5"), deep_pass_rate=Decimal(1))

    def test_terra_triage_is_much_more_expensive_than_luna(self):
        luna = self.representative()["triage_cost_usd"]
        terra = self.representative(triage_model="gpt-5.6-terra")["triage_cost_usd"]
        self.assertEqual(terra, luna * 10)

    def test_unknown_model_is_an_error_not_a_zero_cost(self):
        from src.ai import pricing

        with self.assertRaises(pricing.CostEstimationError):
            self.representative(triage_model="gpt-unknown")

    def test_cli_exit_codes_and_output(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = simulator.main(["--announcements-per-day", "30", "--days", "30", "--deep-per-month", "30"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        for expected in ("triage calls/month:  900", "deep calls/month:    30", "TOTAL:", "buffer remaining:", "ПРИБЛИЗИТЕЛЬНО"):
            self.assertIn(expected, text)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(simulator.main(["--announcements-per-day", "30", "--deep-per-month", "900"]), 1)
            self.assertEqual(simulator.main(["--announcements-per-day", "30"]), 2)

    def test_makes_no_network_calls(self):
        before = safety_guards.violations()
        with redirect_stdout(io.StringIO()):
            simulator.main(["--announcements-per-day", "30", "--deep-per-month", "30"])
        self.assertEqual(safety_guards.violations(), before)


if __name__ == "__main__":
    unittest.main()
