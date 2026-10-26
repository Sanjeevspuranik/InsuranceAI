import unittest

from insuranceai.claims_calc import (
    LineItem,
    PolicyInput,
    Range,
    compulsory_deductible,
    compute_claimable,
    depreciation_rate,
    list_policy_templates,
)


def bumper(parts=(10000, 14000)):
    return LineItem("front bumper", "rubber_plastic", "replace", Range(*parts), Range(1000, 1500), Range(2000, 3000))


class ClaimsCalcTests(unittest.TestCase):
    def test_metal_depreciation_bands(self):
        self.assertEqual(depreciation_rate("metal", 3), 0.0)
        self.assertEqual(depreciation_rate("metal", 30), 0.15)
        self.assertEqual(depreciation_rate("metal", 200), 0.50)
        self.assertEqual(depreciation_rate("glass", 100), 0.0)
        self.assertEqual(depreciation_rate("fibreglass", 1), 0.30)

    def test_compulsory_deductible_by_engine_size(self):
        self.assertEqual(compulsory_deductible(1500), 1000)
        self.assertEqual(compulsory_deductible(1501), 2000)

    def test_standard_comprehensive_applies_depreciation_and_deductible(self):
        result = compute_claimable([bumper()], PolicyInput(idv_inr=500000, engine_cc=1200))
        self.assertEqual(result.gross_repair, Range(13000, 18500))
        # parts 50% + paint 50%: low 5000+1000, high 7000+1500
        self.assertEqual(result.depreciation, Range(6000, 8500))
        self.assertEqual(result.claimable.low, 13000 - 6000 - 1000)
        self.assertEqual(result.claimable.high, 18500 - 8500 - 1000)

    def test_zero_dep_removes_depreciation(self):
        std = compute_claimable([bumper()], PolicyInput(idv_inr=500000))
        zd = compute_claimable([bumper()], PolicyInput(template="comprehensive_zero_dep", idv_inr=500000))
        self.assertEqual(zd.depreciation.high, 0)
        self.assertGreater(zd.claimable.high, std.claimable.high)

    def test_third_party_only_not_claimable(self):
        result = compute_claimable([bumper()], PolicyInput(template="third_party_only"))
        self.assertFalse(result.covered)
        self.assertEqual(result.claimable, Range(0, 0))

    def test_copay_voluntary_deductible_and_floor_at_zero(self):
        small = LineItem("scratch", "metal", "repair", Range(0, 0), Range(500, 800), Range(0, 0))
        result = compute_claimable([small], PolicyInput(voluntary_deductible_inr=5000))
        self.assertEqual(result.claimable, Range(0, 0))
        co = compute_claimable([bumper()], PolicyInput(template="comprehensive_zero_dep", copay_pct=10))
        base = compute_claimable([bumper()], PolicyInput(template="comprehensive_zero_dep"))
        self.assertAlmostEqual(co.claimable.high, base.claimable.high * 0.9)

    def test_idv_cap_and_total_loss_flag(self):
        big = LineItem("body", "metal", "replace", Range(90000, 120000))
        result = compute_claimable([big], PolicyInput(template="comprehensive_zero_dep", idv_inr=100000))
        self.assertLessEqual(result.claimable.high, 100000)
        self.assertTrue(result.possible_total_loss)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            Range(5, 1)
        with self.assertRaises(ValueError):
            PolicyInput(copay_pct=120)
        with self.assertRaises(ValueError):
            compute_claimable([bumper()], PolicyInput(template="nope"))

    def test_templates_listed_and_serializable(self):
        self.assertTrue(
            {"third_party_only", "standard_comprehensive", "comprehensive_zero_dep"} <= set(list_policy_templates())
        )
        data = compute_claimable([bumper()], PolicyInput()).to_dict()
        self.assertTrue(data["disclaimer"])
        self.assertTrue(data["assumptions"])


if __name__ == "__main__":
    unittest.main()
