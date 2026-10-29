import unittest
import importlib.util

DEPENDENCIES_AVAILABLE = all(importlib.util.find_spec(module) is not None for module in ("pandas", "duckdb"))
if DEPENDENCIES_AVAILABLE:
    import pandas as pd
    from evaluation.scripts.run_animals_measurement import GROUND_TRUTH, _aggregate_quality, _top_group_quality


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "ReasonDB runtime dependencies unavailable")
class AnimalsScoringTests(unittest.TestCase):
    def test_labelled_ground_truth_scoring(self):
        self.assertTrue(_aggregate_quality(pd.DataFrame({"count": [27]}), GROUND_TRUTH / "Q1_500.csv")["correct"])
        self.assertTrue(_aggregate_quality(pd.DataFrame({"count": [5]}), GROUND_TRUTH / "Q2_500.csv")["correct"])
        q3 = _top_group_quality(pd.DataFrame({"city": ["Nairobi", "Kisumu"], "count": [2, 8]}), GROUND_TRUTH / "Q3_500.csv", ("city",))
        self.assertEqual(q3["f1"], 1.0)
        q10 = _top_group_quality(pd.DataFrame({"city": ["Kisumu", "Nairobi", "Nakuru"], "stationid": ["Station_C", "Station_A", "Station_B"], "count": [8, 8, 2]}), GROUND_TRUTH / "Q10_500.csv", ("city", "stationid"))
        self.assertEqual(q10["expected_rows"], 2)
        self.assertEqual(q10["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
