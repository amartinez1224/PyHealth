"""Offline survival protocol tests; no model, data download, or lifelines needed."""

import json
import unittest
from pathlib import Path

import numpy as np

from pyhealth.metrics import (
    survival_evaluation_grid,
    survival_metrics_fn,
)


class TestSurvivalMetrics(unittest.TestCase):
    def setUp(self):
        self.d = np.array([1.0, 2.0, 3.0, 4.0])
        self.e = np.array([1.0, 0.0, 1.0, 1.0])
        self.t = survival_evaluation_grid(self.d, self.e, integration_steps=17)
        self.s = np.exp(-np.array([0.8, 0.3, 0.05, 0.1])[:, None] * self.t)

    def score(self, **kwargs):
        args = {
            "duration": self.d,
            "event": self.e,
            "survival": self.s,
            "times": self.t,
            "integration_steps": 17,
        }
        args.update(kwargs)
        return survival_metrics_fn(**args)

    def test_pinned_reference_cases(self):
        path = Path(__file__).parent / "fixtures/survival_metrics_reference.json"
        cases = json.loads(path.read_text())["cases"]
        for case in cases:
            with self.subTest(
                case=case["name"],
                cutoff=case["censoring_threshold"],
                steps=case["integration_steps"],
            ):
                d, e = np.array(case["duration"]), np.array(case["event"])
                options = {
                    k: case[k] for k in ("censoring_threshold", "integration_steps")
                }
                grid = survival_evaluation_grid(d, e, **options)
                s = np.exp(
                    -np.array(case["rates"])[:, None]
                    * grid[None, :] ** np.array(case["powers"])[:, None]
                )
                actual = survival_metrics_fn(d, e, s, grid, **options)
                for metric, expected in case["expected"].items():
                    self.assertAlmostEqual(actual[metric], expected, places=12)

    def test_uncensored_hand_calculation(self):
        # At t=1: mean(.2^2, (1-.8)^2)=.04.
        # At t=2: mean(.1^2, .4^2)=.085; uniform point mean=.0625.
        result = survival_metrics_fn(
            [1, 2], [1, 1], [[0.2, 0.1], [0.8, 0.4]], [1, 2], integration_steps=2
        )
        self.assertEqual(result["c_index"], 1.0)
        self.assertAlmostEqual(result["ibs"], 0.0625)
        expected = (
            2 * np.log(0.8 + 1e-10) + np.log(0.9 + 1e-10) + np.log(0.6 + 1e-10)
        ) / 4
        self.assertAlmostEqual(result["ibll"], expected)

    def test_censoring_weights_hand_calculation(self):
        # d=[1,2,3], censor at 2 gives G=[1,.5,.5].
        # Point losses: .11, .04, .44; the censored row stays in denominator N.
        result = survival_metrics_fn(
            [1, 2, 3],
            [1, 0, 1],
            np.repeat([[0.2], [0.5], [0.8]], 3, axis=1),
            [1, 2, 3],
            metrics=["ibs", "ibll"],
            integration_steps=3,
        )
        self.assertAlmostEqual(result["ibs"], (0.11 + 0.04 + 0.44) / 3)
        self.assertAlmostEqual(
            result["ibll"],
            (6 * np.log(0.8 + 1e-10) + np.log(0.5 + 1e-10) + 2 * np.log(0.2 + 1e-10))
            / 9,
        )

    def test_inverse_squared_concordance_weight(self):
        # First event: 3 concordant pairs at weight 1. Second: 0/1 at 2.25.
        self.assertAlmostEqual(self.score()["c_index"], 3 / 5.25)

    def test_prediction_ties_get_zero_credit(self):
        self.assertEqual(self.score(survival=np.full_like(self.s, 0.5))["c_index"], 0.0)

    def test_event_censor_ties_are_comparable(self):
        result = survival_metrics_fn(
            [1, 1], [1, 0], [[0.2], [0.8]], [1], metrics=["c_index"]
        )
        self.assertEqual(result["c_index"], 1.0)

    def test_event_event_ties_excluded(self):
        with self.assertRaisesRegex(ValueError, "no comparable pairs"):
            survival_metrics_fn(
                [1, 1], [1, 1], [[0.2], [0.8]], [1], metrics=["c_index"]
            )

    def test_equal_cutoff_includes_concordance_anchor(self):
        # G(2)=2/3; anchor is included at equality (integration uses strict >).
        result = survival_metrics_fn(
            [1, 2, 3],
            [0, 1, 1],
            [[0.7], [0.3], [0.8]],
            [2],
            metrics=["c_index"],
            censoring_threshold=2 / 3,
        )
        self.assertEqual(result["c_index"], 1.0)

    def test_zero_g_rows_removed_and_renormalized(self):
        # Last row censored: G(3)=0. Reference excludes that entire row.
        result = survival_metrics_fn(
            [1, 2, 3],
            [1, 1, 0],
            np.full((3, 2), 0.5),
            [1, 2],
            metrics=["ibs"],
            integration_steps=2,
        )
        self.assertAlmostEqual(result["ibs"], 0.25)

    def test_linear_km_between_observations(self):
        # Censoring at 3: G(1)=1, G(2)=.75 by interpolation, G(3)=.5.
        result = survival_metrics_fn(
            [1, 3, 4],
            [1, 0, 1],
            np.full((3, 4), 0.5),
            [1, 2, 3, 4],
            metrics=["ibs"],
            integration_steps=4,
        )
        expected = (0.25 + (1 + 2 / 0.75) / 12 + 0.25 + 0.25) / 4
        self.assertAlmostEqual(result["ibs"], expected)

    def test_linear_survival_interpolation(self):
        d, e = [1, 2, 3], [1, 1, 1]
        coarse = np.array([[0.8, 0.2], [0.9, 0.3], [1.0, 0.4]])
        dense = np.column_stack([coarse[:, 0], coarse.mean(axis=1), coarse[:, 1]])
        a = survival_metrics_fn(d, e, coarse, [1, 3], integration_steps=3)
        b = survival_metrics_fn(d, e, dense, [1, 2, 3], integration_steps=3)
        for name in a:
            self.assertAlmostEqual(a[name], b[name])

    def test_row_permutation_invariance(self):
        perm = [2, 0, 3, 1]
        actual = self.score(
            duration=self.d[perm], event=self.e[perm], survival=self.s[perm]
        )
        for name, expected in self.score().items():
            self.assertAlmostEqual(actual[name], expected)

    def test_inputs_not_mutated(self):
        copies = [x.copy() for x in (self.d, self.e, self.s, self.t)]
        self.score()
        for original, current in zip(copies, (self.d, self.e, self.s, self.t)):
            np.testing.assert_array_equal(original, current)

    def test_noncontiguous_float32_adapter_contract(self):
        # Exported SUPPORT arrays: shifted float32 labels, [N,T] curves.
        d = (self.d + 0.001).astype(np.float32)
        t = np.linspace(0, 5, 128, dtype=np.float32)
        s = np.exp(-np.array([0.8, 0.3, 0.05, 0.1])[:, None] * t).astype(np.float32)
        result = survival_metrics_fn(
            d[::-1], self.e.astype(np.float32)[::-1], s[::-1], t
        )
        self.assertTrue(
            all(isinstance(v, float) and np.isfinite(v) for v in result.values())
        )

    def test_unit_rescaling(self):
        result = self.score(duration=self.d * 100, times=self.t * 100)
        for name, expected in self.score().items():
            self.assertAlmostEqual(result[name], expected)

    def test_all_censored_metrics_subset(self):
        e = np.zeros_like(self.e)
        with self.assertRaisesRegex(ValueError, "no comparable pairs"):
            self.score(event=e)
        result = self.score(event=e, metrics=["ibs", "ibll"])
        self.assertTrue(all(np.isfinite(v) for v in result.values()))

    def test_probability_boundaries_finite(self):
        for value in (0.0, 1.0):
            result = self.score(survival=np.full_like(self.s, value))
            self.assertTrue(all(np.isfinite(v) for v in result.values()))

    def test_grid_union_and_options(self):
        grid = survival_evaluation_grid(self.d, self.e, integration_steps=17)
        self.assertTrue(np.all(np.diff(grid) > 0))
        for time in (1.0, 3.0):
            self.assertIn(time, grid)
        np.testing.assert_array_equal(
            survival_evaluation_grid(self.d, self.e, metrics=["c_index"]), [1.0, 3.0]
        )

    def test_default_steps_and_metric_order(self):
        default = survival_metrics_fn(self.d, self.e, self.s, self.t)
        explicit = survival_metrics_fn(
            self.d, self.e, self.s, self.t, integration_steps=1000
        )
        self.assertEqual(default, explicit)
        self.assertEqual(list(self.score(metrics=["ibll", "ibs"])), ["ibll", "ibs"])

    def test_undefined_integration(self):
        for d, e in (([1], [1]), ([1, 1], [1, 0]), ([1, 2], [0, 0])):
            with self.subTest(d=d, e=e), self.assertRaises(ValueError):
                survival_evaluation_grid(d, e, metrics=["ibs"], censoring_threshold=0.8)

    def test_no_extrapolation(self):
        for t, s in ((self.t[1:], self.s[:, 1:]), (self.t[:-1], self.s[:, :-1])):
            with self.assertRaisesRegex(ValueError, "extrapolation"):
                self.score(times=t, survival=s)

    def test_invalid_inputs(self):
        invalid = [
            {"duration": []},
            {"duration": [-1, 2, 3, 4]},
            {"duration": [1, 2, np.nan, 4]},
            {"duration": [[1, 2, 3, 4]]},
            {"event": [1, 2, 0, 1]},
            {"event": [1, 0]},
            {"event": [1, 0.1, 0, 1]},
            {"event": [1, 0, np.inf, 1]},
            {"times": self.t[::-1]},
            {"times": np.zeros_like(self.t)},
            {"times": -self.t},
            {"times": self.t.astype(complex)},
            {"survival": self.s.T},
            {"survival": np.full_like(self.s, np.nan)},
            {"survival": np.full_like(self.s, 1.01)},
            {"survival": np.full_like(self.s, -0.01)},
            {"survival": self.s[:, ::-1]},
            {"metrics": ["harrell"]},
            {"metrics": []},
            {"metrics": "ibs"},
            {"censoring_threshold": 0},
            {"censoring_threshold": 1},
            {"censoring_threshold": np.nan},
            {"censoring_threshold": True},
            {"integration_steps": 1},
            {"integration_steps": 2.5},
            {"integration_steps": True},
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score(**change)


if __name__ == "__main__":
    unittest.main()
