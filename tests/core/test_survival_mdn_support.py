"""Offline tests for the example-scoped released SUPPORT adapter."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pyhealth.datasets import get_dataloader
from pyhealth.models import SurvivalMDN

SPEC = importlib.util.spec_from_file_location(
    "support_example",
    Path(__file__).resolve().parents[2] / "examples/survival_mdn_support.py",
)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


class TestSupportLoader(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.arrays = {}
        for index, phase in enumerate(("train", "valid", "test")):
            x = np.arange(5 * 27, dtype=np.float32).reshape(5, 27) / 100 + index
            y = np.column_stack((np.linspace(0, 2, 5), [0, 1, 0, 1, 1]))
            # This value distinguishes addition before and after float32 cast.
            y[1, 0] = 1.3562357016817128
            self.arrays[phase] = (x, y)
            self.write(phase, x, y)

    def write(self, phase, x, y, **extra):
        np.savez(
            self.data / f"{phase}_1.npz",
            arr_0=x,
            arr_1=y,
            arr_2=np.full((len(x), 27), -999),
            **extra,
        )

    def test_mapping_schema_processors_and_identity(self):
        result = support.load_support_split(self.data)
        json.dumps(result["metadata"], allow_nan=False)
        train = result["datasets"]["train"]
        self.assertEqual(train.input_schema, {"features": "tensor"})
        self.assertEqual(train.output_schema, {"duration": "tensor", "event": "tensor"})
        self.assertIsNone(SurvivalMDN(train).mode)
        for phase, dataset in result["datasets"].items():
            x, y = self.arrays[phase]
            self.assertIs(
                dataset.input_processors["features"], train.input_processors["features"]
            )
            for key in ("duration", "event"):
                self.assertIs(
                    dataset.output_processors[key], train.output_processors[key]
                )
            batch = next(iter(get_dataloader(dataset, batch_size=5)))
            np.testing.assert_array_equal(batch["features"].numpy(), x)
            np.testing.assert_array_equal(
                batch["duration"].numpy(), (y[:, 0] + 0.001).astype(np.float32)
            )
            np.testing.assert_array_equal(
                batch["event"].numpy(), y[:, 1].astype(np.float32)
            )
            self.assertEqual(batch["features"].shape, (5, 27))
            self.assertEqual(batch["duration"].shape, (5,))
            self.assertEqual(batch["source_row_index"], list(range(5)))
            self.assertEqual(batch["partition"], [phase] * 5)
            self.assertEqual(
                batch["record_id"],
                [f"support-split1-{phase}-row{i:06d}" for i in range(5)],
            )
            self.assertEqual(result["metadata"]["partitions"][phase]["events"], 3)
        self.assertNotEqual(
            np.float32(self.arrays["train"][1][1, 0] + 0.001),
            np.float32(self.arrays["train"][1][1, 0]) + np.float32(0.001),
        )
        self.assertEqual(train[0]["duration"].item(), float(np.float32(0.001)))

    def test_unrelated_pickle_is_not_opened(self):
        x, y = self.arrays["train"]
        self.write("train", x, y, unused=np.array([object()], dtype=object))
        support.load_support_split(self.data)

    def test_all_censored_and_all_event(self):
        for value in (0, 1):
            for phase, (x, y) in self.arrays.items():
                labels = y.copy()
                labels[:, 1] = value
                self.write(phase, x, labels)
            loaded = support.load_support_split(self.data)
            for dataset in loaded["datasets"].values():
                self.assertEqual(dataset.output_processors["event"].dim(), (0,))
                self.assertTrue(
                    all(dataset[i]["event"].item() == value for i in range(5))
                )

    def test_invalid_split_files_and_keys(self):
        for split in (0, 11, 1.0, True, "1", None):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "split"):
                support.load_support_split(self.data, split)
        with self.assertRaisesRegex(FileNotFoundError, "train_2.npz"):
            support.load_support_split(self.data, 2)
        for missing in ("arr_0", "arr_1"):
            arrays = dict(zip(("arr_0", "arr_1"), self.arrays["train"]))
            arrays.pop(missing)
            np.savez(self.data / "train_1.npz", **arrays)
            with self.assertRaisesRegex(ValueError, f"train_1.npz.*{missing}"):
                support.load_support_split(self.data)

    def test_invalid_arrays(self):
        x, y = self.arrays["train"]
        bad_x = [
            np.zeros((0, 27)),
            np.zeros((5, 26)),
            np.zeros(27),
            np.zeros((5, 27, 1)),
            x.astype(complex),
            x.astype(str),
            np.full((5, 27), np.nan),
            np.full((5, 27), np.inf),
            np.full((5, 27), 1e100),
        ]
        bad_y = [
            np.zeros((0, 2)),
            y[:4],
            np.zeros(5),
            np.zeros((5, 3)),
            np.full((5, 2), np.nan),
            np.full((5, 2), np.inf),
            np.column_stack((y[:, 0], np.full(5, 0.5))),
            np.column_stack((np.full(5, -1), y[:, 1])),
            np.column_stack((np.full(5, 1e100), y[:, 1])),
            y.astype(complex),
            y.astype(str),
            y.astype(object),
        ]
        for features, labels, field in [(v, y, "arr_0") for v in bad_x] + [
            (x, v, "arr_1") for v in bad_y
        ]:
            with self.subTest(shape=features.shape, label=labels.shape, field=field):
                self.write("train", features, labels)
                with self.assertRaisesRegex(ValueError, f"train_1.npz.*{field}"):
                    support.load_support_split(self.data)

    def test_grid_training_only_and_seeded_order(self):
        first = support.load_support_split(self.data)["datasets"]
        grid = support._grid(first["train"])
        for phase in ("valid", "test"):
            x, y = self.arrays[phase]
            y = y.copy()
            y[:, 0] += 100
            self.write(phase, x, y)
        second = support.load_support_split(self.data)["datasets"]
        np.testing.assert_array_equal(grid, support._grid(second["train"]))
        self.assertEqual(grid[0], 0)
        self.assertEqual(grid[-1], np.float32(2.001))

        def order():
            return [
                i
                for b in get_dataloader(first["train"], 2, shuffle=True, seed=7)
                for i in b["source_row_index"]
            ]

        self.assertEqual(order(), order())
        self.assertEqual(sorted(order()), list(range(5)))

    def test_output_and_compatibility(self):
        metadata = support.load_support_split(self.data)["metadata"]
        support._check_compatible(metadata, json.loads(json.dumps(metadata)))
        changed = json.loads(json.dumps(metadata))
        changed["feature_width"] = 26
        with self.assertRaisesRegex(ValueError, "feature_width"):
            support._check_compatible(metadata, changed)
        changed = json.loads(json.dumps(metadata))
        changed["partitions"]["test"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "test archive/hash"):
            support._check_compatible(metadata, changed)
        out = self.root / "out"
        support._prepare_output(out)
        support._prepare_output(out)
        (out / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "empty"):
            support._prepare_output(out)
        self.assertEqual((out / "keep.txt").read_text(), "keep")

    def test_prediction_partial_batch_features_only(self):
        # More than 256 rows exercises the final partial prediction batch.
        x, y = self.arrays["test"]
        self.write("test", np.tile(x, (52, 1)), np.tile(y, (52, 1)))
        datasets = support.load_support_split(self.data)["datasets"]
        model = SurvivalMDN(datasets["train"], **support.MODEL_CONFIG)
        original = model.predict_survival
        seen = []

        def predict(grid, **kwargs):
            self.assertEqual(set(kwargs), {"features"})
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(model.training)
            seen.append(len(kwargs["features"]))
            return original(grid, **kwargs)

        with patch.object(model, "predict_survival", side_effect=predict):
            result = support._predict(
                model, datasets["test"], support._grid(datasets["train"])
            )
        self.assertEqual(seen, [256, 4])
        np.testing.assert_array_equal(result["source_row_index"], np.arange(260))
        self.assertEqual(result["survival"].shape, (260, 128))

    def test_prediction_arbitrary_grid_lengths_and_nonzero_start(self):
        datasets = support.load_support_split(self.data)["datasets"]
        model = SurvivalMDN(datasets["train"], **support.MODEL_CONFIG)
        for grid in (
            np.array([0.0, 0.25, 0.5, 1.0, 2.0], dtype=np.float32),
            np.array([0.25, 0.5, 1.0], dtype=np.float32),
        ):
            with self.subTest(grid=grid):
                result = support._predict(model, datasets["test"], grid)
                self.assertEqual(result["survival"].shape, (5, len(grid)))
                np.testing.assert_array_equal(result["times"], grid)
                if grid[0] > 0:
                    self.assertTrue((result["survival"][:, 0] < 1).all())
                else:
                    np.testing.assert_array_equal(result["survival"][:, 0], 1)

    def test_float64_grid_values_collapsing_in_float32(self):
        datasets = support.load_support_split(self.data)["datasets"]
        model = SurvivalMDN(datasets["train"], **support.MODEL_CONFIG)
        grid = np.array([0.0, 0.1, 0.1 + 1e-10, 0.5], dtype=np.float64)
        self.assertLess(grid[1], grid[2])
        self.assertEqual(np.float32(grid[1]), np.float32(grid[2]))

        predictions = support._predict(model, datasets["test"], grid)

        self.assertEqual(predictions["survival"].shape, (5, len(grid)))
        np.testing.assert_array_equal(predictions["times"], grid)
        self.assertEqual(predictions["times"].dtype, np.float64)
        np.testing.assert_array_equal(
            predictions["survival"][:, 1], predictions["survival"][:, 2]
        )

    def test_prediction_rejects_invalid_grids(self):
        invalid_grids = (
            np.array([]),
            np.array([[0.0, 1.0]]),
            np.array([0.0, np.nan]),
            np.array([-0.1, 0.1]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.5]),
            np.array([0.0, np.inf]),
            np.array(["0", "1"]),
        )
        for grid in invalid_grids:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                support._predict(None, object(), grid)

    def test_evaluate_metrics_uses_union_grid_and_full_ordered_cohort(self):
        datasets = support.load_support_split(self.data)["datasets"]
        dataset = datasets["test"]
        helper_grids = (
            np.array([0.0, 0.5, 1.0]),
            np.array([0.25, 0.75, 1.25]),
            np.array([0.5, 1.0, 1.5]),
        )
        helper_calls = []

        def grid_helper(duration, event, **kwargs):
            helper_calls.append((duration.copy(), event.copy(), kwargs.copy()))
            return helper_grids[len(helper_calls) - 1]

        class Predictor:
            def eval(self):
                return self

            def predict_survival(self, times, *, features):
                return torch.exp(-times[None, :].expand(len(features), -1))

        score_calls = []

        def score(duration, event, survival, times, **kwargs):
            score_calls.append((duration.copy(), event.copy(), survival.copy(), times.copy(), kwargs.copy()))
            return {"c_index": 0.5, "ibs": 0.2, "ibll": -0.3}

        with (
            patch.object(support, "survival_evaluation_grid", side_effect=grid_helper),
            patch.object(support, "survival_metrics_fn", side_effect=score),
        ):
            predictions, report = support._evaluate_metrics(Predictor(), dataset)

        expected_duration = np.array(
            [dataset[i]["duration"].item() for i in range(len(dataset))],
            dtype=np.float32,
        )
        expected_event = np.array(
            [dataset[i]["event"].item() for i in range(len(dataset))],
            dtype=np.float32,
        )
        expected_times = np.unique(np.concatenate(helper_grids)).astype(np.float32)
        self.assertEqual(len(helper_calls), 3)
        for index, (duration, event, kwargs) in enumerate(helper_calls):
            np.testing.assert_array_equal(duration, expected_duration)
            np.testing.assert_array_equal(event, expected_event)
            self.assertEqual(kwargs["censoring_threshold"], (1e-8, 0.2, 0.4)[index])
            self.assertEqual(kwargs["integration_steps"], 1000)
        self.assertEqual(predictions["survival"].shape, (len(dataset), len(expected_times)))
        np.testing.assert_array_equal(predictions["times"], expected_times)
        np.testing.assert_array_equal(predictions["duration"], expected_duration)
        np.testing.assert_array_equal(predictions["event"], expected_event)
        np.testing.assert_array_equal(predictions["source_row_index"], np.arange(len(dataset)))
        self.assertEqual(report["protocol_revision"], support.SOURCE["revision"])
        self.assertEqual(report["dtype"], "float32")
        self.assertEqual(report["time_unit"], support.UNITS)
        self.assertEqual(report["sample_count"], len(dataset))
        self.assertEqual(report["time_count"], len(expected_times))
        self.assertEqual(len(report["cutoffs"]), 3)
        self.assertEqual([entry["cutoff"] for entry in report["cutoffs"]], [1e-8, 0.2, 0.4])
        self.assertEqual([entry["scores"]["ibs"] for entry in report["cutoffs"]], [0.2] * 3)
        self.assertEqual(len(score_calls), 3)
        for index, (duration, event, survival, times, kwargs) in enumerate(score_calls):
            np.testing.assert_array_equal(duration, expected_duration)
            np.testing.assert_array_equal(event, expected_event)
            np.testing.assert_array_equal(survival, predictions["survival"])
            np.testing.assert_array_equal(times, predictions["times"])
            self.assertEqual(kwargs["censoring_threshold"], (1e-8, 0.2, 0.4)[index])
            self.assertEqual(kwargs["integration_steps"], 1000)
        json.dumps(report, allow_nan=False)

    def test_evaluate_metrics_names_undefined_cutoff(self):
        dataset = support.load_support_split(self.data)["datasets"]["test"]
        with (
            patch.object(
                support,
                "survival_evaluation_grid",
                side_effect=[np.array([0.0, 1.0]), ValueError("undefined metric")],
            ),
            self.assertRaisesRegex(ValueError, "cutoff 0.2.*undefined metric"),
        ):
            support._evaluate_metrics(None, dataset)

    def test_evaluate_metrics_names_undefined_score_cutoff(self):
        dataset = support.load_support_split(self.data)["datasets"]["test"]
        times = np.array([0.0, 1.0], dtype=np.float32)
        predictions = {
            "survival": np.ones((len(dataset), len(times)), dtype=np.float32),
            "times": times,
        }
        score_results = [
            {"c_index": 0.5, "ibs": 0.2, "ibll": -0.3},
            ValueError("no comparable pairs"),
        ]
        with (
            patch.object(
                support,
                "survival_evaluation_grid",
                return_value=np.array([0.0, 1.0]),
            ),
            patch.object(support, "_predict", return_value=predictions),
            patch.object(support, "survival_metrics_fn", side_effect=score_results),
            self.assertRaisesRegex(ValueError, "cutoff 0.2.*no comparable pairs"),
        ):
            support._evaluate_metrics(None, dataset)

    def test_end_to_end_repeat_and_prediction_only(self):
        # Keep a partial final training batch; validation must be one full batch.
        x, y = self.arrays["train"]
        self.write("train", np.tile(x, (52, 1)), np.tile(y, (52, 1)))
        train_calls = []
        original = support.Trainer.train

        def train(trainer, loader, **kwargs):
            self.assertNotIn("test_dataloader", kwargs)
            self.assertEqual([len(b["duration"]) for b in loader], [256, 4])
            valid_loader = kwargs["val_dataloader"]
            self.assertEqual(len(valid_loader), 1)
            batch = next(iter(valid_loader))
            trainer.model.eval()
            with torch.no_grad():
                expected = trainer.model(**batch)["loss"].item()
            self.assertEqual(trainer.evaluate(valid_loader)["loss"], expected)
            train_calls.append(kwargs)
            return original(trainer, loader, **kwargs)

        args = ["--data-dir", str(self.data), "--epochs", "2", "--patience", "1"]
        with patch.object(support.Trainer, "train", new=train):
            first = support.main(args + ["--output-dir", str(self.root / "first")])
            support.main(args + ["--output-dir", str(self.root / "second")])
        self.assertEqual(len(train_calls), 2)
        self.assertNotIn("metric_evaluation", first)
        self.assertTrue((self.root / "first/test_predictions.npz").is_file())
        self.assertFalse((self.root / "first/test_metric_predictions.npz").exists())
        self.assertFalse((self.root / "first/test_metrics.json").exists())
        with np.load(self.root / "first/test_predictions.npz", allow_pickle=False) as p:
            self.assertEqual(p["times"].shape, (128,))
        support.main(
            [
                "--data-dir",
                str(self.data),
                "--predict-only",
                str(self.root / "first"),
                "--output-dir",
                str(self.root / "replay"),
            ]
        )
        self.assertTrue(first["all_test_rows_checkpoint_reload_identical"])
        for other in ("second", "replay"):
            with (
                np.load(
                    self.root / "first/test_predictions.npz", allow_pickle=False
                ) as a,
                np.load(
                    self.root / other / "test_predictions.npz", allow_pickle=False
                ) as b,
            ):
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            a = torch.load(self.root / "first/checkpoints/best.ckpt", weights_only=True)
            b = torch.load(
                self.root / other / "checkpoints/best.ckpt", weights_only=True
            )
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "empty"):
            support.main(args + ["--output-dir", str(self.root / "first")])
        x, y = self.arrays["test"]
        self.write("test", x + 1, y)
        with self.assertRaisesRegex(ValueError, "test archive/hash"):
            support.main(
                [
                    "--data-dir",
                    str(self.data),
                    "--predict-only",
                    str(self.root / "first"),
                    "--output-dir",
                    str(self.root / "bad-replay"),
                ]
            )
        self.assertFalse((self.root / "bad-replay").exists())

    def test_evaluate_metrics_cli_smoke(self):
        output = self.root / "metric-run"
        original_train = support.Trainer.train
        original_evaluate_metrics = support._evaluate_metrics
        training_completed = []

        def train(trainer, loader, **kwargs):
            self.assertNotIn("test_dataloader", kwargs)
            result = original_train(trainer, loader, **kwargs)
            training_completed.append(True)
            return result

        def evaluate_metrics(model, dataset):
            self.assertTrue(training_completed)
            self.assertEqual(len(dataset), 5)
            return original_evaluate_metrics(model, dataset)

        with (
            patch.object(support.Trainer, "train", new=train),
            patch.object(support, "_evaluate_metrics", side_effect=evaluate_metrics),
        ):
            manifest = support.main(
                [
                    "--data-dir",
                    str(self.data),
                    "--epochs",
                    "1",
                    "--patience",
                    "1",
                    "--output-dir",
                    str(output),
                    "--evaluate-metrics",
                ]
            )

        self.assertEqual(len(manifest["prediction_grid"]), 128)
        self.assertTrue((output / "test_predictions.npz").is_file())
        self.assertTrue((output / "test_metric_predictions.npz").is_file())
        self.assertTrue((output / "test_metrics.json").is_file())
        metric_metadata = manifest["metric_evaluation"]
        self.assertEqual(metric_metadata["prediction_file"], "test_metric_predictions.npz")
        self.assertEqual(metric_metadata["report_file"], "test_metrics.json")
        self.assertEqual(metric_metadata["cutoffs"], [1e-8, 0.2, 0.4])
        self.assertEqual(metric_metadata["integration_steps"], 1000)
        self.assertEqual(metric_metadata["prediction_dtype"], "float32")
        self.assertEqual(metric_metadata["time_dtype"], "float64")
        self.assertTrue(metric_metadata["replay_agreed"])
        with np.load(output / "test_predictions.npz", allow_pickle=False) as legacy:
            self.assertEqual(legacy["times"].shape, (128,))
        with np.load(
            output / "test_metric_predictions.npz", allow_pickle=False
        ) as metric_predictions:
            self.assertEqual(
                metric_predictions["survival"].shape,
                (5, metric_metadata["time_count"]),
            )
            self.assertGreater(metric_metadata["time_count"], 128)
        report = json.loads((output / "test_metrics.json").read_text())
        self.assertEqual(report["sample_count"], 5)
        self.assertEqual(len(report["cutoffs"]), 3)
        self.assertTrue(all("scores" in item for item in report["cutoffs"]))

        source_files = (
            "manifest.json",
            "test_metric_predictions.npz",
            "test_metrics.json",
            "checkpoints/best.ckpt",
        )
        source_contents = {
            name: (output / name).read_bytes() for name in source_files
        }
        replay_output = self.root / "metric-replay"
        with patch.object(
            support.Trainer,
            "train",
            new=lambda *args, **kwargs: self.fail("Trainer.train called during replay"),
        ):
            replay_manifest = support.main(
                [
                    "--data-dir",
                    str(self.data),
                    "--predict-only",
                    str(output),
                    "--output-dir",
                    str(replay_output),
                    "--evaluate-metrics",
                ]
            )
        replay_metadata = replay_manifest["metric_evaluation"]
        self.assertTrue(replay_metadata["replay_agreed"])
        self.assertTrue(replay_metadata["source_run_agreed"])
        self.assertEqual(replay_metadata["source_run_comparison"], "verified")
        replay_report = json.loads((replay_output / "test_metrics.json").read_text())
        self.assertEqual(replay_report, report)
        with (
            np.load(output / "test_metric_predictions.npz", allow_pickle=False) as source,
            np.load(
                replay_output / "test_metric_predictions.npz", allow_pickle=False
            ) as replay,
        ):
            self.assertEqual(set(source.files), set(replay.files))
            for key in source.files:
                np.testing.assert_array_equal(source[key], replay[key])
        source_checkpoint = torch.load(
            output / "checkpoints/best.ckpt", weights_only=True
        )
        replay_checkpoint = torch.load(
            replay_output / "checkpoints/best.ckpt", weights_only=True
        )
        for key in source_checkpoint:
            torch.testing.assert_close(
                source_checkpoint[key], replay_checkpoint[key], atol=0, rtol=0
            )

        no_metric_output = self.root / "metric-disabled-replay"
        with patch.object(
            support.Trainer,
            "train",
            new=lambda *args, **kwargs: self.fail("Trainer.train called during replay"),
        ):
            no_metric_manifest = support.main(
                [
                    "--data-dir",
                    str(self.data),
                    "--predict-only",
                    str(output),
                    "--output-dir",
                    str(no_metric_output),
                ]
            )
        self.assertNotIn("metric_evaluation", no_metric_manifest)
        self.assertFalse((no_metric_output / "test_metric_predictions.npz").exists())
        self.assertFalse((no_metric_output / "test_metrics.json").exists())
        for name, contents in source_contents.items():
            self.assertEqual((output / name).read_bytes(), contents)

    def test_predict_only_metrics_with_legacy_manifest(self):
        output = self.root / "legacy-run"
        source_manifest = support.main(
            [
                "--data-dir",
                str(self.data),
                "--epochs",
                "1",
                "--patience",
                "1",
                "--output-dir",
                str(output),
            ]
        )
        self.assertNotIn("metric_evaluation", source_manifest)
        replay_output = self.root / "legacy-metric-replay"
        with patch.object(
            support.Trainer,
            "train",
            new=lambda *args, **kwargs: self.fail("Trainer.train called during replay"),
        ):
            replay_manifest = support.main(
                [
                    "--data-dir",
                    str(self.data),
                    "--predict-only",
                    str(output),
                    "--output-dir",
                    str(replay_output),
                    "--evaluate-metrics",
                ]
            )
        metric_metadata = replay_manifest["metric_evaluation"]
        self.assertTrue(metric_metadata["replay_agreed"])
        self.assertIsNone(metric_metadata["source_run_agreed"])
        self.assertEqual(metric_metadata["source_run_comparison"], "unavailable")
        self.assertTrue((replay_output / "test_metric_predictions.npz").is_file())
        self.assertTrue((replay_output / "test_metrics.json").is_file())
        self.assertNotIn(
            "metric_evaluation",
            json.loads((output / "manifest.json").read_text()),
        )

    def test_invalid_cli(self):
        for option in (
            ["--epochs", "0"],
            ["--patience", "0"],
            ["--seed", "-1"],
            ["--split", "11"],
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                support.main(
                    [
                        "--data-dir",
                        str(self.data),
                        "--output-dir",
                        str(self.root / "invalid"),
                    ]
                    + option
                )
        self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
