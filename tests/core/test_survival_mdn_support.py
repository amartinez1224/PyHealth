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
