"""Regression checks for sample-weighted evaluation of uneven batches."""

import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pyhealth.datasets.utils import collate_fn_dict
from pyhealth.trainer import Trainer


class _LossModel(nn.Module):
    def __init__(self, mode=None):
        super().__init__()
        self.mode = mode

    def forward(self, x, **kwargs):
        if isinstance(x, tuple):
            x = x[1]
        x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1)
        result = {"loss": x.square().mean()}
        if self.mode is not None:
            result.update(y_true=torch.zeros_like(x), y_prob=x, embedding=x + 1)
        return result


class TestTrainerLossAggregation(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"x": torch.tensor(float(x)), "patient_id": f"p{i}"}
            for i, x in enumerate((1, 2, 3, 10))
        ]
        self.expected = (1 + 4 + 9 + 100) / 4

    def _trainer(self, mode=None):
        return Trainer(
            _LossModel(mode), metrics=["mse"], device="cpu", enable_logging=False
        )

    def test_loss_is_independent_of_batch_size_for_both_modes(self):
        for mode in (None, "regression"):
            trainer = self._trainer(mode)
            for batch_size in (1, 3, 4):
                with self.subTest(mode=mode, batch_size=batch_size):
                    loader = DataLoader(self.samples, batch_size=batch_size)
                    scores = trainer.evaluate(loader)
                    self.assertAlmostEqual(scores["loss"], self.expected)
                    if mode is not None:
                        self.assertAlmostEqual(scores["mse"], self.expected)

    def test_inference_preserves_predictions_additional_outputs_and_ids(self):
        loader = DataLoader(self.samples, batch_size=3)
        y_true, y_prob, loss, extra, ids = self._trainer("regression").inference(
            loader, additional_outputs=["embedding"], return_patient_ids=True
        )
        np.testing.assert_array_equal(y_true, np.zeros((4, 1)))
        np.testing.assert_array_equal(y_prob[:, 0], [1, 2, 3, 10])
        np.testing.assert_array_equal(extra["embedding"], y_prob + 1)
        self.assertEqual(ids, [f"p{i}" for i in range(4)])
        self.assertAlmostEqual(loss, self.expected)

    def test_custom_loss_counts_actual_tensor_and_tuple_batches_without_ids(self):
        trainer = self._trainer()
        tensor_batches = [
            {"x": torch.tensor([1.0, 2.0, 3.0])},
            {"x": torch.tensor([10.0])},
        ]
        tuple_batches = [{"x": (None, batch["x"])} for batch in tensor_batches]
        for batches in (tensor_batches, tuple_batches):
            self.assertAlmostEqual(trainer.evaluate(batches)["loss"], self.expected)

    def test_custom_loss_supports_legacy_list_collation(self):
        loader = DataLoader(
            [{"x": float(x)} for x in (1, 2, 3, 10)],
            batch_size=3,
            collate_fn=collate_fn_dict,
        )
        self.assertAlmostEqual(self._trainer().evaluate(loader)["loss"], self.expected)

    def test_dropped_samples_are_excluded_from_denominator(self):
        for mode in (None, "regression"):
            with self.subTest(mode=mode):
                loader = DataLoader(self.samples, batch_size=3, drop_last=True)
                self.assertAlmostEqual(
                    self._trainer(mode).evaluate(loader)["loss"], 14 / 3
                )

    def test_empty_evaluation_is_rejected(self):
        for mode in (None, "regression"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "empty"):
                self._trainer(mode).evaluate(DataLoader([], batch_size=3))
        with self.assertRaisesRegex(ValueError, "empty"):
            self._trainer("regression").inference([])


if __name__ == "__main__":
    unittest.main()
