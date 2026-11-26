import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pyhealth.data import Event
from pyhealth.datasets import PatientSplit, Support2Dataset, get_dataloader
from pyhealth.tasks import Support2TimeToEvent
from pyhealth.tasks.support2_time_to_event import Support2FeatureProcessor

FIXTURE = (
    Path(__file__).resolve().parents[2] / "test-resources/core/support2/support2.csv"
)


def load_native(root, cache_dir):
    return Support2Dataset(root=str(root), tables=["support2"], cache_dir=cache_dir)


def run_task(*records):
    def get_events(event_type):
        assert event_type == "support2"
        return [Event(event_type, attr_dict=record) for record in records]

    return Support2TimeToEvent()(
        SimpleNamespace(patient_id="patient-x", get_events=get_events)
    )


def fit_processor(*rows):
    processor = Support2FeatureProcessor()
    processor.fit(({"features": row} for row in rows), "features")
    return processor


def write_native(root, records):
    root.mkdir()
    with FIXTURE.open(newline="") as source:
        header = next(csv.reader(source))
    with (root / "support2.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=header)
        writer.writeheader()
        writer.writerows(records)


class TestSupport2TimeToEvent(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_native_fixture_labels(self):
        tmp_path = self.root
        dataset = load_native(FIXTURE.parent, tmp_path)
        task = Support2TimeToEvent()
        expected = {"1": (2029, 0), "2": (4, 1), "3": (47, 1)}
        for patient_id, pair in expected.items():
            (sample,) = task(dataset.get_patient(patient_id))
            assert sample["patient_id"] == patient_id
            assert (sample["duration"], sample["event"]) == pair
            assert set(sample["features"]) == {"age", "sex"}

    def test_missing_and_duplicate_records(self):
        assert run_task() == []
        with self.assertRaisesRegex(ValueError, "patient-x.*2"):
            run_task({}, {})

    def test_unusable_duration(self):
        for duration in [None, "", "broken", 0, -5, "nan", "inf", 1e50, 1e-50]:
            with self.subTest(duration=duration):
                assert run_task({"d.time": duration, "death": 1}) == []

    def test_unusable_event(self):
        for event in [
            None,
            "",
            "broken",
            2,
            -1,
            "nan",
            "inf",
            0.5,
            "1.0000000000000000001",
            "1e-900",
        ]:
            with self.subTest(event=event):
                assert run_task({"d.time": 4, "death": event}) == []

    def test_numeric_strings_and_missing_predictors(self):
        assert run_task({"death": "1"}) == []
        assert run_task({"d.time": "4.5"}) == []
        assert run_task({"d.time": "4.5", "death": "1.0"}) == [
            {
                "patient_id": "patient-x",
                "features": {"age": None, "sex": None},
                "duration": 4.5,
                "event": 1.0,
            }
        ]

    def test_feature_encoding(self):
        processor = fit_processor(
            {"age": 70, "sex": "male"},
            {"age": "40", "sex": "female"},
            {"age": 40, "sex": None},
            {"age": "inf", "sex": " "},
        )
        self.assertAlmostEqual(processor.age_median, 40)
        self.assertAlmostEqual(processor.age_mean, 50)
        self.assertAlmostEqual(processor.age_scale, 200**0.5)
        assert processor.sex_vocabulary == {"female": 2, "male": 3}
        assert processor.schema() == ("value",)
        assert processor.dim() == (1,)
        assert processor.size() == 6
        known = processor.process({"age": "70", "sex": "male"})
        assert known.dtype == torch.float32
        np.testing.assert_allclose(known.tolist(), [2**0.5, 0, 0, 1, 0, 0], atol=1e-6)
        for missing in (None, " ", "invalid", "nan", float("inf")):
            np.testing.assert_allclose(
                processor.process({"age": missing, "sex": ""}).tolist(),
                [-(0.5**0.5), 1, 0, 0, 1, 0],
                atol=1e-6,
            )
        np.testing.assert_allclose(
            processor.process({"age": 50, "sex": "unseen"}).tolist(),
            [0, 0, 0, 0, 0, 1],
            atol=1e-6,
        )

    def test_constant_and_wholly_missing_training_ages(self):
        constant = fit_processor({"age": 60}, {"age": 60})
        assert constant.age_scale == 1
        assert constant.process({"age": 61}).tolist() == [1, 0, 1, 0]
        assert constant.process({}).tolist() == [0, 1, 1, 0]
        missing = fit_processor({"age": None}, {"age": "invalid"})
        assert missing.age_median is None
        assert missing.process({"age": 61}).tolist() == [0, 0, 1, 0]
        assert missing.process({}).tolist() == [0, 1, 1, 0]

    def test_labels_and_nonpredictors_cannot_leak(self):
        record = {"age": "65", "sex": "female", "d.time": "47", "death": "1"}
        (baseline,) = run_task(record)
        processor = fit_processor(baseline["features"])
        expected_features = processor.process(baseline["features"])
        expected_labels = {"duration": 47, "event": 1}
        for field, value, labels in (
            ("death", "0", {"duration": 47, "event": 0}),
            ("d.time", "48", {"duration": 48, "event": 1}),
            ("hospdead", "1", expected_labels),
            ("slos", "100", expected_labels),
            ("surv2m", "0.2", expected_labels),
            ("surv6m", "0.1", expected_labels),
            ("future_measurement", "1234", expected_labels),
        ):
            (sample,) = run_task({**record, field: value})
            assert sample["features"] == baseline["features"]
            assert torch.equal(processor.process(sample["features"]), expected_features)
            assert {key: sample[key] for key in labels} == labels

    def test_held_out_rows_do_not_shape_fitted_processor(self):
        tmp_path = self.root
        records = [
            {
                "sno": f"p{i}",
                "age": 40 + i,
                "sex": ("female" if i % 2 else "male"),
                "d.time": 10 + i,
                "death": i % 2,
            }
            for i in range(20)
        ]
        split = PatientSplit((0.70, 0.15, 0.15), seed=14)

        def prepare(name):
            root = tmp_path / name
            write_native(root, records)
            dataset = load_native(root, tmp_path / "cache")
            return dataset.set_task(Support2TimeToEvent(), split=split)

        train, val, test = prepare("before")
        held_out = set(val.patient_to_index) | set(test.patient_to_index)
        for record in records:
            if record["sno"] in held_out:
                record.update(age="1000", sex="held-out")
        changed_train, changed_val, changed_test = prepare("after")
        assert train.patient_to_index == changed_train.patient_to_index
        assert val.patient_to_index == changed_val.patient_to_index
        assert test.patient_to_index == changed_test.patient_to_index

        processor = train.input_processors["features"]
        changed_processor = changed_train.input_processors["features"]
        for name in ("age_median", "age_mean", "age_scale", "sex_vocabulary"):
            assert getattr(processor, name) == getattr(changed_processor, name)
        assert processor.size() == changed_processor.size() == 6
        baseline = {row["patient_id"]: row["features"] for row in train}
        for row in changed_train:
            assert torch.equal(row["features"], baseline[row["patient_id"]])
        batch = next(iter(get_dataloader(changed_val, batch_size=3)))
        assert batch["features"].dtype == torch.float32
        assert batch["features"].shape == (3, 6)
        assert torch.isfinite(batch["features"]).all()
        assert (batch["features"][:, -1] == 1).all()
        assert batch["duration"].shape == batch["event"].shape == (3,)
