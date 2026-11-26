"""All-cause mortality from study entry in native SUPPORT2 records."""

import math
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from numbers import Real
from typing import Any, ClassVar

import numpy as np
import torch

from pyhealth.data import Patient
from pyhealth.processors import FeatureProcessor

from .base_task import BaseTask

_PREDICTORS = ("age", "sex")
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _category(value: Any) -> str | None:
    return (value.strip() or None) if isinstance(value, str) else None


class Support2FeatureProcessor(FeatureProcessor):
    """Encode study-entry age and sex using statistics fitted on training patients.

    The columns are standardized age, an age-missing indicator, sorted training
    sex categories, and separate missing and unseen sex indicators.

    Examples:
        >>> processor = Support2FeatureProcessor()
        >>> processor.fit([{"features": {"age": "65", "sex": "female"}}], "features")
        >>> processor.process({"age": None, "sex": "male"}).tolist()
        [0.0, 1.0, 0.0, 0.0, 1.0]
    """

    learns_statistics = True
    stores_tensor = True
    dtype = torch.float32

    def __init__(self):
        self.age_median: float | None = None
        self.age_mean = 0.0
        self.age_scale = 1.0
        self.sex_vocabulary: dict[str, int] = {}

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        ages = []
        categories = set()
        for sample in samples:
            features = sample[field]
            age = _finite_float(features.get("age"))
            if age is not None:
                ages.append(age)
            category = _category(features.get("sex"))
            if category is not None:
                categories.add(category)

        self.age_median, self.age_mean, self.age_scale = None, 0.0, 1.0
        if ages:
            # Normalize first so computing moments does not square large ages.
            magnitude = max(abs(min(ages)), abs(max(ages))) or 1.0
            normalized = np.asarray(ages) / magnitude
            self.age_median = float(np.median(normalized)) * magnitude
            self.age_mean = float(normalized.mean()) * magnitude
            self.age_scale = float(normalized.std()) * magnitude or 1.0
        self.sex_vocabulary = {
            category: index
            for index, category in enumerate(sorted(categories), start=2)
        }

    def process(self, value: Mapping[str, Any]) -> torch.Tensor:
        age = _finite_float(value.get("age"))
        result = torch.zeros(self.size(), dtype=self.dtype)
        result[1] = float(age is None)
        if self.age_median is not None:
            imputed = self.age_median if age is None else age
            difference = imputed - self.age_mean
            if math.isfinite(difference):
                standardized = difference / self.age_scale
            else:
                standardized = imputed / self.age_scale - self.age_mean / self.age_scale
            result[0] = max(-_FLOAT32_MAX, min(_FLOAT32_MAX, standardized))
        category = _category(value.get("sex"))
        index = (
            self.size() - 2
            if category is None
            else self.sex_vocabulary.get(category, self.size() - 1)
        )
        result[index] = 1.0
        return result

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def is_token(self) -> bool:
        return False

    def size(self) -> int:
        return 4 + len(self.sex_vocabulary)


class Support2TimeToEvent(BaseTask):
    """Predict all-cause mortality from study entry in SUPPORT2.

    The literal CSV column ``d.time`` is observed follow-up in days. ``death``
    is 1 for an observed death and 0 for right censoring. Use days for
    prediction horizons when the model is trained on these labels directly.
    Age and sex are the only predictors. Patient ID is retained as metadata.

    See the SUPPORT2 variable dictionary (https://archive.ics.uci.edu/dataset/880/support2)
    and study notes (https://hbiostat.org/data/repo/supportdesc) for field meanings
    and assessment timing. Most physiologic assessments were made on study day 3.

    Examples:
        >>> from pyhealth.datasets import PatientSplit, Support2Dataset
        >>> from pyhealth.tasks import Support2TimeToEvent
        >>> dataset = Support2Dataset(root="/path/to/prepared/data", tables=["support2"])
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     Support2TimeToEvent(), split=PatientSplit((0.70, 0.15, 0.15), seed=0)
        ... )
    """

    task_name = "Support2TimeToEvent"
    input_schema: ClassVar = {"features": Support2FeatureProcessor}
    output_schema: ClassVar = {"duration": "tensor", "event": "tensor"}

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        records = patient.get_events(event_type="support2")
        if not records:
            return []
        if len(records) > 1:
            raise ValueError(
                f"Expected one support2 record for patient {patient.patient_id!r}, "
                f"found {len(records)}."
            )

        record = records[0].attr_dict
        duration = _finite_float(record.get("d.time"))
        if duration is None or duration <= 0:
            return []
        with np.errstate(over="ignore"):
            duration32 = np.float32(duration)
        if not np.isfinite(duration32) or duration32 <= 0:
            return []

        event = record.get("death")
        if isinstance(event, str):
            # Preserve exact binary membership even for strings near 0 or 1.
            try:
                event = Decimal(event)
            except InvalidOperation:
                return []
        if isinstance(event, Decimal) and not event.is_finite():
            return []
        if not isinstance(event, (Real, Decimal)) or event not in (0, 1):
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "features": {field: record.get(field) for field in _PREDICTORS},
                "duration": float(duration32),
                "event": float(event),
            }
        ]
