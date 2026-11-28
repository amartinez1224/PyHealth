"""Data preparation and day-based prediction for the native SUPPORT2 notebook."""

import csv
import hashlib
import io
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from pyhealth.datasets import support2
from pyhealth.datasets.configs import load_yaml_config

ARCHIVE_SHA256 = "8ed43980742a18e1847a8dfc5530bc4b30564ad9e4ad1b1b50bbc5d29d8c86fe"
CSV_NAME = "support2.csv"
DAYS_PER_YEAR = 365
EXPORT_DAYS = (0, 30, 90, 180, 365)
RELEASE_LABELS = {"1": (2029, 0), "2": (4, 1), "3": (47, 1)}


def support2_header():
    """Use the same required columns as the native loader."""
    config_path = Path(support2.__file__).parent / "configs" / "support2.yaml"
    table = load_yaml_config(config_path).tables["support2"]
    return [table.patient_id, *table.attributes]


def _read_csv(stream):
    reader = csv.reader(stream, strict=True)
    header = next(reader, [])
    columns = set(header)
    required = set(support2_header())
    raw = header[:1] == ["age"] and len(header) == 47 and columns == required - {"sno"}
    if raw:
        header = ["sno", *header]
    elif len(columns) != len(header) or "" in columns or not required <= columns:
        raise ValueError(
            "Expected the HBiostat source header or a prepared SUPPORT2 header."
        )

    ids = set()
    rows = []
    id_index = header.index("sno")
    for row in reader:
        if len(row) != len(header):
            raise ValueError(
                f"SUPPORT2 CSV line {reader.line_num}: expected {len(header)} fields, "
                f"found {len(row)}."
            )
        patient_id = row[id_index]
        if not patient_id.strip() or patient_id.strip() in ids:
            raise ValueError(
                f"SUPPORT2 CSV line {reader.line_num}: blank or duplicate ID."
            )
        ids.add(patient_id.strip())
        rows.append(row)
    if not rows:
        raise ValueError("The SUPPORT2 CSV has no patient records.")
    return raw, header, rows


def _verify_release(header, rows):
    id_index, duration_index, event_index = (
        header.index(field) for field in ("sno", "d.time", "death")
    )
    labels = {
        row[id_index]: (int(row[duration_index]), int(row[event_index])) for row in rows
    }
    if len(rows) != 9105 or [row[id_index] for row in rows[:3]] != list(RELEASE_LABELS):
        raise ValueError(
            "Unexpected HBiostat SUPPORT2 release size or first three IDs."
        )
    if any(labels[pid] != pair for pid, pair in RELEASE_LABELS.items()):
        raise ValueError(
            "The first three native labels differ from the SUPPORT2 fixture."
        )
    durations = [duration for duration, _ in labels.values()]
    if (min(durations), max(durations)) != (3, 2029):
        raise ValueError("Expected HBiostat SUPPORT2 follow-up of 3–2,029 days.")


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            writer = csv.writer(temporary)
            writer.writerow(header)
            writer.writerows(rows)
            temporary.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def prepare_support2(source, prepared_root):
    """Validate a prepared CSV or make a separate loader-ready copy of a raw CSV/ZIP.

    Return the loader-ready CSV path and the IDs present in the source rows.
    The archive is pinned to the inspected HBiostat release.
    """
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with source.open("rb") as binary:
            if hashlib.file_digest(binary, "sha256").hexdigest() != ARCHIVE_SHA256:
                raise ValueError(
                    "Unexpected SHA-256 for the HBiostat SUPPORT2 archive."
                )
            binary.seek(0)
            with zipfile.ZipFile(binary) as archive, archive.open(CSV_NAME) as member:
                raw, header, rows = _read_csv(
                    io.TextIOWrapper(member, newline="", encoding="utf-8")
                )
        if not raw:
            raise ValueError("Expected an unnamed ID column in the HBiostat archive.")
        _verify_release(header, rows)
    else:
        with source.open(encoding="utf-8-sig", newline="") as stream:
            raw, header, rows = _read_csv(stream)

    path = source
    if raw:
        path = Path(prepared_root) / CSV_NAME
        if path.resolve() == source.resolve():
            raise ValueError("Choose a separate output directory for the prepared CSV.")
        _write_csv(path, header, rows)
    elif path.name != CSV_NAME:
        raise ValueError(f"Name the prepared CSV {CSV_NAME} for the native loader.")
    id_index = header.index("sno")
    return path, {row[id_index] for row in rows}


def write_synthetic_support2(path, *, seed, n_patients):
    """Write native-format synthetic data with median event time of 90 days."""
    rng = np.random.default_rng(seed)
    event_times = rng.exponential(90 / np.log(2), n_patients)
    censor_times = rng.exponential(365, n_patients)
    durations = np.maximum(np.minimum(event_times, censor_times), 0.01)
    ages = rng.uniform(30, 85, n_patients)
    sexes = rng.choice(("female", "male"), n_patients)
    header = support2_header()
    rows = []
    for i in range(n_patients):
        fields = {
            "sno": f"synthetic-{i + 1}",
            "age": ages[i],
            "sex": sexes[i],
            "d.time": durations[i],
            "death": int(event_times[i] <= censor_times[i]),
        }
        rows.append([fields.get(column, "") for column in header])
    _write_csv(Path(path), header, rows)


class ModelTimeLoader:
    """Present duration in years to Trainer while preserving native sample labels."""

    def __init__(self, loader):
        self.loader = loader
        self.batch_size = loader.batch_size

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield {**batch, "duration": batch["duration"] / DAYS_PER_YEAR}


def predict_survival_days(model, days, features):
    """Return survival probabilities at day coordinates for a model trained in years."""
    return model.predict_survival(days / DAYS_PER_YEAR, features=features)


@torch.no_grad()
def export_survival(model, loader, destination):
    """Write one row per test patient, keeping IDs aligned with batch predictions."""
    model.eval()
    days = torch.tensor(EXPORT_DAYS, dtype=torch.float32)
    ids, predictions = [], []
    for batch in loader:
        curve_batch = predict_survival_days(model, days, batch["features"])
        ids.extend(batch["patient_id"])
        predictions.append(curve_batch.cpu())
    curves = torch.cat(predictions).numpy()
    if len(set(ids)) != len(ids):
        raise ValueError("Expected one prediction per held-out patient.")
    _write_csv(
        Path(destination),
        ["patient_id", *(f"survival_day_{day}" for day in EXPORT_DAYS)],
        (
            [patient_id, *curve.tolist()]
            for patient_id, curve in zip(ids, curves, strict=True)
        ),
    )
    return ids, curves
