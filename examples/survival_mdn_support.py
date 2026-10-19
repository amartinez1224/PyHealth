"""Train and reload a fixed SurvivalMDN baseline on preprocessed SUPPORT data (CPU).

See survival_mdn_support_README.md for provenance, units, and comparison limits.
No archive downloads or reference-code imports are performed.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SurvivalMDN
from pyhealth.trainer import Trainer

SOURCE = {
    "repository": "https://github.com/XintianHan/Survival-MDN",
    "revision": "71c61f649f3c8784fe13b092b38e0cbb3a184dbd",
}
OFFSET = 0.001
UNITS = "released SUPPORT time units + 0.001"
MODEL_CONFIG = {
    "feature_key": "features",
    "duration_key": "duration",
    "event_key": "event",
    "hidden_dims": [64, 64, 64],
    "n_components": 5,
    "activation": "prelu",
    "batch_norm": False,
    "min_scale": 1e-4,
}


def _numeric(array, path, field):
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{path}: {field} must be a finite real numeric array")


def load_support_split(data_dir, split=1):
    """Load three released partitions without sorting or fitting normalization.

    Args:
        data_dir: Directory containing train/valid/test_<split>.npz files.
        split: Integer in 1..10. Missing partitions are errors.

    Returns:
        Dictionary of in-memory processed ``datasets`` and JSON ``metadata``.
        Record IDs identify archive rows, not verified original patient IDs.

    Examples:
        >>> result = load_support_split("data/support", split=1)  # doctest: +SKIP
        >>> result["datasets"]["test"][0]["record_id"]  # doctest: +SKIP
        'support-split1-test-row000000'
    """
    if isinstance(split, bool) or not isinstance(split, int) or not 1 <= split <= 10:
        raise ValueError("split must be an integer in 1..10")
    datasets, partitions = {}, {}
    for phase in ("train", "valid", "test"):
        path = Path(data_dir) / f"{phase}_{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"{path}: required partition missing")
        with np.load(path, allow_pickle=False) as archive:
            for field in ("arr_0", "arr_1"):
                if field not in archive.files:
                    raise ValueError(f"{path}: missing {field}")
            # Do not access unrelated keys: even object arrays remain unopened.
            try:
                features, labels = archive["arr_0"], archive["arr_1"]
            except ValueError as exc:
                raise ValueError(f"{path}: arr_0/arr_1 must be numeric") from exc
            keys = list(archive.files)
        _numeric(features, path, "arr_0")
        _numeric(labels, path, "arr_1")
        if features.ndim != 2 or features.shape[1] != 27:
            raise ValueError(f"{path}: arr_0 must have shape [N, 27]")
        if labels.ndim != 2 or labels.shape[1] != 2:
            raise ValueError(f"{path}: arr_1 must have shape [N, 2]")
        if len(features) == 0 or len(features) != len(labels):
            raise ValueError(f"{path}: arr_0/arr_1 require matching positive N")
        if not np.isin(labels[:, 1], [0, 1]).all():
            raise ValueError(f"{path}: arr_1 event must be exactly 0 or 1")
        if (labels[:, 0] < 0).any():
            raise ValueError(f"{path}: arr_1 duration must be nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            x = features.astype(np.float32)
            # Add before casting, matching the reference's archive precision.
            duration = (labels[:, 0] + OFFSET).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"{path}: arr_0 overflows float32")
        if not np.isfinite(duration).all() or not (duration > 0).all():
            raise ValueError(f"{path}: arr_1 transformed duration invalid in float32")
        samples = []
        for row in range(len(x)):
            identifier = f"support-split{split}-{phase}-row{row:06d}"
            samples.append(
                {
                    "patient_id": identifier,
                    "record_id": identifier,
                    "source_row_index": row,
                    "partition": phase,
                    "released_duration": labels[row, 0].item(),
                    "features": x[row].tolist(),
                    "duration": float(duration[row]),
                    "event": float(labels[row, 1]),
                }
            )
        train = datasets.get("train")
        datasets[phase] = create_sample_dataset(
            samples,
            {"features": "tensor"},
            {"duration": "tensor", "event": "tensor"},
            in_memory=True,
            input_processors=train.input_processors if train is not None else None,
            output_processors=train.output_processors if train is not None else None,
        )
        partitions[phase] = {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": len(x),
            "events": int(labels[:, 1].sum()),
            "censored": int(len(x) - labels[:, 1].sum()),
            "archive_keys": keys,
            "feature_shape": list(features.shape),
            "label_shape": list(labels.shape),
            "feature_dtype": str(features.dtype),
            "label_dtype": str(labels.dtype),
            "all_selected_values_finite": True,
            "released_duration_range": [
                float(labels[:, 0].min()),
                float(labels[:, 0].max()),
            ],
            "feature_mean": features.mean(axis=0, dtype=np.float64).tolist(),
            "feature_std": features.std(axis=0, dtype=np.float64).tolist(),
        }
    return {
        "datasets": datasets,
        "metadata": {
            "split": split,
            "feature_width": 27,
            "duration_offset": OFFSET,
            "time_units": UNITS,
            "original_calendar_unit_mapping": "unknown",
            "identity": "archive rows; original patient IDs unavailable",
            "input_schema": {"features": "tensor"},
            "output_schema": {"duration": "tensor", "event": "tensor"},
            "partitions": partitions,
        },
    }


def _prepare_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{path}: output directory must be empty or nonexistent")
    path.mkdir(parents=True, exist_ok=True)


def _check_compatible(metadata, saved):
    for key in (
        "split",
        "feature_width",
        "duration_offset",
        "time_units",
        "original_calendar_unit_mapping",
        "input_schema",
        "output_schema",
    ):
        if metadata[key] != saved[key]:
            raise ValueError(f"manifest incompatible: {key}")
    for phase in ("train", "valid", "test"):
        if metadata["partitions"][phase] != saved["partitions"][phase]:
            raise ValueError(f"manifest incompatible: {phase} archive/hash")


def _grid(train):
    maximum = max(float(train[i]["duration"]) for i in range(len(train)))
    return np.linspace(0, maximum, 128, dtype=np.float32)


def _predict(model, dataset, grid):
    model.eval()
    curves, indices, identifiers = [], [], []
    with torch.no_grad():
        for batch in get_dataloader(dataset, batch_size=256, shuffle=False):
            curves.append(
                model.predict_survival(
                    torch.from_numpy(grid), features=batch["features"]
                ).numpy()
            )
            indices.extend(batch["source_row_index"])
            identifiers.extend(batch["record_id"])
    survival = np.concatenate(curves)
    expected_ids = [dataset[i]["record_id"] for i in range(len(dataset))]
    if indices != list(range(len(dataset))) or identifiers != expected_ids:
        raise AssertionError("Prediction rows lost, repeated, or reordered")
    if (
        survival.shape != (len(dataset), 128)
        or not np.isfinite(survival).all()
        or not ((survival >= 0) & (survival <= 1)).all()
        or not (np.diff(survival, axis=1) <= 1e-6).all()
        or not (survival[:, 0] == 1).all()
    ):
        raise AssertionError("Invalid survival curves")
    return {
        "survival": survival,
        "times": grid,
        "duration": np.array(
            [dataset[i]["duration"].item() for i in range(len(dataset))],
            dtype=np.float32,
        ),
        "released_duration": np.array(
            [dataset[i]["released_duration"] for i in range(len(dataset))]
        ),
        "event": np.array(
            [dataset[i]["event"].item() for i in range(len(dataset))], dtype=np.float32
        ),
        "source_row_index": np.asarray(indices, dtype=np.int64),
        "record_id": np.asarray(identifiers, dtype=str),
        "partition": np.full(len(dataset), "test"),
    }


def _environment():
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pyhealth", "litdata")
        },
        "pyhealth_git_sha": git("rev-parse", "HEAD"),
        "pyhealth_git_dirty": bool(git("status", "--porcelain")),
        "thread_count": torch.get_num_threads(),
        "interop_thread_count": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _reload(dataset, config, checkpoint):
    model = SurvivalMDN(dataset, **config).cpu().float()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model


def main(argv=None):
    """Train/select/export, or replay an existing baseline without fitting.

    Examples:
        >>> main(["--data-dir", "data/support", "--epochs", "2",
        ...       "--output-dir", "output/support-smoke"])  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predict-only", type=Path)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or not 0 <= args.seed < 2**32:
        parser.error("epochs/patience must be positive; seed must be in [0, 2**32)")
    if not 1 <= args.split <= 10:
        parser.error("split must be in 1..10")
    saved = None
    if args.predict_only:
        if args.output_dir.resolve() == args.predict_only.resolve():
            parser.error("prediction-only output must differ from the source run")
        saved = json.loads((args.predict_only / "manifest.json").read_text())
        args.split, args.seed = saved["data"]["split"], saved["config"]["seed"]
        if saved["device"] != "cpu" or saved["dtype"] != "float32":
            raise ValueError("manifest requires CPU float32")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    loaded = load_support_split(args.data_dir, args.split)
    datasets, metadata = loaded["datasets"], loaded["metadata"]
    if saved:
        _check_compatible(metadata, saved["data"])
    _prepare_output(args.output_dir)
    for phase, audit in metadata["partitions"].items():
        print(
            f"{phase}: {audit['count']} rows, {audit['events']} events, {audit['censored']} censored"
        )
    print(f"Time convention: {metadata['time_units']}; calendar mapping unknown")
    train, valid, test = (datasets[p] for p in ("train", "valid", "test"))
    if saved:
        manifest = saved.copy()
        config = saved["config"]
        grid = np.asarray(saved["prediction_grid"], dtype=np.float32)
        if grid.shape != (128,) or not np.array_equal(grid, _grid(train)):
            raise ValueError("manifest prediction grid incompatible with training data")
        checkpoint = args.predict_only / saved["checkpoint"]
        model = _reload(train, config["model"], checkpoint)
        destination = args.output_dir / "checkpoints"
        destination.mkdir()
        shutil.copyfile(checkpoint, destination / "best.ckpt")
        manifest["execution"] = "prediction-only"
        manifest["replay_environment"] = _environment()
    else:
        config = {
            "model": MODEL_CONFIG.copy(),
            "seed": args.seed,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0,
            "train_batch_size": 256,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "max_grad_norm": 100,
            "monitor": "validation loss",
            "minimize": True,
            "load_best_model_at_last": True,
            "drop_last": False,
            "validation_batch_size": len(valid),
            "loader_seed": args.seed,
        }
        model = SurvivalMDN(train, **config["model"]).cpu().float()
        trainer = Trainer(
            model,
            device="cpu",
            output_path=str(args.output_dir),
            exp_name="checkpoints",
        )
        valid_loader = get_dataloader(valid, batch_size=len(valid), shuffle=False)
        train_eval = get_dataloader(train, batch_size=len(train), shuffle=False)
        initial_train = trainer.evaluate(train_eval)["loss"]
        initial_valid = trainer.evaluate(valid_loader)["loss"]
        print(
            f"Initial training NLL: {initial_train:.9f}; validation NLL: {initial_valid:.9f}"
        )
        started = time.perf_counter()
        trainer.train(
            get_dataloader(train, batch_size=256, shuffle=True, seed=args.seed),
            val_dataloader=valid_loader,
            epochs=args.epochs,
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 0.001},
            weight_decay=0,
            max_grad_norm=100,
            monitor="loss",
            monitor_criterion="min",
            patience=args.patience,
            load_best_model_at_last=True,
        )
        elapsed = time.perf_counter() - started
        selected_loss = trainer.evaluate(valid_loader)["loss"]
        if not np.isfinite([initial_train, initial_valid, selected_loss]).all():
            raise ValueError("Nonfinite initial/restored NLL")
        print(f"Restored validation NLL: {selected_loss:.9f}")
        grid = _grid(train)
        manifest = {
            "format_version": 1,
            "execution": "training",
            "baseline": "PyHealth core baseline on released SUPPORT",
            "source": SOURCE,
            "data": metadata,
            "config": config,
            "device": "cpu",
            "dtype": "float32",
            "environment": _environment(),
            "checkpoint": "checkpoints/best.ckpt",
            "initial_training_nll": initial_train,
            "initial_validation_nll": initial_valid,
            "restored_validation_nll": selected_loss,
            "training_seconds": elapsed,
            "prediction_grid": grid.tolist(),
        }
    predictions = _predict(model, test, grid)
    reloaded = _reload(train, config["model"], args.output_dir / manifest["checkpoint"])
    repeated = _predict(reloaded, test, grid)
    for key in predictions:
        np.testing.assert_array_equal(predictions[key], repeated[key])
    manifest["all_test_rows_checkpoint_reload_identical"] = True
    np.savez(args.output_dir / "test_predictions.npz", **predictions)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(f"All {len(test)} test rows exported; fresh-model reload identical")
    print(f"Artifacts: {args.output_dir.resolve()}")
    return manifest


if __name__ == "__main__":
    main()
