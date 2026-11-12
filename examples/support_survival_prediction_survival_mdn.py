r"""Ablation of patient-specific mixture weights in SurvivalMDN on SUPPORT.

Research question:
    The SurvivalMDN predicts a Gaussian mixture whose means, scales *and*
    mixture weights all depend on the patient. The means and scales already
    carry patient-specific information. How much do patient-specific mixture
    weights add on top of that?

Hypothesis:
    Patient-specific weights give a lower test integrated Brier score (IBS)
    than a single shared weight vector. Uniform weights (1/K for everyone)
    are a reference point showing what learning any weights at all buys.

Experimental setup:
    * Data: released SUPPORT split 1 (train/valid/test), or ``--synthetic``
      random data of the same shape for a code check without the real files.
    * Model: ``SurvivalMDN`` with ``n_components=5``. Means and scales are
      patient-specific in every run; only ``weight_mode`` changes:
      ``uniform`` (fixed 1/K), ``shared`` (one learned vector), ``patient``
      (predicted from features, the default model behavior).
    * Grid: 3 weight modes x 2 hidden sizes ([32, 32, 32] and [64, 64, 64])
      x 3 seeds (0, 1, 2) = 18 runs.
    * Held constant: Adam, learning rate 0.001, batch size 256, gradient
      clipping at 100, maximum epochs, patience, and early stopping on
      validation NLL with the best-epoch weights restored. The test set is
      used only for the final evaluation of each run.
    * Metrics on the test set: IBS (primary, lower is better), IPCW C-index
      and IBLL (higher is better) at censoring cutoff 1e-8, plus the same
      three metrics at cutoffs 0.2 and 0.4 as a robustness check.
    * Reported: mean and std over seeds for every (mode, hidden size) config,
      and *matched* differences (patient minus shared, patient minus uniform)
      computed within each (seed, hidden size) pair, which cancels
      seed-to-seed variation shared by the compared runs.

Results:
    PLACEHOLDER: not yet run on the real SUPPORT data. Replace this section
    with the summary table and matched differences printed by

        python examples/support_survival_prediction_survival_mdn.py \
            --data-dir <support-dir> --output-dir output/weight-ablation

    No numbers are reported here until that run is done.

Usage:
    python examples/support_survival_prediction_survival_mdn.py --synthetic \
        --epochs 2 --output-dir output/weight-ablation-smoke
"""

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import SampleDataset, create_sample_dataset, get_dataloader
from pyhealth.models import SurvivalMDN
from pyhealth.trainer import Trainer

# Reuse the SUPPORT loader and metrics from the baseline script.
from survival_mdn_support import (
    MODEL_CONFIG,
    OFFSET,
    _evaluate_metrics,
    load_support_split,
)

WEIGHT_MODES = ("uniform", "shared", "patient")
HIDDEN_SIZES = (32, 64)
N_LAYERS = 3
FEATURE_WIDTH = 27
# Training settings used for every run.
LEARNING_RATE = 0.001
BATCH_SIZE = 256
MAX_GRAD_NORM = 100
# Censoring cutoffs for the metrics. The first one is the main result.
CUTOFFS = (1e-8, 0.2, 0.4)
METRICS = ("ibs", "c_index", "ibll")
# Pairs of modes to compare (patient minus the other mode).
DIFFERENCES = (("patient", "shared"), ("patient", "uniform"))


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run is repeatable.

    Args:
        seed: Non-negative integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_datasets(
    sizes: Sequence[int] = (400, 150, 250), seed: int = 0
) -> dict[str, SampleDataset]:
    """Create fake SUPPORT-shaped train/valid/test datasets.

    Event times are exponential with a rate that depends on two features, and
    censoring times are independent exponentials, so both events and censored
    patients appear in every partition. The data only exercises the code; it
    is not meant to say anything about the models.

    Args:
        sizes: Number of rows in the train, valid and test partitions.
        seed: Seed of the random generator.

    Returns:
        Dictionary mapping ``"train"``, ``"valid"`` and ``"test"`` to datasets
        with the same fields as :func:`load_support_split`.
    """
    rng = np.random.default_rng(seed)
    datasets: dict[str, SampleDataset] = {}
    for phase, size in zip(("train", "valid", "test"), sizes):
        features = rng.normal(size=(size, FEATURE_WIDTH)).astype(np.float32)
        rate = np.exp(0.5 * features[:, 0] - 0.3 * features[:, 1])
        event_time = rng.exponential(1.0 / rate)
        censor_time = rng.exponential(2.0, size=size)
        released = np.minimum(event_time, censor_time)
        event = (event_time <= censor_time).astype(np.float32)
        samples = []
        for row in range(size):
            identifier = f"synthetic-{phase}-row{row:06d}"
            samples.append(
                {
                    "patient_id": identifier,
                    "record_id": identifier,
                    "source_row_index": row,
                    "partition": phase,
                    "released_duration": float(released[row]),
                    "features": features[row].tolist(),
                    "duration": float(np.float32(released[row] + OFFSET)),
                    "event": float(event[row]),
                }
            )
        train = datasets.get("train")
        datasets[phase] = create_sample_dataset(
            samples,
            {"features": "tensor"},
            {"duration": "tensor", "event": "tensor"},
            in_memory=True,
            input_processors=train.input_processors if train else None,
            output_processors=train.output_processors if train else None,
        )
    return datasets


def run_once(
    datasets: dict[str, SampleDataset],
    weight_mode: str,
    hidden_size: int,
    seed: int,
    epochs: int,
    patience: int,
) -> dict[str, float]:
    """Train one model, select it on validation NLL, evaluate it on test.

    Args:
        datasets: Train, valid and test datasets.
        weight_mode: ``"uniform"``, ``"shared"`` or ``"patient"``.
        hidden_size: Width of each of the three hidden layers.
        seed: Seed for initialization and training batch order.
        epochs: Maximum number of epochs.
        patience: Epochs without validation improvement before stopping.

    Returns:
        One result row: identifiers, parameter count, training time, best
        epoch (1-based), epochs run, and test metrics at every cutoff. Metric
        columns for the primary cutoff are unsuffixed (``ibs``); the others
        carry a ``_cut<value>`` suffix.
    """
    train, valid, test = (datasets[phase] for phase in ("train", "valid", "test"))
    set_seed(seed)
    config = {
        **MODEL_CONFIG,
        "hidden_dims": [hidden_size] * N_LAYERS,
        "weight_mode": weight_mode,
    }
    model = SurvivalMDN(train, **config).cpu().float()
    n_params = sum(p.numel() for p in model.parameters())
    trainer = Trainer(model, device="cpu", enable_logging=False)
    valid_loader = get_dataloader(valid, batch_size=len(valid), shuffle=False)
    # Save the validation loss after each epoch so we can find the best
    # epoch. The trainer does not return it.
    validation_nll: list[float] = []
    evaluate = trainer.evaluate

    def recording_evaluate(dataloader) -> dict[str, float]:
        scores = evaluate(dataloader)
        validation_nll.append(scores["loss"])
        return scores

    trainer.evaluate = recording_evaluate
    started = time.perf_counter()
    trainer.train(
        get_dataloader(train, batch_size=BATCH_SIZE, shuffle=True, seed=seed),
        val_dataloader=valid_loader,
        epochs=epochs,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": LEARNING_RATE},
        weight_decay=0,
        max_grad_norm=MAX_GRAD_NORM,
        monitor="loss",
        monitor_criterion="min",
        patience=patience,
        load_best_model_at_last=True,
    )
    train_seconds = time.perf_counter() - started
    trainer.evaluate = evaluate
    # Use the test set only once, after loading the best model.
    _, report = _evaluate_metrics(model, test)
    row: dict[str, float] = {
        "weight_mode": weight_mode,
        "hidden_size": hidden_size,
        "seed": seed,
        "n_params": n_params,
        "train_seconds": train_seconds,
        "best_epoch": int(np.argmin(validation_nll)) + 1,
        "epochs_run": len(validation_nll),
    }
    for cutoff, result in zip(CUTOFFS, report["cutoffs"]):
        suffix = "" if cutoff == CUTOFFS[0] else f"_cut{cutoff}"
        for metric in METRICS:
            row[f"{metric}{suffix}"] = result["scores"][metric]
    return row


def metric_columns(results: pd.DataFrame) -> list[str]:
    """List the metric columns of a results table.

    Args:
        results: Per-run results from :func:`run_once`.

    Returns:
        Names of all metric columns, primary cutoff first.
    """
    return [c for c in results.columns if c.split("_cut")[0] in METRICS]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std across seeds for each configuration.

    Args:
        results: Per-run results.

    Returns:
        Table indexed by (weight_mode, hidden_size) with ``<column>_mean`` and
        ``<column>_std`` for metrics, parameter count, time and best epoch.
    """
    columns = metric_columns(results) + ["n_params", "train_seconds", "best_epoch"]
    grouped = results.groupby(["weight_mode", "hidden_size"])[columns]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    return summary


def matched_differences(results: pd.DataFrame) -> pd.DataFrame:
    """Compute patient-minus-baseline differences within matched runs.

    Runs are matched on (seed, hidden size): the same initialization seed and
    network width, differing only in weight mode. IBS differences below zero
    favor patient-specific weights; for C-index and IBLL, above zero does.

    Args:
        results: Per-run results containing the needed weight modes.

    Returns:
        Table indexed by (comparison, hidden_size) with the mean and std over
        seeds of each metric difference. Comparisons whose modes are missing
        from ``results`` are skipped.
    """
    columns = metric_columns(results)
    wide = results.set_index(["hidden_size", "seed", "weight_mode"])[columns]
    rows = []
    for left, right in DIFFERENCES:
        modes = wide.index.get_level_values("weight_mode")
        if left not in modes or right not in modes:
            continue
        diff = wide.xs(left, level="weight_mode") - wide.xs(
            right, level="weight_mode"
        )
        for hidden_size, group in diff.groupby(level="hidden_size"):
            row = {"comparison": f"{left} - {right}", "hidden_size": hidden_size}
            row["n_pairs"] = len(group)
            for column in columns:
                row[f"{column}_mean"] = group[column].mean()
                row[f"{column}_std"] = group[column].std()
            rows.append(row)
    return pd.DataFrame(rows).set_index(["comparison", "hidden_size"])


def format_table(table: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render the chosen columns of a summary table as aligned text.

    Args:
        table: Table from :func:`summarize` or :func:`matched_differences`.
        columns: Column names (without ``_mean``/``_std``) to show as
            ``mean ± std``.

    Returns:
        Multi-line string.
    """
    shown = pd.DataFrame(index=table.index)
    for column in columns:
        shown[column] = [
            f"{mean:.4f} ± {std:.4f}"
            for mean, std in zip(table[f"{column}_mean"], table[f"{column}_std"])
        ]
    return shown.to_string()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and validate command line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, help="Released SUPPORT files")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use small fake data of the same shape instead of --data-dir",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weight-modes", nargs="+", choices=WEIGHT_MODES, default=WEIGHT_MODES
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=HIDDEN_SIZES,
        help="Width of each of the three hidden layers",
    )
    args = parser.parse_args(argv)
    if not args.synthetic and args.data_dir is None:
        parser.error("provide --data-dir or --synthetic")
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    if any(seed < 0 for seed in args.seeds) or any(h < 1 for h in args.hidden_sizes):
        parser.error("seeds must be non-negative and hidden sizes positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Run the weight-mode ablation and write results to ``--output-dir``.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Per-run results table, also saved as ``runs.csv``.
    """
    args = parse_args(argv)
    # Same settings as the baseline script, so results are repeatable.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    logging.getLogger("pyhealth.trainer").setLevel(logging.WARNING)
    if args.synthetic:
        datasets = make_synthetic_datasets()
    else:
        datasets = load_support_split(args.data_dir, split=1)["datasets"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        (mode, size, seed)
        for size in args.hidden_sizes
        for seed in args.seeds
        for mode in args.weight_modes
    ]
    rows = []
    for number, (mode, size, seed) in enumerate(configs, start=1):
        row = run_once(datasets, mode, size, seed, args.epochs, args.patience)
        rows.append(row)
        print(
            f"[{number}/{len(configs)}] {mode:8s} hidden={size:3d} seed={seed} "
            f"IBS={row['ibs']:.4f} C={row['c_index']:.4f} IBLL={row['ibll']:.4f} "
            f"best_epoch={row['best_epoch']} time={row['train_seconds']:.1f}s"
        )
        # Save after each run so results are kept if the job stops.
        pd.DataFrame(rows).to_csv(args.output_dir / "runs.csv", index=False)
    results = pd.DataFrame(rows)
    summary = summarize(results)
    summary.to_csv(args.output_dir / "summary.csv")
    print("\nMean ± std across seeds (test set, cutoff 1e-8):")
    print(format_table(summary, [*METRICS, "n_params", "best_epoch"]))
    differences = matched_differences(results)
    if len(differences):
        differences.to_csv(args.output_dir / "matched_differences.csv")
        print("\nMatched differences (same seed and hidden size):")
        print("IBS < 0 favors the first mode; C-index and IBLL > 0 do.")
        print(format_table(differences, METRICS))
    print(f"\nResults written to {args.output_dir.resolve()}")
    return results


if __name__ == "__main__":
    main()
