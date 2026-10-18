"""Train SurvivalMDN on synthetic right-censored observations without downloads.

Run from the repository root::

    python examples/survival_mdn_synthetic.py --epochs 2 --device cpu

All times use one arbitrary synthetic time unit. Features determine latent
Gaussian mixture weights and means; event times are softplus-transformed latent
draws. Independent exponential censoring gives observed durations and binary
event indicators.
The model evaluates observed and censored observations in log space, centering
component log densities and log survival probabilities before adding weights
to preserve mixture gradients. This example uses moderate times; deep-tail
gradient accuracy is checked separately by the model's regression tests.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SurvivalMDN
from pyhealth.trainer import Trainer


def _make_samples(seed, count=320, width=4):
    """Generate fixed-width features and independently censored event times."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(count, width))
    probability = 1.0 / (1.0 + np.exp(-features[:, 0]))
    component = rng.random(count) < probability
    means = np.where(component, 1.2, -0.8) + 0.3 * features[:, 1]
    latent = means + 0.5 * rng.normal(size=count)
    event_times = np.logaddexp(0.0, latent)
    # Uniform draws strictly inside (0, 1) make censoring times positive.
    uniform = np.clip(rng.random(count), np.finfo(float).eps, 1.0 - 1e-16)
    censoring_times = -2.5 * np.log(uniform)
    durations = np.minimum(event_times, censoring_times)
    events = event_times <= censoring_times
    samples = [
        {
            "patient_id": f"synthetic-{index}",
            "features": features[index].tolist(),
            "duration": float(durations[index]),
            "event": float(events[index]),
        }
        for index in range(count)
    ]
    # The collator pads ragged vectors: validate source widths before processing.
    if any(len(sample["features"]) != width for sample in samples):
        raise ValueError("Every raw feature vector must have the same width")
    return samples


def main(argv=None):
    """Train, export curves, and verify a configured checkpoint round trip.

    Args:
        argv: Optional command-line argument list, primarily for smoke tests.

    Examples:
        >>> main(["--epochs", "2", "--output-dir", "/tmp/mdn-demo"])  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/survival_mdn_synthetic")
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples = _make_samples(args.seed)
    input_schema = {"features": "tensor"}
    output_schema = {"duration": "tensor", "event": "tensor"}
    train_dataset = create_sample_dataset(
        samples[:256], input_schema, output_schema, in_memory=True
    )
    # Reuse fitted training processors, as with held-out real observations.
    validation_dataset = create_sample_dataset(
        samples[256:],
        input_schema,
        output_schema,
        input_processors=train_dataset.input_processors,
        output_processors=train_dataset.output_processors,
        in_memory=True,
    )
    train_loader = get_dataloader(
        train_dataset, batch_size=32, shuffle=True, seed=args.seed
    )
    train_evaluation_loader = get_dataloader(train_dataset, batch_size=32)
    validation_loader = get_dataloader(validation_dataset, batch_size=32)
    config = {
        "feature_key": "features",
        "duration_key": "duration",
        "event_key": "event",
        "hidden_dims": [32, 32],
        "n_components": 3,
        "activation": "prelu",
        "batch_norm": False,
        "min_scale": 1e-4,
    }
    model = SurvivalMDN(train_dataset, **config)
    trainer = Trainer(model=model, device=args.device, enable_logging=False)
    initial_loss = trainer.evaluate(train_evaluation_loader)["loss"]
    print(f"Initial training loss: {initial_loss:.6f}")
    print(
        "Censored fraction: "
        f"training={1 - np.mean([s['event'] for s in samples[:256]]):.3f}, "
        f"validation={1 - np.mean([s['event'] for s in samples[256:]]):.3f}"
    )
    # Trainer averages batch losses. Equal-sized validation batches ensure this
    # agrees with a per-patient mean; general uneven-batch aggregation is separate.
    trainer.train(
        train_loader,
        val_dataloader=validation_loader,
        epochs=args.epochs,
        monitor="loss",
        monitor_criterion="min",
        patience=3,
        load_best_model_at_last=True,
    )
    training_loss = trainer.evaluate(train_evaluation_loader)["loss"]
    validation_loss = trainer.evaluate(validation_loader)["loss"]
    print(f"Restored-model training loss: {training_loss:.6f}")
    print(f"Restored-model validation loss: {validation_loss:.6f}")

    features = next(iter(validation_loader))["features"][:3]
    grid = torch.linspace(0.0, 6.0, 61, device=args.device)
    model.eval()
    with torch.no_grad():
        # Prediction needs features only, with no duration/event labels.
        curves = model.predict_survival(grid, features=features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "state_dict.pt"
    config_path = args.output_dir / "config.json"
    torch.save(model.state_dict(), weights_path)
    metadata = {
        "model_config": config,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "feature_width": 4,
        "seed": args.seed,
        "time_unit": "arbitrary synthetic unit",
        "initial_training_loss": initial_loss,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
    }
    config_path.write_text(json.dumps(metadata, indent=2) + "\n")
    # A state dictionary contains neither constructor settings nor processors.
    # Reconstruct using the saved settings and a compatible processed dataset.
    saved_config = json.loads(config_path.read_text())["model_config"]
    reloaded = SurvivalMDN(train_dataset, **saved_config).to(args.device)
    reloaded.load_state_dict(
        torch.load(weights_path, map_location=args.device, weights_only=True)
    )
    reloaded.eval()
    with torch.no_grad():
        reloaded_curves = reloaded.predict_survival(grid, features=features)
    torch.testing.assert_close(curves, reloaded_curves, atol=0.0, rtol=0.0)
    np.savez(
        args.output_dir / "survival_curves.npz",
        times=grid.cpu().numpy(),
        survival=curves.cpu().numpy(),
        features=features.cpu().numpy(),
    )
    print(f"Prediction grid (every tenth point): {grid[::10].cpu().tolist()}")
    print(f"First survival curve: {curves[0, ::10].cpu().tolist()}")
    print("Checkpoint reload: survival curves match exactly")
    print(f"Artifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
