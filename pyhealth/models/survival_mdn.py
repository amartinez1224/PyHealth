"""Gaussian mixture survival model with a softplus time transformation."""

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..processors import TensorProcessor
from .base_model import BaseModel


def _transformed_terms(times, log_weights, means, scales):
    """Align paired [B] or grid [B, T] times with component parameters."""
    if times.ndim == 1:
        times = times[:, None]
    elif times.ndim == 2:
        times = times[:, :, None]
        log_weights = log_weights[:, None, :]
        means = means[:, None, :]
        scales = scales[:, None, :]
    else:
        raise ValueError("Internal times must have rank one or two.")
    log_inverse_derivative = torch.log(-torch.expm1(-times))
    latent = times + log_inverse_derivative
    return latent, log_inverse_derivative, log_weights, means, scales


def _log_density(times, log_weights, means, scales):
    """Transformed mixture log density for strictly positive times."""
    latent, log_inverse, weights, means, scales = _transformed_terms(
        times, log_weights, means, scales
    )
    standardized = (latent - means) / scales
    normal = -0.5 * standardized.square() - scales.log() - 0.5 * math.log(2 * math.pi)
    return torch.logsumexp(weights + normal, dim=-1) - log_inverse[..., 0]


def _log_survival(times, log_weights, means, scales):
    """Transformed mixture log survival for strictly positive times."""
    latent, _, weights, means, scales = _transformed_terms(
        times, log_weights, means, scales
    )
    standardized = (latent - means) / scales
    return torch.logsumexp(weights + torch.special.log_ndtr(-standardized), dim=-1)


class SurvivalMDN(BaseModel):
    """Patient-specific Gaussian mixture for right-censored positive times.

    Models ``T = softplus(Z)`` with a feature-dependent Gaussian mixture for
    ``Z``. Features must be fixed-width finite floating vectors processed by a
    fitted floating :class:`~pyhealth.processors.TensorProcessor`. The collator
    can pad ragged vectors, so ensure equal raw widths before processing.
    Scalar duration/event processors must expose a single ``value`` channel.
    Event 1 means observed; 0 means right censored. Durations must be positive,
    finite, and use the same units as prediction times. No scaling is implicit.

    Args:
        dataset: Nonempty processed dataset with exactly the selected schemas.
        feature_key: Dense vector input field. Selected field names must be
            distinct nonempty strings other than PyTorch's reserved ``self``.
        duration_key: Positive duration output field.
        event_key: Binary event output field.
        hidden_dims: Nonempty sequence of positive hidden widths.
        n_components: Positive number of Gaussian components.
        activation: ``"prelu"`` or ``"relu"``.
        batch_norm: Normalize hidden layers before activation. Training batches
            then need at least two patients; singleton evaluation is supported.
        min_scale: Finite positive floor added to softplus scale outputs.

    Notes:
        Mean head weights start at zero and biases at evenly spaced centers
        in [-3, 3] (zero for one component). Other heads use PyTorch defaults.
        Stable log operations, the scale floor, optional BatchNorm, and the
        absence of reference output clamps are deliberate reference differences.
        CPU float32/float64 and CUDA are supported; mixed precision/autocast is
        outside this contract. Rescale times yourself for extreme magnitudes.

        ``mode=None`` lets Trainer train/evaluate the custom scalar loss. Its
        generic ``inference()`` expects classification outputs; use
        :meth:`predict_survival` instead. Checkpoints need constructor settings
        and a compatible processed dataset alongside the state dictionary.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": "p0", "features": [0.2, -1.0],
        ...      "duration": 1.5, "event": 1.0},
        ...     {"patient_id": "p1", "features": [0.8, 0.3],
        ...      "duration": 2.0, "event": 0.0},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples, input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ...     in_memory=True)
        >>> model = SurvivalMDN(dataset)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
        >>> output = model(**batch)
        >>> output["loss"].backward()
        >>> model(features=batch["features"])["means"].shape
        torch.Size([2, 5])
        >>> _ = model.eval()
        >>> with torch.no_grad():
        ...     curves = model.predict_survival(torch.tensor([0., 1., 2.]), **batch)
        >>> curves.shape
        torch.Size([2, 3])
    """

    def __init__(
        self,
        dataset,
        feature_key="features",
        duration_key="duration",
        event_key="event",
        hidden_dims=(64, 64, 64),
        n_components=5,
        activation="prelu",
        batch_norm=False,
        min_scale=1e-4,
    ):
        keys = (feature_key, duration_key, event_key)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Field names must be nonempty strings.")
        if "self" in keys:
            raise ValueError("Field name 'self' is reserved by PyTorch keyword calls.")
        if len(set(keys)) != 3:
            raise ValueError(
                "Feature, duration and event field names must be distinct."
            )
        if (
            not isinstance(hidden_dims, Sequence)
            or not hidden_dims
            or any(
                isinstance(width, bool) or not isinstance(width, int) or width <= 0
                for width in hidden_dims
            )
        ):
            raise ValueError(
                "hidden_dims must be a nonempty sequence of positive integers."
            )
        if (
            isinstance(n_components, bool)
            or not isinstance(n_components, int)
            or n_components <= 0
        ):
            raise ValueError("n_components must be a positive integer.")
        if activation not in ("prelu", "relu"):
            raise ValueError("activation must be 'prelu' or 'relu'.")
        if (
            isinstance(min_scale, bool)
            or not isinstance(min_scale, (int, float))
            or not math.isfinite(min_scale)
            or min_scale <= 0
        ):
            raise ValueError("min_scale must be finite and positive.")
        if dataset is None or len(dataset) == 0:
            raise ValueError("A nonempty processed dataset is required.")
        try:
            if set(dataset.input_schema) != {feature_key} or set(
                dataset.output_schema
            ) != {duration_key, event_key}:
                raise ValueError(
                    "Dataset schemas must contain exactly the selected fields."
                )
            processor = dataset.input_processors[feature_key]
            if not isinstance(processor, TensorProcessor) or not (
                processor.dtype.is_floating_point
                and processor.schema() == ("value",)
                and processor.dim() == (1,)
                and isinstance(processor.size(), int)
                and processor.size() > 0
            ):
                raise ValueError(
                    "Features require a fitted floating vector TensorProcessor."
                )
            width = processor.size()
            for key in (duration_key, event_key):
                label = dataset.output_processors[key]
                if (
                    label.schema() != ("value",)
                    or label.size() != 1
                    or label.dim() not in ((0,), (1,))
                ):
                    raise ValueError("Labels require fitted scalar value processors.")
        except (AttributeError, KeyError, NotImplementedError, RuntimeError) as error:
            raise ValueError(
                "Dataset requires fitted feature and label metadata."
            ) from error
        super().__init__(dataset)
        self.mode = None
        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.input_width = width
        self.batch_norm = batch_norm
        self.min_scale = min_scale
        layers = []
        for hidden_width in hidden_dims:
            layers.append(nn.Linear(width, hidden_width, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_width))
            layers.append(nn.PReLU() if activation == "prelu" else nn.ReLU())
            width = hidden_width
        self.hidden = nn.Sequential(*layers)
        self.weight_head = nn.Linear(width, n_components)
        self.mean_head = nn.Linear(width, n_components)
        self.scale_head = nn.Linear(width, n_components)
        nn.init.zeros_(self.mean_head.weight)
        with torch.no_grad():
            self.mean_head.bias.copy_(
                torch.linspace(-3, 3, n_components)
                if n_components > 1
                else torch.zeros(1)
            )

    def _convert(self, value, name, *, floating=False):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a raw tensor, not a tuple/list.")
        if floating and not value.is_floating_point():
            raise TypeError(f"{name} must be a floating tensor.")
        if value.is_complex() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite real values.")
        weight = self.weight_head.weight
        if weight.dtype not in (torch.float32, torch.float64):
            raise TypeError("Model dtype must be float32 or float64.")
        value = value.to(device=weight.device, dtype=weight.dtype)
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} exceeds the model dtype numerical range.")
        return value

    def _parameters_from_features(self, kwargs):
        if self.feature_key not in kwargs:
            raise ValueError(f"Missing feature field {self.feature_key!r}.")
        features = self._convert(kwargs[self.feature_key], "features", floating=True)
        if features.ndim != 2 or features.shape[1] != self.input_width:
            raise ValueError(f"features must have shape [B, {self.input_width}].")
        if features.shape[0] == 0:
            raise ValueError("Empty batches are unsupported.")
        if self.batch_norm and self.training and features.shape[0] == 1:
            raise ValueError(
                "BatchNorm training requires at least two patients per batch."
            )
        hidden = self.hidden(features)
        outputs = {
            "log_weights": F.log_softmax(self.weight_head(hidden), dim=-1),
            "means": self.mean_head(hidden),
            "scales": F.softplus(self.scale_head(hidden)) + self.min_scale,
        }
        if (
            any(not torch.isfinite(value).all() for value in outputs.values())
            or (outputs["scales"] <= 0).any()
        ):
            raise ValueError(
                "Predicted parameters exceed the supported numerical range."
            )
        return outputs

    def _label(self, value, name, batch_size):
        value = self._convert(value, name)
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape [B] or [B, 1] matching features.")
        return value

    def forward(self, **kwargs):
        """Return [B, K] log_weights, means, scales and optional scalar loss.

        Both duration/event labels or neither must be supplied. Unrelated
        metadata is ignored. No classification-style outputs are returned.

        Examples:
            >>> output = model(**batch)  # doctest: +SKIP
            >>> output["loss"].backward()  # doctest: +SKIP
        """
        outputs = self._parameters_from_features(kwargs)
        has_duration = self.duration_key in kwargs
        has_event = self.event_key in kwargs
        if has_duration != has_event:
            raise ValueError(
                "Both duration and event labels must be supplied together."
            )
        if not has_duration:
            return outputs
        raw_event = kwargs[self.event_key]
        if not isinstance(raw_event, torch.Tensor):
            raise TypeError("event must be a raw tensor.")
        if raw_event.is_complex() or not ((raw_event == 0) | (raw_event == 1)).all():
            raise ValueError("event values must be exactly 0 or 1.")
        batch_size = outputs["means"].shape[0]
        duration = self._label(kwargs[self.duration_key], "duration", batch_size)
        event = self._label(raw_event, "event", batch_size)
        if (duration <= 0).any():
            raise ValueError(
                "duration must be strictly positive after dtype conversion."
            )
        contributions = []
        for selected, helper in (
            (event == 1, _log_density),
            (event == 0, _log_survival),
        ):
            if selected.any():
                parameters = [
                    outputs[key][selected] for key in ("log_weights", "means", "scales")
                ]
                likelihood = helper(duration[selected], *parameters)
                if not torch.isfinite(likelihood).all():
                    raise ValueError(
                        "Selected log likelihood exceeds the numerical range."
                    )
                contributions.append((likelihood / batch_size).sum())
        # Divide before summing to avoid overflow in a representable mean.
        outputs["loss"] = -torch.stack(contributions).sum()
        if not torch.isfinite(outputs["loss"]):
            raise ValueError("Mean loss exceeds the supported numerical range.")
        return outputs

    def predict_survival(self, times, /, **kwargs):
        """Return differentiable [B, T] survival probabilities on a shared grid.

        Args:
            times: Positional-only, nonempty 1-D floating tensor of finite,
                nonnegative, nondecreasing times. Duplicates are allowed.
            **kwargs: Features plus optional ignored labels and metadata.

        Returns:
            Probabilities with exactly one at zero. The time derivative at
            zero is defined as zero. This method does not change model mode.
            Use ``eval()`` and ``torch.no_grad()`` for ordinary inference.

        Examples:
            >>> _ = model.eval()  # doctest: +SKIP
            >>> with torch.no_grad():  # doctest: +SKIP
            ...     curves = model.predict_survival(torch.tensor([0., 1.]), **batch)
        """
        outputs = self._parameters_from_features(kwargs)
        # Validate ordering and sign before conversion can round away violations.
        if (
            isinstance(times, torch.Tensor)
            and times.ndim == 1
            and not times.is_complex()
            and ((times < 0).any() or (times[1:] < times[:-1]).any())
        ):
            raise ValueError("times must be nonnegative and nondecreasing.")
        times = self._convert(times, "times", floating=True)
        if times.ndim != 1 or times.numel() == 0:
            raise ValueError("times must be a nonempty 1-D tensor.")
        if (times < 0).any() or (times[1:] < times[:-1]).any():
            raise ValueError("times must be nonnegative and nondecreasing.")
        batch_size = outputs["means"].shape[0]
        # Multiplication before summation avoids overflow for finite parameters.
        connection = sum((value * 0).sum() for value in outputs.values())
        probabilities = (
            torch.ones(
                batch_size, times.numel(), dtype=times.dtype, device=times.device
            )
            + connection
            + times[None, :] * 0
        )
        positive = times > 0
        if positive.any():
            grid = times[positive].expand(batch_size, -1)
            log_survival = _log_survival(grid, **outputs)
            if not torch.isfinite(log_survival).all():
                raise ValueError("Predicted survival exceeds the numerical range.")
            probabilities = probabilities.clone()
            probabilities[:, positive] = log_survival.exp().clamp(0, 1)
        return probabilities
