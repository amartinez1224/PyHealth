"""Right-censored survival evaluation using the Survival-MDN/SODEN protocol.

This is a NumPy-only, whole-partition API, independent of Trainer. See the
API guide for reference-specific tie, censoring, and integration conventions.
"""

from collections.abc import Sequence

import numpy as np

__all__ = ["survival_evaluation_grid", "survival_metrics_fn"]

_METRICS = ("c_index", "ibs", "ibll")
_MIN_G = 1e-8


def _array(value, name, ndim):
    value = np.asarray(value)
    if value.dtype.kind not in "biuf" or value.ndim != ndim or value.size == 0:
        raise ValueError(f"{name} must be a nonempty {ndim}-D real numeric array")
    value = value.astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _labels(duration, event):
    duration = _array(duration, "duration", 1)
    event = _array(event, "event", 1)
    if duration.shape != event.shape or (duration < 0).any():
        raise ValueError("duration/event must match and duration must be nonnegative")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("event must be exactly 0 (censored) or 1 (observed)")
    return duration, event


def _options(metrics, censoring_threshold, integration_steps):
    names = _METRICS if metrics is None else tuple(metrics)
    if not names or any(name not in _METRICS for name in names):
        raise ValueError(f"metrics must be a nonempty sequence from {_METRICS}")
    if (
        not np.isscalar(censoring_threshold)
        or isinstance(censoring_threshold, (str, bool, np.bool_, complex))
        or not np.isfinite(censoring_threshold)
        or not _MIN_G <= censoring_threshold < 1
    ):
        raise ValueError("censoring_threshold must be finite and in [1e-8, 1)")
    if (
        isinstance(integration_steps, (bool, np.bool_))
        or not isinstance(integration_steps, (int, np.integer))
        or integration_steps < 2
    ):
        raise ValueError("integration_steps must be an integer >= 2")
    return names


def _censoring_curve(duration, event):
    """KM of censoring, with all tied observations in the same risk set."""
    knots, inverse, counts = np.unique(
        duration, return_inverse=True, return_counts=True
    )
    censored = np.bincount(inverse, weights=1 - event)
    at_risk = len(duration) - np.r_[0, np.cumsum(counts[:-1])]
    # Accumulate in log space, matching the reference KM arithmetic even at
    # cutoff boundaries. A terminal all-censored risk set gives log(0)=-inf.
    with np.errstate(divide="ignore"):
        probability = np.exp(np.cumsum(np.log(at_risk - censored) - np.log(at_risk)))
    # lifelines' KM timeline includes zero. At observed times the lookup is
    # right-continuous; between knots the reference requests linear interpolation.
    if knots[0] > 0:
        knots = np.r_[0.0, knots]
        probability = np.r_[1.0, probability]
    return knots, probability


def _protocol(duration, event, names, threshold, steps):
    knots, probability = _censoring_curve(duration, event)
    g_duration = np.interp(duration, knots, probability)
    anchors = np.flatnonzero((event == 1) & (g_duration >= threshold))
    integration = np.array([], dtype=float)
    if "ibs" in names or "ibll" in names:
        horizon_cutoff = 0.0 if threshold == _MIN_G else threshold
        eligible = duration[g_duration > horizon_cutoff]
        if eligible.size == 0 or eligible.max() <= duration.min():
            raise ValueError("No positive-width integration interval at this cutoff")
        integration = np.linspace(duration.min(), eligible.max(), steps)
    required = []
    if "c_index" in names:
        # Omit anchors that cannot form a pair; they need no prediction column.
        anchors = np.array(
            [i for i in anchors if np.any(_comparators(duration, event, i))],
            dtype=int,
        )
        if not len(anchors):
            raise ValueError("C-index is undefined: no comparable pairs at this cutoff")
        required.append(duration[anchors])
    if integration.size:
        required.append(integration)
    return np.unique(np.concatenate(required)), integration, anchors, g_duration


def _comparators(duration, event, i):
    return (duration > duration[i]) | ((duration == duration[i]) & (event == 0))


def survival_evaluation_grid(
    duration: np.ndarray,
    event: np.ndarray,
    metrics: Sequence[str] | None = None,
    *,
    censoring_threshold: float = 1e-8,
    integration_steps: int = 1000,
) -> np.ndarray:
    """Return the sorted unique times needed for direct model evaluation.

    Args:
        duration: Observed follow-up times, shape [N], in model time units.
        event: Shape [N]; 1 for observed events and 0 for right censoring.
        metrics: Subset of ``c_index``, ``ibs``, ``ibll``; all by default.
        censoring_threshold: Minimum censoring survival; paper cutoffs are
            1e-8, 0.2, and 0.4. C-index uses >=; integration horizon uses >
            (baseline 1e-8 selects its horizon using G > 0).
        integration_steps: Number of uniform integration points, default 1000.

    Returns:
        Float64 grid containing comparable event times and/or integration times.
        Use the same labels and options when subsequently scoring predictions.

    Raises:
        ValueError: Invalid inputs or an undefined requested metric.
    """
    duration, event = _labels(duration, event)
    names = _options(metrics, censoring_threshold, integration_steps)
    return _protocol(duration, event, names, censoring_threshold, integration_steps)[0]


def _curves(survival, times, n):
    times = _array(times, "times", 1)
    survival = _array(survival, "survival", 2)
    if (times < 0).any() or (np.diff(times) <= 0).any():
        raise ValueError("times must be nonnegative and strictly increasing")
    if survival.shape != (n, len(times)):
        raise ValueError("survival must have shape [N, T], aligned to labels and times")
    if ((survival < 0) | (survival > 1)).any():
        raise ValueError("survival must lie in [0, 1]")
    if (np.diff(survival, axis=1) > 1e-6).any():
        raise ValueError("survival must be nonincreasing (tolerance 1e-6)")
    return survival, times


def _interpolate(survival, times, query):
    if query.min() < times[0] or query.max() > times[-1]:
        raise ValueError(
            "times must cover all required evaluation times; extrapolation is not "
            "allowed. Use survival_evaluation_grid and predict again."
        )
    return np.asarray([np.interp(query, times, row) for row in survival])


def survival_metrics_fn(
    duration: np.ndarray,
    event: np.ndarray,
    survival: np.ndarray,
    times: np.ndarray,
    metrics: Sequence[str] | None = None,
    *,
    censoring_threshold: float = 1e-8,
    integration_steps: int = 1000,
) -> dict[str, float]:
    """Compute reference-protocol IPCW C-index, IBS and IBLL on one partition.

    C-index is time-dependent, with inverse squared censoring weights. IBS is
    better when lower; C-index and IBLL are better when higher. IBLL is the
    signed log-likelihood, not its negative. Censoring KM is fitted on the
    supplied evaluation labels, matching Survival-MDN/SODEN, not on training
    labels. This function must be called once on the full evaluation partition;
    averaging scores computed per batch is not equivalent.

    Args:
        duration: Finite nonnegative observed durations, shape [N]. For SUPPORT,
            use the exported ``duration``, already offset by 0.001 once.
        event: Shape [N]; exactly 1 for observed events, 0 for right censoring.
        survival: Finite survival probabilities, shape [N, T], in label order.
        times: Shared nonnegative, strictly increasing prediction grid [T],
            in the same model units as duration. Missing evaluation times are
            linearly interpolated; extrapolation is rejected. Use the grid
            helper to avoid interpolation when predicting in float64.
        metrics: Nonempty subset of ``c_index``, ``ibs``, ``ibll``; all by default.
        censoring_threshold: Cutoff in [1e-8, 1); paper values: 1e-8, 0.2, 0.4.
        integration_steps: Uniform integration grid size, default 1000.

    Returns:
        Dictionary mapping each requested metric name to a Python float.

    Raises:
        ValueError: Invalid arrays/options, insufficient grid coverage, no
            comparable pairs, or no positive-width integration interval.

    Notes:
        The reference uses linear KM interpolation, a uniform point mean from
        min(duration) to max(duration[G > cutoff]) (G > 0 for the baseline),
        excludes G <= 1e-8 rows
        and integration points, and adds 1e-10 inside logarithms. These are
        deliberate reproduction conventions, not a generic IPCW estimator.
        An event tied with a censoring time is comparable; prediction ties
        get zero credit. See the API guide for limitations and exact formulas.
        Runtime is O(N^2 + N*T + N*integration_steps); no N-by-N array is built.
    """
    duration, event = _labels(duration, event)
    names = _options(metrics, censoring_threshold, integration_steps)
    survival, times = _curves(survival, times, len(duration))
    required, grid, anchors, g_duration = _protocol(
        duration, event, names, censoring_threshold, integration_steps
    )
    predictions = _interpolate(survival, times, required)
    result = {}
    if "c_index" in names:
        numerator, denominator = 0.0, 0.0
        for i in anchors:
            others = _comparators(duration, event, i)
            column = predictions[:, np.searchsorted(required, duration[i])]
            weight = 1.0 / g_duration[i] ** 2
            numerator += weight * np.count_nonzero(column[i] < column[others])
            denominator += weight * np.count_nonzero(others)
        result["c_index"] = float(numerator / denominator)
    if grid.size:
        knots, probability = _censoring_curve(duration, event)
        g_grid = np.interp(grid, knots, probability)
        keep_rows, keep_times = g_duration > _MIN_G, g_grid > _MIN_G
        if not keep_rows.any() or not keep_times.any():
            raise ValueError("No stable censoring weights for integration")
        s = predictions[keep_rows][:, np.searchsorted(required, grid[keep_times])]
        u = duration[keep_rows, None]
        t = grid[None, keep_times]
        failed = (u <= t) & (event[keep_rows, None] == 1)
        alive = u > t
        failure_weight = failed / g_duration[keep_rows, None]
        survival_weight = alive / g_grid[None, keep_times]
        if "ibs" in names:
            result["ibs"] = float(
                np.mean(s**2 * failure_weight + (1 - s) ** 2 * survival_weight)
            )
        if "ibll" in names:
            result["ibll"] = float(
                np.mean(
                    np.log(1 - s + 1e-10) * failure_weight
                    + np.log(s + 1e-10) * survival_weight
                )
            )
    return {name: result[name] for name in names}
