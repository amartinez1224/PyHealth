pyhealth.metrics.survival
=========================

Standalone evaluation of right-censored survival curves for the SurvivalMDN
reproduction. The implementation uses NumPy only; neither lifelines nor a
survival package is a runtime dependency. Call it on a complete evaluation
partition after collecting predictions, independently of ``Trainer``.

.. currentmodule:: pyhealth.metrics.survival

.. autofunction:: survival_metrics_fn

.. autofunction:: survival_evaluation_grid

Protocol and interpretation
---------------------------

This API independently implements the evaluation protocol in the authors'
``metrics.py`` at revision ``71c61f649f3c8784fe13b092b38e0cbb3a184dbd``.
Han et al. Section 5.3 describes the metrics and points to SODEN's implementation.
The executable protocol has details that differ from the paper's displayed
continuous-time equations. Those details are preserved here for reproduction,
rather than silently substituting a different IPCW estimator.

Let :math:`u_i` denote observed duration, :math:`\delta_i` the event indicator,
:math:`S_i(t)` predicted survival, and :math:`G(t)` censoring survival.

* Fit Kaplan-Meier to the **evaluation partition**, with censoring indicators
  ``1 - event``. Training labels are not required.
  All observations at a tied time are in the same risk set. At observed times
  use the post-jump value; between knots use linear interpolation, as requested
  by the reference's ``KaplanMeierFitter.predict(interpolate=True)``.
* Report separate scores at ``censoring_threshold=1e-8``, ``0.2``, and ``0.4``.
  Cutoffs are probabilities, not time quantiles or fractions of observations.
* For concordance, an observed event is an anchor if :math:`G(u_i)\geq c`.
  Its comparators have :math:`u_j>u_i`, or the same duration with
  :math:`\delta_j=0`. Event-event time ties are excluded. Credit is one only
  when :math:`S_i(u_i)<S_j(u_i)`; equal predictions receive zero, not one-half.
  Each comparable pair receives weight :math:`1/G(u_i)^2`.
* For IBS/IBLL, choose :math:`\tau=\max\{u_i:G(u_i)>c\}`. For the baseline
  cutoff ``1e-8`` only, use ``G > 0`` to select the horizon, matching the
  reference meters' ``eps=0`` baseline. Construct 1,000 uniform points from
  :math:`\min_i u_i` to :math:`\tau`, inclusive. The reference uses their
  arithmetic mean, not trapezoidal integration over :math:`[0,\tau]`.
* Remove observations whose :math:`G(u_i)\leq10^{-8}` from IBS/IBLL entirely
  and normalize by the remaining count :math:`N_*`. Remove integration points
  with :math:`G(t)\leq10^{-8}` and average over the remaining points. This
  reference-specific exclusion can change the estimand; it is not silently
  applied to concordance comparators.

On retained rows and grid points, the point scores are:

.. math::

    BS(t)=\frac{1}{N_*}\sum_i\left[
    \frac{S_i(t)^2\mathbf{1}(u_i\leq t,\delta_i=1)}{G(u_i)}
    +\frac{(1-S_i(t))^2\mathbf{1}(u_i>t)}{G(t)}\right],

.. math::

    BLL(t)=\frac{1}{N_*}\sum_i\left[
    \frac{\log(1-S_i(t)+10^{-10})\mathbf{1}(u_i\leq t,\delta_i=1)}{G(u_i)}
    +\frac{\log(S_i(t)+10^{-10})\mathbf{1}(u_i>t)}{G(t)}\right].

IBS is lower-is-better; IBLL and C-index are higher-is-better. IBLL is signed,
usually negative. The reference's additive log epsilon keeps boundary
predictions finite and can give a tiny positive value for a perfect prediction.
Finite-sample IPCW Brier scores need not be bounded above by one.

Do not call these functions per batch and average the results: that changes
the censoring estimate, comparable pairs, and integration interval. These
weights assume independent censoring; they do not condition censoring on
patient features. This API supports neither competing risks nor left truncation.
Undefined requested scores raise ``ValueError`` rather than returning a
plausible-looking zero. All-censored data can still request IBS/IBLL alone if
a positive-width interval remains. No comparable pairs makes C-index undefined.

SUPPORT saved predictions
-------------------------

The SUPPORT adapter's ordinary ``test_predictions.npz`` remains a 128-point
baseline export, with a grid from zero to the maximum transformed training
duration. It is intended for inspection and visualization; its finite grid may
not cover every event time and integration point required by scoring.

Opt in to direct metric-grid predictions after training/checkpoint selection:

.. code-block:: bash

  python examples/survival_mdn_support.py \
    --data-dir ../Survival-MDN/data/support --split 1 --seed 0 \
    --output-dir output/support-baseline-metrics --evaluate-metrics

Replay that saved checkpoint and regenerate metrics in a new directory:

.. code-block:: bash

  python examples/survival_mdn_support.py \
    --data-dir ../Survival-MDN/data/support \
    --predict-only output/support-baseline-metrics \
    --output-dir output/support-metrics-reloaded --evaluate-metrics

The flag is invocation-scoped. Without it, training or replay writes no metric
artifacts and a replay does not carry stale ``metric_evaluation`` metadata from
its source manifest. Replay never trains. When source metric artifacts exist,
the adapter verifies regenerated prediction arrays and scores against them; an
older manifest without metric metadata is supported with source comparison
reported as unavailable. The empty-output-directory protection applies to both
modes.

When enabled, the adapter builds a grid for each censoring cutoff
``1e-8``, ``0.2``, and ``0.4``, using 1,000 integration points per cutoff. It
predicts once on the sorted unique union of all required times, and scores the
whole test partition once per cutoff. This produces separate
``test_metric_predictions.npz`` and ``test_metrics.json`` artifacts. The
128-point ``test_predictions.npz`` and manifest ``prediction_grid`` stay
unchanged. The metric prediction archive contains the usual label and row
metadata fields, with ``survival [N_test, T_metric]`` and ``times
[T_metric]``.

Use the metric artifact's ``duration``, ``event``, ``survival`` and ``times``
without sorting rows or changing units. ``duration`` already includes the
single ``+0.001`` offset added before float32 conversion. Do not add it again
and do not substitute ``released_duration``. No calendar-day conversion is
known. Model inference is CPU float32: helper times are cast to float32 before
prediction and receive ordinary float32 rounding.

C-index and IBLL are higher-is-better; IBS is lower-is-better. IBLL is a signed
log-likelihood, usually negative. Censoring is estimated on the complete test
evaluation partition, as specified by the reference protocol. Test labels
determine metric evaluation times and censoring estimates, but do not influence
training, validation, model selection, or checkpoint choice. The three scores
are computed on the full cohort, not per-batch averages.

These metrics characterize this PyHealth baseline; enabling them does not
reproduce the paper's tuned benchmark. The adapter's architecture, numerical
choices, sampler, optimizer, and training configuration remain as documented in
the [SUPPORT baseline guide](../../../examples/survival_mdn_support_README.md).

.. code-block:: python

    import numpy as np
    from pyhealth.metrics import survival_metrics_fn

  with np.load(
    "output/support-baseline-metrics/test_metric_predictions.npz",
    allow_pickle=False,
  ) as p:
        scores = {
            str(cutoff): survival_metrics_fn(
                p["duration"], p["event"], p["survival"], p["times"],
                censoring_threshold=cutoff,
            )
            for cutoff in (1e-8, 0.2, 0.4)
        }

      The adapter passes ``integration_steps=1000`` to both grid generation and
      scoring. Its direct predictions use the helper grid rather than interpolating
      the 128-point visualization curves. Grid construction uses test labels only to
      set evaluation query times; these labels and test scores are never fed into
      training or checkpoint selection.

Direct predictions on required times
------------------------------------

The helper can be used without the SUPPORT adapter. For a loaded ``model`` and
an ordered test ``loader`` with labels collected in that same order:

.. code-block:: python

    import numpy as np
    import torch
    from pyhealth.metrics import survival_evaluation_grid, survival_metrics_fn

    cutoffs = (1e-8, 0.2, 0.4)
    grid = np.unique(np.concatenate([
        survival_evaluation_grid(duration, event, censoring_threshold=c)
        for c in cutoffs
    ]))
    model.eval()
    with torch.no_grad():
        survival = np.concatenate([
            model.predict_survival(torch.from_numpy(grid), **batch).cpu().numpy()
            for batch in loader  # shuffle=False, drop_last=False
        ], axis=0)
    scores = {
        str(c): survival_metrics_fn(
            duration, event, survival, grid, censoring_threshold=c,
        )
        for c in cutoffs
    }

Use the original helper grid as ``times``; the model internally casts query
times to its floating dtype, so ordinary model roundoff remains. Float64
predictions permit the closest controlled numerical comparison. Keep the same
cutoff and ``integration_steps`` options for grid generation and scoring. The
grid is deterministic from evaluation labels and does not fit or tune the
predictive model. Evaluation labels must not drive model/checkpoint selection.
The model allows duplicate times; this metrics API requires a strictly
increasing grid to make interpolation unambiguous. The helper returns one.

Validation and reference fixture
--------------------------------

``tests/core/test_survival_metrics.py`` contains hand-computed examples,
censoring/tie/shape/range tests, permutation and unit-rescaling checks, and 24
controlled comparisons with the authors' original metric methods. Fixtures
cover four cohorts, three cutoffs, and both 17 and 1,000 integration points.
The checked-in fixture needs only NumPy at test time.

To regenerate, obtain ``metrics.py`` from the pinned revision and run:

.. code-block:: bash

    python -m pip install lifelines==0.30.0
    python tools/generate_survival_metrics_reference.py /path/to/metrics.py
    python -m unittest discover -s tests/core -p 'test_survival_metrics.py' -v

The developer tool checks the source SHA-256 and executes the three original
``value()`` method bodies on analytic curves. It does not import the reference
training entry point, require Torch, fetch code, or depend on the new metric
implementation. The fixture records versions and provenance. Agreement tests
validate metric arithmetic at matched inputs, not trained-model or SUPPORT
benchmark equivalence. No benchmark scores are claimed by this module.

References
----------

* Han, Goldstein, and Ranganath (2022),
  `Survival Mixture Density Networks <https://proceedings.mlr.press/v182/han22a.html>`_,
  Section 5.3.
* `Pinned authors' metric implementation
  <https://github.com/XintianHan/Survival-MDN/blob/71c61f649f3c8784fe13b092b38e0cbb3a184dbd/metrics.py>`_.
* `SODEN metric implementation <https://github.com/jiaqima/SODEN/blob/main/metrics.py>`_.
