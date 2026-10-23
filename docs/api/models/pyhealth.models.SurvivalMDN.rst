pyhealth.models.SurvivalMDN
==========================

``SurvivalMDN`` models a positive, continuous event time from one dense feature
vector per patient. It supports right censoring: ``event=1`` means the event
was observed; ``event=0`` means observation ended before the event. Its mixture
weights, Gaussian means, and Gaussian standard deviations all depend on the
patient. Interval censoring, competing risks, survival metrics, and clinical
dataset preprocessing are outside this initial implementation.

Data and prediction contract
----------------------------

Use fixed-width, finite floating feature vectors and strictly positive finite
observed durations, in one consistent time unit. Prediction grids use the same
unit; no implicit normalization is performed. Check raw vector widths before
processing: the collator can pad ragged vectors, and model shape validation
cannot distinguish that padding from actual zeros.

Selected field names must be distinct, nonempty strings. The name ``self``
is reserved by PyTorch's keyword-call wrapper and is rejected at construction.

The processed dataset must contain the selected input and two output fields.
Additional fields are allowed. Compatible fitted processors must expose numeric
tensor metadata; the built-in tensor processors are the simplest option. Features have processor schema ``("value",)``,
``dim() == (1,)``, and positive ``size()``. Labels have the same processor
schema, size one, and dimension zero or one. Use tensor processing for events,
so datasets with only events or only censored patients remain valid:

.. code-block:: python

    import torch
    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.models import SurvivalMDN

    samples = [
        {"patient_id": "p0", "features": [0.2, -1.0],
         "duration": 1.5, "event": 1.0},
        {"patient_id": "p1", "features": [0.8, 0.3],
         "duration": 2.0, "event": 0.0},
    ]
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        in_memory=True,
    )
    batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
    model = SurvivalMDN(dataset)
    output = model(**batch)
    output["loss"].backward()
    model.eval()
    with torch.no_grad():
        parameters = model(features=batch["features"])
        survival = model.predict_survival(
            torch.tensor([0.0, 0.5, 1.0, 2.0]),
            features=batch["features"],
        )
    assert survival.shape == (2, 4)

Batches contain raw tensors: features ``[B, D]``, and labels ``[B]`` or
``[B, 1]``. Tuple/list batches are unsupported. Both labels must be supplied
together or both omitted. Unrelated metadata is ignored. ``forward`` always
returns ``log_weights``, ``means``, and ``scales``, each ``[B, K]``; it adds a
scalar ``loss`` only when both labels are present. It has no classification
``y_true``, ``y_prob``, or ``logit`` output.

``predict_survival(grid, **batch)`` returns ``[B, T]`` and ignores label fields,
even incomplete or invalid ones. Its positional-only grid also permits a custom
feature field named ``times``. The grid is a nonempty, finite, floating 1-D
tensor with nonnegative, nondecreasing values; duplicates are allowed. At zero,
survival is exactly one and the grid derivative is defined as zero. Prediction
remains differentiable and preserves the current training/evaluation mode;
call ``model.eval()`` and ``torch.no_grad()`` explicitly for inference.

CPU float32/float64 are supported, as is CUDA when available. Inputs move to the
network's device and floating dtype without detachment. Autocast and mixed
precision are outside this contract. No finite floating dtype can represent
arbitrary tail magnitudes: nonfinite parameters or selected training likelihoods
raise numerical-range errors. Caller-controlled time rescaling can keep the
problem in a useful numerical range. Tests cover times from ``1e-8`` to
``100``, scales down to ``0.02``, and separated latent means; those controlled
cases are stress checks, not universal bounds for every parameter combination.

Distribution and likelihood
---------------------------

Following Han, Goldstein, and Ranganath (2022), let the latent distribution be

.. math::

    Z\mid x \sim \sum_{k=1}^K \pi_k(x)
        \mathcal{N}(\mu_k(x),\sigma_k(x)^2),\qquad
    T=\operatorname{softplus}(Z).

For positive times, the stable inverse and log Jacobian are

.. math::

    z(t)=t+\log(-\operatorname{expm1}(-t)),\qquad
    J(t)=-\log(-\operatorname{expm1}(-t)).

Writing :math:`a_k=(z(t)-\mu_k)/\sigma_k`, the transformed log density and
survival are

.. math::

    \log f(t\mid x) &=
        \operatorname{logsumexp}_k\left[
        \log\pi_k+\log\mathcal{N}(z(t);\mu_k,\sigma_k^2)\right]+J(t),\\
    \log S(t\mid x) &=
        \operatorname{logsumexp}_k\left[
        \log\pi_k+\log\Phi(-a_k)\right],\\
    \mathcal{L} &= -\frac1B\sum_i\left[
        \delta_i\log f(t_i\mid x_i)+(1-\delta_i)\log S(t_i\mid x_i)\right].

The Jacobian belongs only in the density. Gaussian log survival uses
``torch.special.log_ndtr`` for nonpositive standardized times. For positive
standardized times ``a``, it uses the equivalent expression
``-a**2 / 2 + log(erfcx(a / sqrt(2))) - log(2)``. This preserves tail gradients
that direct float32 ``log_ndtr`` backward can lose through cancellation.
Both component log densities and log survival probabilities are centered using
a detached maximum before adding mixture log weights. This preserves weight
precision and gradient normalization when several components share a deep tail.
Neither branch subtracts a rounded CDF from one.
Only observed patients evaluate the density contribution; only censored
patients evaluate the survival contribution. This avoids multiplying unused
infinite terms by zero. A continuous density may exceed one, so negative mean
negative log likelihood is valid.

Architecture and training
-------------------------

Defaults are three hidden layers of width 64, separate PReLU activations,
five components, no BatchNorm, and ``min_scale=1e-4``. ReLU is also supported.
Separate heads predict logits, latent means, and raw scales. The model uses
``log_softmax`` weights and ``softplus(raw_scale) + min_scale`` standard
deviations. Mean-head weights start at zero and biases at evenly spaced
centers in ``[-3, 3]``; the single-component center is zero. Other heads use
ordinary PyTorch initialization.

Optional BatchNorm sits between each hidden linear layer and activation, with
hidden linear biases disabled. BatchNorm training requires at least two
patients per batch; singleton evaluation works. The default supports singleton
training.

``model.mode`` is ``None``. ``Trainer.train`` consumes the custom scalar loss;
``Trainer.evaluate`` reports loss only. Monitor ``loss`` with
``monitor_criterion="min"`` and a short patience. Generic
``Trainer.inference()`` expects classification-style outputs and does not
support this output contract; use ``predict_survival`` instead.
Trainer weights validation batch losses by their sample counts, so a short
final batch contributes proportionally to the per-patient validation mean.

The self-contained synthetic walkthrough is
``examples/survival_mdn_synthetic.ipynb``. It generates independent censoring
and latent-mixture event times, trains on 256 patients, validates on 64, plots
survival curves and mixture parameters, and checks exact checkpoint reload
agreement. Setup instructions are in the notebook. CPU is the verified default.

Paper and implementation differences
-------------------------------------

Section 5.4 of the paper describes exponential scale outputs. The authors'
released code uses softplus, as does this model, with an added ``min_scale``
floor. The released default network uses BatchNorm and clamps its outputs;
this model makes BatchNorm optional and leaves the heads unclamped. These
choices affect optimization. The controlled fixture below compares the
conditional distributions at matched effective parameters, not trained-model
or benchmark equivalence.

Reference validation and reproduction
-------------------------------------

The paper's Section 2.1 likelihood and Section 3.2 softplus change of variables
match the equations above. The reference fixture at
``tests/core/fixtures/survival_mdn_reference.json`` contains original synthetic
inputs, float64 CPU results, package versions, Git revision, and measured
errors. It controls logits, means, and effective scales directly, bypassing
initialization, feature networks, and BatchNorm. Raw scales are separately
chosen to match effective standard deviations, accounting for this model's
scale floor. Normal CI reads the JSON and never imports the authors' code.

Analytic tests target ``atol=1e-8, rtol=1e-6`` at moderate float64 inputs.
The measured maximum absolute errors are ``2.49e-6`` for log density,
``4.08e-7`` for survival at durations, and ``9.24e-7`` for grid survival.
The mean paper-NLL difference is ``9.26e-7``; the reference training criterion
minus its paper NLL is ``-6.59e-6``.
Reference parity uses ``atol=1e-4, rtol=1e-4`` to account for the authors'
``1e-6`` adjustments to inverse softplus, its Jacobian, density aggregation,
and cumulative hazard. The fixture records the measured discrepancies.
Density quadrature instead uses absolute error ``1e-4``, grid refinement, and
an analytic omitted-tail bound.

The actual reference criterion is ``main.py:survival_loss``, whose reduction
is a patient mean of ``-event*log(clamp(pdf/S, min=1e-8)) - log(S+1e-6)``.
Without the adjustments this is the paper's NLL. On the fixture the hazard
floor is inactive; the extra cumulative-hazard adjustment changes that NLL
by ``-mean(log1p(1e-6/S))``. The fixture records both values separately.
In extreme tails, subtracting the CDF, exponential overflow, and the added
epsilons can produce much larger reference discrepancies. Stable operations
are retained rather than reproducing those numerical limitations.

To regenerate the fixture, check out the reference revision recorded in its
provenance and use a separate environment with PyTorch, NumPy, and lifelines:

.. code-block:: bash

    python tools/generate_survival_mdn_reference.py ../Survival-MDN \
        tests/core/fixtures/survival_mdn_reference.json

The developer tool loads the reference distribution and extracts its training
criterion without importing the training entry point. Normal tests use the
saved fixture and require neither the reference checkout nor lifelines.

References
----------

* Xintian Han, Mark Goldstein, and Rajesh Ranganath.
  `Survival Mixture Density Networks
  <https://proceedings.mlr.press/v182/han22a.html>`_.
  Proceedings of Machine Learning Research 182:1–25, 2022.
* `Authors' implementation <https://github.com/XintianHan/Survival-MDN>`_.

API
---

.. autoclass:: pyhealth.models.SurvivalMDN
    :members:
    :show-inheritance:
