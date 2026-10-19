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

The processed dataset must contain exactly one input and two output fields,
with fitted tensor processors. Features have processor schema ``("value",)``,
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

Released SUPPORT baseline
-------------------------

``examples/survival_mdn_support.py`` trains this model on preprocessed SUPPORT
data from the Survival-MDN repository, preserving partitions and archive row
identity.
It trains a fixed CPU baseline, selects on validation NLL, reloads the best
checkpoint, and exports every test curve. See
`the SUPPORT example guide
<https://github.com/sunlabuiuc/PyHealth/blob/master/examples/survival_mdn_support_README.md>`_
for commands, fingerprints, preprocessing audit, and reference differences.
The companion ``examples/survival_mdn_support.ipynb`` provides an interactive
walkthrough of local data inspection, baseline training/reload, aggregate curves,
and exact prediction-only replay. Its committed version contains no outputs.
Times use released SUPPORT units plus a single ``0.001``
offset; no calendar-day mapping is known. Benchmark records and run artifacts
are not distributed with PyHealth.

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
Tests compare density gradients to analytic Gaussian derivatives and survival
gradients to an independent inverse Mills-ratio expansion at standardized times
from 5,000 to 1,000,000, including the default scale floor. Identical components
must produce zero weight-logit gradients. These checks do not imply stability
for arbitrary parameter magnitudes.
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
training. The reference Survival-MDN network uses BatchNorm/PReLU, hard-clamps means
and raw scales, and uses different numerical adjustments. This implementation
uses stable formulas, a scale floor, optional BatchNorm, and unclamped output
heads. These choices affect the resulting distributions and training behavior.

``model.mode`` is ``None``. ``Trainer.train`` consumes the custom scalar loss;
``Trainer.evaluate`` reports loss only. Monitor ``loss`` with
``monitor_criterion="min"`` and a short patience. Generic
``Trainer.inference()`` expects classification-style outputs and does not
support this output contract; use ``predict_survival`` instead.
Trainer averages validation losses by batch, which can differ from a patient
mean for unequal batches. The synthetic example uses equal-size validation
batches:

.. code-block:: bash

    python examples/survival_mdn_synthetic.py --epochs 2 --seed 0 --device cpu \
        --output-dir /tmp/pyhealth-survival-mdn-smoke

The example generates independent censoring and latent-mixture event times,
trains on 256 patients, validates on 64, and uses batch size 32. It saves a
grid, survival probabilities, constructor settings, and a state dictionary,
and checks checkpoint reload equivalence. Reconstruct with the same settings
and a compatible processed dataset; weights alone do not store configuration
or processor metadata.

For a cell-by-cell walkthrough, open
``examples/survival_mdn_synthetic.ipynb`` in a notebook kernel with this checkout
installed. It uses the script's synthetic data generator and adds plots of
observed follow-up, patient survival curves, and component weights, followed
by a checkpoint round trip. CPU is the verified default, including on Apple
Silicon; MPS execution is not validated. Setup instructions are in the notebook.

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

To reproduce, use a separate Python environment with PyTorch, NumPy, and
lifelines installed, check out the reference at the revision in the fixture,
and save the original harness below as ``/tmp/survival_mdn_reference.py``.
From the PyHealth root run:

.. code-block:: bash

    python /tmp/survival_mdn_reference.py ../Survival-MDN \
        tests/core/fixtures/survival_mdn_reference.json

The harness loads only ``mdn_models.py`` and its dependencies. It extracts the
single criterion function using AST; importing ``main.py`` would parse CLI
arguments, load data, and launch training. This harness is documentation for
external validation, not production or CI code:

.. code-block:: python

    """Original controlled validation harness; run outside PyHealth/normal CI."""

    import ast
    import importlib.util
    import json
    import math
    import platform
    import subprocess
    import sys
    from pathlib import Path

    import lifelines
    import numpy
    import torch

    source, destination = map(Path, sys.argv[1:])
    sys.path.insert(0, str(source.resolve()))
    spec = importlib.util.spec_from_file_location(
        "reference_mdn", source / "mdn_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = next(
        n
        for n in ast.parse((source / "main.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "survival_loss"
    )
    scope = {"torch": torch, "sys": sys}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "survival_loss_ast", "exec"),
        scope,
    )
    inputs = {
        "logits": [[0.2, -0.3, 0.7], [-0.6, 0.4, 0.1], [0.5, 0.0, -0.4]],
        "means": [[-0.7, 0.4, 1.2], [-0.2, 0.8, 1.5], [-1.0, 0.2, 1.0]],
        "scales": [[0.8, 1.0, 1.2], [1.1, 0.9, 0.7], [0.9, 1.3, 0.8]],
        "duration": [0.5, 1.1, 2.0],
        "event": [1.0, 0.0, 1.0],
        "grid": [0.25, 0.75, 1.5, 2.5],
        "min_scale": 0.0001,
    }
    t = {k: torch.tensor(v, dtype=torch.float64) for k, v in inputs.items()}
    # Match effective standard deviations, not raw scale-head outputs.
    raw = t["scales"] + torch.log(-torch.expm1(-t["scales"]))
    params = torch.stack((t["logits"], t["means"], raw), dim=1)
    dist = module.SurvivalMDN_Dist(params)
    w, mu, sigma = dist.pred_params_to_parameters(params)
    ld = dist.log_prob(t["duration"])
    surv = 1 - dist.cdf(t["duration"])
    grid = t["grid"].expand(3, -1)
    sg = 1 - dist.cdf(grid)
    criterion = scope["survival_loss"](
        {"lambda": ld.exp() / surv, "Lambda": -(surv + 1e-6).log()}, t["event"]
    )
    nll = -(t["event"] * ld + (1 - t["event"]) * surv.log()).mean()


    # Stable paper formulas, independently from the PyHealth implementation.
    def stable(times):
        z = times + torch.log(-torch.expm1(-times))
        jac = -torch.log(-torch.expm1(-times))
        m, s, weights = mu, sigma, w
        if times.ndim == 2:
            m, s, weights = mu[:, None], sigma[:, None], w[:, None]
        a = (z[..., None] - m) / s
        density = (
            torch.logsumexp(
                weights.log()
                - s.log()
                - 0.5 * a.square()
                - 0.5 * math.log(2 * math.pi),
                -1,
            )
            + jac
        )
        survival = torch.logsumexp(weights.log() + torch.special.log_ndtr(-a), -1)
        return density, survival.exp(), survival


    stable_ld, stable_s, stable_ls = stable(t["duration"])
    _, stable_sg, _ = stable(grid)
    stable_nll = -(t["event"] * stable_ld + (1 - t["event"]) * stable_ls).mean()


    def plain(value):
        return value.tolist() if isinstance(value, torch.Tensor) else value


    result = {
        "provenance": {
            "repository": "https://github.com/XintianHan/Survival-MDN",
            "revision": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "paper": (
                "Han, Goldstein, Ranganath (2022), "
                "Survival Mixture Density Networks, PMLR 182"
            ),
            "dtype": "torch.float64",
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "lifelines": lifelines.__version__,
            "criterion": (
                "main.py survival_loss AST; "
                "mean(-event*log(clamp(pdf/S,min=1e-8))-log(S+1e-6))"
            ),
            "command": (
                "python /tmp/survival_mdn_reference.py ../Survival-MDN "
                "tests/core/fixtures/survival_mdn_reference.json"
            ),
            "harness": (
                "Original full source in "
                "docs/api/models/pyhealth.models.SurvivalMDN.rst; no main.py import"
            ),
        },
        "inputs": inputs,
        "reference": {
            k: plain(v)
            for k, v in dict(
                weights=w,
                means=mu,
                scales=sigma,
                log_density=ld,
                survival_at_duration=surv,
                survival_grid=sg,
                paper_nll=nll,
                training_criterion=criterion,
            ).items()
        },
        "stable": {
            k: plain(v)
            for k, v in dict(
                log_density=stable_ld,
                survival_at_duration=stable_s,
                survival_grid=stable_sg,
                paper_nll=stable_nll,
            ).items()
        },
        "comparison": {
            "max_abs_log_density": (ld - stable_ld).abs().max().item(),
            "max_abs_survival_at_duration": (surv - stable_s).abs().max().item(),
            "max_abs_survival_grid": (sg - stable_sg).abs().max().item(),
            "abs_paper_nll": abs((nll - stable_nll).item()),
            "criterion_minus_reference_paper_nll": (criterion - nll).item(),
            "atol": 1e-4,
            "rtol": 1e-4,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparison"], indent=2))

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
