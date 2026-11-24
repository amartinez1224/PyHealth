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


def _main():
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
    # Execute only the extracted reference criterion, avoiding its training entry point.
    exec(  # noqa: S102
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
                "python tools/generate_survival_mdn_reference.py ../Survival-MDN "
                "tests/core/fixtures/survival_mdn_reference.json"
            ),
            "harness": (
                "Original full source in "
                "tools/generate_survival_mdn_reference.py; no main.py import"
            ),
        },
        "inputs": inputs,
        "reference": {
            k: plain(v)
            for k, v in {
                "weights": w,
                "means": mu,
                "scales": sigma,
                "log_density": ld,
                "survival_at_duration": surv,
                "survival_grid": sg,
                "paper_nll": nll,
                "training_criterion": criterion,
            }.items()
        },
        "stable": {
            k: plain(v)
            for k, v in {
                "log_density": stable_ld,
                "survival_at_duration": stable_s,
                "survival_grid": stable_sg,
                "paper_nll": stable_nll,
            }.items()
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


if __name__ == "__main__":
    _main()
