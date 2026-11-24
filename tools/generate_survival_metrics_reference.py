"""Regenerate controlled metric fixtures from the pinned authors' metrics.py.

Developer-only requirements: numpy and lifelines==0.30.0. No reference imports
or downloads occur. Only the three original value() method bodies are executed;
the reference's Torch accumulation interface is not needed for NumPy fixtures.
Usage: python tools/generate_survival_metrics_reference.py /path/to/metrics.py
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import lifelines
import numpy as np
from lifelines import KaplanMeierFitter

REVISION = "71c61f649f3c8784fe13b092b38e0cbb3a184dbd"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/core/fixtures/survival_metrics_reference.json"),
    )
    args = parser.parse_args()
    source = args.source.read_bytes()
    expected_hash = "3fae4569596aebef4a1917cc7b4423fe69e2a6131750772f312f0b36e089e42c"
    if hashlib.sha256(source).hexdigest() != expected_hash:
        raise ValueError("Expected metrics.py from pinned Survival-MDN revision")
    methods = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name in (
            "IPWCIndexMeter",
            "BrierScoreMeter",
            "BinomialLogLikelihoodMeter",
        ):
            method = next(
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "value"
            )
            scope = {"np": np, "KaplanMeierFitter": KaplanMeierFitter}
            exec(  # noqa: S102 - execute only methods from SHA-256-verified source
                compile(
                    ast.Module(body=[method], type_ignores=[]), str(args.source), "exec"
                ),
                scope,
            )
            methods[node.name] = scope["value"]
    cases = []
    inputs = [
        ("mixed_ties", [1, 2, 2, 3, 4, 5, 6], [1, 1, 0, 1, 0, 1, 0]),
        ("no_censoring", [1, 2, 3, 4], [1, 1, 1, 1]),
        ("zero_time", [0, 1, 2, 3, 4, 5], [0, 1, 1, 0, 1, 1]),
        ("crossing", [0.5, 1, 2, 3, 4, 5], [1, 0, 1, 0, 1, 1]),
    ]
    for name, duration, event in inputs:
        d, e = np.array(duration, dtype=float), np.array(event, dtype=float)
        rates = np.linspace(0.1, 0.8, len(d))[::-1]
        powers = (
            np.linspace(0.6, 1.8, len(d)) if name == "crossing" else np.ones(len(d))
        )
        km = KaplanMeierFitter().fit(d, event_observed=1 - e)
        g = km.predict(d, interpolate=True).to_numpy()
        for threshold in (1e-8, 0.2, 0.4):
            for steps in (17, 1000):
                # For baseline IBS/IBLL the original class uses eps=0, not 1e-8.
                # On these fixtures both select the same last positive-G time.
                eps = 0 if threshold == 1e-8 else threshold
                grid = np.linspace(d.min(), max(d[g > eps]), steps)
                curves = np.exp(-rates[:, None] * grid[None, :] ** powers[:, None])
                state = SimpleNamespace(t=d, labels=e, eps=eps, survival_seqs=curves.T)
                ibs = methods["BrierScoreMeter"](state)
                ibll = methods["BinomialLogLikelihoodMeter"](state)
                state = SimpleNamespace(
                    t=d,
                    labels=e,
                    eps=threshold,
                    eval_t=np.unique(d),
                    cum_hazard_seqs=(
                        rates[:, None] * np.unique(d)[None, :] ** powers[:, None]
                    ).T,
                )
                cindex = methods["IPWCIndexMeter"](state)[0]
                cases.append(
                    {
                        "name": name,
                        "duration": duration,
                        "event": event,
                        "rates": rates.tolist(),
                        "powers": powers.tolist(),
                        "censoring_threshold": threshold,
                        "integration_steps": steps,
                        "expected": {
                            "c_index": cindex,
                            "ibs": float(ibs),
                            "ibll": float(ibll),
                        },
                    }
                )
    result = {
        "provenance": {
            "repository": "https://github.com/XintianHan/Survival-MDN",
            "revision": REVISION,
            "file": "metrics.py",
            "sha256": expected_hash,
            "lifelines": lifelines.__version__,
            "numpy": np.__version__,
            "method": "Execute original value() ASTs on analytic Weibull survival curves",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
