"""Offline mathematical, contract, and Trainer tests for SurvivalMDN."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SurvivalMDN
from pyhealth.models.survival_mdn import _log_density, _log_survival
from pyhealth.processors import TensorProcessor
from pyhealth.trainer import Trainer


def _dataset(events=(1.0, 0.0, 1.0, 0.0), keys=None):
    feature, duration, event = keys or ("features", "duration", "event")
    samples = [
        {
            "patient_id": f"p{i}",
            feature: [0.2 + 0.1 * i, (-1.0) ** i],
            duration: 0.5 + 0.4 * i,
            event: float(observed),
        }
        for i, observed in enumerate(events)
    ]
    return create_sample_dataset(
        samples,
        {feature: "tensor"},
        {duration: "tensor", event: "tensor"},
        in_memory=True,
    )


def _batch(dataset):
    return next(iter(get_dataloader(dataset, batch_size=len(dataset), shuffle=False)))


def _fixed_parameters(batch=2):
    logits = torch.tensor([-1.0, 0.4, 0.1], dtype=torch.float64)
    return (
        logits.log_softmax(-1).expand(batch, -1).clone(),
        torch.tensor([-0.8, 0.2, 1.3], dtype=torch.float64).expand(batch, -1).clone(),
        torch.tensor([0.5, 0.8, 0.6], dtype=torch.float64).expand(batch, -1).clone(),
    )


def _independent_distribution(times, log_weights, means, scales):
    """Moderate-range oracle using scalar math and Normal's erfc formula."""
    density, survival = [], []
    for t, weights, centers, widths in zip(
        times.detach(), log_weights.detach(), means.detach(), scales.detach()
    ):
        z = math.log(math.expm1(float(t)))
        jacobian = 1.0 / (1.0 - math.exp(-float(t)))
        f, s = 0.0, 0.0
        for weight, mean, scale in zip(weights, centers, widths):
            a = (z - float(mean)) / float(scale)
            p = math.exp(float(weight))
            f += p * math.exp(-0.5 * a * a) / (float(scale) * math.sqrt(2 * math.pi))
            s += p * 0.5 * math.erfc(a / math.sqrt(2.0))
        density.append(f * jacobian)
        survival.append(s)
    return torch.tensor(density, dtype=torch.float64), torch.tensor(
        survival, dtype=torch.float64
    )


class TestSurvivalDistribution(unittest.TestCase):
    def test_single_component_analytic_and_jacobian(self):
        times = torch.tensor([0.2, 0.7, 2.0], dtype=torch.float64, requires_grad=True)
        weights = torch.zeros(3, 1, dtype=torch.float64)
        means = torch.full((3, 1), 0.3, dtype=torch.float64)
        scales = torch.full((3, 1), 0.8, dtype=torch.float64)
        density, survival = _independent_distribution(times, weights, means, scales)
        torch.testing.assert_close(
            _log_density(times, weights, means, scales).exp(),
            density,
            atol=1e-8,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            _log_survival(times, weights, means, scales).exp(),
            survival,
            atol=1e-8,
            rtol=1e-6,
        )
        # Differentiating the inverse transform independently checks its Jacobian.
        latent = torch.log(torch.expm1(times))
        derivative = torch.autograd.grad(latent.sum(), times)[0]
        torch.testing.assert_close(derivative, 1 / (-torch.expm1(-times)))

    def test_multi_component_oracle_and_permutation(self):
        times = torch.tensor([0.4, 1.8], dtype=torch.float64)
        parameters = _fixed_parameters()
        density, survival = _independent_distribution(times, *parameters)
        torch.testing.assert_close(
            _log_density(times, *parameters).exp(), density, atol=1e-8, rtol=1e-6
        )
        torch.testing.assert_close(
            _log_survival(times, *parameters).exp(), survival, atol=1e-8, rtol=1e-6
        )
        permuted = tuple(p[:, [2, 0, 1]] for p in parameters)
        for helper in (_log_density, _log_survival):
            torch.testing.assert_close(
                helper(times, *parameters), helper(times, *permuted)
            )

    def test_broadcast_rank_not_equal_axis_sizes(self):
        for batch, points in ((1, 1), (2, 2), (2, 3)):
            parameters = _fixed_parameters(batch)
            times = torch.linspace(0.2, 2.0, points, dtype=torch.float64)
            grid = times.expand(batch, -1)
            for helper in (_log_density, _log_survival):
                actual = helper(grid, *parameters)
                self.assertEqual(actual.shape, (batch, points))
                for j in range(points):
                    torch.testing.assert_close(
                        actual[:, j], helper(grid[:, j], *parameters)
                    )

    def test_density_normalization_with_refinement_and_tail_bound(self):
        parameters = _fixed_parameters(1)
        low, high = 1e-5, 12.0
        # This moderate mixture has less than 1e-10 probability outside the grid.
        # Independent Normal tail probabilities bound omitted transformed mass.
        boundary = torch.tensor([low, high], dtype=torch.float64)
        _, survival = _independent_distribution(
            boundary, *(p.expand(2, -1) for p in parameters)
        )
        omitted = 1 - survival[0] + survival[1]
        self.assertLess(float(omitted), 1e-10)
        integrals = []
        for points in (10001, 20001):
            grid = torch.linspace(low, high, points, dtype=torch.float64)
            density = _log_density(grid[None, :], *parameters).exp()[0]
            integrals.append(torch.trapezoid(density, grid))
        self.assertLess(float(abs(integrals[-1] - 1)), 1e-4)
        self.assertLess(float(abs(integrals[-1] - integrals[0])), 1e-4)

    def test_survival_derivative_is_negative_density(self):
        times = torch.tensor([0.4, 1.8], dtype=torch.float64, requires_grad=True)
        parameters = _fixed_parameters()
        survival = _log_survival(times, *parameters).exp()
        derivative = torch.autograd.grad(survival.sum(), times)[0]
        torch.testing.assert_close(
            -derivative, _log_density(times, *parameters).exp(), atol=1e-8, rtol=1e-6
        )

    def test_float64_helper_gradcheck(self):
        times = torch.tensor([0.6, 1.3], dtype=torch.float64, requires_grad=True)
        parameters = tuple(p.requires_grad_() for p in _fixed_parameters())
        for helper in (_log_density, _log_survival):
            self.assertTrue(torch.autograd.gradcheck(helper, (times, *parameters)))

    def test_stress_values_and_tail_gradients(self):
        for dtype in (torch.float32, torch.float64):
            times = torch.tensor([1e-8, 100.0, 5.0], dtype=dtype, requires_grad=True)
            logits = torch.tensor([[0.0, -80.0]] * 3, dtype=dtype, requires_grad=True)
            means = torch.tensor([[-5.0, 5.0]] * 3, dtype=dtype, requires_grad=True)
            scales = torch.tensor([[0.02, 0.5]] * 3, dtype=dtype, requires_grad=True)
            for helper in (_log_density, _log_survival):
                value = helper(times, logits.log_softmax(-1), means, scales)
                self.assertTrue(torch.isfinite(value).all())
                gradients = torch.autograd.grad(
                    value.sum(), (times, logits, means, scales), retain_graph=True
                )
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        normal = torch.distributions.Normal(0.0, 1.0)
        self.assertEqual(float(1 - normal.cdf(torch.tensor(10.0))), 0.0)
        tail = _log_survival(
            torch.tensor([10.0]),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.ones(1, 1),
        )
        self.assertTrue(torch.isfinite(tail).all())
        self.assertGreater(float(tail.exp()), 0)

    def test_deep_tail_gradient_accuracy_and_mixture_normalization(self):
        # Independent inverse Mills-ratio expansion; at a >= 5000 the
        # omitted terms are far below either dtype's test tolerance.
        for dtype in (torch.float32, torch.float64):
            for scale in (0.02, 0.001, 1e-4):
                for components in (1, 3):
                    with self.subTest(dtype=dtype, scale=scale, k=components):
                        times = torch.tensor([100.0], dtype=dtype, requires_grad=True)
                        means = torch.zeros(
                            1, components, dtype=dtype, requires_grad=True
                        )
                        scales = torch.full(
                            (1, components), scale, dtype=dtype, requires_grad=True
                        )
                        logits = torch.arange(components, dtype=dtype)[None, :]
                        logits.requires_grad_()
                        log_weights = logits.log_softmax(-1)
                        value = _log_survival(times, log_weights, means, scales)
                        dt, dm, ds, dw = torch.autograd.grad(
                            value.sum(), (times, means, scales, logits)
                        )
                        a = 100.0 / float(scales.detach()[0, 0])
                        mills = a + 1 / a - 2 / a**3 + 10 / a**5
                        weights = log_weights.detach().exp()
                        tolerance = 2e-6 if dtype == torch.float32 else 1e-10
                        for actual, expected in (
                            (dt, torch.full_like(dt, -mills / scale)),
                            (dm, weights * (mills / scale)),
                            (ds, weights * (mills * a / scale)),
                        ):
                            torch.testing.assert_close(
                                actual, expected, rtol=tolerance, atol=0
                            )
                        # Identical components make the likelihood independent
                        # of weights, even when the log likelihood is huge.
                        torch.testing.assert_close(
                            dw, torch.zeros_like(dw), rtol=0, atol=tolerance
                        )

    def test_deep_tail_density_gradient_accuracy_and_mixture_normalization(self):
        # At t=100, inverse softplus and its derivative equal t and 1 to
        # either dtype's precision. Use analytic single-Gaussian derivatives;
        # identical components must reduce to that same Gaussian.
        for dtype in (torch.float32, torch.float64):
            for scale in (0.02, 0.001, 1e-4):
                for components in (1, 3):
                    with self.subTest(dtype=dtype, scale=scale, k=components):
                        times = torch.tensor([100.0], dtype=dtype, requires_grad=True)
                        means = torch.zeros(
                            1, components, dtype=dtype, requires_grad=True
                        )
                        scales = torch.full(
                            (1, components), scale, dtype=dtype, requires_grad=True
                        )
                        logits = torch.arange(components, dtype=dtype)[None, :]
                        logits.requires_grad_()
                        log_weights = logits.log_softmax(-1)
                        value = _log_density(times, log_weights, means, scales)
                        dt, dm, ds, dw = torch.autograd.grad(
                            value.sum(), (times, means, scales, logits)
                        )
                        a = 100.0 / float(scales.detach()[0, 0])
                        weights = log_weights.detach().exp()
                        tolerance = 2e-6 if dtype == torch.float32 else 1e-10
                        for actual, expected in (
                            (dt, torch.full_like(dt, -a / scale)),
                            (dm, weights * (a / scale)),
                            (ds, weights * ((a * a - 1) / scale)),
                        ):
                            torch.testing.assert_close(
                                actual, expected, rtol=tolerance, atol=0
                            )
                        torch.testing.assert_close(
                            dw, torch.zeros_like(dw), rtol=0, atol=tolerance
                        )


class TestSurvivalReference(unittest.TestCase):
    def test_recorded_author_distribution_and_actual_criterion(self):
        path = Path(__file__).parent / "fixtures" / "survival_mdn_reference.json"
        fixture = json.loads(path.read_text())
        inputs, reference = fixture["inputs"], fixture["reference"]

        def tensor(key):
            return torch.tensor(inputs[key], dtype=torch.float64)

        log_weights = tensor("logits").log_softmax(-1)
        means, scales = tensor("means"), tensor("scales")
        floor = inputs["min_scale"]
        raw_scales = torch.log(torch.expm1(scales - floor))
        effective_scales = torch.nn.functional.softplus(raw_scales) + floor
        torch.testing.assert_close(
            effective_scales,
            torch.tensor(reference["scales"], dtype=torch.float64),
            atol=1e-8,
            rtol=1e-6,
        )
        duration, event, grid = tensor("duration"), tensor("event"), tensor("grid")
        log_density = _log_density(duration, log_weights, means, scales)
        log_survival = _log_survival(duration, log_weights, means, scales)
        grid_survival = _log_survival(
            grid.expand(duration.numel(), -1), log_weights, means, scales
        ).exp()
        loss = -torch.where(event.bool(), log_density, log_survival).mean()
        for actual, key in (
            (log_weights.exp(), "weights"),
            (means, "means"),
            (scales, "scales"),
            (log_density, "log_density"),
            (log_survival.exp(), "survival_at_duration"),
            (grid_survival, "survival_grid"),
            (loss, "paper_nll"),
            (loss, "training_criterion"),
        ):
            with self.subTest(key=key):
                torch.testing.assert_close(
                    actual,
                    torch.tensor(reference[key], dtype=torch.float64),
                    atol=1e-4,
                    rtol=1e-4,
                )


class TestSurvivalMDN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _dataset()

    def setUp(self):
        torch.manual_seed(17)
        self.model = SurvivalMDN(self.dataset, hidden_dims=(8, 6), n_components=3)
        self.batch = _batch(self.dataset)

    def test_actual_tensor_collation_and_output_contract(self):
        self.assertEqual(self.batch["features"].shape, (4, 2))
        self.assertEqual(self.batch["duration"].shape, (4,))
        self.assertEqual(self.batch["event"].shape, (4,))
        self.assertIsNone(self.model.mode)
        output = self.model(**self.batch)
        self.assertEqual(set(output), {"log_weights", "means", "scales", "loss"})
        self.assertEqual(output["loss"].shape, ())
        for name in ("log_weights", "means", "scales"):
            self.assertEqual(output[name].shape, (4, 3))
            self.assertTrue(torch.isfinite(output[name]).all())
        torch.testing.assert_close(output["log_weights"].exp().sum(-1), torch.ones(4))
        self.assertTrue((output["scales"] > self.model.min_scale).all())
        output["loss"].backward()
        unlabeled = self.model(features=self.batch["features"])
        self.assertEqual(set(unlabeled), {"log_weights", "means", "scales"})

    def test_network_initialization_and_batch_norm(self):
        torch.testing.assert_close(self.model.mean_head.weight, torch.zeros(3, 6))
        torch.testing.assert_close(self.model.mean_head.bias, torch.linspace(-3, 3, 3))
        model = SurvivalMDN(
            self.dataset, hidden_dims=(7, 5), n_components=1, batch_norm=True
        )
        modules = list(model.hidden.children())
        self.assertEqual(sum(isinstance(m, nn.BatchNorm1d) for m in modules), 2)
        activations = [m for m in modules if isinstance(m, nn.PReLU)]
        self.assertEqual(len(activations), 2)
        self.assertIsNot(activations[0], activations[1])
        for module in modules:
            if isinstance(module, nn.Linear):
                self.assertIsNone(module.bias)
        torch.testing.assert_close(model.mean_head.bias, torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "[Ss]ingleton|at least two|batch"):
            model(features=self.batch["features"][:1])
        model.eval()
        self.assertEqual(
            model(features=self.batch["features"][:1])["means"].shape, (1, 1)
        )
        # Divisible batch sizes exercise BatchNorm with the actual loader.
        model.train()
        for batch in get_dataloader(self.dataset, batch_size=2, shuffle=False):
            model(**batch)["loss"].backward()

    def test_singletons_custom_names_and_repeated_times(self):
        dataset = _dataset(keys=("times", "followup", "observed"))
        model = SurvivalMDN(
            dataset,
            feature_key="times",
            duration_key="followup",
            event_key="observed",
            hidden_dims=(3,),
            n_components=1,
            activation="relu",
        ).eval()
        features = _batch(dataset)["times"][:1]
        output = model(times=features)
        self.assertEqual(output["means"].shape, (1, 1))
        torch.testing.assert_close(output["log_weights"], torch.zeros(1, 1))
        grid = torch.tensor([0.0, 0.2, 0.2, 100.0])
        survival = model.predict_survival(grid, times=features).detach()
        self.assertEqual(survival.shape, (1, 4))
        self.assertEqual(float(survival[0, 0]), 1.0)
        self.assertEqual(float(survival[0, 1]), float(survival[0, 2]))
        self.assertLess(float(survival[0, -1]), 1e-10)
        self.assertEqual(model.predict_survival(grid[:1], times=features).shape, (1, 1))

    def test_curves_label_independence_and_batch_equivalence(self):
        self.model.eval()
        grid = torch.tensor([0.0, 0.2, 0.7, 2.0, 100.0])
        original = self.model(**self.batch)
        changed = dict(
            self.batch,
            duration=self.batch["duration"] + 3,
            event=1 - self.batch["event"],
        )
        output = self.model(**changed)
        for key in ("log_weights", "means", "scales"):
            torch.testing.assert_close(original[key], output[key])
        curves = self.model.predict_survival(grid, **self.batch)
        self.assertTrue(((curves >= 0) & (curves <= 1)).all())
        self.assertTrue((curves[:, 1:] <= curves[:, :-1]).all())
        torch.testing.assert_close(curves[:, 0], torch.ones(4))
        for labels in (
            {"duration": None},
            {"event": "invalid"},
            {"duration": -2, "event": float("nan")},
        ):
            torch.testing.assert_close(
                curves,
                self.model.predict_survival(
                    grid, features=self.batch["features"], **labels
                ),
            )
        individual = torch.cat(
            [
                self.model.predict_survival(grid, features=row[None, :])
                for row in self.batch["features"]
            ]
        )
        torch.testing.assert_close(curves, individual)

    def test_likelihood_selected_reductions_and_head_gradients(self):
        for events in (torch.zeros(4), torch.ones(4), self.batch["event"]):
            self.model.zero_grad()
            output = self.model(**dict(self.batch, event=events))
            density, survival = _independent_distribution(
                self.batch["duration"],
                output["log_weights"],
                output["means"],
                output["scales"],
            )
            expected = -torch.stack(
                [
                    density[i].log() if event else survival[i].log()
                    for i, event in enumerate(events)
                ]
            ).mean()
            torch.testing.assert_close(
                output["loss"].double(), expected, atol=1e-6, rtol=1e-6
            )
            output["loss"].backward()
            for head in (
                self.model.weight_head,
                self.model.mean_head,
                self.model.scale_head,
            ):
                gradients = [p.grad for p in head.parameters()]
                self.assertTrue(
                    all(g is not None and torch.isfinite(g).all() for g in gradients)
                )
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        singleton = SurvivalMDN(self.dataset, hidden_dims=(4,), n_components=1)
        singleton(**self.batch)["loss"].backward()
        for parameter in singleton.weight_head.parameters():
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))

    def test_zero_grid_gradients_and_differentiable_conversion(self):
        for values in ([0.0, 0.0], [0.0, 0.4, 1.0]):
            self.model.zero_grad()
            features = self.batch["features"].double().requires_grad_()
            grid = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            original_mode = self.model.training
            result = self.model.predict_survival(grid, features=features)
            self.assertEqual(self.model.training, original_mode)
            result.sum().backward()
            self.assertIsNotNone(features.grad)
            self.assertTrue(torch.isfinite(features.grad).all())
            self.assertTrue(torch.isfinite(grid.grad).all())
            self.assertTrue((grid.grad[grid == 0] == 0).all())
            if not any(values):
                torch.testing.assert_close(features.grad, torch.zeros_like(features))
                for parameter in self.model.parameters():
                    if parameter.numel():
                        self.assertIsNotNone(parameter.grad)
                        torch.testing.assert_close(
                            parameter.grad, torch.zeros_like(parameter)
                        )

    def test_dtypes_devices_and_state_dict_round_trip(self):
        for dtype in (torch.float32, torch.float64):
            model = self.model.to(dtype=dtype).eval()
            output = model(**self.batch)
            self.assertEqual(output["means"].dtype, dtype)
            output["loss"].backward()
        settings = {"hidden_dims": (8, 6), "n_components": 3}
        grid = torch.tensor([0.0, 0.5, 2.0])
        expected = self.model.predict_survival(grid, **self.batch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(
                {"settings": settings, "state_dict": self.model.state_dict()}, path
            )
            saved = torch.load(path, weights_only=True)
            restored = SurvivalMDN(_dataset(), **saved["settings"]).double().eval()
            restored.load_state_dict(saved["state_dict"])
        torch.testing.assert_close(
            expected, restored.predict_survival(grid, **self.batch)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_move_and_cpu_inputs(self):
        self.model.to("cuda")
        output = self.model(**self.batch)
        self.assertEqual(output["means"].device.type, "cuda")
        output["loss"].backward()
        curves = self.model.predict_survival(torch.tensor([0.0, 1.0]), **self.batch)
        self.assertEqual(curves.device.type, "cuda")

    def test_invalid_constructor_settings(self):
        cases = [
            {"hidden_dims": ()},
            {"hidden_dims": (0,)},
            {"hidden_dims": (-1,)},
            {"hidden_dims": (1.5,)},
            {"hidden_dims": (True,)},
            {"n_components": 0},
            {"n_components": -1},
            {"n_components": 1.5},
            {"n_components": True},
            {"activation": "tanh"},
            {"min_scale": 0},
            {"min_scale": -1},
            {"min_scale": float("nan")},
            {"min_scale": float("inf")},
            {"feature_key": ""},
            {"feature_key": "self"},
            {"duration_key": "self"},
            {"event_key": "self"},
            {"event_key": "duration"},
            {"duration_key": None},
        ]
        for kwargs in cases:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises((ValueError, TypeError)),
            ):
                SurvivalMDN(self.dataset, **kwargs)

    def test_extra_fields_and_compatible_custom_processor(self):
        class CompatibleProcessor:
            """Expose the same numeric contract without TensorProcessor inheritance."""

            def __init__(self, delegate):
                self.delegate = delegate

            def __getattr__(self, name):
                return getattr(self.delegate, name)

        samples = [
            {
                "patient_id": f"p{i}",
                "features": [0.2 + i, -0.5],
                "extra": [float(i)],
                "duration": 0.5 + i,
                "event": float(i % 2),
                "other_label": float(i),
            }
            for i in range(4)
        ]
        dataset = create_sample_dataset(
            samples,
            {"features": "tensor", "extra": "tensor"},
            {"duration": "tensor", "event": "tensor", "other_label": "tensor"},
            in_memory=True,
        )
        dataset.input_processors["features"] = CompatibleProcessor(
            dataset.input_processors["features"]
        )
        model = SurvivalMDN(dataset, hidden_dims=(4,), n_components=2).eval()
        batch = _batch(dataset)
        output = model(**batch)
        output["loss"].backward()
        changed = dict(
            batch,
            extra=torch.full((4, 1), float("nan")),
            other_label=torch.full((4,), float("nan")),
        )
        torch.testing.assert_close(output["loss"], model(**changed)["loss"])
        self.assertIsNone(model.mode)

    def test_invalid_dataset_metadata(self):
        for dataset in (None, []):
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                SurvivalMDN(dataset)
        for mutate in (
            lambda d: d.output_schema.pop("duration"),
            lambda d: d.input_processors.pop("features"),
            lambda d: d.input_processors.update(features=TensorProcessor()),
            lambda d: setattr(d.input_processors["features"], "dtype", torch.int64),
            lambda d: setattr(d.input_processors["features"], "_n_dim", 2),
            lambda d: setattr(d.input_processors["features"], "_feature_dim", 0),
            lambda d: setattr(d.output_processors["event"], "_feature_dim", 2),
            lambda d: setattr(d.output_processors["duration"], "_n_dim", 2),
        ):
            dataset = copy.copy(self.dataset)
            for attr in (
                "input_schema",
                "output_schema",
                "input_processors",
                "output_processors",
            ):
                setattr(dataset, attr, copy.deepcopy(getattr(self.dataset, attr)))
            mutate(dataset)
            with (
                self.subTest(mutate=mutate),
                self.assertRaises((ValueError, TypeError)),
            ):
                SurvivalMDN(dataset)

    def test_invalid_features_labels_and_prediction_grids(self):
        for features in (
            [[1.0, 2.0]],
            (torch.ones(1, 2),),
            torch.ones(2),
            torch.ones(1, 3),
            torch.ones(0, 2),
            torch.tensor([[float("nan"), 1.0]]),
            torch.tensor([[float("inf"), 1.0]]),
            torch.tensor([[1e300, 0.0]], dtype=torch.float64),
        ):
            with (
                self.subTest(features=features),
                self.assertRaises((ValueError, TypeError)),
            ):
                self.model(features=features)
        with self.assertRaises((ValueError, TypeError, KeyError)):
            self.model()
        for labels in ({"duration": torch.ones(4)}, {"event": torch.ones(4)}):
            with self.assertRaises((ValueError, TypeError)):
                self.model(features=self.batch["features"], **labels)
        for duration in (
            torch.zeros(4),
            -torch.ones(4),
            torch.full((4,), float("nan")),
            torch.full((4,), float("inf")),
            torch.ones(3),
            torch.ones(4, 2),
            [1.0] * 4,
            torch.full((4,), 1e-300, dtype=torch.float64),
        ):
            with (
                self.subTest(duration=duration),
                self.assertRaises((ValueError, TypeError)),
            ):
                self.model(**dict(self.batch, duration=duration))
        for event in (
            torch.full((4,), 0.5),
            torch.full((4,), 2),
            torch.full((4,), float("nan")),
            torch.full((4,), float("inf")),
            torch.ones(3),
            torch.ones(4, 2),
            [1.0] * 4,
            torch.full((4,), 1 + 1e-10, dtype=torch.float64),
        ):
            with self.subTest(event=event), self.assertRaises((ValueError, TypeError)):
                self.model(**dict(self.batch, event=event))
        for grid in (
            [],
            torch.tensor([]),
            torch.tensor(1.0),
            torch.ones(2, 2),
            torch.tensor([-1.0]),
            torch.tensor([-1e-300], dtype=torch.float64),
            torch.tensor([1.0 + 1e-10, 1.0], dtype=torch.float64),
            torch.tensor([1.0, 0.0]),
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
            torch.tensor([1]),
            torch.tensor([1e300], dtype=torch.float64),
        ):
            with self.subTest(grid=grid), self.assertRaises((ValueError, TypeError)):
                self.model.predict_survival(grid, features=self.batch["features"])
        self.model.half()
        with self.assertRaises((ValueError, TypeError)):
            self.model(features=self.batch["features"])

    def test_unused_likelihood_helpers_are_not_evaluated(self):
        for events, unused in (
            (torch.ones(4), "_log_survival"),
            (torch.zeros(4), "_log_density"),
        ):
            with patch(
                f"pyhealth.models.survival_mdn.{unused}",
                side_effect=AssertionError("Unused likelihood branch was evaluated"),
            ):
                output = self.model(**dict(self.batch, event=events))
                output["loss"].backward()

    def test_unused_density_cannot_poison_all_censored_backward(self):
        # Density overflows to -inf, but the selected survival log likelihood is 0.
        with torch.no_grad():
            self.model.mean_head.bias.fill_(1e30)
        output = self.model(**dict(self.batch, event=torch.zeros(4)))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_column_labels_and_booleans(self):
        output = self.model(**self.batch)
        columns = dict(
            self.batch,
            duration=self.batch["duration"][:, None],
            event=self.batch["event"][:, None].bool(),
        )
        torch.testing.assert_close(output["loss"], self.model(**columns)["loss"])

    def test_scale_floor_underflow_fails_clearly(self):
        model = SurvivalMDN(self.dataset, min_scale=1e-300)
        with torch.no_grad():
            model.scale_head.weight.zero_()
            model.scale_head.bias.fill_(-1000)
        with self.assertRaisesRegex(ValueError, "numerical range"):
            model(features=self.batch["features"])

    def test_nonfinite_parameters_and_selected_likelihood_fail_clearly(self):
        with torch.no_grad():
            self.model.mean_head.bias.fill_(float("inf"))
        with self.assertRaisesRegex(
            (ValueError, RuntimeError), "finite|numerical|range"
        ):
            self.model(features=self.batch["features"])
        with torch.no_grad():
            self.model.mean_head.bias.fill_(1e30)
        with self.assertRaisesRegex(
            (ValueError, RuntimeError), "finite|numerical|range"
        ):
            self.model(**self.batch)


class TestSurvivalIntegration(unittest.TestCase):
    def test_all_censored_and_all_observed_processed_datasets(self):
        for events in ((0.0,) * 4, (1.0,) * 4):
            dataset = _dataset(events)
            model = SurvivalMDN(dataset, hidden_dims=(4,), n_components=2)
            output = model(**_batch(dataset))
            output["loss"].backward()
            self.assertTrue(torch.isfinite(output["loss"]))
            self.assertTrue(
                all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                )
            )

    def test_short_deterministic_optimization_improves_loss(self):
        torch.manual_seed(23)
        dataset = _dataset()
        model = SurvivalMDN(dataset, hidden_dims=(8,), n_components=2)
        batch = _batch(dataset)
        initial = float(model(**batch)["loss"].detach())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        for _ in range(30):
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
        self.assertLess(float(model(**batch)["loss"].detach()), initial - 0.1)

    def test_trainer_evaluation_early_stop_and_best_state_restoration(self):
        torch.manual_seed(31)
        dataset = _dataset()
        model = SurvivalMDN(dataset, hidden_dims=(4,), n_components=2)
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        trainer = Trainer(model, device="cpu", enable_logging=False)
        self.assertIsNone(trainer.exp_path)
        self.assertTrue(math.isfinite(trainer.evaluate(loader)["loss"]))
        snapshots = []
        scores = iter((3.0, 1.0, 2.0, 4.0, 5.0))

        def controlled_validation(dataloader):
            self.assertIs(dataloader, loader)
            # Only validation scoring is controlled; train forward/backward is real.
            snapshots.append(
                {k: v.detach().clone() for k, v in model.state_dict().items()}
            )
            return {"loss": next(scores)}

        with patch.object(trainer, "evaluate", side_effect=controlled_validation):
            trainer.train(
                loader,
                loader,
                epochs=10,
                monitor="loss",
                monitor_criterion="min",
                patience=2,
                optimizer_params={"lr": 0.01},
                load_best_model_at_last=True,
            )
        self.assertEqual(len(snapshots), 4)
        self.assertTrue(
            any(not torch.equal(snapshots[0][k], snapshots[1][k]) for k in snapshots[0])
        )
        self.assertTrue(
            any(not torch.equal(snapshots[1][k], snapshots[3][k]) for k in snapshots[1])
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, snapshots[1][name], atol=0, rtol=0)
        self.assertTrue(math.isfinite(trainer.evaluate(loader)["loss"]))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "temporary.ckpt")
            trainer.save_ckpt(path)
            trainer.load_ckpt(path)


if __name__ == "__main__":
    unittest.main()
