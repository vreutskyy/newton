# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check engagement-aware material history against independent scalar references."""

import itertools
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.tendon_alm_material import _alm_zero_force_limit
from newton._src.solvers.tendon_material_state import TendonMaterialState
from newton.tests.test_tendon_alm_cooperative import _cooperative, _scalar, _workspace
from newton.tests.test_tendon_material_slack import _reference as _slack_reference
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def _limits(state: TendonMaterialState, limits: wp.array[float], feasible: wp.array[int]):
    seg = wp.tid()
    valid, limit = _alm_zero_force_limit(
        state.alm, seg, state.ea_low, state.ea_ratio, state.transition_strain, state.transition_width
    )
    limits[seg] = limit
    feasible[seg] = int(valid)


def _force(extension, length, compliance, damping, viscous, multiplier, penalty, dt, nonlinear):
    """Evaluate the signed physical/ALM force without a frozen affine approximation."""
    if nonlinear:
        rest = max(length - extension, 1.0e-8)
        strain = max(extension, 0.0) / rest
        ea = 500.0 * (1.0 + 4.5 * (1.0 + np.tanh((strain - 0.01) / 0.003)))
        compliance = rest / ea
    metric = 1.0 + compliance * damping / dt
    damped = min(viscous, damping * max(extension, 0.0) / dt)
    numerator = extension + compliance * damped
    denominator = compliance
    if penalty > 0.0:
        numerator += metric * multiplier / penalty
        denominator += metric / penalty
    return numerator / denominator


def _arrays(state):
    """Read exactly the represented input values, promoted to reference precision."""
    return {
        name: getattr(state.alm, name).numpy().astype(float)
        for name in ("length", "physical_compliance", "damping", "damping_force", "multiplier", "penalty")
    }


def _span_force(data, seg, extension, dt, nonlinear):
    return _force(extension, *(data[name][seg] for name in data), dt, nonlinear)


def _pair_tensions(data, i, left, right, flow, dt, nonlinear):
    return (
        max(_span_force(data, 2 * i, left - flow, dt, nonlinear), 0.0),
        max(_span_force(data, 2 * i + 1, right + flow, dt, nonlinear), 0.0),
    )


def _endpoint(data, seg, dt, nonlinear):
    """Bisect the full force, retaining the physical minimum-rest-length bound."""
    bound = float(np.float32(data["length"][seg]) - np.float32(1.0e-6))
    if _span_force(data, seg, bound, dt, nonlinear) <= 0.0:
        return bound
    lower, upper = -1.0, bound
    for _ in range(60):
        if _span_force(data, seg, lower, dt, nonlinear) <= 0.0:
            break
        lower *= 2.0
    else:
        raise AssertionError("No zero-force lower interval in this test case")
    for _ in range(90):
        middle = 0.5 * (lower + upper)
        if _span_force(data, seg, middle, dt, nonlinear) > 0.0:
            upper = middle
        else:
            lower = middle
    return lower


def test_engagement_two_span_reference(test, device):
    """Select the first capstan solution, including a zero-force plateau."""
    cases = list(
        itertools.product(
            ((-0.3, 0.2), (0.2, -0.3), (0.2, 0.05), (0.05, 0.2), (-0.2, -0.3), (0.02, 0.02)),
            ((-100, -20), (100, 20), (0, 0), (100, -20), (-100, 20), (50, 500)),
            (1.0, 1.7, 4.0),
        )
    )
    for nonlinear in (False, True):
        counts = np.full(len(cases), 2, np.int32)
        state = _workspace(device, counts, 2, nonlinear=nonlinear)
        state.alm.dt = 0.01
        state.ids.assign(np.arange(2 * len(cases), dtype=np.int32))
        state.alm.length.fill_(1.0)
        state.alm.physical_compliance.assign(np.tile([0.001, 0.002], len(cases)))
        state.alm.damping.assign(np.tile([100, 20], len(cases)))
        state.alm.penalty.zero_()
        state.alm.multiplier.zero_()
        state.alm.reference.assign(np.array([e for e, _, _ in cases], np.float32).ravel())
        state.alm.damping_force.assign(np.array([b for _, b, _ in cases], np.float32).ravel())
        state.cap.assign(np.repeat([ratio for _, _, ratio in cases], 2))
        data = _arrays(state)
        reference = state.alm.reference.numpy().astype(float).reshape(-1, 2)
        caps = state.cap.numpy()[::2].astype(float)
        wp.launch(_scalar, len(cases), inputs=[state, wp.array(counts, dtype=int, device=device), 2], device=device)
        output = state.output.numpy().reshape(-1, 2)
        status = state.status.numpy()[::2]
        for i, (left, right) in enumerate(reference):
            with test.subTest(nonlinear=nonlinear, case=i):
                test.assertGreater(status[i], 0)

                tl, tr = _pair_tensions(data, i, left, right, 0.0, state.alm.dt, nonlinear)
                flow = 0.0
                if tl > caps[i] * tr or tr > caps[i] * tl:
                    direction = 1.0 if tl > caps[i] * tr else -1.0
                    lower = 0.0
                    upper = 1.0 - 1.0e-6 - (right if direction > 0 else left)
                    for _ in range(90):
                        middle = 0.5 * (lower + upper)
                        tl, tr = _pair_tensions(data, i, left, right, direction * middle, state.alm.dt, nonlinear)
                        residual = tl - caps[i] * tr if direction > 0 else tr - caps[i] * tl
                        if residual > 0.0:
                            lower = middle
                        else:
                            upper = middle
                    flow = direction * upper
                np.testing.assert_allclose(output[i], [left - flow, right + flow], atol=2.0e-5, rtol=0.0)


def test_engagement_slack_chain(test, device, capture=False):
    """Minimize net flow with nonlinear damping, dual history, bounds, and packed IDs."""
    if capture and not device.is_cuda:
        test.skipTest("CUDA graph capture requires CUDA")
    counts = np.array([1, 2, 5, 17, 31, 32], np.int32)
    stride = 37
    for nonlinear, stored_dual in itertools.product((False, True), (False, True)):
        state = _workspace(device, counts, stride, nonlinear=nonlinear)
        state.alm.dt = 0.01
        size = len(counts) * stride
        state.cap.fill_(1.7)
        lengths = np.ones(size, np.float32)
        lengths[::11] = 3.0e-6  # Some unloading endpoints are clipped by minimum rest length.
        state.alm.length.assign(lengths)
        state.alm.physical_compliance.assign(np.resize([0.001, 0.002], size))
        state.alm.damping.assign(np.resize([100.0, 20.0], size))
        state.alm.damping_force.assign(np.resize([-100.0, 20.0, 0.0], size))
        state.alm.penalty.fill_(10000.0 if stored_dual else 0.0)
        state.alm.multiplier.fill_(3.0 if stored_dual else 0.0)
        data = _arrays(state)
        ids = state.ids.numpy()
        rng = np.random.default_rng(1817)
        reference = np.full(size, -0.1, np.float32)
        expected = []
        for component, count in enumerate(counts):
            row = component * stride
            limits = np.array([_endpoint(data, seg, state.alm.dt, nonlinear) for seg in ids[row : row + count]])
            deficit = rng.uniform(-0.04, 0.04, count)
            deficit[-1] -= max(deficit.sum(), 0.0) + 0.02
            reference[row : row + count] = limits + deficit
            target, _ = _slack_reference(reference[row : row + count], limits)
            expected.append(target)
        state.alm.reference.assign(reference)
        device_counts = wp.array(counts, dtype=int, device=device)
        kernel, dim = (_cooperative, 32 * len(counts)) if device.is_cuda else (_scalar, len(counts))
        wp.launch(kernel, dim, inputs=[state, device_counts, stride], device=device, block_dim=32)
        if capture:
            with wp.ScopedCapture(device=device) as graph:
                wp.launch(kernel, dim, inputs=[state, device_counts, stride], device=device, block_dim=32)
            for _ in range(2):
                wp.capture_launch(graph.graph)
        output = state.output.numpy()
        status = state.status.numpy()
        force = state.alm.force.numpy()
        for component, count in enumerate(counts):
            row = component * stride
            with test.subTest(nonlinear=nonlinear, stored_dual=stored_dual, count=count):
                test.assertIn(status[row], (3, 4))
                np.testing.assert_allclose(output[row : row + count], expected[component], atol=2.0e-6, rtol=0.0)
                np.testing.assert_array_equal(force[row : row + count], 0.0)
                test.assertAlmostEqual(
                    output[row : row + count].astype(float).sum(),
                    reference[row : row + count].astype(float).sum(),
                    delta=2.0e-6,
                )
        # Reusing an already feasible allocation must not erode its history.
        state.alm.reference.assign(output)
        wp.launch(kernel, dim, inputs=[state, device_counts, stride], device=device, block_dim=32)
        for component, count in enumerate(counts):
            row = component * stride
            np.testing.assert_array_equal(state.output.numpy()[row : row + count], output[row : row + count])


def test_engagement_zero_endpoint(test, device):
    """Bound the true zero-force interval across damping, dual history, and span scales."""
    cases = list(
        itertools.product((1.0e-7, 3.0e-6, 0.01, 1.0), (-10000.0, -100.0, 0.0, 20.0), (0.0, 100.0), (0.0, 3.0, 1000.0))
    )
    for nonlinear in (False, True):
        count = len(cases)
        state = _workspace(device, np.ones(count, np.int32), 1, nonlinear=nonlinear)
        state.alm.dt = 0.01
        state.alm.length.assign([case[0] for case in cases])
        state.alm.damping_force.assign([case[1] for case in cases])
        state.alm.damping.assign([case[2] for case in cases])
        state.alm.multiplier.assign([case[3] for case in cases])
        state.alm.penalty.fill_(1000.0)
        limits, feasible = wp.empty(count, device=device), wp.empty(count, dtype=int, device=device)
        wp.launch(_limits, count, inputs=[state, limits, feasible], device=device)
        actual, valid, data = limits.numpy(), feasible.numpy(), _arrays(state)
        for seg in range(count):
            with test.subTest(nonlinear=nonlinear, case=cases[seg]):
                u = data["multiplier"][seg] / data["penalty"][seg]
                b = min(data["damping_force"][seg], 0.0) + u * data["damping"][seg] / state.alm.dt
                ea_zero = 500.0 * (1.0 + 4.5 * (1.0 + np.tanh(-0.01 / 0.003)))
                if nonlinear and b >= ea_zero:
                    test.assertEqual(valid[seg], 0)
                    continue
                target = _endpoint(data, seg, state.alm.dt, nonlinear)
                test.assertEqual(valid[seg], 1)
                test.assertAlmostEqual(float(actual[seg]), target, delta=3.0e-6 * max(abs(target), 1.0e-6))


def test_engagement_slack_graph(test, device):
    """Preserve the least-transfer history under repeated CUDA graph replay."""
    test_engagement_slack_chain(test, device, capture=True)


class TestTendonALMSlack(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(
        TestTendonALMSlack, "test_two_span_reference", test_engagement_two_span_reference, devices=[_device]
    )
    add_function_test(TestTendonALMSlack, "test_slack_chain", test_engagement_slack_chain, devices=[_device])
    add_function_test(TestTendonALMSlack, "test_slack_graph", test_engagement_slack_graph, devices=[_device])
    add_function_test(TestTendonALMSlack, "test_zero_endpoint", test_engagement_zero_endpoint, devices=[_device])


if __name__ == "__main__":
    unittest.main()
