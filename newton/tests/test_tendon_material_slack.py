# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check zero-force material allocation against an independent decimal reference."""

import unittest
from decimal import Decimal, localcontext

import numpy as np
import warp as wp

from newton._src.solvers.tendon_damping import linear_damped_tension
from newton._src.solvers.tendon_material_slack import TendonSlackBlock, project_slack_component
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _project(
    starts: wp.array[int],
    active: wp.array[int],
    length: wp.array[float],
    compliance: wp.array[float],
    damping: wp.array[float],
    stretch: wp.array[float],
    links: wp.array[int],
    transfer: wp.array[float],
    slack: wp.array[int],
    blocks: wp.array[TendonSlackBlock],
):
    """Project disjoint components with potentially inactive slots."""
    i = wp.tid()
    project_slack_component(
        starts[i], starts[i + 1] - 1, active, length, compliance, damping, stretch, links, transfer, slack, blocks
    )


@wp.kernel
def _force(stretch: wp.array[float], compliance: wp.array[float], damping: wp.array[float], output: wp.array[float]):
    """Evaluate the same zero endpoint used by material sweeps and VBD."""
    i = wp.tid()
    output[i] = linear_damped_tension(stretch[i], compliance[i], damping[i], 1.0)


def _reference(stretch, limits):
    """Compute a high-precision minimum-transfer projection independently."""
    with localcontext() as context:
        context.prec = 80
        limits = [Decimal.from_float(float(v)) for v in limits]
        deficits = [Decimal.from_float(float(v)) - limit for v, limit in zip(stretch, limits, strict=True)]
        capacity = -sum(deficits, Decimal(0))
        assert capacity >= 0
        prefix = Decimal(0)
        values, stack = [], []
        for deficit in deficits[:-1]:
            prefix += deficit
            values.append(-prefix)
            stack.append([-prefix, 1])
            while len(stack) > 1 and stack[-2][0] / stack[-2][1] > stack[-1][0] / stack[-1][1]:
                left, right = stack[-2:]
                stack[-2:] = [[left[0] + right[0], left[1] + right[1]]]
        fit = [min(max(total / count, Decimal(0)), capacity) for total, count in stack for _ in range(count)]
        output = [
            float(limit - (right - left))
            for limit, left, right in zip(limits, [Decimal(0), *fit], [*fit, capacity], strict=True)
        ]
        transfer = [float(z - y) for z, y in zip(fit, values, strict=True)]
        return np.array(output), np.array(transfer)


def _run(device, stretch, length, compliance, damping, *, active=None, starts=None, capture=False):
    """Run a projection without converting inactive spans into material links."""
    n = len(stretch)
    active = np.ones(n, dtype=np.int32) if active is None else active
    starts = [0, n] if starts is None else starts
    arrays = [
        wp.array(v, dtype=dtype, device=device)
        for v, dtype in (
            (starts, int),
            (active, int),
            (length, float),
            (compliance, float),
            (damping, float),
            (stretch, float),
            (np.arange(1, n + 1), int),
        )
    ]
    transfer = wp.zeros(n + 1, device=device)
    slack = wp.zeros(n, dtype=int, device=device)
    blocks = wp.zeros(n, dtype=TendonSlackBlock, device=device)
    inputs = [*arrays, transfer, slack, blocks]
    wp.launch(_project, dim=len(starts) - 1, inputs=inputs, device=device)
    if capture:
        with wp.ScopedCapture(device=device) as graph:
            wp.launch(_project, dim=len(starts) - 1, inputs=inputs, device=device)
        for _ in range(2):
            arrays[5].assign(np.asarray(stretch, dtype=np.float32))
            slack.zero_()
            transfer.zero_()
            wp.capture_launch(graph.graph)
    force = wp.empty(n, device=device)
    wp.launch(_force, dim=n, inputs=[arrays[5], arrays[3], arrays[4], force], device=device)
    return arrays[5].numpy(), transfer.numpy(), slack.numpy(), force.numpy()


def test_slack_projection_reference(test, device):
    """Preserve bounds and minimum transfer across scales, damping, and long routes."""
    rng = np.random.default_rng(81732)
    cases = []
    for n in (1, 2, 5, 16, 32, 64, 129):
        for scale in (1e-6, 1.0, 1e6):
            for damped in (False, True):
                compliance = np.full(n, 0.1, dtype=np.float32)
                damping = (rng.uniform(-0.5, 0.5, n) * scale).astype(np.float32) if damped else np.zeros(n, np.float32)
                length = np.full(n, 2 * scale, dtype=np.float32)
                limit = np.minimum(-compliance * np.minimum(damping, 0.0), length - np.float32(1e-6))
                deficit = (rng.uniform(-1, 1, n) * scale).astype(np.float32)
                deficit[-1] -= np.float32(max(float(deficit.astype(float).sum()), 0) + 0.1 * scale)
                cases.append((length, compliance, damping, limit + deficit))
    for values in ([1e10, 1, -1e10, -2], [1e20, 1, -1e20, -2], [1, -1, 0, 0], [0, -1, 1, 0]):
        n = len(values)
        cases.append(
            (np.full(n, 2, np.float32), np.ones(n, np.float32), np.zeros(n, np.float32), np.array(values, np.float32))
        )
    for index, (length, compliance, damping, stretch) in enumerate(cases):
        with test.subTest(case=index, n=len(stretch)):
            limit = np.minimum(-compliance * np.minimum(damping, 0.0), length - np.float32(1e-6))
            expected, expected_q = _reference(stretch, limit)
            output, transfer, slack, force = _run(device, stretch, length, compliance, damping)
            np.testing.assert_array_equal(slack, 1)
            test.assertTrue(np.all(output <= limit))
            np.testing.assert_array_equal(force, 0)
            for actual, reference in ((output, expected), (transfer[1:-1], expected_q)):
                scale = max(float(np.max(np.abs(reference), initial=0)), 1e-30)
                np.testing.assert_allclose(actual, reference, rtol=5e-7, atol=scale * 5e-7)
            # Compare the small conserved residual, not the much larger cancelling entries.
            with localcontext() as context:
                context.prec = 80
                before = sum((Decimal.from_float(float(v)) for v in stretch), Decimal(0))
                after = sum((Decimal.from_float(float(v)) for v in output), Decimal(0))
                test.assertLessEqual(abs(float(after - before)), max(float(np.abs(expected).sum()), 1e-30) * 1e-6)


def test_slack_projection_components(test, device):
    """Keep taut components and inactive slots unchanged, including during graph replay."""
    stretch = np.array([0.4, 777, -0.5, 0.1, 0.2, -0.1, -0.2], np.float32)
    length = np.full(7, 2, np.float32)
    compliance = np.full(7, 0.1, np.float32)
    damping = np.zeros(7, np.float32)
    active = np.array([1, 0, 1, 1, 1, 1, 1], np.int32)
    output, transfer, slack, force = _run(
        device, stretch, length, compliance, damping, active=active, starts=[0, 3, 5, 7], capture=device.is_cuda
    )
    np.testing.assert_allclose(output, [0, 777, -0.1, 0.1, 0.2, -0.1, -0.2], atol=2e-8)
    np.testing.assert_array_equal(slack, [1, 0, 1, 0, 0, 1, 1])
    np.testing.assert_allclose(transfer, [0, 0.4, 0, 0, 0, 0, 0, 0], atol=1e-8)
    np.testing.assert_array_equal(force[[0, 2, 5, 6]], 0)


class TestTendonMaterialSlack(unittest.TestCase):
    """Test exact zero-force material projection."""


for name, function in (
    ("test_slack_projection_reference", test_slack_projection_reference),
    ("test_slack_projection_components", test_slack_projection_components),
):
    add_function_test(TestTendonMaterialSlack, name, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
