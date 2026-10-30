# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent references for damping engagement and finite-friction transfer."""

import unittest
from itertools import product

import numpy as np
import warp as wp

import newton
from newton._src.solvers.tendon_damping import linear_damped_tension, linear_damped_transfer
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def _pair_projection(cases: wp.array2d[float], output: wp.array[float], force: wp.array2d[float]):
    i = wp.tid()
    flow = linear_damped_transfer(
        cases[i, 0],
        cases[i, 1],
        cases[i, 2],
        cases[i, 3],
        cases[i, 4],
        cases[i, 5],
        cases[i, 6],
        cases[i, 7],
        cases[i, 8],
        cases[i, 9],
    )
    output[i] = flow
    force[i, 0] = linear_damped_tension(cases[i, 0] - flow, cases[i, 1], cases[i, 2], cases[i, 3])
    force[i, 1] = linear_damped_tension(cases[i, 4] + flow, cases[i, 5], cases[i, 6], cases[i, 7])


def _force(extension, compliance, raw, slope):
    return np.maximum(extension / compliance + np.minimum(raw, slope * np.maximum(extension, 0)), 0)


def test_linear_transfer_engagement_reference(test, device):
    """Check every damping branch, unloaded plateaus, bounds, and CUDA graph replay."""
    rng = np.random.default_rng(2976)
    count = 4096
    cases = np.zeros((count, 10), dtype=np.float32)
    cases[:, [0, 4]] = rng.uniform(-0.1, 0.2, (count, 2))
    cases[:, [1, 5]] = 10.0 ** rng.uniform(-9, -2, (count, 2))
    cases[:, [2, 6]] = rng.uniform(-50, 300, (count, 2))
    cases[:, [3, 7]] = 10.0 ** rng.uniform(-1, 6, (count, 2))
    cases[:, 8] = rng.uniform(1, 8, count)
    cases[:, 9] = rng.uniform(1.0e-6, 0.3, count)
    cases[::7, [2, 6]] = 0
    cases[::11, 8] = 1
    # Include a broad zero-force plateau: the first solution, not its far end,
    # must be chosen even when positive damping is large on the slack side.
    cases[:3] = [
        [0.02, 0.001, 200, 10000, -0.2, 0.002, 50, 10000, 1.7, 0.5],
        [0.02, 0.001, -10, 10000, -0.2, 0.002, 50, 10000, 1.7, 0.5],
        [0.02, 0.001, 200, 10000, 0.02, 0.002, 50, 10000, 1.7, 0.001],
    ]
    data = cases.astype(float)
    high, ch, bh, dh, low, cl, bl, dl, ratio, bound = data.T
    # The projection helper is only called for a violated capstan face.
    violated = _force(high, ch, bh, dh) > ratio * _force(low, cl, bl, dl)
    cases = cases[violated]
    high, ch, bh, dh, low, cl, bl, dl, ratio, bound = cases.astype(float).T
    lower = np.zeros(len(cases))
    upper = bound.copy()
    for _ in range(80):
        middle = (lower + upper) * 0.5
        residual = _force(high - middle, ch, bh, dh) - ratio * _force(low + middle, cl, bl, dl)
        lower = np.where(residual > 0, middle, lower)
        upper = np.where(residual > 0, upper, middle)
    with wp.ScopedDevice(device):
        values = wp.array(cases)
        output = wp.zeros(len(cases))
        force = wp.zeros((len(cases), 2))
        wp.launch(_pair_projection, len(cases), inputs=[values, output, force])
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                wp.launch(_pair_projection, len(cases), inputs=[values, output, force])
            wp.capture_launch(capture.graph)
        np.testing.assert_allclose(output.numpy(), upper, atol=4.0e-8, rtol=2.0e-6)
        test.assertTrue(np.all(force.numpy() >= 0))
        test.assertAlmostEqual(float(output.numpy()[0]), 0.02, delta=4.0e-8)
        test.assertEqual(float(force.numpy()[0, 0]), 0.0)
        test.assertEqual(float(force.numpy()[0, 1]), 0.0)


def test_implicit_engagement_reference(test, device):
    """Match backward Euler through slack, engagement, and taut loading/unloading."""
    dt = 1.0 / 120.0
    cases = (
        (0.25, -3.0),
        (0.25, 30.0),
        (0.25, 90.0),
        (0.25, 120.0),
        (1.2, -3.0),
        (1.2, 3.0),
        (1.2, -30.0),
        (1.001, -3.0),
    )
    with wp.ScopedDevice(device):
        for nonlinear, direct in product((False, True), repeat=2):
            options = {"tendon_material_direct": direct}
            if nonlinear:
                options.update(
                    tendon_sigmoid_ea_low=500.0,
                    tendon_sigmoid_ea_ratio=10.0,
                    tendon_sigmoid_transition_strain=0.01,
                    tendon_sigmoid_transition_width=0.003,
                )
            for start, velocity in cases:
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                anchor = builder.add_body(mass=0.0, is_kinematic=True)
                body = builder.add_body(xform=wp.transform(p=(start, 0, 0)), mass=1.0, inertia=wp.mat33(np.eye(3)))
                builder.add_tendon()
                builder.add_tendon_link(body=anchor, link_type=newton.TendonLinkType.ATTACHMENT)
                builder.add_tendon_link(
                    body=body,
                    link_type=newton.TendonLinkType.ATTACHMENT,
                    compliance=0.001,
                    damping=100.0,
                    rest_length=1.0,
                )
                builder.color()
                model = builder.finalize()
                # Use represented initial values in the independent scalar equation.
                initial = float(model.body_q.numpy()[body, 0])
                free = initial + dt * velocity
                lo, hi = 0.0, free
                for _ in range(80):
                    x = (lo + hi) * 0.5
                    extension = x - 1.0
                    elastic = 1000.0 * max(extension, 0.0)
                    if nonlinear:
                        ea = 500.0 * (1.0 + 4.5 * (1.0 + np.tanh((max(extension, 0.0) - 0.01) / 0.003)))
                        elastic = ea * max(extension, 0.0)
                    damped = 100.0 * min((x - initial) / dt, max(extension, 0.0) / dt)
                    residual = x - free + dt * dt * max(elastic + damped, 0.0)
                    if residual > 0.0:
                        hi = x
                    else:
                        lo = x
                expected = (lo + hi) * 0.5
                for name, solver_type in (("xpbd", newton.solvers.SolverXPBD), ("vbd", newton.solvers.SolverVBD)):
                    with test.subTest(solver=name, nonlinear=nonlinear, direct=direct, start=start, velocity=velocity):
                        solver = solver_type(model, iterations=32, **options)
                        state, output = model.state(), model.state()
                        state.body_qd.assign([[0, 0, 0, 0, 0, 0], [velocity, 0, 0, 0, 0, 0]])
                        solver.step(state, output, model.control(), None, dt)
                        solver.check_tendon_material()
                        test.assertAlmostEqual(float(output.body_q.numpy()[body, 0]), expected, delta=2.0e-6)
                        test.assertAlmostEqual(
                            float(output.body_qd.numpy()[body, 0]), (expected - initial) / dt, delta=2.0e-4
                        )


class TestTendonDamping(unittest.TestCase):
    pass


add_function_test(
    TestTendonDamping,
    "test_linear_transfer_engagement_reference",
    test_linear_transfer_engagement_reference,
    devices=get_test_devices(),
)
add_function_test(
    TestTendonDamping,
    "test_implicit_engagement_reference",
    test_implicit_engagement_reference,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    unittest.main()
