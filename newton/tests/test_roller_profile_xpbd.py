# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate prescribed roller profiles in XPBD independently of the showcase motion."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.cable.example_roller_profiles import Example
from newton.geometry import RollerProfileCircle, RollerProfileEllipse, RollerProfileSector
from newton.solvers import SolverVBD, SolverXPBD
from newton.tests.test_roller_profile_vbd import _build, _load, test_profile_example_graph
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _endpoint_load(model, solver, state, body):
    """Sum the physical endpoint forces and moments at the current route."""
    points_l = solver.tendon_seg_attachment_l.numpy()
    points_r = solver.tendon_seg_attachment_r.numpy()
    directions = points_r - points_l
    lengths = np.linalg.norm(directions, axis=1)
    directions /= lengths[:, None]
    rest = solver.tendon_seg_rest_length.numpy()
    material = (lengths - rest) / solver.tendon_seg_active_compliance.numpy()
    if solver.tendon_sigmoid_ea_low > 0:
        strain = np.maximum(lengths.astype(np.float64) / rest - 1, 0)
        transition = np.tanh(
            (strain - solver.tendon_sigmoid_transition_strain) / solver.tendon_sigmoid_transition_width
        )
        ea = solver.tendon_sigmoid_ea_low * (1 + (solver.tendon_sigmoid_ea_ratio - 1) * (1 + transition) / 2)
        material = ea * strain
    tension = np.maximum(material + solver.tendon_seg_damping_tension.numpy(), 0)
    pose = wp.transform(*state.body_q.numpy()[body])
    com = np.array(wp.transform_point(pose, wp.vec3(*model.body_com.numpy()[body])))
    force, torque = np.zeros(3), np.zeros(3)
    bodies = model.tendon_link_body.numpy()
    for segment, direction in enumerate(directions):
        for link, point, sign in (
            (solver.tendon_seg_active_link_l.numpy()[segment], points_l[segment], 1),
            (solver.tendon_seg_active_link_r.numpy()[segment], points_r[segment], -1),
        ):
            if bodies[link] == body:
                load = sign * tension[segment] * direction
                force += load
                torque += np.cross(point - com, load)
    return np.concatenate((force, torque))


def test_profile_impulse(test, device):
    """Apply full endpoint moments once, including finite-friction and damping loads."""
    for profile in (RollerProfileCircle(0.05), RollerProfileEllipse(0.08, 0.02), RollerProfileSector(0.08, 2.2)):
        for mu, damping in ((0.0, 0.0), (0.3, 0.0), (0.3, 1.0)):
            _, model, solver, roller = _build(profile, device, mu=mu, damping=damping, solver_type=SolverXPBD)
            state = model.state()
            velocities = state.body_qd.numpy()
            velocities[roller, 5] = 0.3
            state.body_qd.assign(velocities)
            _load(model, solver, state, tensions=(8, 2))
            expected = _endpoint_load(model, solver, state, roller)
            deltas = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
            dt = 1e-3
            solver.joint_linear_relaxation = 1.0
            solver._solve_tendon_stretch(state.body_q, state.body_qd, deltas, dt)
            np.testing.assert_allclose(deltas.numpy()[roller] / dt, expected, atol=2e-4, rtol=3e-4)


def test_profile_step(test, device):
    """Match a small XPBD step to cable force and conserve the accepted route's material."""
    for profile in (RollerProfileCircle(0.05), RollerProfileEllipse(0.08, 0.02), RollerProfileSector(0.08, 2.2)):
        _, model, solver, roller = _build(profile, device, movable=True, solver_type=SolverXPBD)
        a, b = model.state(), model.state()
        _load(model, solver, a)
        load = _endpoint_load(model, solver, a, roller)
        total = np.sum(solver.tendon_seg_rest_length.numpy(), dtype=np.float64)
        total += np.sum(solver.tendon_profile_wrap_length.numpy(), dtype=np.float64)
        dt = 1e-4
        solver.step(a, b, None, None, dt)
        velocity = b.body_qd.numpy()[roller]
        np.testing.assert_allclose(velocity[:3], load[:3] * dt, rtol=0.01, atol=2e-6)
        np.testing.assert_allclose(velocity[3:], load[3:] * dt / 0.1, rtol=0.01, atol=2e-6)
        for _ in range(20):
            a, b = b, a
            solver.step(a, b, None, None, dt)
            measured = np.sum(solver.tendon_seg_rest_length.numpy(), dtype=np.float64)
            measured += np.sum(solver.tendon_profile_wrap_length.numpy(), dtype=np.float64)
            test.assertAlmostEqual(measured, total, delta=3e-6)
            test.assertFalse(np.any(solver.tendon_profile_wrap_status.numpy()))
            test.assertFalse(np.any(solver.tendon_profile_tangent_status.numpy()))
            tension = (
                np.maximum(solver.tendon_seg_length.numpy() - solver.tendon_seg_rest_length.numpy(), 0)
                / solver.tendon_seg_active_compliance.numpy()
            )
            np.testing.assert_allclose(solver.tendon_seg_material_tension.numpy(), tension, atol=2e-5)


def test_nonlinear_profile_loads(test, device):
    """Check nonlinear/damped loads against independent constitutive and endpoint sums."""
    for profile in (RollerProfileCircle(0.05), RollerProfileEllipse(0.08, 0.02), RollerProfileSector(0.08, 2.2)):
        for mu in (0.0, 0.3):
            for movable in (False, True):
                _, model, solver, roller = _build(
                    profile, device, mu=mu, damping=0.2, movable=movable, solver_type=SolverXPBD
                )
                solver.tendon_sigmoid_ea_low = 200.0
                solver.tendon_sigmoid_ea_ratio = 4.0
                solver.tendon_sigmoid_transition_strain = 0.02
                solver.tendon_sigmoid_transition_width = 0.008
                a = model.state()
                velocities = a.body_qd.numpy()
                velocities[roller, 5] = 0.3
                a.body_qd.assign(velocities)
                dt = 1e-4
                _load(model, solver, a, tensions=(8, 2), dt=dt)
                expected = _endpoint_load(model, solver, a, roller)
                deltas = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
                solver.joint_linear_relaxation = 1.0
                solver._solve_tendon_stretch(a.body_q, a.body_qd, deltas, dt)
                # In the small-step limit the implicit impulse approaches F*dt.
                # Measure the impulse directly: differencing float32 poses would
                # quantize the tiny angular displacement of the movable body.
                np.testing.assert_allclose(deltas.numpy()[roller] / dt, expected, rtol=5e-4, atol=3e-4)


class TestRollerProfileXPBD(unittest.TestCase):
    pass


def test_circle_low_iteration_cancellation(test, device):
    """Cancel frictionless circular moments without iterating independent span impulses."""
    with wp.ScopedDevice(device):
        e = Example(None, SimpleNamespace(solver="xpbd", iterations=32, substeps=10, friction=0.0))
        for _ in range(12):
            e.step()
            e.test_post_step()
        circle = e.roller_indices[0]
        test.assertLess(abs(float(e.state_0.body_qd.numpy()[circle, 4])), 1.0e-5)
        impulse = e.solver.tendon_seg_lambda.numpy()
        test.assertAlmostEqual(float(impulse[0]), float(impulse[1]), delta=1.0e-8)


def test_circle_authoring_equivalence(test, device):
    """Use the same mechanics for an explicit circle profile and a radius-authored roller."""
    for solver_type in (SolverXPBD, SolverVBD):
        for mu in (0.0, 0.3):
            simulations = []
            for radius_only in (False, True):
                _, model, solver, _ = _build(
                    RollerProfileCircle(0.05),
                    device,
                    mu=mu,
                    damping=0.2,
                    movable=True,
                    solver_type=solver_type,
                    radius_only=radius_only,
                )
                a, b = model.state(), model.state()
                rest = solver.tendon_seg_length.numpy() - np.array([8.0, 2.0]) * model.tendon_seg_compliance.numpy()
                solver.tendon_seg_rest_length.assign(rest)
                for _ in range(20):
                    solver.step(a, b, None, None, 1.0e-3)
                    a, b = b, a
                simulations.append((a.body_q.numpy(), a.body_qd.numpy(), solver.tendon_seg_rest_length.numpy()))
            for explicit, radius in zip(*simulations, strict=True):
                np.testing.assert_allclose(explicit, radius, rtol=1.0e-3, atol=2.0e-5)


def test_frictional_sector_transition(test, device):
    """Keep the hinge fixed through the showcase sector's stick/slip transition."""
    with wp.ScopedDevice(device):
        e = Example(None, SimpleNamespace(solver="xpbd", iterations=32, substeps=10, friction=0.3))
        for _ in range(65):
            e.step()
            e.test_post_step()


for fn in (
    test_profile_impulse,
    test_profile_step,
    test_nonlinear_profile_loads,
    test_circle_low_iteration_cancellation,
    test_circle_authoring_equivalence,
    test_frictional_sector_transition,
):
    add_function_test(TestRollerProfileXPBD, fn.__name__, fn, devices=get_test_devices())
add_function_test(
    TestRollerProfileXPBD,
    "test_profile_example_graph",
    test_profile_example_graph,
    devices=get_test_devices(),
    solver_name="xpbd",
)


if __name__ == "__main__":
    unittest.main()
