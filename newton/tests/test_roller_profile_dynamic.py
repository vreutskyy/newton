# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate dynamic circles in otherwise prescribed planar profile routes."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.geometry import RollerProfileCircle, RollerProfileEllipse, RollerProfileSector
from newton.solvers import SolverVBD, SolverXPBD
from newton.tests.test_roller_profile_routing import _reference_length
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build(
    device,
    solver_type,
    *,
    orientation=1,
    initially_active=False,
    explicit=True,
    reverse=False,
    free_end=False,
    copies=1,
    dynamic=True,
    iterations=2,
    circular_only=False,
):
    builder = newton.ModelBuilder(gravity=0.0)
    profiles = [
        None,
        RollerProfileEllipse(0.05, 0.025),
        RollerProfileCircle(0.025),
        RollerProfileSector(0.05, 2.2),
        None,
    ]
    if reverse:
        profiles[1], profiles[3] = profiles[3], profiles[1]
    if circular_only:
        profiles[1] = profiles[3] = RollerProfileCircle(0.05)
    centers = [(-0.4, 0.18), (-0.2, 0), (0, -0.045 if initially_active else 0.07), (0.2, 0), (0.4, 0.18)]
    route = []
    for copy in range(copies):
        builder.add_tendon()
        for i, (center, profile) in enumerate(zip(centers, profiles, strict=True)):
            angle = -orientation * math.pi / 2 if isinstance(profile, RollerProfileSector) else 0.0
            body = builder.add_body(
                xform=wp.transform(
                    p=(center[0] + 2 * copy, orientation * center[1], 0),
                    q=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), angle),
                ),
                is_kinematic=not (free_end and i == 4),
                mass=0.1,
                inertia=wp.mat33(np.eye(3) * 1e-4),
            )
            kwargs = {"profile": profile} if profile is not None else {}
            if isinstance(profile, RollerProfileCircle) and not explicit:
                kwargs = {"radius": profile.radius}
            builder.add_tendon_link(
                body,
                newton.TendonLinkType.ATTACHMENT if profile is None else newton.TendonLinkType.ROLLING,
                dynamic=dynamic and i == 2,
                orientation=orientation,
                compliance=1e-3,
                damping=2.0,
                mu=0.3,
                **kwargs,
            )
            route.append((body, (0, 0, 0), profile))
    builder.color()
    model = builder.finalize(device=device)
    return model, solver_type(model, iterations=iterations), route


def _material(solver, state, route, orientation, tendon=0):
    active = solver.tendon_link_active.numpy()
    poses = state.body_q.numpy()
    reference_poses = np.column_stack((poses[:, :2], 2 * np.arctan2(poses[:, 5], poses[:, 6])))
    start, end = solver.model.tendon_start.numpy()[tendon : tendon + 2]
    path = [route[i] for i in range(start, end) if active[i]]
    geometric_length = _reference_length(path, reference_poses, orientation)
    spans = solver.tendon_seg_active.numpy().astype(bool)
    stretch = solver.tendon_seg_length.numpy() - solver.tendon_seg_rest_length.numpy()
    segment_slice = slice(start - tendon, end - tendon - 1)
    return geometric_length - float(np.sum(stretch[segment_slice][spans[segment_slice]], dtype=np.float64))


def test_dynamic_circle_mixed_profiles(test, device):
    """Conserve material through repeated circle switches beside ellipses and sectors."""
    with wp.ScopedDevice(device):
        for solver_type in (SolverXPBD, SolverVBD):
            for orientation in (-1, 1):
                for initially_active in (False, True):
                    for explicit in (False, True):
                        with test.subTest(
                            solver=solver_type.__name__,
                            orientation=orientation,
                            active=initially_active,
                            explicit=explicit,
                        ):
                            model, solver, route = _build(
                                device,
                                solver_type,
                                orientation=orientation,
                                initially_active=initially_active,
                                explicit=explicit,
                                reverse=initially_active,
                            )
                            state, output = model.state(), model.state()
                            test.assertEqual(bool(solver.tendon_link_active.numpy()[2]), initially_active)
                            total = _material(solver, state, route, orientation)
                            test.assertAlmostEqual(float(solver.tendon_total_cable.numpy()[0]), total, delta=2e-7)
                            for frame, active in enumerate((True, True, False, False, True, False)):
                                poses = state.body_q.numpy()
                                poses[2, 1] = orientation * (-0.045 if active else 0.07)
                                for neighbor in (1, 3):
                                    base = model.body_q.numpy()[neighbor]
                                    angle = 2 * math.atan2(base[5], base[6]) + 0.1 * math.sin(frame)
                                    poses[neighbor, 3:] = (0, 0, math.sin(angle / 2), math.cos(angle / 2))
                                state.body_q.assign(poses)
                                solver.step(state, output, model.control(), None, 1e-3)
                                state, output = output, state
                                test.assertEqual(bool(solver.tendon_link_active.numpy()[2]), active)
                                test.assertFalse(np.any(solver.tendon_profile_tangent_status.numpy()))
                                test.assertFalse(np.any(solver.tendon_profile_wrap_status.numpy()))
                                test.assertAlmostEqual(_material(solver, state, route, orientation), total, delta=3e-6)
                                np.testing.assert_array_equal(solver.tendon_seg_active.numpy(), [1, 1, int(active), 1])
                                test.assertEqual(int(solver.tendon_link_cone_seg_l.numpy()[3]), 2 if active else 1)


def test_dynamic_profile_hysteresis(test, device):
    """Use the actual ellipse/sector bypass and retain the circle's radius-relative gap."""
    with wp.ScopedDevice(device):
        for solver_type in (SolverXPBD, SolverVBD):
            for orientation in (-1, 1):
                model, solver, _ = _build(device, solver_type, orientation=orientation)
                state = model.state()
                left = solver.tendon_seg_attachment_l.numpy()[1]
                right = solver.tendon_seg_attachment_r.numpy()[1]
                normal = orientation * np.cross([0, 0, 1], right - left)
                normal /= np.linalg.norm(normal)
                radius, tol = 0.025, solver.tendon_activation_tol
                for distance, expected in (
                    (radius * (1 - 0.5 * tol), False),
                    (radius * (1 - 1.5 * tol), True),
                    (radius * (1 - 0.5 * tol), True),
                    (radius * (1 + 0.5 * tol), False),
                ):
                    poses = state.body_q.numpy()
                    poses[2, :3] = 0.5 * (left + right) + distance * normal
                    state.body_q.assign(poses)
                    solver._update_tendon_link_active(model, state.body_q)
                    test.assertEqual(bool(solver.tendon_link_active.numpy()[2]), expected)


def test_dynamic_profiles_multiple_tendons(test, device):
    """Keep segment history and cone indices local to each independently switching route."""
    with wp.ScopedDevice(device):
        for solver_type in (SolverXPBD, SolverVBD):
            model, solver, route = _build(device, solver_type, copies=2)
            state, output = model.state(), model.state()
            totals = solver.tendon_total_cable.numpy()
            for active_pair in ((True, False), (False, True), (True, True), (False, False)) * 2:
                poses = state.body_q.numpy()
                for tendon, active in enumerate(active_pair):
                    poses[5 * tendon + 2, 1] = -0.045 if active else 0.07
                state.body_q.assign(poses)
                solver.step(state, output, model.control(), None, 1e-3)
                state, output = output, state
                for tendon, active in enumerate(active_pair):
                    test.assertEqual(bool(solver.tendon_link_active.numpy()[5 * tendon + 2]), active)
                    test.assertAlmostEqual(_material(solver, state, route, 1, tendon), totals[tendon], delta=3e-6)


def test_dynamic_circle_legacy_equivalence(test, device):
    """Match explicit-circle switching to the existing radius-only route under load."""
    with wp.ScopedDevice(device):
        for solver_type in (SolverXPBD, SolverVBD):
            cases = []
            for explicit in (False, True):
                model, solver, _ = _build(device, solver_type, explicit=explicit, circular_only=True, free_end=True)
                test.assertEqual(model.tendon_profile_routing, explicit)
                cases.append((model, solver, [model.state(), model.state()]))
            for active in (True, True, False, False) * 5:
                for model, solver, states in cases:
                    wp.launch(_move_circle, 1, inputs=[states[0].body_q, states[0].body_f, active], device=device)
                    solver.step(states[0], states[1], model.control(), None, 1e-3)
                    states.reverse()
                np.testing.assert_allclose(
                    cases[0][2][0].body_q.numpy(), cases[1][2][0].body_q.numpy(), atol=3e-6, rtol=0
                )
                np.testing.assert_allclose(
                    cases[0][1].tendon_seg_rest_length.numpy(),
                    cases[1][1].tendon_seg_rest_length.numpy(),
                    atol=5e-7,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    cases[0][1].tendon_seg_material_tension.numpy(),
                    cases[1][1].tendon_seg_material_tension.numpy(),
                    atol=1e-3,
                    rtol=0,
                )


@wp.kernel
def _move_circle(q: wp.array[wp.transform], forces: wp.array[wp.spatial_vector], active: bool):
    q[2] = wp.transform(wp.vec3(0.0, -0.045 if active else 0.07, 0.0), wp.quat_identity())
    forces[4] = wp.spatial_vector(0.0, 0.2, 0.0, 0.0, 0.0, 0.0)


def test_dynamic_profile_graph(test, device):
    """Replay activation/deactivation with a moving loaded endpoint, matching uncaptured steps."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires CUDA")
    with wp.ScopedDevice(device):
        for solver_type in (SolverXPBD, SolverVBD):
            graph_model, graph_solver, route = _build(device, solver_type, free_end=True)
            plain_model, plain_solver, _ = _build(device, solver_type, free_end=True)
            graph_states = [graph_model.state(), graph_model.state()]
            plain_states = [plain_model.state(), plain_model.state()]

            def cycle(model, solver, states):
                for i, active in enumerate((True, False)):
                    wp.launch(_move_circle, 1, inputs=[states[i].body_q, states[i].body_f, active], device=device)
                    solver.step(states[i], states[1 - i], model.control(), None, 1e-3)

            # Compile every path before capture, using only the plain instance.
            cycle(plain_model, plain_solver, plain_states)
            plain_model, plain_solver, _ = _build(device, solver_type, free_end=True)
            plain_states = [plain_model.state(), plain_model.state()]
            with wp.ScopedCapture(device=device) as capture:
                cycle(graph_model, graph_solver, graph_states)
            total = float(graph_solver.tendon_total_cable.numpy()[0])
            for _ in range(10):
                wp.capture_launch(capture.graph)
                cycle(plain_model, plain_solver, plain_states)
                np.testing.assert_allclose(graph_states[0].body_q.numpy(), plain_states[0].body_q.numpy(), atol=2e-7)
                for name in (
                    "tendon_link_active",
                    "tendon_seg_rest_length",
                    "tendon_seg_material_tension",
                    "tendon_profile_parameter_l",
                    "tendon_profile_parameter_r",
                ):
                    np.testing.assert_allclose(
                        getattr(graph_solver, name).numpy(), getattr(plain_solver, name).numpy(), atol=2e-7
                    )
                test.assertAlmostEqual(_material(graph_solver, graph_states[0], route, 1), total, delta=3e-6)
            test.assertGreater(
                np.linalg.norm(graph_states[0].body_q.numpy()[4, :3] - graph_model.body_q.numpy()[4, :3]), 1e-5
            )


class TestRollerProfileDynamic(unittest.TestCase):
    def test_non_circular_dynamic_rejected(self):
        """Keep non-circular activation out of the supported scope."""
        for profile in (RollerProfileEllipse(0.05, 0.025), RollerProfileSector(0.05, 2.2)):
            builder = newton.ModelBuilder()
            body = builder.add_body()
            builder.add_tendon()
            builder.add_tendon_link(body, newton.TendonLinkType.ATTACHMENT)
            with self.assertRaisesRegex(ValueError, "[Cc]ircular"):
                builder.add_tendon_link(body, newton.TendonLinkType.ROLLING, profile=profile, dynamic=True)


for fn in (
    test_dynamic_circle_mixed_profiles,
    test_dynamic_profile_hysteresis,
    test_dynamic_profile_graph,
    test_dynamic_profiles_multiple_tendons,
    test_dynamic_circle_legacy_equivalence,
):
    add_function_test(TestRollerProfileDynamic, fn.__name__, fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
