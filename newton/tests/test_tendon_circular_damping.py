# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the circular penalty formulation against engagement-aware damping."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import Axis, TendonLinkType
from newton.tests.test_tendon_capstan import build_kinematic_rolling_transport
from newton.tests.test_tendon_vbd import _make_tendon_vbd_solver, _set_serial_body_coloring
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_damped_tendon_matches_unilateral_force_law(test, device):
    """Match the implicit unilateral spring-plus-damper force on taut and slack spans.

    Motion that only takes up slack must not contribute positive damping.
    The implicit solution remains continuous when the cable becomes taut.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor = builder.add_body(mass=0.0, is_kinematic=True)
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.25, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_tendon()
        builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-3,
            damping=100.0,
            rest_length=1.0,
        )
        builder.color()
        model = builder.finalize()
        dt = 1.0 / 120.0
        for solver_name in ("xpbd", "vbd"):
            # 120 m/s ends the step taut (free length 1.25 m > rest 1.0 m) and exercises the active branch.
            for velocity in (-3.0, 0.0, 3.0, 30.0, 120.0):
                with test.subTest(solver=solver_name, velocity=velocity):
                    if solver_name == "xpbd":
                        solver = newton.solvers.SolverXPBD(model, iterations=20, joint_linear_relaxation=1.0)
                    else:
                        solver = _make_tendon_vbd_solver(model)
                    state_0, state_1 = model.state(), model.state()
                    state_0.body_qd.assign(
                        [wp.spatial_vector(0.0), wp.spatial_vector(velocity, 0.0, 0.0, 0.0, 0.0, 0.0)]
                    )
                    solver.step(state_0, state_1, model.control(), None, dt)

                    # The step starts slack: damping acts only on extension
                    # beyond rest length, not on the preceding slack travel.
                    free_length = 0.25 + dt * velocity
                    stiffness = 1.0 / 1.0e-3 + 100.0 / dt
                    tension = max(stiffness * (free_length - 1.0), 0.0) / (1.0 + dt * dt * stiffness)
                    expected_velocity = velocity - dt * tension
                    test.assertAlmostEqual(float(state_1.body_qd.numpy()[body][0]), expected_velocity, delta=1.0e-4)
                    expected_length = free_length - dt * dt * tension
                    test.assertAlmostEqual(float(state_1.body_q.numpy()[body][0]), expected_length, delta=1.0e-5)
                    test.assertAlmostEqual(
                        float(solver.tendon_seg_material_tension.numpy()[0]),
                        max(expected_length - 1.0, 0.0) / 1.0e-3,
                        delta=1.0e-2,
                    )
                    reported_lambda = float(solver.tendon_seg_lambda.numpy()[0])
                    if solver_name == "vbd":
                        test.assertEqual(reported_lambda, 0.0, "VBD must not report a synthetic XPBD multiplier")
                    else:
                        test.assertAlmostEqual(-reported_lambda / dt, tension, delta=1.0e-3)


def test_vbd_rolling_diagnostics_match_material_projection(test, device):
    """Report the same damped tension used by the accepted native material solve.

    A spinning, loaded roller exercises the contact-point length rate that
    differs from the legacy spin-removed rate. Publishing diagnostics must
    not change that force split after the material state has been accepted.
    """
    with wp.ScopedDevice(device):
        dt = 1.0 / 60.0
        for direct, nonlinear in ((False, False), (True, False), (False, True), (True, True)):
            # Exercise sliding and sticking inside the direct solver's
            # documented capstan-ratio range [1, 4].
            for friction in (0.25, 1.0):
                with test.subTest(direct=direct, friction=friction, nonlinear=nonlinear):
                    model, pulley = build_kinematic_rolling_transport(mu=friction, pre_stretch=0.08)
                    model.tendon_seg_compliance.fill_(1.0e-3)
                    model.tendon_seg_damping.fill_(2.0)
                    _set_serial_body_coloring(model)
                    solver = newton.solvers.SolverVBD(
                        model, tendon_material_direct=direct, tendon_sigmoid_ea_low=500.0 if nonlinear else 0.0
                    )
                    test.assertLessEqual(float(np.max(solver.tendon_link_cap_ratio.numpy())), 4.0)
                    state = model.state()
                    solver.body_q_prev.assign(state.body_q)
                    previous_pose = wp.clone(state.body_q)
                    solver._snapshot_tendon_step_state()

                    poses = state.body_q.numpy().copy()
                    angle = 0.1
                    poses[pulley, 3:] = [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]
                    state.body_q.assign(poses)
                    solver._finalize_tendon_routing(state, dt)
                    solver.check_tendon_material()
                    expected_damping = solver.tendon_seg_damping_tension.numpy().copy()
                    expected_total = solver.tendon_material_modes.numpy()["tension"].copy()
                    test.assertGreater(float(np.max(np.abs(expected_damping))), 0.1)
                    test.assertTrue(np.all(expected_total > 0.0))

                    def publish_diagnostics(solver, state, previous_pose):
                        wp.copy(solver.body_q_prev, previous_pose)
                        solver._snapshot_tendon_segment_length_reference(state.body_q, dt)
                        # Rigid finalization overwrites the pose history before
                        # publishing diagnostics in an actual solver step.
                        wp.copy(solver.body_q_prev, state.body_q)
                        solver._update_tendon_segment_diagnostics(state, dt)

                    publish_diagnostics(solver, state, previous_pose)
                    if device.is_cuda:
                        with wp.ScopedCapture(device=device) as capture:
                            publish_diagnostics(solver, state, previous_pose)
                        wp.capture_launch(capture.graph)
                    reported_damping = solver.tendon_seg_damping_tension.numpy()
                    reported_total = np.maximum(solver.tendon_seg_material_tension.numpy() + reported_damping, 0.0)
                    np.testing.assert_allclose(reported_damping, expected_damping, atol=1.0e-3, rtol=0)
                    np.testing.assert_allclose(reported_total, expected_total, atol=1.0e-3, rtol=0)


def test_vbd_diagnostics_preserve_sub_ulp_extension(test, device):
    """Preserve a stiff span's projected extension below the length resolution."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        bodies = [
            builder.add_body(xform=wp.transform(p=wp.vec3(*position)), mass=0.0, is_kinematic=True)
            for position in ((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        ]
        builder.add_tendon()
        builder.add_tendon_link(body=bodies[0], link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=bodies[1],
            link_type=int(TendonLinkType.PINHOLE),
            mu=0.25,
            compliance=1.0e-8,
            rest_length=1.0,
        )
        builder.add_tendon_link(
            body=bodies[2],
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-3,
            rest_length=0.99,
        )
        builder.color()
        model = builder.finalize()
        dt = 1.0 / 60.0
        for direct in (False, True):
            with test.subTest(direct=direct):
                with test.assertWarnsRegex(UserWarning, "compliance per unit length varies"):
                    solver = newton.solvers.SolverVBD(model, tendon_material_direct=direct)
                state = model.state()
                solver.body_q_prev.assign(state.body_q)
                initial_rest = solver.tendon_seg_rest_length.numpy().copy()
                solver._snapshot_tendon_step_state()
                solver._finalize_tendon_routing(state, dt)
                solver.check_tendon_material()
                compliance = solver.tendon_seg_active_compliance.numpy().astype(np.float64)
                ratio = float(solver.tendon_link_cap_ratio.numpy()[1])
                extension = 2.0 - float(np.sum(initial_rest.astype(np.float64)))
                # The right span starts loaded and slips toward the left span
                # until T_right = ratio * T_left, conserving total material.
                left_tension = extension / (compliance[0] + ratio * compliance[1])
                expected = [left_tension, ratio * left_tension]
                projected = solver.tendon_material_modes.numpy()["tension"].copy()
                np.testing.assert_allclose(projected, expected, atol=1.0e-3, rtol=0)
                test.assertGreater(float(solver.tendon_seg_stretch.numpy()[0]), 0.0)
                test.assertLess(float(solver.tendon_seg_stretch.numpy()[0]), float(np.spacing(np.float32(1.0))))

                def publish(solver, state):
                    solver._snapshot_tendon_segment_length_reference(state.body_q, dt)
                    solver._update_tendon_segment_diagnostics(state, dt)

                publish(solver, state)
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as capture:
                        publish(solver, state)
                    wp.capture_launch(capture.graph)
                reported = np.maximum(
                    solver.tendon_seg_material_tension.numpy() + solver.tendon_seg_damping_tension.numpy(), 0.0
                )
                np.testing.assert_allclose(reported, expected, atol=1.0e-3, rtol=0)
                np.testing.assert_allclose(reported, projected, atol=1.0e-4, rtol=0)


def test_vbd_segment_alm_preserves_spin_removed_diagnostics(test, device):
    """Keep per-segment ALM's existing damping semantics for a prescribed spin."""
    with wp.ScopedDevice(device):
        model, pulley = build_kinematic_rolling_transport(mu=1.0, pre_stretch=0.08)
        model.tendon_seg_compliance.fill_(1.0e-3)
        model.tendon_seg_damping.fill_(2.0)
        _set_serial_body_coloring(model)
        solver = newton.solvers.SolverVBD(
            model, tendon_material_direct=True, tendon_alm=True, tendon_alm_per_segment=True
        )
        state_0, state_1 = model.state(), model.state()
        dt = 1.0 / 60.0
        solver.step(state_0, state_1, model.control(), None, dt)
        state_0, state_1 = state_1, state_0
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        poses = state_0.body_q.numpy().copy()
        angle = 0.1
        poses[pulley, 3:] = [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]
        state_0.body_q.assign(poses)
        solver.step(state_0, state_1, model.control(), None, dt)
        solver.check_tendon_material()

        test.assertTrue(solver._tendon_alm_rows)
        test.assertTrue(solver._tendon_material_state.legacy_alm_damping)
        test.assertGreater(float(np.max(np.abs(solver.tendon_seg_rest_length.numpy() - initial_rest))), 1.0e-4)
        np.testing.assert_allclose(solver.tendon_seg_damping_tension.numpy(), 0.0, atol=1.0e-4, rtol=0)


class TestTendonCircularDamping(unittest.TestCase):
    pass


for device in get_test_devices():
    add_function_test(
        TestTendonCircularDamping,
        "test_damped_tendon_matches_unilateral_force_law",
        test_damped_tendon_matches_unilateral_force_law,
        devices=[device],
    )
    add_function_test(
        TestTendonCircularDamping,
        "test_vbd_rolling_diagnostics_match_material_projection",
        test_vbd_rolling_diagnostics_match_material_projection,
        devices=[device],
    )
    add_function_test(
        TestTendonCircularDamping,
        "test_vbd_diagnostics_preserve_sub_ulp_extension",
        test_vbd_diagnostics_preserve_sub_ulp_extension,
        devices=[device],
    )
    add_function_test(
        TestTendonCircularDamping,
        "test_vbd_segment_alm_preserves_spin_removed_diagnostics",
        test_vbd_segment_alm_preserves_spin_removed_diagnostics,
        devices=[device],
    )

if __name__ == "__main__":
    unittest.main()
