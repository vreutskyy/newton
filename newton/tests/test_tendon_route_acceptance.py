# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check supported overlapping routes and latched invalid accepted wraps."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_tendon_capstan import (
    _oriented_route_material_length,
    _oriented_wrap_angle_at_link,
    build_interpenetrating_neighbor_route,
    build_oriented_dynamic_route,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_same_winding_overlap_activation(test, device):
    """Equal-radius external tangent remains x=-radius despite circle overlap."""
    for solver_type in (newton.solvers.SolverVBD, newton.solvers.SolverXPBD):
        model, upper, link = build_interpenetrating_neighbor_route(device)
        model.body_color_groups = [wp.array([i], dtype=int, device=device) for i in range(model.body_count)]
        solver = solver_type(model, iterations=1)
        poses = model.body_q.numpy()
        poses[upper, 2] = -0.25
        candidate = int(model.tendon_link_body.numpy()[link])
        poses[candidate, 2] = -0.375
        # The external tangent is x=-0.3. Candidate radius is 0.1.
        for initial in (False, True):
            for x, expected in ((-0.25, True), (0.0, False)):
                active = solver.tendon_link_active.numpy()
                active[link] = initial
                solver.tendon_link_active.assign(active)
                poses[candidate, 0] = x
                model.body_q.assign(poses)
                solver._update_tendon_link_active(model, model.body_q)
                test.assertEqual(bool(solver.tendon_link_active.numpy()[link]), expected)


def test_initial_invalid_route_check(test, device):
    """An invalid initial route cannot become a successful direct solve."""
    for solver_type in (newton.solvers.SolverVBD, newton.solvers.SolverXPBD):
        model, _, link = build_oriented_dynamic_route(1, device, dynamic=False)
        model.body_color_groups = [wp.array([i], dtype=int, device=device) for i in range(model.body_count)]
        model.tendon_seg_compliance.fill_(0.01)
        solver = solver_type(model, iterations=1, tendon_material_direct=True)
        with test.assertRaisesRegex(RuntimeError, f"Unsupported accepted tendon route: tendon 0, ROLLING link {link}"):
            solver.check_tendon_material()


def test_accepted_invalid_route_check(test, device):
    """Intermediate wraps do not latch; accepted wraps persist across graph replay."""
    for capture in (False, True) if device.is_cuda else (False,):
        model, body, link = build_oriented_dynamic_route(1, device, dynamic=False)
        model.body_color_groups = [wp.array([i], dtype=int, device=device) for i in range(model.body_count)]
        model.tendon_seg_compliance.fill_(0.01)
        poses = model.body_q.numpy()
        poses[body, 0] = 0.0
        model.body_q.assign(poses)
        solver = newton.solvers.SolverVBD(
            model, iterations=1, tendon_material_direct=True, tendon_alm=True, tendon_alm_per_segment=True
        )
        state = model.state()
        solver.body_q_prev.assign(state.body_q)
        solver._snapshot_tendon_step_state()
        solver._update_tendon_routing(state, 1.0 / 1200.0, False)
        poses[body, 0] = 0.15
        state.body_q.assign(poses)
        solver._update_tendon_routing(state, 1.0 / 1200.0, False)
        solver.check_tendon_material()
        if capture:
            with wp.ScopedCapture(device=device) as captured:
                solver._finalize_tendon_routing(state, 1.0 / 1200.0)
            wp.capture_launch(captured.graph)
        else:
            solver._finalize_tendon_routing(state, 1.0 / 1200.0)
        with test.assertRaisesRegex(RuntimeError, f"Unsupported accepted tendon route: tendon 0, ROLLING link {link}"):
            solver.check_tendon_material()
        poses[body, 0] = 0.0
        state.body_q.assign(poses)
        solver._finalize_tendon_routing(state, 1.0 / 1200.0)
        with test.assertRaisesRegex(RuntimeError, "Unsupported accepted tendon route"):
            solver.check_tendon_material()


def _check_dynamic_route_cleanup(test, device, *, nonlinear=False, direct=True):
    dt = 1.0 / 1200.0
    modes = ("penalty", "shared_alm", "segment_alm") if direct else ("penalty", "shared_alm")
    for mode in modes:
        for capture in (False, True) if device.is_cuda else (False,):
            with test.subTest(mode=mode, capture=capture), wp.ScopedDevice(device):
                model, body, link = build_oriented_dynamic_route(1, device)
                model.body_color_groups = [wp.array([i], dtype=int) for i in range(model.body_count)]
                model.tendon_seg_compliance.fill_(0.01)
                model.tendon_seg_rest_length.fill_(0.47)
                model.tendon_link_mu.fill_(0.25)
                model.tendon_seg_damping.fill_(0.01)
                poses = model.body_q.numpy()
                poses[body, 0] = 0.0
                model.body_q.assign(poses)
                solver = newton.solvers.SolverVBD(
                    model,
                    iterations=4,
                    tendon_material_direct=direct,
                    tendon_alm=mode != "penalty",
                    tendon_alm_per_segment=mode == "segment_alm",
                    tendon_sigmoid_ea_low=500.0 if nonlinear else 0.0,
                    tendon_sigmoid_ea_ratio=10.0,
                    tendon_sigmoid_transition_strain=0.01,
                    tendon_sigmoid_transition_width=0.003,
                )
                state = model.state()
                solver.body_q_prev.assign(state.body_q)
                solver._snapshot_tendon_step_state()
                solver._update_tendon_routing(state, dt, False)
                initial_inventory = _oriented_route_material_length(solver, model, state, link)
                test.assertTrue(solver.tendon_link_active.numpy()[link])

                # The accepted pose crosses the bypass chord during this step.
                poses[body, 0] = 0.1001
                state.body_q.assign(poses)
                solver._update_tendon_routing(state, dt, False)
                test.assertLess(_oriented_wrap_angle_at_link(solver, model, state, link), -1.0e-4)
                solver.check_tendon_material()
                if capture:
                    with wp.ScopedCapture(device=device) as captured:
                        solver._finalize_tendon_routing(state, dt)
                    wp.capture_launch(captured.graph)
                else:
                    solver._finalize_tendon_routing(state, dt)

                test.assertFalse(solver.tendon_link_active.numpy()[link])
                solver.check_tendon_material()
                inventory = _oriented_route_material_length(solver, model, state, link)
                test.assertAlmostEqual(inventory, initial_inventory, delta=2.0e-7)
                test.assertTrue(np.isfinite(solver.tendon_seg_rest_length.numpy()).all())
                active = solver.tendon_seg_active.numpy().astype(bool)
                test.assertEqual(int(active.sum()), 1)
                tension = np.maximum(
                    solver.tendon_seg_stretch.numpy() / solver.tendon_seg_active_compliance.numpy(), 0.0
                )
                test.assertGreater(float(tension[active][0]), 0.1)

                # Replaying the same accepted state must not merge its arc twice.
                if capture:
                    wp.capture_launch(captured.graph)
                else:
                    solver._finalize_tendon_routing(state, dt)
                solver.check_tendon_material()
                test.assertAlmostEqual(
                    _oriented_route_material_length(solver, model, state, link), inventory, delta=2.0e-7
                )


def test_direct_dynamic_route_cleanup(test, device):
    """Merge a detached dynamic roller before checking the accepted direct route."""
    _check_dynamic_route_cleanup(test, device)


def test_direct_nonlinear_dynamic_route_cleanup(test, device):
    """Preserve the sigmoid material inventory through a damped dynamic detachment."""
    _check_dynamic_route_cleanup(test, device, nonlinear=True)


def test_sweeps_dynamic_route_cleanup(test, device):
    """Preserve damped sweep material through end-of-step dynamic release."""
    _check_dynamic_route_cleanup(test, device, direct=False)


def test_sweeps_nonlinear_dynamic_route_cleanup(test, device):
    """Preserve damped sigmoid sweep material through end-of-step dynamic release."""
    _check_dynamic_route_cleanup(test, device, nonlinear=True, direct=False)


def test_dynamic_cleanup_preserves_unchanged_route(test, device):
    """Leave material and force unchanged when no dynamic roller detaches."""
    for direct in (False, True):
        for nonlinear in (False, True):
            results = []
            for dynamic in (False, True):
                with test.subTest(direct=direct, nonlinear=nonlinear, dynamic=dynamic):
                    model, body, link = build_oriented_dynamic_route(1, device, dynamic=dynamic)
                    model.body_color_groups = [wp.array([i], dtype=int, device=device) for i in range(model.body_count)]
                    model.tendon_seg_compliance.fill_(0.01)
                    model.tendon_seg_rest_length.assign(np.array([0.45, 0.49], dtype=np.float32))
                    model.tendon_link_mu.fill_(0.25)
                    model.tendon_seg_damping.fill_(0.01)
                    poses = model.body_q.numpy()
                    poses[body, 0] = 0.0
                    model.body_q.assign(poses)
                    solver = newton.solvers.SolverVBD(
                        model,
                        iterations=4,
                        tendon_material_direct=direct,
                        tendon_max_sweeps=1,
                        tendon_sigmoid_ea_low=500.0 if nonlinear else 0.0,
                        tendon_sigmoid_ea_ratio=10.0,
                        tendon_sigmoid_transition_strain=0.01,
                        tendon_sigmoid_transition_width=0.003,
                    )
                    state = model.state()
                    solver.body_q_prev.assign(state.body_q)
                    solver._snapshot_tendon_step_state()
                    poses[body, 0] = 0.02
                    state.body_q.assign(poses)
                    solver._finalize_tendon_routing(state, 1.0 / 1200.0)
                    solver.check_tendon_material()
                    test.assertTrue(solver.tendon_link_active.numpy()[link])
                    results.append(
                        (
                            solver.tendon_seg_rest_length.numpy(),
                            solver.tendon_seg_stretch.numpy(),
                            solver.tendon_seg_damping_tension.numpy(),
                            solver._tendon_link_material_transfer.numpy(),
                        )
                    )
            for fixed, dynamic in zip(*results, strict=True):
                np.testing.assert_allclose(dynamic, fixed, rtol=0.0, atol=1.0e-7)


class TestTendonRouteAcceptance(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_same_winding_overlap_activation,
        test_initial_invalid_route_check,
        test_accepted_invalid_route_check,
        test_direct_dynamic_route_cleanup,
        test_direct_nonlinear_dynamic_route_cleanup,
        test_sweeps_dynamic_route_cleanup,
        test_sweeps_nonlinear_dynamic_route_cleanup,
        test_dynamic_cleanup_preserves_unchanged_route,
    ):
        # The invalid accepted-route case deliberately emits the solver diagnostic;
        # its RuntimeError and latched link identity are asserted in the test.
        add_function_test(
            TestTendonRouteAcceptance,
            _test.__name__,
            _test,
            devices=[_device],
            check_output=_test is not test_accepted_invalid_route_check,
        )

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
