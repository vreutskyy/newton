# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material-sweep regressions independent of rigid-body convergence."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.tendon import TendonLinkType
from newton._src.solvers.tendon_kernels import _material_sweep_contact_error, solve_tendon_material
from newton._src.solvers.tendon_material_slack import TendonSlackBlock
from newton._src.solvers.tendon_material_state import TendonMaterialState
from newton._src.solvers.vbd.tendon_kernels import _rolling_spin_axis_component
from newton._src.solvers.xpbd.tendon_kernels import solve_tendon_slip
from newton.tests.test_tendon_material_integration import _chain_model
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_xpbd_spin_reaction_bounds_accumulated_impulse(test, device):
    """Bound the total friction impulse and retract it when the cable unloads."""
    with wp.ScopedDevice(device):
        for mu in (0.0, 0.1, 10.0):
            with test.subTest(mu=mu):
                lambdas = wp.zeros(2)
                tension = wp.full(2, 100.0)
                reaction = wp.zeros(3, dtype=wp.vec3)
                deltas = wp.zeros(1, dtype=wp.spatial_vector)
                inputs = [
                    wp.array([wp.transform_identity()], dtype=wp.transform),
                    wp.array([-1, 0, -1], dtype=int),
                    wp.zeros(3, dtype=int),
                    wp.array(
                        [int(TendonLinkType.ATTACHMENT), int(TendonLinkType.ROLLING), int(TendonLinkType.ATTACHMENT)],
                        dtype=int,
                    ),
                    wp.array([0.0, 1.0, 0.0], dtype=float),
                    wp.array([0.0, mu, 0.0], dtype=float),
                    wp.ones(3, dtype=bool),
                    wp.zeros(3, dtype=wp.vec3),
                    wp.array([[0.0, 1.0, 0.0]] * 3, dtype=wp.vec3),
                    wp.array([[-1.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=wp.vec3),
                    wp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, -1.0]], dtype=wp.vec3),
                    wp.full(2, 0.01),
                    tension,
                    wp.zeros(2),
                    lambdas,
                    0.0,
                    0.01,
                    reaction,
                ]
                accumulated = 0.0
                for pair in ([-1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0], [-0.5, -0.5], [0.0, -1.0]):
                    lambdas.assign(np.asarray(pair, dtype=np.float32))
                    deltas.zero_()
                    wp.launch(solve_tendon_slip, dim=3, inputs=inputs, outputs=[deltas])
                    accumulated += float(deltas.numpy()[0, 4])
                    rho = np.exp(min(mu * np.pi, 20.0))
                    bound = 2.0 * (rho - 1.0) / (rho + 1.0)
                    expected = np.clip(pair[0] - pair[1], -bound, bound)
                    test.assertAlmostEqual(accumulated, float(expected), delta=1.0e-6)
                tension.zero_()
                deltas.zero_()
                wp.launch(solve_tendon_slip, dim=3, inputs=inputs, outputs=[deltas])
                accumulated += float(deltas.numpy()[0, 4])
                test.assertAlmostEqual(accumulated, 0.0, delta=1.0e-6)


def material_tension(length, rest):
    strain = max(length - rest, 0.0) / rest
    ea = 2000.0 * (1.0 + 15.0 * 0.5 * (1.0 + np.tanh((strain - 0.005) / 0.0008)))
    return ea * strain


def material_rest(length, tension):
    lo, hi = length * 0.5, length
    for _ in range(70):
        mid = (lo + hi) * 0.5
        if material_tension(length, mid) > tension:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def damped_material_rest(length, tension, raw_damping, damping_rate):
    """Invert the full engaged force independently of the production projection."""
    lo, hi = length * 0.5, length
    for _ in range(70):
        mid = (lo + hi) * 0.5
        force = max(material_tension(length, mid) + min(raw_damping, damping_rate * (length - mid)), 0.0)
        if force > tension:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def solve_material_case(
    device,
    tensions,
    ratios,
    rolling_left=None,
    rolling_right=None,
    sweeps=256,
    tol=1.0e-6,
    atol=1.0e-6,
    nonlinear=False,
    damping_tensions=None,
    segment_compliance=0.01,
    direct=False,
    capture=False,
    segment_lengths=None,
):
    """Hold geometry fixed and exercise the production material-transfer kernel."""
    count = len(tensions)
    length = (
        np.ones(count, dtype=np.float32) if segment_lengths is None else np.asarray(segment_lengths, dtype=np.float32)
    )
    compliance = np.full(count, segment_compliance, dtype=np.float32)
    rest = length - compliance * np.asarray(tensions, dtype=np.float32)
    damping_values = (
        np.zeros(count, dtype=np.float32)
        if damping_tensions is None
        else np.asarray(damping_tensions, dtype=np.float32)
    )
    if nonlinear:
        rest = np.array(
            [material_rest(l, t - d) for l, t, d in zip(length, tensions, damping_values, strict=True)],
            dtype=np.float32,
        )
    else:
        rest += compliance * damping_values
    active = wp.ones(count, dtype=int, device=device)
    length_gpu = wp.array(length, device=device)
    compliance_gpu = wp.array(compliance, device=device)
    damping_coefficient = wp.ones(count, device=device)
    rest_gpu = wp.array(rest, device=device)
    stretch = wp.zeros(count, device=device)
    damping = wp.zeros(count, device=device)
    positions = np.concatenate(([0.0], np.cumsum(length)))
    points_l = wp.array([[x, 0, 0] for x in positions[:-1]], dtype=wp.vec3, device=device)
    points_r = wp.array([[x, 0, 0] for x in positions[1:]], dtype=wp.vec3, device=device)
    poses = wp.array([wp.transform_identity()] * (count + 1), dtype=wp.transform, device=device)
    link_active = wp.ones(count + 1, dtype=bool, device=device)
    sweep_count = wp.zeros(1, dtype=int, device=device)
    # Identity poses and unit +x span directions make each length rate the
    # difference of adjacent x velocities. Use damping coefficient one.
    velocities = np.zeros((count + 1, 6), dtype=np.float32)
    velocities[1:, 0] = np.cumsum(damping_values)
    inputs = [
        poses,
        wp.array(velocities, dtype=wp.spatial_vector, device=device),
        poses,
        wp.zeros(count + 1, dtype=wp.vec3, device=device),
        wp.array([0, count + 1], dtype=int, device=device),
        wp.array(np.arange(count + 1), dtype=int, device=device),
        wp.array(
            [int(TendonLinkType.ATTACHMENT)]
            + [int(TendonLinkType.ROLLING)] * (count - 1)
            + [int(TendonLinkType.ATTACHMENT)],
            dtype=int,
            device=device,
        ),
        wp.ones(count + 1, device=device),
        wp.zeros(count + 1, dtype=wp.vec3, device=device),
        wp.array([[0, 0, 1]] * (count + 1), dtype=wp.vec3, device=device),
        rest_gpu,
        wp.array(rest, device=device),
        wp.array(rest, device=device),
        stretch,
        damping,
        active,
        wp.array(np.arange(count), dtype=int, device=device),
        wp.array(np.arange(1, count + 1), dtype=int, device=device),
        compliance_gpu,
        damping_coefficient,
        link_active,
        link_active,
        wp.zeros(count + 1, device=device),
        points_l,
        points_r,
        length_gpu,
        points_l,
        points_r,
        wp.array(np.zeros(count) if rolling_left is None else rolling_left, dtype=float, device=device),
        wp.array(np.zeros(count) if rolling_right is None else rolling_right, dtype=float, device=device),
        wp.array([-1, *range(count - 1), -1], dtype=int, device=device),
        wp.array([-1, *range(1, count), -1], dtype=int, device=device),
        wp.array([1.0, *ratios, 1.0], dtype=float, device=device),
        sweep_count,
        0,
        1.0 / 60.0,
        1,
        1,
        1,
        sweeps,
        tol,
        atol,
        2000.0 if nonlinear else 0.0,
        16.0,
        0.005,
        0.0008,
        wp.zeros(count + 1, device=device),
    ]
    inputs.insert(8, wp.ones(count + 1, dtype=int, device=device))
    workspace = TendonMaterialState()
    workspace.slack_enabled = True
    workspace.slack = wp.zeros(count, dtype=int, device=device)
    workspace.slack_blocks = wp.zeros(count, dtype=TendonSlackBlock, device=device)
    kernel, lanes, block_dim = solve_tendon_material, 1, 256
    if direct:
        model = _chain_model(device, span_count=count)
        solver = newton.solvers.SolverXPBD(
            model,
            tendon_material_direct=True,
            tendon_sigmoid_ea_low=2000.0 if nonlinear else 0.0,
            tendon_sigmoid_ea_ratio=16.0,
            tendon_sigmoid_transition_strain=0.005,
            tendon_sigmoid_transition_width=0.0008,
        )
        workspace = solver._tendon_material_state
        workspace.alm.length = length_gpu
        workspace.alm.physical_compliance = compliance_gpu
        workspace.alm.damping = damping_coefficient
        workspace.alm.damping_force = damping
        kernel = solver._tendon_material_kernel
        lanes = solver._tendon_material_lanes
        block_dim = solver._tendon_material_block_dim
    inputs[-1:-1] = [workspace, 0]
    wp.launch(kernel, dim=lanes, inputs=inputs, device=device, block_dim=block_dim)
    if capture:
        with wp.ScopedCapture(device=device) as graph:
            wp.launch(kernel, dim=lanes, inputs=inputs, device=device, block_dim=block_dim)
        wp.capture_launch(graph.graph)
    if direct:
        solver.check_tendon_material()
    stretch_np = stretch.numpy()
    result = np.maximum(stretch_np / compliance + damping.numpy(), 0.0)
    if nonlinear:
        result = np.maximum(
            [material_tension(float(l), float(l - d)) for l, d in zip(length, stretch_np, strict=True)]
            + damping.numpy(),
            0.0,
        )
    return result, rest_gpu.numpy(), rest, int(sweep_count.numpy()[0])


def test_adaptive_sweeps_check_final_force_scale(test, device):
    """Resolve small final tensions after a much larger transient material load."""
    for reverse in (False, True):
        initial = [-40.0, 40.0, 0.03]
        ratios = [1.2, 1.4]
        if reverse:
            initial = initial[::-1]
            ratios = ratios[::-1]
        with test.subTest(reverse=reverse):
            tension, rest, initial_rest, count = solve_material_case(device, initial, ratios, tol=1.0e-3)
            full, _, _, _ = solve_material_case(device, initial, ratios, tol=0.0)
            for left, right, rho in zip(tension[:-1], tension[1:], ratios, strict=True):
                error = max(left - rho * right, right - rho * left, 0.0)
                test.assertLessEqual(error, 1.0e-6 + 1.0e-3 * max(left, right))
            np.testing.assert_allclose(tension, full, atol=3.0e-6, rtol=5.0e-3)
            np.testing.assert_allclose(rest.sum(), initial_rest.sum(), atol=3.0e-7, rtol=0)
            test.assertLess(count, 256, "Converged material should stop before exhausting its budget")


def test_adaptive_sweeps_exhausted_span(test, device):
    """Accept a material-bound optimum, not an impossible capstan equality."""
    for reverse in (False, True):
        for capture in (False, True) if device.is_cuda else (False,):
            lengths, tensions = [1.0, 0.001], [5.0, 0.0]
            if reverse:
                lengths, tensions = lengths[::-1], tensions[::-1]
            with test.subTest(reverse=reverse, capture=capture):
                force, rest, initial, count = solve_material_case(
                    device, tensions, [2.0], segment_lengths=lengths, capture=capture
                )
                donor = 0 if reverse else 1
                receiver = 1 - donor
                expected_rest = initial.copy()
                expected_rest[receiver] += expected_rest[donor] - 1.0e-6
                expected_rest[donor] = 1.0e-6
                np.testing.assert_allclose(rest, expected_rest, atol=1.0e-7, rtol=0)
                test.assertGreater(force[receiver], 2.0 * force[donor])
                test.assertLessEqual(count, 2, "The donor bound already prevents any further transfer")
                full, full_rest, _, full_count = solve_material_case(
                    device, tensions, [2.0], segment_lengths=lengths, tol=0.0, atol=100.0
                )
                test.assertEqual(full_count, 256, "Zero relative tolerance disables early-out, including atol")
                np.testing.assert_array_equal(force, full)
                np.testing.assert_array_equal(rest, full_rest)


def test_adaptive_sweeps_replenished_donor(test, device):
    """Recheck a depleted middle span after its other neighbor replenishes it."""
    for reverse in (False, True):
        lengths, tensions, ratios = [1.0, 0.1, 1.0], [50.0, 0.0, -25.0], [1.2, 1.4]
        expected = 25.0 * np.array([1.68, 1.4, 1.0]) / 4.08
        if reverse:
            lengths, tensions, ratios, expected = lengths[::-1], tensions[::-1], ratios[::-1], expected[::-1]
        with test.subTest(reverse=reverse):
            force, rest, initial, count = solve_material_case(
                device, tensions, ratios, segment_lengths=lengths, tol=1.0e-6
            )
            test.assertGreater(count, 1)
            test.assertLess(count, 256)
            np.testing.assert_allclose(force, expected, atol=3.0e-5, rtol=0)
            np.testing.assert_allclose(rest.sum(), initial.sum(), atol=2.0e-7, rtol=0)
            test.assertGreater(rest[1], 1.0e-6)


def test_adaptive_sweeps_preserve_small_corrections(test, device):
    """Resolve small force corrections after transferring much more material."""
    for reverse in (False, True):
        initial = [-1.0e5, 1.0e5, 100.0]
        ratios = [1.2, 1.4]
        if reverse:
            initial = initial[::-1]
            ratios = ratios[::-1]
        with test.subTest(reverse=reverse):
            tension, rest, initial_rest, count = solve_material_case(
                device, initial, ratios, tol=1.0e-6, segment_compliance=1.0e-8
            )
            for left, right, rho in zip(tension[:-1], tension[1:], ratios, strict=True):
                error = max(left - rho * right, right - rho * left, 0.0)
                test.assertLessEqual(error, 1.0e-6 + 1.0e-6 * max(left, right))
            test.assertLess(count, 256)
            np.testing.assert_allclose(rest.sum(), initial_rest.sum(), atol=3.0e-7, rtol=0)


def test_adaptive_sweeps_stop_at_satisfied_projection(test, device):
    """Stop once the final friction law is satisfied, even after a large update."""
    for reverse in (False, True):
        initial = [30.0, 0.0] if not reverse else [0.0, 30.0]
        expected = [20.0, 10.0] if not reverse else [10.0, 20.0]
        with test.subTest(reverse=reverse):
            tension, rest, initial_rest, count = solve_material_case(device, initial, [2.0], tol=1.0e-3)
            np.testing.assert_allclose(tension, expected, atol=1.0e-5, rtol=0)
            np.testing.assert_allclose(rest.sum(), initial_rest.sum(), atol=3.0e-7, rtol=0)
            test.assertEqual(count, 1, "A satisfied two-span projection needs no further sweep")


def test_adaptive_sweeps_absolute_tolerance(test, device):
    """Stop an unloading cable without chasing a shrinking relative force scale."""
    tension, rest, initial, count = solve_material_case(
        device, [-40.0, 40.0, -0.01], [1.2, 1.4], tol=1.0e-3, atol=1.0e-6
    )
    test.assertLessEqual(float(tension.max()), 2.0e-6)
    test.assertLess(count, 128)
    np.testing.assert_allclose(rest.sum(), initial.sum(), atol=3.0e-7, rtol=0)
    _, _, _, full_count = solve_material_case(device, [0, 0, 0], [1.2, 1.4], sweeps=16, tol=0.0)
    test.assertEqual(full_count, 16, "Zero relative tolerance retains the explicit full-budget policy")


def test_adaptive_sweeps_honor_large_budget(test, device):
    """Resolve a long finite-friction chain beyond the former 256-sweep cap."""
    span_count = 32
    rho = 1.01
    initial = [10.0] + [0.0] * (span_count - 1)
    ratios = [rho] * (span_count - 1)
    for capture in (False, True) if device.is_cuda else (False,):
        with test.subTest(capture=capture):
            tension, rest, initial_rest, count = solve_material_case(
                device, initial, ratios, sweeps=4096, tol=1.0e-5, atol=1.0e-7, capture=capture
            )
            # Material moves right across every junction, so T_i = T_0 / rho**i.
            # Conservation fixes T_0 independently of either material solver.
            scale = np.power(rho, -np.arange(span_count))
            extension = np.sum(1.0 - initial_rest.astype(np.float64))
            expected = scale * extension / (0.01 * scale.sum())
            test.assertGreater(count, 256)
            test.assertLess(count, 4096, "The larger budget should permit adaptive convergence")
            np.testing.assert_allclose(tension, expected, atol=2.0e-4, rtol=0.0)
            np.testing.assert_allclose(rest.sum(), initial_rest.sum(), atol=4.0e-6, rtol=0.0)

    _, _, _, count = solve_material_case(device, [1.0, 1.0], [rho], sweeps=512, tol=0.0)
    test.assertEqual(count, 512, "Disabling early-out must execute the requested budget")


def test_solver_entry_points_honor_large_sweep_budget(test, device):
    """Accept and execute a large material budget through both public solvers."""
    model = _chain_model(device)
    budget = 1024
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            solver = solver_type(model, iterations=1, tendon_max_sweeps=budget, tendon_settle_tol=0.0)
            state, result, control = model.state(), model.state(), model.control()
            solver.step(state, result, control, None, 1.0 / 240.0)
            np.testing.assert_array_equal(solver.tendon_cone_sweep_count.numpy(), [budget])
            if wp.get_device(device).is_cuda:
                with wp.ScopedCapture(device=device) as captured:
                    solver.step(state, result, control, None, 1.0 / 240.0)
                wp.capture_launch(captured.graph)
                np.testing.assert_array_equal(solver.tendon_cone_sweep_count.numpy(), [budget])


def test_slack_component_has_exact_zero_force(test, device):
    """Release fully slack finite-friction components without fictitious stiffness."""
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        for reverse in (False, True):
            with test.subTest(solver=solver_type.__name__, reverse=reverse):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                body = builder.add_body(mass=0.0, is_kinematic=True)
                builder.add_tendon()
                rests = [1.4, 0.6, 1.01]
                if reverse:
                    rests.reverse()
                for index, point in enumerate(((0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0))):
                    builder.add_tendon_link(
                        body,
                        offset=point,
                        link_type=TendonLinkType.PINHOLE if index in (1, 2) else TendonLinkType.ATTACHMENT,
                        mu=0.2,
                        compliance=0.01,
                        rest_length=1.0 if index == 0 else rests[index - 1],
                    )
                builder.color()
                model = builder.finalize(device=device)
                solver = solver_type(model, iterations=1)
                initial = solver.tendon_seg_rest_length.numpy().astype(np.float64)
                state, next_state = model.state(), model.state()
                solver.step(state, next_state, model.control(), None, 1.0 / 240.0)
                test.assertTrue(np.all(solver.tendon_link_cap_ratio.numpy()[1:3] > 1.0))
                test.assertTrue(np.all(solver.tendon_seg_stretch.numpy() <= 0.0))
                np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy().sum(), initial.sum(), atol=3.0e-7)
                if solver_type is newton.solvers.SolverVBD:
                    np.testing.assert_array_equal(solver.tendon_material_modes.numpy()["tension"], 0.0)
                    np.testing.assert_array_equal(solver.tendon_material_modes.numpy()["stiffness"], 0.0)


def test_slack_projection_stops_at_anchor(test, device):
    """Keep a loaded component isolated from slack across an interior anchor."""
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_body(mass=0.0, is_kinematic=True)
            builder.add_tendon()
            rests = [1.01, 0.6, 1.4, 0.9, 0.8]
            points = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 2, 0), (3, 2, 0))
            for index, point in enumerate(points):
                builder.add_tendon_link(
                    body,
                    offset=point,
                    link_type=TendonLinkType.ATTACHMENT if index in (0, 3, 5) else TendonLinkType.PINHOLE,
                    mu=0.2,
                    compliance=0.01,
                    rest_length=1.0 if index == 0 else rests[index - 1],
                )
            builder.color()
            model = builder.finalize(device=device)
            solver = solver_type(model, iterations=1)
            initial = solver.tendon_seg_rest_length.numpy().astype(np.float64)
            solver.step(model.state(), model.state(), model.control(), None, 1.0 / 240.0)
            np.testing.assert_array_equal(solver._tendon_material_state.slack.numpy(), [1, 1, 1, 0, 0])
            test.assertTrue(np.all(solver.tendon_seg_stretch.numpy()[:3] <= 0.0))
            test.assertTrue(np.all(solver.tendon_seg_stretch.numpy()[3:] > 0.0))
            final = solver.tendon_seg_rest_length.numpy().astype(np.float64)
            for region in (slice(0, 3), slice(3, 5)):
                np.testing.assert_allclose(final[region].sum(), initial[region].sum(), atol=3e-7, rtol=0)


@wp.kernel
def _evaluate_contact_error(values: wp.array2d[float], result: wp.array[float]):
    """Evaluate force residuals for independently specified contact states."""
    i = wp.tid()
    result[i] = _material_sweep_contact_error(
        values[i, 0], values[i, 1], values[i, 2], values[i, 3], values[i, 4] != 0.0, values[i, 5] != 0.0
    )


def test_material_convergence_requires_slip_boundary(test, device):
    """Reject excess net slip even when the resulting tension lies inside the band."""
    rows = [
        [1.5, 1.0, 2.0, 0.0],
        [1.5, 1.0, 2.0, 0.1],
        [1.0, 1.5, 2.0, -0.1],
        [2.0, 1.0, 2.0, 0.1],
        [1.0, 2.0, 2.0, -0.1],
        [3.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 2.0, 0.1],
    ]
    rows = [[*row, 1.0, 1.0] for row in rows]
    errors = wp.empty(len(rows), dtype=float, device=device)
    wp.launch(
        _evaluate_contact_error,
        dim=len(rows),
        inputs=[wp.array(rows, dtype=float, device=device)],
        outputs=[errors],
        device=device,
    )
    np.testing.assert_allclose(errors.numpy(), [0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0], atol=1e-7, rtol=0)


def test_material_convergence_bound_directions(test, device):
    """An exhausted donor can stop transfer, but cannot prevent release from its bound."""
    rows = [
        [3.0, 1.0, 2.0, 0.1, 1.0, 0.0],  # upper bound, force drives into it
        [1.0, 3.0, 2.0, -0.1, 0.0, 1.0],  # lower bound, force drives into it
        [1.5, 1.0, 2.0, 0.1, 1.0, 0.0],  # upper bound must release
        [1.0, 1.5, 2.0, -0.1, 0.0, 1.0],  # lower bound must release
        [3.0, 1.0, 2.0, 0.0, 1.0, 0.0],  # no slip, upper bound
        [1.0, 3.0, 2.0, 0.0, 1.0, 0.0],  # no slip, can move away
        [3.0, 1.0, 2.0, 0.1, 1.0, 1.0],  # neighbor replenished donor
        [1.0, 3.0, 2.0, -0.1, 1.0, 1.0],
    ]
    errors = wp.empty(len(rows), dtype=float, device=device)
    wp.launch(
        _evaluate_contact_error,
        dim=len(rows),
        inputs=[wp.array(rows, dtype=float, device=device)],
        outputs=[errors],
        device=device,
    )
    np.testing.assert_allclose(errors.numpy(), [0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0], atol=1e-7, rtol=0)


def test_material_sweep_tolerances_are_validated(test, device):
    """Reject invalid tolerances and sweep budgets consistently in both solvers."""
    model = newton.ModelBuilder().finalize(device=device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        for budget in (0, -1, 2**31, 1.5, float("nan"), float("inf")):
            with test.subTest(solver=solver_type.__name__, budget=budget):
                with test.assertRaisesRegex(ValueError, "tendon_max_sweeps"):
                    solver_type(model, tendon_max_sweeps=budget)
        for name in ("tendon_settle_tol", "tendon_settle_atol"):
            for value in (-1.0, float("nan"), float("inf")):
                with test.subTest(solver=solver_type.__name__, parameter=name, value=value):
                    with test.assertRaisesRegex(ValueError, name):
                        solver_type(model, **{name: value})


def test_net_slip_reaches_capstan_boundary(test, device):
    """Neighbor updates must retract excess slip instead of stopping inside the cone."""
    rho = np.array([1.4, 1.0004])
    for reverse in (False, True):
        ratios = rho[::-1] if reverse else rho
        for tol in (0.0, 1.0e-6):
            with test.subTest(reverse=reverse, tol=tol):
                tension, rest, initial, _ = solve_material_case(device, [9, 1, 9], ratios, tol=tol)
                initial_tension = (1.0 - initial.astype(float)) / float(np.float32(0.01))
                # Both end spans feed the middle. Thus both contacts must finish
                # on their slipping boundaries, not merely somewhere in the cone.
                middle = initial_tension.sum() / (1 + ratios[0] + ratios[1])
                expected = np.array([ratios[0] * middle, middle, ratios[1] * middle])
                np.testing.assert_allclose(tension, expected, atol=3.0e-5, rtol=1.0e-5)
                np.testing.assert_allclose(rest.sum(), initial.sum(), atol=3.0e-7, rtol=0)


def test_rolling_trial_precedes_friction_projection(test, device):
    """A sticking roller must preserve each endpoint's full material displacement."""
    for reverse in (False, True):
        left = [0.0, 0.002] if not reverse else [0.0, 0.0]
        right = [0.0, 0.0] if not reverse else [0.002, 0.0]
        for sweeps in (1, 4, 256):
            with test.subTest(reverse=reverse, sweeps=sweeps):
                tension, rest, initial, _ = solve_material_case(
                    device, [4, 4], [2], left, right, sweeps=sweeps, tol=0.0
                )
                expected_rest = initial + np.array([right[0], left[1]])
                expected_tension = (1.0 - expected_rest) / float(np.float32(0.01))
                np.testing.assert_allclose(rest, expected_rest, atol=1.0e-7, rtol=0)
                np.testing.assert_allclose(tension, expected_tension, atol=1.0e-5, rtol=0)


def test_damped_nonlinear_net_slip(test, device):
    """The accumulated-slip condition also applies to nonlinear, damped material."""
    for nonlinear in (False, True):
        for damping in ([0, 0, 0], [0.5, -0.2, 0.3]):
            with test.subTest(nonlinear=nonlinear, damping=damping):
                rho = np.array([1.4, 1.0004])
                tension, rest, initial, _ = solve_material_case(
                    device, [9, 1, 9], rho, nonlinear=nonlinear, damping_tensions=damping
                )
                if nonlinear:
                    lo, hi = 1.0, 9.0
                    for _ in range(70):
                        mid = (lo + hi) * 0.5
                        total = sum(
                            damped_material_rest(1.0, t, d, 60.0)
                            for t, d in zip([rho[0] * mid, mid, rho[1] * mid], damping, strict=True)
                        )
                        if total > initial.astype(float).sum():
                            lo = mid
                        else:
                            hi = mid
                    middle = (lo + hi) * 0.5
                else:
                    initial_tension = (1.0 - initial.astype(float)) / float(np.float32(0.01)) + damping
                    middle = initial_tension.sum() / (1 + rho[0] + rho[1])
                np.testing.assert_allclose(tension, [rho[0] * middle, middle, rho[1] * middle], atol=0.002, rtol=3.0e-4)
                np.testing.assert_allclose(rest.sum(), initial.sum(), atol=3.0e-7, rtol=0)


def test_damping_engagement_material_projection(test, device):
    """Match direct and sweep transfer to the full finite-friction force law."""
    for nonlinear in (False, True):
        for direct in (False, True):
            for raw in ([0.5, -0.2, 0.3], [20.0, 0.0, 50.0], [-2.0, 40.0, 0.5]):
                with test.subTest(nonlinear=nonlinear, direct=direct, damping=raw):
                    ratios = np.array([1.4, 1.2])
                    tension, rest, initial, _ = solve_material_case(
                        device,
                        [9, 1, 9],
                        ratios,
                        nonlinear=nonlinear,
                        damping_tensions=raw,
                        direct=direct,
                        capture=device.is_cuda,
                    )
                    extension = 1.0 - rest.astype(float)

                    # The initial allocation is generated from nominal tensions;
                    # the reference uses the actual resulting rest lengths.
                    elastic = extension / float(np.float32(0.01))
                    if nonlinear:
                        elastic = np.array([material_tension(1.0, 1.0 - x) for x in extension])
                    actual = np.maximum(elastic + np.minimum(raw, 60.0 * np.maximum(extension, 0.0)), 0.0)
                    np.testing.assert_allclose(tension, actual, atol=0.002, rtol=3.0e-4)
                    np.testing.assert_allclose(rest.sum(), initial.sum(), atol=3.0e-7, rtol=0)
                    flow = np.cumsum(rest.astype(float) - initial.astype(float))
                    for i, ratio in enumerate(ratios):
                        left, right = actual[i : i + 2]
                        tolerance = 0.003 + 4.0e-4 * max(left, right)
                        test.assertLessEqual(left - ratio * right, tolerance)
                        test.assertLessEqual(right - ratio * left, tolerance)
                        if flow[i] > 1.0e-6:
                            test.assertLessEqual(abs(left - ratio * right), tolerance)
                        elif flow[i] < -1.0e-6:
                            test.assertLessEqual(abs(right - ratio * left), tolerance)


def test_slack_span_and_sticking_contact(test, device):
    """Preserve slack material and sticking contacts while taking up slack before tension."""
    for initial, rho, expected in (
        ([-4, 0, 9], [1, 1], [5 / 3] * 3),
        ([-4, -2, -1], [1.2, 1.5], [0, 0, 0]),
        ([3, 4, 3], [1.5, 1.5], [3, 4, 3]),
    ):
        tension, rest, old_rest, _ = solve_material_case(device, initial, rho)
        np.testing.assert_allclose(tension, expected, atol=2.0e-5, rtol=0)
        np.testing.assert_allclose(rest.sum(), old_rest.sum(), atol=3.0e-7, rtol=0)
        if min(initial) >= 0 or max(initial) <= 0:
            np.testing.assert_allclose(rest, old_rest, atol=1.0e-7, rtol=0)


def test_rolling_grip_requires_tension(test, device):
    """Grip a loaded cable, but cancel rolling transport without tension or friction."""
    for initial, rho, grip in (([4, 4], 2.0, True), ([0, 0], 2.0, False), ([4, 4], 1.0, False)):
        with test.subTest(initial=initial, rho=rho):
            tension, rest, before, _ = solve_material_case(
                device, initial, [rho], rolling_left=[0, 0.002], rolling_right=[-0.002, 0]
            )
            transport = np.array([-0.002, 0.002]) if grip else np.zeros(2)
            np.testing.assert_allclose(rest, before + transport, atol=1.0e-7, rtol=0)
            np.testing.assert_allclose(tension, (1.0 - before - transport) / 0.01, atol=2.0e-5, rtol=0)


def test_rolling_trial_respects_rest_floor(test, device):
    """Project an exhausted no-slip trial as a pair without creating material."""
    tension, rest, before, _ = solve_material_case(
        device, [99.9998, 3], [1.0e8], rolling_left=[0, 0.01], rolling_right=[-0.01, 0]
    )
    test.assertTrue(np.isfinite(tension).all())
    np.testing.assert_allclose(rest[0], 1.0e-6, atol=3.0e-8, rtol=0)
    np.testing.assert_allclose(rest.astype(float).sum(), before.astype(float).sum(), atol=1.0e-7, rtol=0)


class TestTendonMaterialSweeps(unittest.TestCase):
    pass


@wp.kernel
def _evaluate_roller_spin(
    poses: wp.array[wp.transform],
    link_body: wp.array[int],
    link_type: wp.array[int],
    offsets: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    cone_l: wp.array[int],
    cone_r: wp.array[int],
    cap_ratio: wp.array[float],
    rest: wp.array[float],
    point_l: wp.array[wp.vec3],
    point_r: wp.array[wp.vec3],
    compliance: wp.array[float],
    damping: wp.array[float],
    active: wp.array[int],
    link_l: wp.array[int],
    link_r: wp.array[int],
    result: wp.array[wp.vec3],
):
    seg = wp.tid()
    tension = wp.max((1.0 - rest[seg]) / compliance[seg], 0.0)
    result[seg] = _rolling_spin_axis_component(
        1.0 / 60.0,
        poses,
        poses,
        offsets,
        link_body,
        link_type,
        offsets,
        axes,
        cone_l,
        cone_r,
        cap_ratio,
        rest,
        point_l,
        point_r,
        compliance,
        damping,
        active,
        link_l,
        link_r,
        1,
        seg,
        tension,
        wp.vec3(0.5, 1.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.0,
        1.0,
        0.005,
        0.0008,
    )


def test_vbd_spin_reaction_matches_capstan_projection(test, device):
    """Transmit the full sticking moment and limit only an out-of-cone trial."""
    for rho, tensions in ((1.0, [4, 3]), (2.0, [4, 3]), (2.0, [9, 1]), (2.0, [1, 9]), (2.0, [9, 0])):
        with test.subTest(rho=rho, tensions=tensions):
            rest = 1.0 - 0.01 * np.array(tensions, dtype=np.float32)
            output = wp.zeros(2, dtype=wp.vec3, device=device)
            wp.launch(
                _evaluate_roller_spin,
                dim=2,
                inputs=[
                    wp.array([wp.transform_identity()] * 3, dtype=wp.transform, device=device),
                    wp.array([0, 1, 2], dtype=int, device=device),
                    wp.array(
                        [int(TendonLinkType.ATTACHMENT), int(TendonLinkType.ROLLING), int(TendonLinkType.ATTACHMENT)],
                        dtype=int,
                        device=device,
                    ),
                    wp.array([[0, 0, 0], [0.5, 0, 0], [0, 0, 0]], dtype=wp.vec3, device=device),
                    wp.array([[0, 0, 1]] * 3, dtype=wp.vec3, device=device),
                    wp.array([-1, 0, -1], dtype=int, device=device),
                    wp.array([-1, 1, -1], dtype=int, device=device),
                    wp.array([1, rho, 1], dtype=float, device=device),
                    wp.array(rest, device=device),
                    wp.array([[-0.5, 1, 0], [1.5, 0, 0]], dtype=wp.vec3, device=device),
                    wp.array([[0.5, 1, 0], [1.5, -1, 0]], dtype=wp.vec3, device=device),
                    wp.full(2, 0.01, device=device),
                    wp.zeros(2, device=device),
                    wp.ones(2, dtype=int, device=device),
                    wp.array([0, 1], dtype=int, device=device),
                    wp.array([1, 2], dtype=int, device=device),
                    output,
                ],
                device=device,
            )
            scale = min(1.0, (rho - 1.0) / (rho + 1.0) * sum(tensions) / abs(tensions[0] - tensions[1]))
            np.testing.assert_allclose(output.numpy(), [[0, 0, -(1 - scale)]] * 2, atol=2.0e-6, rtol=0)


devices = get_test_devices()
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_replenished_donor",
    test_adaptive_sweeps_replenished_donor,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_material_convergence_bound_directions",
    test_material_convergence_bound_directions,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_exhausted_span",
    test_adaptive_sweeps_exhausted_span,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_stop_at_satisfied_projection",
    test_adaptive_sweeps_stop_at_satisfied_projection,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_absolute_tolerance",
    test_adaptive_sweeps_absolute_tolerance,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_material_convergence_requires_slip_boundary",
    test_material_convergence_requires_slip_boundary,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_material_sweep_tolerances_are_validated",
    test_material_sweep_tolerances_are_validated,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_check_final_force_scale",
    test_adaptive_sweeps_check_final_force_scale,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_honor_large_budget",
    test_adaptive_sweeps_honor_large_budget,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_solver_entry_points_honor_large_sweep_budget",
    test_solver_entry_points_honor_large_sweep_budget,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_slack_component_has_exact_zero_force",
    test_slack_component_has_exact_zero_force,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_slack_projection_stops_at_anchor",
    test_slack_projection_stops_at_anchor,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_preserve_small_corrections",
    test_adaptive_sweeps_preserve_small_corrections,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps, "test_rolling_grip_requires_tension", test_rolling_grip_requires_tension, devices=devices
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_rolling_trial_respects_rest_floor",
    test_rolling_trial_respects_rest_floor,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_net_slip_reaches_capstan_boundary",
    test_net_slip_reaches_capstan_boundary,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_rolling_trial_precedes_friction_projection",
    test_rolling_trial_precedes_friction_projection,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps, "test_damped_nonlinear_net_slip", test_damped_nonlinear_net_slip, devices=devices
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_damping_engagement_material_projection",
    test_damping_engagement_material_projection,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_slack_span_and_sticking_contact",
    test_slack_span_and_sticking_contact,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_vbd_spin_reaction_matches_capstan_projection",
    test_vbd_spin_reaction_matches_capstan_projection,
    devices=devices,
)


add_function_test(
    TestTendonMaterialSweeps,
    "test_xpbd_spin_reaction_bounds_accumulated_impulse",
    test_xpbd_spin_reaction_bounds_accumulated_impulse,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main()
