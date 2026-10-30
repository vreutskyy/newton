# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check slip metadata against the material actually transferred by direct solve."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.tendon_material import solve_tendon_material_component
from newton._src.solvers.tendon_material_state import TendonMaterialState, publish_tendon_material
from newton.tests.test_tendon_material_integration import _chain_model
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _project_and_publish(
    state: TendonMaterialState,
    active: wp.array[int],
    damping: wp.array[float],
    stretch: wp.array[float],
    transfer: wp.array[float],
):
    row = 4 * wp.tid()
    for i in range(4):
        seg = row + i
        state.initial[seg] = stretch[seg] + state.compliance[seg] * damping[seg]
    solve_tendon_material_component(
        state.initial,
        state.compliance,
        state.cap,
        state.upper,
        state.output,
        state.status,
        state.faces,
        state.valid,
        state.pieces,
        4,
        1,
        row,
        1,
        1,
    )
    if state.status[row] > 0:
        publish_tendon_material(state, row, 4, active, state.compliance, damping, stretch, transfer)


def test_sticking_transfer_is_zero(test, device):
    """Remove roundoff slip at sticking boundaries without clipping real small or slack flow."""
    rng = np.random.default_rng(32914)
    count = 64
    compliance = 10.0 ** rng.uniform(-8, -2, (count, 4))
    scale = 10.0 ** rng.uniform(-3, 3, (count, 1))
    force = np.tile([2.0, 1.0, 1.0, 2.0], (count, 1)) * scale
    flow = np.zeros((count, 4))
    flow[:, 0] = 10.0 ** rng.uniform(-12, -3, count)
    flow[:, 2] = -(10.0 ** rng.uniform(-12, -3, count))
    offset = compliance * rng.uniform(-2.0, 2.0, (count, 4)) * scale
    trial = compliance * force + flow - np.roll(flow, 1, axis=1) - offset
    # A slack component must still exchange material. A separate taut component
    # has real transfers far below a length-based epsilon and must retain them.
    trial[0], compliance[0], offset[0] = [0.01, -0.03, 0.01, -0.03], 1.0, 0.0
    compliance[1], offset[1] = 1.0e-6, 0.0
    flow[1] = [1.0e-10, 0.0, -2.0e-10, 0.0]
    trial[1] = compliance[1] * np.array([0.002, 0.001, 0.001, 0.002]) + flow[1] - np.roll(flow[1], 1)
    damping = (offset / compliance).astype(np.float32)
    trial = trial.astype(np.float32)
    next_link = np.arange(count * 4, dtype=np.int32).reshape(count, 4)
    next_link[:, -1] = -1
    with wp.ScopedDevice(device):
        state = TendonMaterialState()
        state.initial = wp.zeros(count * 4)
        state.compliance = wp.array(compliance.ravel(), dtype=float)
        state.cap = wp.full(count * 4, 2.0)
        state.upper = wp.full(count * 4, 1.0e6)
        state.output = wp.zeros(count * 4)
        state.next_link = wp.array(next_link.ravel(), dtype=int)
        for name in ("status", "faces", "valid", "pieces"):
            setattr(state, name, wp.zeros(count * 4, dtype=int))
        stretch = wp.array(trial.ravel(), dtype=float)
        transfer = wp.zeros(count * 4)
        args = [state, wp.ones(count * 4, dtype=int), wp.array(damping.ravel(), dtype=float), stretch, transfer]
        wp.launch(_project_and_publish, dim=count, inputs=args)
        graph = None
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                wp.launch(_project_and_publish, dim=count, inputs=args)
            graph = capture.graph
        for replay in range(3):
            if replay:
                stretch.assign(trial.ravel())
                if graph is None:
                    wp.launch(_project_and_publish, dim=count, inputs=args)
                else:
                    wp.capture_launch(graph)
            test.assertTrue(np.all(state.status.numpy()[::4] > 0))
            valid = state.valid.numpy()[::4]
            faces = state.faces.numpy().reshape(count, 4)
            sticking = (valid[:, None] != 0) & (faces == 0)
            sticking[:, -1] = False
            test.assertGreater(np.count_nonzero(sticking), 0)
            actual = transfer.numpy().reshape(count, 4)
            np.testing.assert_array_equal(actual[sticking], 0.0)
            test.assertEqual(valid[0], 0)
            # The first deficit requires q0=.01; the third is shared equally
            # by q1=-.005 and q2=.005 to minimize squared net transfer.
            np.testing.assert_allclose(actual[0], [0.01, -0.005, 0.005, 0.0], atol=2.0e-9, rtol=0.0)
            np.testing.assert_allclose(actual[1], flow[1], atol=2.0e-15, rtol=0.0)
            # Rest-length gain must still agree with the signed interface flow
            # to float storage accuracy; canonicalization changes no stretch.
            gain = trial.astype(float) - stretch.numpy().reshape(count, 4).astype(float)
            error = gain - (actual - np.roll(actual, 1, axis=1))
            bound = 8 * np.finfo(np.float32).eps * (np.abs(trial).sum(axis=1) + np.abs(offset).sum(axis=1))
            test.assertTrue(np.all(np.abs(error).max(axis=1) <= bound))


def test_direct_slip_matches_transfer(test, device):
    """Publish signed slip independently on either side of an internal anchor."""
    for nonlinear in (False, True):
        for reverse in (False, True):
            for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
                with test.subTest(nonlinear=nonlinear, reverse=reverse, solver=solver_type.__name__):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    body = builder.add_body(mass=0.0, is_kinematic=True)
                    builder.add_tendon()
                    rests = [0.97, 1.0, 1.0, 0.97]
                    if reverse:
                        # The two components deliberately slip in opposite directions.
                        rests = [1.0, 0.97, 0.97, 1.0]
                    for index, point in enumerate(((0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 2, 0))):
                        builder.add_tendon_link(
                            body,
                            offset=point,
                            link_type=(
                                newton.TendonLinkType.PINHOLE if index % 2 else newton.TendonLinkType.ATTACHMENT
                            ),
                            mu=float(np.log(2.0) / (np.pi / 2.0)),
                            compliance=0.001,
                            damping=0.0,
                            rest_length=1.0 if index == 0 else rests[index - 1],
                        )
                    builder.color()
                    model = builder.finalize(device=device)
                    options = {}
                    if nonlinear:
                        options.update(
                            tendon_sigmoid_ea_low=500.0,
                            tendon_sigmoid_ea_ratio=10.0,
                            tendon_sigmoid_transition_strain=0.01,
                            tendon_sigmoid_transition_width=0.003,
                        )
                    if solver_type is newton.solvers.SolverVBD:
                        options["tendon_material_direct_settle_iterations"] = 0
                    solver = solver_type(model, iterations=1, tendon_material_direct=True, **options)
                    initial = solver.tendon_seg_rest_length.numpy().copy()
                    # Seed all slots to expose stale endpoints and non-sliding anchors too.
                    solver._tendon_link_material_transfer.fill_(7.0)
                    state, next_state = model.state(), model.state()
                    solver.step(state, next_state, model.control(), None, 1.0 / 240.0)
                    solver.check_tendon_material()
                    rest = solver.tendon_seg_rest_length.numpy()
                    slip = solver._tendon_link_material_transfer.numpy()
                    np.testing.assert_allclose(slip[[1, 3]], (rest - initial)[[0, 2]], atol=2.0e-7, rtol=0.0)
                    np.testing.assert_array_equal(slip[[0, 2, 4]], 0.0)
                    np.testing.assert_allclose(rest.reshape(2, 2).sum(axis=1), initial.reshape(2, 2).sum(axis=1))
                    sign = -1.0 if reverse else 1.0
                    test.assertGreater(sign * slip[1], 1.0e-4)
                    test.assertLess(sign * slip[3], -1.0e-4)
                    if not nonlinear:
                        np.testing.assert_allclose(slip[[1, 3]], sign * np.array([0.01, -0.01]), atol=2.0e-7)
                    if solver_type is newton.solvers.SolverVBD:
                        # The sparse metric is opt-in for assembly, never for local updates.
                        solver._update_tendon_routing(next_state, 1.0 / 240.0, False, sliding_curvature=True)
                        modes = solver.tendon_material_modes.numpy()
                        np.testing.assert_array_equal(modes["first"], [0, 0, 2, 2])
                        weights = modes["transfer_weight"][[0, 2]]
                        test.assertTrue(np.all((weights > 0) & (weights < 1)))
                        np.testing.assert_array_equal(modes["transfer_weight"][[1, 3]], 0.0)
                        if not nonlinear:
                            np.testing.assert_allclose(weights, 1.0 / 3.0, atol=2.0e-6)


def test_direct_slip_hold_and_rejection(test, device):
    """Clear stale slip for held/rejected trials, but preserve skipped-route caches."""
    for nonlinear in (False, True):
        with test.subTest(nonlinear=nonlinear):
            model = _chain_model(device, rest_lengths=[0.97, 1.0])
            solver = newton.solvers.SolverVBD(
                model,
                iterations=1,
                tendon_material_direct=True,
                tendon_material_direct_settle_iterations=0,
                tendon_sigmoid_ea_low=500.0 if nonlinear else 0.0,
            )
            state = model.state()
            solver.step(state, model.state(), model.control(), None, 1.0 / 240.0)
            workspace = solver._tendon_material_state
            slip = solver._tendon_link_material_transfer

            # A hold computes a no-slip trial, so the previous projection's
            # slip must not soften the next body solve.
            slip.fill_(7.0)
            solver._update_tendon_routing(state, 1.0 / 240.0, False, skip_material=True)
            np.testing.assert_array_equal(slip.numpy(), 0.0)
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    slip.fill_(7.0)
                    solver._update_tendon_routing(state, 1.0 / 240.0, False, skip_material=True)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(slip.numpy(), 0.0)

            # Selective route updates leave unprocessed routes unchanged.
            workspace.process_changed_routes = True
            workspace.changed_routes = wp.zeros(model.tendon_count, dtype=int, device=device)
            slip.fill_(7.0)
            solver._update_tendon_routing(state, 1.0 / 240.0, False)
            np.testing.assert_array_equal(slip.numpy(), 7.0)
            workspace.process_changed_routes = False

            # A rejected in-iteration trial may be retried, but cannot publish
            # slip from a partially solved component or a previous projection.
            model.tendon_seg_compliance.zero_()
            solver._update_tendon_routing(state, 1.0 / 240.0, False, latch_failure=False)
            np.testing.assert_array_equal(slip.numpy(), 0.0)
            np.testing.assert_array_equal(workspace.failure.numpy(), 0)


def test_material_hold_refreshes_tension(test, device, *, nonlinear=False, damping=0.0):
    """Hold rest lengths, not the forces from the previously evaluated pose."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    bodies = [
        builder.add_body(xform=wp.transform(p=point), mass=0.0, is_kinematic=True)
        for point in ((0, 0, 0), (1, 0, 0), (1, 1, 0))
    ]
    builder.add_tendon()
    for index, body in enumerate(bodies):
        builder.add_tendon_link(
            body,
            link_type=newton.TendonLinkType.PINHOLE if index == 1 else newton.TendonLinkType.ATTACHMENT,
            mu=0.2,
            compliance=0.001,
            damping=damping,
            rest_length=0.99,
        )
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=1,
        tendon_material_direct=True,
        tendon_sigmoid_ea_low=500.0 if nonlinear else 0.0,
        tendon_sigmoid_ea_ratio=10.0,
        tendon_sigmoid_transition_strain=0.01,
        tendon_sigmoid_transition_width=0.003,
    )
    state = model.state()
    solver.step(state, model.state(), model.control(), None, 1.0 / 240.0)
    before = solver.tendon_seg_rest_length.numpy().copy()
    initial_length = solver.tendon_seg_length.numpy().copy()
    poses = state.body_q.numpy()
    poses[-1, 1] += 0.01
    state.body_q.assign(poses)
    solver._update_tendon_routing(state, 1.0 / 240.0, False, skip_material=True)
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            solver._update_tendon_routing(state, 1.0 / 240.0, False, skip_material=True)
        wp.capture_launch(capture.graph)
    rest = solver.tendon_seg_rest_length.numpy()
    np.testing.assert_array_equal(rest, before)
    length = solver.tendon_seg_length.numpy()
    material = (length - rest) / model.tendon_seg_compliance.numpy()
    if nonlinear:
        strain = np.maximum(length - rest, 0.0) / rest
        material = 500.0 * (1.0 + 4.5 * (1.0 + np.tanh((strain - 0.01) / 0.003))) * strain
    damping_force = damping * (length - initial_length) * 240.0
    expected = np.maximum(material + damping_force, 0.0)
    np.testing.assert_allclose(solver.tendon_seg_damping_tension.numpy(), damping_force, atol=1.0e-5, rtol=2.0e-6)
    np.testing.assert_allclose(solver.tendon_material_modes.numpy()["tension"], expected, atol=1.0e-5, rtol=2.0e-6)
    np.testing.assert_array_equal(solver._tendon_link_material_transfer.numpy(), 0.0)


class TestTendonMaterialSlip(unittest.TestCase):
    """Validate linear/scalar and nonlinear/cooperative slip publication."""


for device in get_test_devices():
    add_function_test(
        TestTendonMaterialSlip, "test_sticking_transfer_is_zero", test_sticking_transfer_is_zero, devices=[device]
    )
    add_function_test(
        TestTendonMaterialSlip, "test_direct_slip_matches_transfer", test_direct_slip_matches_transfer, devices=[device]
    )
    add_function_test(
        TestTendonMaterialSlip,
        "test_direct_slip_hold_and_rejection",
        test_direct_slip_hold_and_rejection,
        devices=[device],
    )

    for nonlinear in (False, True):
        for damping in (0.0, 2.0):
            add_function_test(
                TestTendonMaterialSlip,
                f"test_material_hold_refreshes_tension_nonlinear_{nonlinear}_damping_{int(damping)}",
                test_material_hold_refreshes_tension,
                devices=[device],
                nonlinear=nonlinear,
                damping=damping,
            )


if __name__ == "__main__":
    wp.init()
    unittest.main()
