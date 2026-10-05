# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise exact ellipse/sector routes through the VBD integration."""

import math
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.tendon_kernels import (
    TendonForceElementAdjacencyInfo,
    TendonMaterialMode,
    evaluate_tendon_force_hessians,
    prepare_tendon_material_modes,
)
from newton.geometry import RollerProfileCircle, RollerProfileEllipse, RollerProfileSector
from newton.solvers import SolverVBD
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _forces(
    body: int,
    q: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    adjacency: TendonForceElementAdjacencyInfo,
    modes: wp.array[TendonMaterialMode],
    bodies: wp.array[int],
    attachment_l: wp.array[wp.vec3],
    attachment_r: wp.array[wp.vec3],
    active: wp.array[int],
    link_l: wp.array[int],
    link_r: wp.array[int],
    result: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
):
    force, torque, h_ll, h_al, h_aa = evaluate_tendon_force_hessians(
        body,
        q,
        com,
        adjacency,
        modes,
        bodies,
        attachment_l,
        attachment_r,
        active,
        link_l,
        link_r,
    )
    result[0] = force
    result[1] = torque
    hessian[0] = h_ll
    hessian[1] = h_al
    hessian[2] = h_aa


def _build(
    profile, device, *, mu=0.0, damping=0.0, movable=False, orientation=1, solver_type=SolverVBD, radius_only=False
):
    builder = newton.ModelBuilder(gravity=0.0)
    anchor_l = builder.add_body(xform=wp.transform(p=(-0.3, 0.1 * orientation, 0.0)), is_kinematic=True)
    anchor_r = builder.add_body(xform=wp.transform(p=(0.25, 0.16 * orientation, 0.0)), is_kinematic=True)
    roller = builder.add_body(
        xform=wp.transform(
            p=(0.017, -0.011 * orientation, 0), q=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.43 * orientation)
        ),
        mass=1.0,
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
        is_kinematic=not movable,
    )
    builder.add_tendon()
    builder.add_tendon_link(anchor_l, newton.TendonLinkType.ATTACHMENT)
    builder.add_tendon_link(
        roller,
        newton.TendonLinkType.ROLLING,
        **({"radius": profile.radius} if radius_only else {"profile": profile}),
        compliance=1e-3,
        damping=damping,
        mu=mu,
        orientation=orientation,
    )
    builder.add_tendon_link(anchor_r, newton.TendonLinkType.ATTACHMENT, compliance=1e-3, damping=damping)
    builder.color()
    model = builder.finalize(device=device)
    solver = solver_type(model, iterations=16, tendon_settle_tol=1e-6)
    return builder, model, solver, roller


def _load(model, solver, state, tensions=(4.0, 4.0), dt=0.001):
    compliance = model.tendon_seg_compliance.numpy()
    rest = solver.tendon_seg_length.numpy() - np.asarray(tensions) * compliance
    solver.tendon_seg_rest_length.assign(rest)
    solver._snapshot_tendon_step_state()
    solver._prepare_tendon_route(model, state.body_q, 1e-8)
    if isinstance(solver, SolverVBD):
        wp.copy(solver.body_q_prev, state.body_q)
        solver._update_tendon_routing(state, dt, True)
    else:
        solver._update_tendon_attachments(state.body_q)
        solver._update_tendon_cone_rows(model, state.body_q, True)
        solver._solve_tendon_material(state.body_q, state.body_qd, state.body_q, dt)


def _measure(model, solver, state, body, dt=0.001):
    result = wp.zeros(2, dtype=wp.vec3, device=model.device)
    hessian = wp.zeros(3, dtype=wp.mat33, device=model.device)
    wp.launch(
        _forces,
        dim=1,
        inputs=[
            body,
            state.body_q,
            model.body_com,
            solver.tendon_adjacency,
            solver.tendon_material_modes,
            model.tendon_link_body,
            solver.tendon_seg_attachment_l,
            solver.tendon_seg_attachment_r,
            solver.tendon_seg_active,
            solver.tendon_seg_active_link_l,
            solver.tendon_seg_active_link_r,
        ],
        outputs=[result, hessian],
        device=model.device,
    )
    return result.numpy(), hessian.numpy()


def test_profile_loads(test, device):
    """Preserve non-circular frictionless torque and physical endpoint forces in VBD."""
    for profile in (
        RollerProfileEllipse(0.08, 0.02),
        RollerProfileSector(0.08, 2.2),
        RollerProfileCircle(0.05),
    ):
        _, model, solver, roller = _build(profile, device)
        state = model.state()
        _load(model, solver, state)
        loads, hessian = _measure(model, solver, state, roller)
        a = solver.tendon_seg_attachment_l.numpy()
        b = solver.tendon_seg_attachment_r.numpy()
        directions = (b - a) / np.linalg.norm(b - a, axis=1)[:, None]
        tension = (
            solver.tendon_seg_length.numpy() - solver.tendon_seg_rest_length.numpy()
        ) / model.tendon_seg_compliance.numpy()
        com = state.body_q.numpy()[roller, :3]
        expected_force = -tension[0] * directions[0] + tension[1] * directions[1]
        expected_torque = np.cross(b[0] - com, -tension[0] * directions[0]) + np.cross(
            a[1] - com, tension[1] * directions[1]
        )
        # Local-to-world reconstruction and float32 norms contribute about one
        # position ULP / compliance; this bound remains below 0.003% of the load.
        np.testing.assert_allclose(loads, [expected_force, expected_torque], atol=1e-4)
        if isinstance(profile, RollerProfileCircle):
            test.assertAlmostEqual(float(loads[1, 2]), 0.0, delta=3e-6)
        else:
            test.assertGreater(abs(float(loads[1, 2])), 0.005)
        matrix = np.block([[hessian[0], hessian[1].T], [hessian[1], hessian[2]]])
        test.assertGreater(np.min(np.linalg.eigvalsh(matrix)), -0.003)


def test_profile_friction(test, device):
    """Project finite-friction tension onto the actual tangent-turn capstan bound."""
    for profile in (RollerProfileEllipse(0.08, 0.02), RollerProfileSector(0.08, 2.2)):
        _, model, solver, _ = _build(profile, device, mu=0.3)
        state = model.state()
        _load(model, solver, state, (10.0, 1.0))
        tension = (
            solver.tendon_seg_length.numpy() - solver.tendon_seg_rest_length.numpy()
        ) / model.tendon_seg_compliance.numpy()
        a = solver.tendon_seg_attachment_l.numpy()
        b = solver.tendon_seg_attachment_r.numpy()
        d = (b - a) / np.linalg.norm(b - a, axis=1)[:, None]
        turn = math.atan2(np.linalg.norm(np.cross(d[0], d[1])), np.dot(d[0], d[1]))
        ratio = math.exp(0.3 * turn)
        test.assertGreater(ratio, 1.05)
        test.assertAlmostEqual(float(solver.tendon_link_cap_ratio.numpy()[1]), ratio, delta=1e-6)
        test.assertAlmostEqual(float(tension[0] / tension[1]), ratio, delta=6e-4)
        test.assertAlmostEqual(float(np.sum(tension)), 11.0, delta=0.002)


def test_stiff_circle_preserves_projected_balance(test, device):
    """Keep equal projected tensions equal when float32 rest lengths cannot encode their stretch."""
    for compliance in (1e-4, 1e-6, 1e-8):
        with test.subTest(compliance=compliance):
            _, model, solver, roller = _build(RollerProfileCircle(0.05), device)
            model.tendon_seg_compliance.fill_(compliance)
            state = model.state()
            poses = state.body_q.numpy()
            poses[0, 0] = -1.2
            state.body_q.assign(poses)
            solver._update_tendon_attachments(state.body_q)
            _load(model, solver, state)
            loads, _ = _measure(model, solver, state, roller)
            test.assertLess(abs(float(loads[1, 2])), 5e-7)


def test_circle_sliding_curvature(test, device):
    """Do not assign spin stiffness to a centered frictionless circular route."""
    _, model, solver, roller = _build(RollerProfileCircle(0.05), device)
    state = model.state()
    _load(model, solver, state)
    _, hessian = _measure(model, solver, state, roller)
    # After free material transfer, rotating the centered circle changes
    # neither path length nor tension. Its spin row and column must vanish.
    np.testing.assert_allclose(hessian[1][2], 0.0, atol=1e-5, rtol=0)
    np.testing.assert_allclose(hessian[2][2], 0.0, atol=1e-5, rtol=0)


def test_damped_material_modes_use_total_tension(test, device):
    """Retain damping-supported tension and its free material mode at nonpositive elastic stretch."""
    stretch = np.array([0.0006, -0.001, -0.0001, 0.0, -0.001, 0.0006, 0.0006])
    damping_force = np.array([0.0, 80.0, 80.0, 80.0, 20.0, -40.0, -20.0])
    count = len(stretch)
    compliance = np.full(count, 2.0e-5)
    expected = np.maximum(stretch / compliance + damping_force, 0.0)
    cone_l = np.full(count + 1, -1)
    cone_r = np.full(count + 1, -1)
    cone_l[1], cone_r[1] = 0, 1

    def array(values, dtype=float):
        return wp.array(values, dtype=dtype, device=device)

    modes = wp.empty(count, dtype=TendonMaterialMode, device=device)
    wp.launch(
        prepare_tendon_material_modes,
        dim=1,
        inputs=[
            array([0, count + 1], int),
            array(cone_l, int),
            array(cone_r, int),
            array(np.ones(count + 1)),
            array(1.0 - stretch),
            array(np.ones(count)),
            array(stretch),
            array(compliance),
            array(np.full(count, 80.0)),
            array(damping_force),
            array(np.ones(count), int),
            0.001,
            0.0,
            20.0,
            0.001,
            0.001,
        ],
        outputs=[modes],
        device=device,
    )
    result = modes.numpy()
    np.testing.assert_allclose(result["tension"], expected, atol=1.0e-5)
    np.testing.assert_allclose(result["stiffness"], np.where(expected > 0.0, 130000.0, 0.0), rtol=1.0e-6)
    np.testing.assert_array_equal(result["first"][:2], [0, 0])
    np.testing.assert_array_equal(result["last"][:2], [1, 1])


def test_material_curvature_reference(test, device):
    """Match independent dense elimination, including repeated bodies and cone boundaries."""
    rng = np.random.default_rng(2976)
    for count in (1, 2, 8, 32):
        for damping_scale in (0.0, 2.0):
            with test.subTest(count=count, damping=damping_scale):
                dt = 0.003
                comp = rng.uniform(1e-4, 1e-3, count)
                damping = rng.uniform(0, damping_scale, count)
                left = rng.uniform(-1, 1, (count, 3))
                right = rng.uniform(-1, 1, (count, 3))
                length = np.linalg.norm(right - left, axis=1)
                force = np.full(count, 4.0)
                stretch = comp * force
                rest = length - stretch
                ratio = np.ones(count + 1)
                active = np.ones(count, dtype=np.int32)
                bodies = np.arange(count + 1, dtype=np.int32) % 3
                if count > 2:
                    # A finite-friction boundary remains on the sticking tangent
                    # even when the projected force is exactly on its slip bound.
                    ratio[2] = 1.7
                    force[2] *= ratio[2]
                    stretch[2] = comp[2] * force[2]
                    bodies[-1] = bodies[-2]  # one same-body span
                    rest[-2] = 1e-6  # hard depletion breaks the free mode
                cone_l = np.arange(-1, count, dtype=np.int32)
                cone_r = np.arange(count + 1, dtype=np.int32)
                cone_r[[0, -1]] = -1
                cone_l[-1] = -1
                link_l = np.arange(count, dtype=np.int32)
                link_r = np.arange(1, count + 1, dtype=np.int32)
                if count > 2:
                    # A deactivated guide leaves an inactive slot inside a
                    # material block, while the preceding span bypasses it.
                    gap = count // 2
                    active[gap] = 0
                    force[gap] = 0
                    link_r[gap - 1] = gap + 1
                    cone_l[gap] = cone_r[gap] = -1
                    cone_l[gap + 1] = gap - 1
                modes = wp.empty(count, dtype=TendonMaterialMode, device=device)

                def array(value, dtype=float):
                    return wp.array(value, dtype=dtype, device=device)

                wp.launch(
                    prepare_tendon_material_modes,
                    dim=1,
                    inputs=[
                        array([0, count + 1], int),
                        array(cone_l, int),
                        array(cone_r, int),
                        array(ratio),
                        array(rest),
                        array(length),
                        array(stretch),
                        array(comp),
                        array(damping),
                        array(np.zeros(count)),
                        array(active, int),
                        dt,
                        0.0,
                        20.0,
                        0.001,
                        0.001,
                    ],
                    outputs=[modes],
                    device=device,
                )
                # Eliminate one independent material-transfer variable for
                # each freely sliding interface from the full diagonal energy.
                transfer = []
                for link in range(1, count):
                    a, b = cone_l[link], cone_r[link]
                    if (
                        a >= 0
                        and b >= 0
                        and active[a]
                        and active[b]
                        and ratio[link] == 1
                        and min(rest[a], rest[b]) > 1.001e-6
                    ):
                        column = np.zeros(count)
                        column[a], column[b] = 1, -1
                        transfer.append(column)
                stiffness = np.diag(active * (1 / comp + damping / dt))
                condensed = stiffness.copy()
                if transfer:
                    transfer = np.asarray(transfer).T
                    condensed -= (
                        stiffness
                        @ transfer
                        @ np.linalg.solve(transfer.T @ stiffness @ transfer, transfer.T @ stiffness)
                    )
                adjacency = TendonForceElementAdjacencyInfo()
                adjacency.body_adj_segments = array(np.tile(np.arange(count), 3), int)
                adjacency.body_adj_segments_offsets = array(np.arange(4) * count, int)
                directions = (right - left) / length[:, None]
                for body in range(3):
                    jacobian = np.zeros((count, 6))
                    for seg in range(count):
                        if bodies[link_l[seg]] == body:
                            jacobian[seg] += np.r_[directions[seg], np.cross(left[seg], directions[seg])]
                        if bodies[link_r[seg]] == body:
                            jacobian[seg] -= np.r_[directions[seg], np.cross(right[seg], directions[seg])]
                    result = wp.zeros(2, dtype=wp.vec3, device=device)
                    hessian = wp.zeros(3, dtype=wp.mat33, device=device)
                    wp.launch(
                        _forces,
                        dim=1,
                        inputs=[
                            body,
                            array([wp.transform_identity()] * 3, wp.transform),
                            array(np.zeros((3, 3)), wp.vec3),
                            adjacency,
                            modes,
                            array(bodies, int),
                            array(left, wp.vec3),
                            array(right, wp.vec3),
                            array(active, int),
                            array(link_l, int),
                            array(link_r, int),
                        ],
                        outputs=[result, hessian],
                        device=device,
                    )
                    blocks = hessian.numpy()
                    actual = np.block([[blocks[0], blocks[1].T], [blocks[1], blocks[2]]])
                    np.testing.assert_allclose(actual, jacobian.T @ condensed @ jacobian, rtol=3e-5, atol=0.005)
                    np.testing.assert_allclose(result.numpy().ravel(), jacobian.T @ force, rtol=1e-5, atol=5e-6)


def _tangent_route(profile, normal, point, device, *, orientation=1):
    """Construct a straight cable touching a prescribed profile."""
    tangent = orientation * np.array([-normal[1], normal[0]])
    builder = newton.ModelBuilder(gravity=0.0)
    left = builder.add_body(xform=wp.transform(p=(*(point - 0.3 * tangent), 0.0)), is_kinematic=True)
    right = builder.add_body(xform=wp.transform(p=(*(point + 0.4 * tangent), 0.0)), is_kinematic=True)
    roller = builder.add_body(is_kinematic=True)
    builder.add_tendon()
    builder.add_tendon_link(left, newton.TendonLinkType.ATTACHMENT)
    builder.add_tendon_link(
        roller, newton.TendonLinkType.ROLLING, profile=profile, compliance=1e-3, mu=0.3, orientation=orientation
    )
    builder.add_tendon_link(right, newton.TendonLinkType.ATTACHMENT, compliance=1e-3)
    builder.color()
    model = builder.finalize(device=device)
    return SolverVBD(model, iterations=1)


def test_profile_zero_wrap(test, device):
    """Keep tangent routes at zero wrap instead of inventing a full perimeter."""
    for profile in (RollerProfileCircle(0.05), RollerProfileEllipse(0.08, 0.02), RollerProfileSector(0.08, 2.2)):
        for angle in np.linspace(-math.pi, math.pi, 25):
            normal = np.array([math.cos(angle), math.sin(angle)])
            if isinstance(profile, RollerProfileCircle):
                point = profile.radius * normal
            elif isinstance(profile, RollerProfileEllipse):
                radii = np.array([profile.a, profile.b])
                point = radii**2 * normal / np.linalg.norm(radii * normal)
            else:
                theta = np.clip(angle, -profile.angle / 2, profile.angle / 2)
                point = profile.radius * np.array([math.cos(theta), math.sin(theta)])
                if point @ normal <= 0:
                    point = np.zeros(2)
            for orientation in (-1, 1):
                solver = _tangent_route(profile, normal, point, device, orientation=orientation)
                with test.subTest(profile=profile, angle=angle, orientation=orientation):
                    test.assertLess(float(solver.tendon_profile_wrap_length.numpy()[1]), 2e-6)
                    test.assertAlmostEqual(float(solver.tendon_total_cable.numpy()[0]), 0.7, delta=2e-6)


def test_sector_flat_edge(test, device):
    """Preserve cable along a straight sector edge despite zero tangent turning."""
    for angle in (0.5, 2.2, math.pi):
        profile = RollerProfileSector(0.08, angle)
        for side in (-1, 1):
            edge = np.array([math.cos(angle / 2), side * math.sin(angle / 2)])
            normal = np.array([-math.sin(angle / 2), side * math.cos(angle / 2)])
            for orientation in (-1, 1):
                solver = _tangent_route(profile, normal, 0.04 * edge, device, orientation=orientation)
                test.assertAlmostEqual(float(solver.tendon_total_cable.numpy()[0]), 0.7, delta=2e-6)
                test.assertAlmostEqual(float(solver.tendon_link_cap_ratio.numpy()[1]), 1.0, delta=2e-6)


def test_profile_transport(test, device):
    """Conserve material in the actual solver while an ellipse or pizza slice rotates."""
    for profile in (RollerProfileEllipse(0.08, 0.02), RollerProfileSector(0.08, 2.2)):
        for mu in (0.0, 0.3, 10.0):
            _, model, solver, roller = _build(profile, device, mu=mu)
            state = model.state()
            _load(model, solver, state)
            total = float(
                np.sum(solver.tendon_seg_rest_length.numpy()) + np.sum(solver.tendon_profile_wrap_length.numpy())
            )
            poses = state.body_q.numpy()
            for angle in np.linspace(0.43, 0.63, 30):
                solver._snapshot_tendon_step_state()
                wp.copy(solver.body_q_prev, state.body_q)
                poses[roller, 3:] = (0, 0, math.sin(angle / 2), math.cos(angle / 2))
                state.body_q.assign(poses)
                solver._prepare_tendon_route(model, state.body_q, 1e-8)
                solver._update_tendon_routing(state, 0.001, True)
                measured = float(
                    np.sum(solver.tendon_seg_rest_length.numpy()) + np.sum(solver.tendon_profile_wrap_length.numpy())
                )
                test.assertAlmostEqual(measured, total, delta=2e-6)
                test.assertFalse(np.any(solver.tendon_profile_tangent_status.numpy()))
                test.assertFalse(np.any(solver.tendon_profile_wrap_status.numpy()))


def test_profile_step(test, device):
    """Advance loaded non-circular rollers with VBD and retain finite conserved state."""
    for profile in (RollerProfileEllipse(0.08, 0.02), RollerProfileSector(0.08, 2.2)):
        _, model, solver, roller = _build(profile, device, movable=True)
        a, b = model.state(), model.state()
        _load(model, solver, a)
        initial_load, _ = _measure(model, solver, a, roller)
        total = float(np.sum(solver.tendon_seg_rest_length.numpy()) + np.sum(solver.tendon_profile_wrap_length.numpy()))
        solver.step(a, b, None, None, 1e-3)
        velocity = b.body_qd.numpy()[roller]
        np.testing.assert_allclose(velocity[:3], initial_load[0] * 1e-3, rtol=0.03, atol=1e-5)
        np.testing.assert_allclose(velocity[3:], initial_load[1] * 1e-2, rtol=0.05, atol=2e-5)
        for _ in range(20):
            a, b = b, a
            solver.step(a, b, None, None, 1e-3)
        test.assertTrue(np.isfinite(b.body_q.numpy()).all())
        test.assertGreater(np.linalg.norm(b.body_q.numpy()[roller, :3] - model.body_q.numpy()[roller, :3]), 1e-6)
        # Re-evaluate geometry at the accepted state before comparing material.
        solver._update_profile_attachments(b.body_q, rolling=False)
        solver._update_profile_cones(b.body_q, True)
        measured = float(
            np.sum(solver.tendon_seg_rest_length.numpy()) + np.sum(solver.tendon_profile_wrap_length.numpy())
        )
        test.assertAlmostEqual(measured, total, delta=3e-6)


def test_profile_frictionless_motion(test, device):
    """Keep a frictionless circle stationary while its loaded cable ends move."""
    from newton.examples.cable.example_roller_profiles import Example  # noqa: PLC0415

    with wp.ScopedDevice(device):
        for friction, damping in ((0.0, 1.0), (0.0, 0.0), (0.3, 1.0)):
            example = Example(None, SimpleNamespace(substeps=10, iterations=8, friction=friction))
            example.model.tendon_seg_damping.fill_(damping)
            circle = example.roller_indices[0]
            for _ in range(6):
                example.step()
            velocity = float(example.state_0.body_qd.numpy()[circle, 4])
            if friction == 0.0:
                # Neither the circular boundary nor the cable's length depends
                # on its angle. Sliding must not accelerate this centered hinge.
                test.assertAlmostEqual(velocity, 0.0, delta=0.003)
                np.testing.assert_allclose(example.state_0.body_q.numpy()[circle, 3:], [0, 0, 0, 1], atol=2e-4, rtol=0)
            else:
                # The correction must not suppress legitimate friction torque.
                test.assertLess(velocity, -0.1)


def _reference_path_length(profile, anchors, center, angle):
    """Measure a planar route using independent float64 support bisection."""
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    def support(normal):
        local = rotation.T @ normal
        if isinstance(profile, RollerProfileSector):
            theta = np.clip(math.atan2(local[1], local[0]), -profile.angle / 2, profile.angle / 2)
            point = profile.radius * np.array([math.cos(theta), math.sin(theta)])
            if point @ local <= 0:
                point = np.zeros(2)
                coordinate = 0.0
            else:
                coordinate = profile.radius * (1 + theta + profile.angle / 2)
        else:
            radii = np.array([profile.a, profile.b])
            point = radii**2 * local / np.linalg.norm(radii * local)
            coordinate = math.atan2(point[1] / profile.b, point[0] / profile.a)
        return center + rotation @ point, coordinate

    points, coordinates = [], []
    for incoming, anchor in enumerate(anchors):
        along = (center - anchor) if incoming == 0 else (anchor - center)
        along = along / np.linalg.norm(along)
        across = np.array([-along[1], along[0]])
        low, high = -math.pi / 2, math.pi / 2
        for _ in range(60):
            middle = (low + high) / 2
            normal = math.sin(middle) * along - math.cos(middle) * across
            point, coordinate = support(normal)
            span = point - anchor if incoming == 0 else anchor - point
            if span @ normal < 0:
                low = middle
            else:
                high = middle
        points.append(point)
        coordinates.append(coordinate)
    if isinstance(profile, RollerProfileSector):
        arc = (coordinates[1] - coordinates[0]) % (profile.radius * (2 + profile.angle))
    else:
        delta = (coordinates[1] - coordinates[0]) % (2 * math.pi)
        nodes, weights = np.polynomial.legendre.leggauss(64)
        theta = coordinates[0] + (nodes + 1) * delta / 2
        arc = delta / 2 * np.dot(weights, np.hypot(profile.a * np.sin(theta), profile.b * np.cos(theta)))
    return np.linalg.norm(points[0] - anchors[0]) + arc + np.linalg.norm(anchors[1] - points[1])


def test_profile_force_reference(test, device):
    """Check non-circular VBD loads against an independent routed-length gradient."""
    for profile in (RollerProfileEllipse(0.08, 0.02), RollerProfileSector(0.08, 2.2)):
        _, model, solver, roller = _build(profile, device)
        state = model.state()
        for angle in (-0.6, 0.0, 0.43, 1.2):
            poses = model.body_q.numpy()
            poses[roller, 3:] = (0, 0, math.sin(angle / 2), math.cos(angle / 2))
            state.body_q.assign(poses)
            solver._update_profile_attachments(state.body_q, rolling=False)
            _load(model, solver, state)
            force, _ = _measure(model, solver, state, roller)
            center_angle = np.array([*poses[roller, :2], angle], dtype=np.float64)
            epsilon = 1e-5
            gradient = []
            for axis in range(3):
                shift = np.eye(3)[axis] * epsilon
                plus, minus = center_angle + shift, center_angle - shift
                gradient.append(
                    (
                        _reference_path_length(profile, poses[:2, :2], plus[:2], plus[2])
                        - _reference_path_length(profile, poses[:2, :2], minus[:2], minus[2])
                    )
                    / (2 * epsilon)
                )
            np.testing.assert_allclose(
                [force[0, 0], force[0, 1], force[1, 2]], -4 * np.array(gradient), atol=2e-4, rtol=2e-4
            )


def test_profile_example_graph(test, device, solver_name="vbd"):
    """Match graph and uncaptured trajectories, clocks, and odd/even buffers."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    from newton.examples.cable.example_roller_profiles import Example  # noqa: PLC0415

    with wp.ScopedDevice(device):
        for substeps in (3, 4):
            graph = Example(None, SimpleNamespace(substeps=substeps, iterations=4, solver=solver_name))
            plain = Example(
                None, SimpleNamespace(substeps=substeps, iterations=4, solver=solver_name, disable_cuda_graph=True)
            )
            test.assertEqual(int(graph.frame.numpy()[0]), 0)
            np.testing.assert_array_equal(graph.state_0.body_q.numpy(), graph.model.body_q.numpy())
            for frame in range(8):
                graph.step()
                plain.step()
                np.testing.assert_allclose(
                    graph.state_0.body_q.numpy(), plain.state_0.body_q.numpy(), atol=2e-7, rtol=0
                )
                np.testing.assert_allclose(
                    graph.solver.tendon_seg_rest_length.numpy(),
                    plain.solver.tendon_seg_rest_length.numpy(),
                    atol=2e-7,
                    rtol=0,
                )
                test.assertEqual(int(graph.frame.numpy()[0]), frame + 1)
                t = frame * graph.frame_dt + (substeps - 1) * graph.sim_dt
                expected_z = 0.70 - 0.04 * (1.0 - math.cos(2 * math.pi * t / 4.0))
                np.testing.assert_allclose(
                    graph.state_0.body_q.numpy()[graph.anchors.numpy(), 2], expected_z, atol=1e-7, rtol=0
                )


class TestRollerProfileVBD(unittest.TestCase):
    def test_coloring_separates_material_coupling(self):
        """Bodies coupled through free material must not share a VBD update color."""
        builder = newton.ModelBuilder(gravity=0.0)
        bodies = [
            builder.add_body(xform=wp.transform(p=(i, i % 2, 0)), mass=1.0, inertia=wp.mat33(np.eye(3)))
            for i in range(4)
        ]
        builder.add_tendon()
        for i, body in enumerate(bodies):
            builder.add_tendon_link(
                body,
                newton.TendonLinkType.ATTACHMENT if i in (0, 3) else newton.TendonLinkType.PINHOLE,
                compliance=1e-3,
            )
        builder.color()
        model = builder.finalize(device="cpu")
        colors = np.full(model.body_count, -1)
        for color, group in enumerate(model.body_color_groups):
            colors[group.numpy()] = color
        self.assertEqual(len(set(colors)), 4)
        SolverVBD(model, iterations=1)
        # Adjacent endpoints are separate, but material connects the two pairs.
        model.body_color_groups = [wp.array([0, 2], dtype=int), wp.array([1, 3], dtype=int)]
        with self.assertRaisesRegex(ValueError, "material-coupled tendon bodies"):
            SolverVBD(model, iterations=1)

    def test_mixed_radius_circle_axis(self):
        """Radius-only circles need no authored in-plane axis in a mixed model."""
        builder = newton.ModelBuilder(gravity=0.0)
        for i in range(2):
            left = builder.add_body(xform=wp.transform(p=(i, -0.3, 0.1)), is_kinematic=True)
            roller = builder.add_body(xform=wp.transform(p=(i, 0, 0)), is_kinematic=True)
            right = builder.add_body(xform=wp.transform(p=(i, 0.3, 0.1)), is_kinematic=True)
            builder.add_tendon()
            builder.add_tendon_link(left, newton.TendonLinkType.ATTACHMENT)
            shape = (
                {"radius": 0.05} if i == 0 else {"profile": RollerProfileEllipse(0.08, 0.04), "profile_axis": (0, 1, 0)}
            )
            builder.add_tendon_link(roller, newton.TendonLinkType.ROLLING, axis=(1, 0, 0), compliance=1e-4, **shape)
            builder.add_tendon_link(right, newton.TendonLinkType.ATTACHMENT, compliance=1e-4)
        builder.color()
        model = builder.finalize(device="cpu")
        solver = SolverVBD(model, iterations=4)
        self.assertFalse(np.any(solver.tendon_profile_tangent_status.numpy()))
        entry = solver.tendon_seg_attachment_r_local.numpy()[0]
        exit = solver.tendon_seg_attachment_l_local.numpy()[1]
        np.testing.assert_allclose(np.linalg.norm([entry, exit], axis=1), 0.05, atol=1e-7)

    def test_circle_default_and_explicit_geometry(self):
        """Radius-only links retain their path and agree with the explicit circle."""
        contacts = []
        for explicit in (False, True):
            builder = newton.ModelBuilder(gravity=0.0)
            left = builder.add_body(xform=wp.transform(p=(-0.3, 0.1, 0)), is_kinematic=True)
            roller = builder.add_body(is_kinematic=True)
            right = builder.add_body(xform=wp.transform(p=(0.3, 0.1, 0)), is_kinematic=True)
            builder.add_tendon()
            builder.add_tendon_link(left, newton.TendonLinkType.ATTACHMENT)
            shape = {"profile": RollerProfileCircle(0.05)} if explicit else {"radius": 0.05}
            builder.add_tendon_link(roller, newton.TendonLinkType.ROLLING, compliance=1e-4, **shape)
            builder.add_tendon_link(right, newton.TendonLinkType.ATTACHMENT, compliance=1e-4)
            builder.color()
            model = builder.finalize(device="cpu")
            self.assertEqual(model.tendon_profile_routing, explicit)
            solver = SolverVBD(model, iterations=4)
            contacts.append(np.array([solver.tendon_seg_attachment_l.numpy(), solver.tendon_seg_attachment_r.numpy()]))
        np.testing.assert_allclose(contacts[0], contacts[1], atol=2e-7, rtol=0)


for fn in (
    test_profile_loads,
    test_profile_friction,
    test_stiff_circle_preserves_projected_balance,
    test_circle_sliding_curvature,
    test_material_curvature_reference,
    test_damped_material_modes_use_total_tension,
    test_profile_zero_wrap,
    test_sector_flat_edge,
    test_profile_transport,
    test_profile_step,
    test_profile_frictionless_motion,
    test_profile_force_reference,
    test_profile_example_graph,
):
    add_function_test(TestRollerProfileVBD, fn.__name__, fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
