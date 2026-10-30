# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse tendon fixtures registered by test_vbd_sparse_articulation."""

import numpy as np
import warp as wp

import newton
from newton.examples.cable.example_tendon_pulley import Example as PulleyExample
from newton.tests.test_tendon_capstan import _oriented_route_material_length
from newton.tests.test_tendon_vbd import _install_vbd_example_solver, _make_example_args


def _build_straight_route(device, count=2, separate_articulations=False, fixed_first=False, closed=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    for i in range(count):
        pose = wp.transform(wp.vec3(float(i) / (count - 1), 0.0, 0.0), wp.quat_identity())
        fixed = fixed_first and i == 0
        body = builder.add_link(
            xform=pose,
            mass=0.0 if fixed else 1.0,
            inertia=wp.mat33(np.zeros((3, 3)) if fixed else np.eye(3)),
        )
        joints.append(builder.add_joint_prismatic(-1, body, parent_xform=pose, axis=(1.0, 0.0, 0.0)))
    if separate_articulations:
        for joint in joints:
            builder.add_articulation([joint])
    else:
        if closed:
            joints.append(
                builder.add_joint_prismatic(
                    0,
                    count - 1,
                    parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
                    axis=(1.0, 0.0, 0.0),
                )
            )
        builder.add_articulation(joints, allow_closed_loops=closed)
    builder.add_tendon()
    for i in range(count):
        kind = newton.TendonLinkType.PINHOLE if 0 < i < count - 1 else newton.TendonLinkType.ATTACHMENT
        builder.add_tendon_link(
            i, link_type=int(kind), rest_length=0.8 / (count - 1), compliance=1.0e-3, damping=0.0, mu=0.0
        )
    builder.color()
    return builder.finalize(device=device)


def _check_implicit_step(
    test,
    device,
    count=2,
    separate_articulations=False,
    fixed_first=False,
    closed=False,
    sparse_mode="block_sparse_joints",
):
    model = _build_straight_route(device, count, separate_articulations, fixed_first, closed)
    solver = newton.solvers.SolverVBD(
        model, iterations=1, rigid_articulation_solve=sparse_mode, rigid_articulation_relaxation=1.0
    )
    state, result = model.state(), model.state()
    dt = 0.01
    solver.step(state, result, model.control(), None, dt)
    # Freely sliding spans are springs in series. Internal collinear pinholes
    # have zero path gradient, but the two end bodies remain coupled.
    stiffness = 1000.0 / (count - 1)
    moving_ends = 1 if fixed_first else 2
    dx = dt * dt * stiffness * 0.2 / (1.0 + moving_ends * dt * dt * stiffness)
    expected = np.linspace(0.0, 1.0, count)
    if not fixed_first:
        expected[0] += dx
    expected[-1] -= dx
    np.testing.assert_allclose(result.body_q.numpy()[:, 0], expected, atol=2.0e-6, rtol=0.0)
    test.assertEqual(solver.rigid_articulation_sparse_layout.articulation_count, 1)


def test_sparse_tendon_implicit_step(test, device, sparse_mode="block_sparse_joints"):
    """Match the analytical implicit step of a two-body tensile spring."""
    _check_implicit_step(test, device, sparse_mode=sparse_mode)


def test_sparse_tendon_local_fallback_metric(test, device, sparse_mode="block_sparse_joints"):
    """Sparse selection must not change friction curvature on locally solved bodies."""

    def two_steps(solver, state, next_state, control):
        solver.step(state, next_state, control, None, 1.0 / 600.0)
        solver.step(next_state, state, control, None, 1.0 / 600.0)

    for mixed in (False, True):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        guide = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()), mass=0.0)
        ends = []
        for x, mass in ((-0.3, 1.0), (0.3, 1.6)):
            ends.append(
                builder.add_link(
                    xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()),
                    mass=mass,
                    inertia=wp.mat33(np.eye(3) * 0.01),
                )
            )
        builder.add_tendon()
        for body in (ends[0], guide, ends[1]):
            builder.add_tendon_link(
                body,
                link_type=newton.TendonLinkType.PINHOLE if body == guide else newton.TendonLinkType.ATTACHMENT,
                compliance=1.0e-5,
                damping=0.0,
                mu=0.1,
            )
        if mixed:
            pose = wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity())
            body = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(-1, body, parent_xform=pose, axis=(1.0, 0.0, 0.0))
            builder.add_articulation([joint])
        builder.color()
        model = builder.finalize(device=device)
        results = []
        for mode in ("local", sparse_mode):
            for capture in (False, True) if device.is_cuda else (False,):
                solver = newton.solvers.SolverVBD(model, iterations=4, rigid_articulation_solve=mode)
                state, next_state, control = model.state(), model.state(), model.control()

                two_steps(solver, state, next_state, control)  # Compile before capture with the same history.
                if capture:
                    with wp.ScopedCapture(device=device) as captured:
                        two_steps(solver, state, next_state, control)
                    for _ in range(29):
                        wp.capture_launch(captured.graph)
                else:
                    for _ in range(29):
                        two_steps(solver, state, next_state, control)
                results.append(state.body_q.numpy()[ends])
                if mode == sparse_mode:
                    test.assertEqual(solver.rigid_articulation_sparse_layout is not None, mixed)
        with test.subTest(mixed=mixed):
            for result in results[1:]:
                np.testing.assert_array_equal(results[0], result)


def test_sparse_tendon_merges_articulations(test, device, sparse_mode="block_sparse_joints"):
    """Solve tendon-connected articulations as one coupled system."""
    _check_implicit_step(test, device, sparse_mode=sparse_mode, separate_articulations=True)


def test_sparse_tendon_world_isolation(test, device, sparse_mode="block_sparse_joints"):
    """Keep heterogeneous worlds independent when merging tendon-connected groups."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    expected = []
    for rest, compliance in ((0.8, 1.0e-3), (0.9, 2.0e-3), (1.1, 1.0e-3)):
        builder.begin_world()
        bodies = []
        for x in (0.0, 1.0):
            pose = wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity())
            body = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(-1, body, parent_xform=pose, axis=(1.0, 0.0, 0.0))
            builder.add_articulation([joint])
            bodies.append(body)
        builder.add_tendon()
        for body in bodies:
            builder.add_tendon_link(
                body,
                link_type=int(newton.TendonLinkType.ATTACHMENT),
                rest_length=rest,
                compliance=compliance,
                damping=0.0,
            )
        builder.end_world()
        dx = 0.01**2 * max(1.0 - rest, 0.0) / (compliance + 2.0 * 0.01**2)
        expected.extend((dx, 1.0 - dx))
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model, iterations=1, rigid_articulation_solve=sparse_mode, rigid_articulation_relaxation=1.0
    )
    layout = solver.rigid_articulation_sparse_layout
    test.assertEqual(layout.articulation_count, 3)
    test.assertEqual(layout.tendon_pair_count, 3)
    result = model.state()
    solver.step(model.state(), result, model.control(), None, 0.01)
    np.testing.assert_allclose(result.body_q.numpy()[:, 0], expected, atol=2.0e-6, rtol=0.0)


def test_sparse_tendon_nonadjacent_coupling(test, device, sparse_mode="block_sparse_joints"):
    """Retain endpoint coupling across a freely sliding internal guide."""
    _check_implicit_step(test, device, sparse_mode=sparse_mode, count=3)


def test_sparse_tendon_fixed_body(test, device, sparse_mode="block_sparse_joints"):
    """Keep a fixed tendon endpoint out of the free-body correction."""
    _check_implicit_step(test, device, sparse_mode=sparse_mode, fixed_first=True)


def test_sparse_tendon_closed_loop(test, device, sparse_mode="block_sparse_joints"):
    """Include closed-loop joint constraints in the tendon-coupled solve."""
    _check_implicit_step(test, device, sparse_mode=sparse_mode, closed=True)


def test_sparse_tendon_loaded_closed_loop(test, device, sparse_mode="block_sparse_joints"):
    """Transmit tendon force through a closing joint, matching a coupled reference."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    anchor = builder.add_body(mass=0.0)
    bodies, joints = [], []
    for x in (0.0, 1.0):
        pose = wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity())
        body = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
        bodies.append(body)
        joints.append(builder.add_joint_prismatic(-1, body, parent_xform=pose, axis=(1.0, 0.0, 0.0)))
    joints.append(
        builder.add_joint_fixed(
            bodies[0], bodies[1], parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())
        )
    )
    builder.add_articulation(joints, allow_closed_loops=True)
    builder.add_tendon()
    for body, offset in ((anchor, (-1.0, 0.0, 0.0)), (bodies[1], (0.0, 0.0, 0.0))):
        builder.add_tendon_link(
            body,
            offset=offset,
            link_type=int(newton.TendonLinkType.ATTACHMENT),
            rest_length=1.5,
            compliance=1.0e-3,
            damping=0.0,
        )
    builder.color()
    model = builder.finalize(device=device)
    dt, joint_stiffness = 0.01, 1.0e5
    tendon_stiffness = 1.0 / float(model.tendon_seg_compliance.numpy()[0])
    # The closing joint has finite authored stiffness, not an exact weld.
    # M/dt^2 * dx + K_joint * [dx0-dx1, dx1-dx0] + [0, K_tendon*(0.5+dx1)] = 0.
    matrix = np.diag(model.body_mass.numpy()[bodies].astype(np.float64) / dt**2)
    matrix += joint_stiffness * np.array([[1.0, -1.0], [-1.0, 1.0]])
    matrix[1, 1] += tendon_stiffness
    expected = np.array([0.0, 1.0]) + np.linalg.solve(matrix, [0.0, -0.5 * tendon_stiffness])
    for mode in ("local", sparse_mode):
        for direct in (False, True):
            for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
                with test.subTest(mode=mode, direct=direct, capture=capture):
                    options = {
                        "iterations": 64,
                        "rigid_articulation_solve": mode,
                        "rigid_articulation_relaxation": 1.0,
                        "rigid_joint_linear_ke": joint_stiffness,
                        "rigid_joint_linear_kd": 0.0,
                        "tendon_material_direct": direct,
                    }
                    if capture:
                        warm = newton.solvers.SolverVBD(model, **options)
                        warm.step(model.state(), model.state(), model.control(), None, dt)
                    solver = newton.solvers.SolverVBD(model, **options)
                    state, result, control = model.state(), model.state(), model.control()
                    if capture:
                        with wp.ScopedCapture(device=device) as captured:
                            solver.step(state, result, control, None, dt)
                        wp.capture_launch(captured.graph)
                    else:
                        solver.step(state, result, control, None, dt)
                    np.testing.assert_allclose(result.body_q.numpy()[bodies, 0], expected, atol=2.0e-6, rtol=0.0)
                    test.assertLess(float(result.body_q.numpy()[bodies[0], 0]), -0.02)


def test_sparse_tendon_frictional_balance(test, device, sparse_mode="block_sparse_joints"):
    """Converge stiff sliding spans to an independent implicit capstan balance."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    guide_body = builder.add_body(mass=0.0)
    joints, bodies = [], []
    for axis, mass in (((-1.0, 0.0, 0.0), 0.001), ((0.0, 0.0, 1.0), 0.01)):
        pose = wp.transform(wp.vec3(*axis), wp.quat_identity())
        body = builder.add_link(xform=pose, mass=mass, inertia=wp.mat33(np.eye(3)))
        joints.append(builder.add_joint_prismatic(-1, body, parent_xform=pose, axis=axis))
        bodies.append(body)
    builder.add_articulation(joints)
    builder.add_tendon()
    for body, kind, friction in (
        (bodies[0], newton.TendonLinkType.ATTACHMENT, 0.0),
        (guide_body, newton.TendonLinkType.PINHOLE, 0.3),
        (bodies[1], newton.TendonLinkType.ATTACHMENT, 0.0),
    ):
        builder.add_tendon_link(
            body, link_type=int(kind), rest_length=0.9999, compliance=1.0e-6, damping=0.0, mu=friction
        )
    builder.color()
    model = builder.finalize(device=device)
    dt = 0.001
    masses = model.body_mass.numpy()[bodies].astype(np.float64)
    for direct in (False, True):
        with test.subTest(direct=direct):
            solver = newton.solvers.SolverVBD(
                model,
                iterations=32,
                rigid_articulation_solve=sparse_mode,
                rigid_articulation_relaxation=1.0,
                tendon_material_direct=direct,
                tendon_material_direct_settle_iterations=0,
            )
            state, result, control = model.state(), model.state(), model.control()
            force = state.body_f.numpy()
            force[bodies[0], 0] = -10.0
            state.body_f.assign(force)

            # Orthogonal slider axes keep the pinhole turn fixed at pi/2.
            # L_i = p_i - dt^2*T_i/m_i and sum(L_i-c_i*T_i) = sum(rest_i).
            # The trial exceeds the capstan bound, so T_left = rho*T_right.
            compliance = solver.tendon_seg_active_compliance.numpy().astype(np.float64)
            rest = solver.tendon_seg_rest_length.numpy().astype(np.float64)
            prediction = np.ones(2) + dt * dt * np.array([10.0, 0.0]) / masses
            a = compliance + dt * dt / masses
            rho = np.exp(float(np.float32(0.3)) * np.pi / 2.0)
            sticking = (prediction - rest) / a
            test.assertGreater(sticking[0], rho * sticking[1])
            tension_right = (prediction.sum() - rest.sum()) / (rho * a[0] + a[1])
            tension = np.array([rho * tension_right, tension_right])
            expected = prediction - dt * dt * tension / masses

            solver.step(state, result, control, None, dt)
            pose = result.body_q.numpy()
            actual = np.array([-pose[bodies[0], 0], pose[bodies[1], 2]])
            np.testing.assert_allclose(actual, expected, atol=2.0e-6, rtol=0.0)
            # Float32 lengths near 1 m resolve about 0.1 N at this stiffness.
            np.testing.assert_allclose(solver.tendon_seg_material_tension.numpy(), tension, atol=0.15, rtol=0.0)


def test_sparse_tendon_nonlinear_damped_step(test, device, sparse_mode="block_sparse_joints"):
    """Match a nonlinear tensile spring's implicit step, including damping."""
    model = _build_straight_route(device)
    dt = 0.01
    rest = float(model.tendon_seg_rest_length.numpy()[0])
    inverse_mass = model.body_inv_mass.numpy().astype(np.float64)
    ea_low, ea_ratio, transition, width = 500.0, 5.0, 0.15, 0.04
    for damping in (0.0, 6.0):
        model.tendon_seg_damping.fill_(damping)

        def tension(length, damping=damping):
            strain = max(length - rest, 0.0) / rest
            blend = 0.5 * (1.0 + np.tanh((strain - transition) / width))
            elastic = ea_low * (1.0 + (ea_ratio - 1.0) * blend) * strain
            # The initially taut span contracts, so damping opposes unloading.
            return max(elastic + damping * (length - 1.0) / dt, 0.0)

        # Eliminate the two translations: L = 1 - dt^2*(1/m0+1/m1)*T(L).
        lower, upper = rest, 1.0
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            residual = middle - 1.0 + dt * dt * inverse_mass.sum() * tension(middle)
            if residual > 0.0:
                upper = middle
            else:
                lower = middle
        force = tension(0.5 * (lower + upper))
        expected = np.array([0.0, 1.0]) + dt * dt * inverse_mass * np.array([force, -force])
        test.assertGreater(force, 1.0)
        for mode in ("local", sparse_mode):
            for direct in (False, True):
                for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
                    with test.subTest(damping=damping, mode=mode, direct=direct, capture=capture):
                        options = {
                            "iterations": 32,
                            "rigid_articulation_solve": mode,
                            "rigid_articulation_relaxation": 1.0,
                            "tendon_material_direct": direct,
                            "tendon_sigmoid_ea_low": ea_low,
                            "tendon_sigmoid_ea_ratio": ea_ratio,
                            "tendon_sigmoid_transition_strain": transition,
                            "tendon_sigmoid_transition_width": width,
                        }
                        if capture:
                            warm = newton.solvers.SolverVBD(model, **options)
                            warm.step(model.state(), model.state(), model.control(), None, dt)
                        solver = newton.solvers.SolverVBD(model, **options)
                        state, result, control = model.state(), model.state(), model.control()
                        if capture:
                            with wp.ScopedCapture(device=device) as captured:
                                solver.step(state, result, control, None, dt)
                            wp.capture_launch(captured.graph)
                        else:
                            solver.step(state, result, control, None, dt)
                        np.testing.assert_allclose(result.body_q.numpy()[:, 0], expected, atol=3.0e-6, rtol=0.0)
                        total = solver.tendon_seg_material_tension.numpy() + solver.tendon_seg_damping_tension.numpy()
                        np.testing.assert_allclose(total, [force], atol=0.02, rtol=0.0)


def test_sparse_tendon_offset_anchor_torque(test, device, sparse_mode="block_sparse_joints"):
    """Match translation and rotation against a dense offset-anchor reference."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    for x in (0.0, 1.0):
        body = builder.add_link(
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        joints.append(builder.add_joint_free(body))
    builder.add_articulation(joints)
    builder.add_tendon()
    for body in range(2):
        builder.add_tendon_link(
            body,
            offset=(0.0, 0.1, 0.0),
            link_type=int(newton.TendonLinkType.ATTACHMENT),
            rest_length=0.8,
            compliance=1.0e-3,
            damping=0.0,
        )
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(
        model, iterations=1, rigid_articulation_solve=sparse_mode, rigid_articulation_relaxation=1.0
    )
    state, result = model.state(), model.state()
    solver.step(state, result, model.control(), None, 0.01)
    # Independent dense reference, ordered as translation then world rotation.
    gradient = np.array([1, 0, 0, 0, 0, -0.1, -1, 0, 0, 0, 0, 0.1])
    hessian = np.eye(12) * 1.0e4 + 1000.0 * np.outer(gradient, gradient)
    delta = np.linalg.solve(hessian, 200.0 * gradient).reshape(2, 6)
    pose = result.body_q.numpy()
    np.testing.assert_allclose(pose[:, :3] - np.array([[0, 0, 0], [1, 0, 0]]), delta[:, :3], atol=1.0e-6)
    np.testing.assert_allclose(2.0 * np.arctan2(pose[:, 5], pose[:, 6]), delta[:, 5], atol=1.0e-6)


def test_sparse_tendon_capture(test, device, sparse_mode="block_sparse_joints"):
    """Match captured and uncaptured sparse tendon trajectories."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires CUDA")
    model = _build_straight_route(device, count=3)
    results = []

    def two_steps(solver, state, other, control):
        solver.step(state, other, control, None, 0.01)
        solver.step(other, state, control, None, 0.01)

    for capture in (False, True):
        solver = newton.solvers.SolverVBD(model, iterations=4, rigid_articulation_solve=sparse_mode)
        state, other, control = model.state(), model.state(), model.control()

        two_steps(solver, state, other, control)
        if capture:
            with wp.ScopedCapture(device=device) as captured:
                two_steps(solver, state, other, control)
            for _ in range(4):
                wp.capture_launch(captured.graph)
        else:
            for _ in range(4):
                two_steps(solver, state, other, control)
        results.append(state.body_q.numpy())
    np.testing.assert_allclose(results[0], results[1], atol=1.0e-6, rtol=0.0)


def test_sparse_tendon_pulley_reference(test, device, sparse_mode="block_sparse_joints"):
    """Match a planar offset-anchor pendulum against float64 implicit Euler."""
    with wp.ScopedDevice(device):
        example = PulleyExample(None, _make_example_args(20))
        _install_vbd_example_solver(example, {"rigid_articulation_solve": sparse_mode, "iterations": 32})
    model, solver = example.model, example.solver
    pose = model.body_q.numpy().astype(np.float64)
    mass = float(model.body_mass.numpy()[1])
    inertia = float(model.body_inertia.numpy()[1, 1, 1])
    metric = np.array([mass, mass, inertia])
    anchor = pose[0, [0, 2]]
    radius = float(model.tendon_link_offset.numpy()[1, 2])
    rest = float(solver.tendon_seg_rest_length.numpy()[0])
    compliance = float(solver.tendon_seg_active_compliance.numpy()[0])
    damping = float(solver.tendon_seg_active_damping.numpy()[0])
    gravity = model.gravity.numpy()[0, [0, 2]].astype(np.float64)
    dt = example.sim_dt
    q = np.array([pose[1, 0], pose[1, 2], 0.0])
    velocity = np.zeros(3)

    def radial(x):
        return radius * np.array([np.sin(x[2]), np.cos(x[2])])

    # Solve the three planar equations independently, including the nonlinear
    # offset-anchor geometry. No VBD Hessian or tendon force kernel is reused.
    for _ in range(20):
        for _ in range(example.sim_substeps):
            previous = q.copy()
            previous_length = np.linalg.norm(previous[:2] + radial(previous) - anchor)
            predicted = previous + dt * velocity
            predicted[:2] += gravity * dt * dt

            def residual(x, previous_length=previous_length, predicted=predicted):
                arm = radial(x)
                delta = x[:2] + arm - anchor
                length = np.linalg.norm(delta)
                stretch = length - rest
                damped = min(damping * (length - previous_length) / dt, damping * max(stretch, 0.0) / dt)
                tension = max(stretch / compliance + damped, 0.0)
                force = -tension * delta / length
                torque = arm[1] * force[0] - arm[0] * force[1]
                return x - predicted - dt * dt * np.array([force[0], force[1], torque]) / metric

            q = predicted.copy()
            for _ in range(12):
                jacobian = np.empty((3, 3))
                for axis in range(3):
                    shift = np.zeros(3)
                    shift[axis] = 1.0e-7
                    jacobian[:, axis] = (residual(q + shift) - residual(q - shift)) / 2.0e-7
                increment = np.linalg.solve(jacobian, -residual(q))
                q += increment
                if np.max(np.abs(increment)) < 1.0e-13:
                    break
            test.assertLess(np.max(np.abs(residual(q))), 1.0e-11)
            velocity = (q - previous) / dt

        example.simulate()
        solver.check_tendon_material()
        actual = example.state_0.body_q.numpy()[1].astype(np.float64)
        angle = 2.0 * np.arctan2(actual[4], actual[6])
        # Stored float32 poses and lengths limit the force accuracy. Check the
        # accumulated trajectory instead of requiring local-solver roundoff.
        np.testing.assert_allclose(actual[[0, 2]], q[:2], atol=1.0e-5, rtol=0.0)
        test.assertLess(abs(angle - q[2]), 1.0e-4)


def _build_articulated_dynamic_roller(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    anchors = []
    joints = []
    for z in (-0.5, 0.5):
        pose = wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity())
        body = builder.add_link(xform=pose, mass=0.0)
        anchors.append(body)
        joints.append(builder.add_joint_fixed(-1, body, parent_xform=pose))
    pose = wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity())
    roller = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(0.01 * np.eye(3)))
    joints.append(builder.add_joint_prismatic(-1, roller, parent_xform=pose, axis=newton.Axis.X))
    builder.add_articulation(joints)
    builder.add_tendon()
    builder.add_tendon_link(anchors[0], link_type=int(newton.TendonLinkType.ATTACHMENT), axis=(0.0, 1.0, 0.0))
    guide = builder.add_tendon_link(
        roller,
        link_type=int(newton.TendonLinkType.ROLLING),
        axis=(0.0, 1.0, 0.0),
        radius=0.1,
        orientation=1,
        dynamic=True,
        mu=0.25,
        compliance=0.01,
    )
    builder.add_tendon_link(
        anchors[1], link_type=int(newton.TendonLinkType.ATTACHMENT), axis=(0.0, 1.0, 0.0), compliance=0.01
    )
    builder.color()
    return builder.finalize(device=device), roller, guide


def test_sparse_articulated_dynamic_roller(test, device, sparse_mode="block_sparse_joints"):
    """Match local VBD through activation and release of an articulated roller."""
    model, roller, guide = _build_articulated_dynamic_roller(device)
    dt = 1.0 / 120.0
    for direct in (False, True):
        reference = None
        for mode in ("local", sparse_mode):
            for capture in (False, True) if device.is_cuda else (False,):
                with test.subTest(direct=direct, mode=mode, capture=capture):
                    options = {"iterations": 32, "tendon_material_direct": direct, "rigid_articulation_solve": mode}
                    # Compile before capture without advancing the measured solver's history.
                    warm = newton.solvers.SolverVBD(model, **options)
                    warm.step(model.state(), model.state(), model.control(), None, dt)
                    solver = newton.solvers.SolverVBD(model, **options)
                    state, result, control = model.state(), model.state(), model.control()
                    initial_inventory = _oriented_route_material_length(solver, model, state, guide)
                    if mode == sparse_mode:
                        test.assertIn(roller, solver.rigid_articulation_sparse_layout.articulation_bodies.numpy())
                    if capture:
                        with wp.ScopedCapture(device=device) as captured:
                            solver.step(state, result, control, None, dt)

                    active, poses, velocities = [], [], []
                    for speed in (-24.0, None, 24.0, None):
                        if speed is not None:
                            qd = state.body_qd.numpy()
                            qd[roller, 0] = speed
                            state.body_qd.assign(qd)
                        if capture:
                            wp.capture_launch(captured.graph)
                        else:
                            solver.step(state, result, control, None, dt)
                        active.append(bool(solver.tendon_link_active.numpy()[guide]))
                        poses.append(result.body_q.numpy())
                        velocities.append(result.body_qd.numpy())
                        solver.check_tendon_material()
                        test.assertAlmostEqual(
                            _oriented_route_material_length(solver, model, result, guide),
                            initial_inventory,
                            delta=2.0e-7,
                        )
                        state.assign(result)

                    test.assertEqual(active, [False, True, True, False])
                    test.assertGreater(velocities[1][roller, 0], -23.9999, "The active roller must receive cable force")
                    test.assertEqual(solver.tendon_seg_active.numpy().tolist(), [1, 0])
                    test.assertEqual(float(solver.tendon_seg_material_tension.numpy()[1]), 0.0)
                    actual = np.asarray(poses), np.asarray(velocities)
                    test.assertTrue(all(np.isfinite(value).all() for value in actual))
                    if reference is None:
                        reference = actual
                    else:
                        np.testing.assert_allclose(actual[0], reference[0], rtol=0.0, atol=2.0e-5)
                        np.testing.assert_allclose(actual[1], reference[1], rtol=0.0, atol=2.0e-3)


def test_sparse_tendon_alm_rejected(test, device, sparse_mode="block_sparse_joints"):
    """Reject unsupported tendon ALM combinations at solver construction."""
    model = _build_straight_route(device)
    for option in ("tendon_alm", "tendon_alm_per_segment"):
        with test.assertRaisesRegex(ValueError, "Tendon ALM currently requires"):
            newton.solvers.SolverVBD(model, rigid_articulation_solve=sparse_mode, **{option: True})


def test_sparse_tendon_elastic_scope(test, device, sparse_mode="block_sparse_joints"):
    """Preserve independent elastic groups and reject unvalidated tendon/modal coupling."""

    def build(tendon_endpoints):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for x in (0.0, 1.0):
            pose = wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity())
            body = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(-1, body, parent_xform=pose, axis=(1.0, 0.0, 0.0))
            builder.add_articulation([joint])
        basis = newton.ModalBasis(
            sample_points=[[0.0, 0.0, 0.0]],
            sample_phi=[[[1.0, 0.0, 0.0]]],
            sample_mass=[1.0],
            mode_stiffness=[50.0],
            mode_damping=[0.0],
        )
        pose = wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity())
        elastic = builder.add_link_elastic(
            xform=pose,
            mass=3.0,
            inertia=wp.mat33(np.eye(3)),
            modal_basis=basis,
            mode_q=[0.01],
            inertia_tolerance=None,
        )
        rigid = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation(
            [
                builder.add_joint_fixed(-1, elastic, parent_xform=pose),
                builder.add_joint_fixed(elastic, rigid),
            ]
        )
        if tendon_endpoints is not None:
            builder.add_tendon()
            for body in tendon_endpoints:
                builder.add_tendon_link(
                    body,
                    link_type=int(newton.TendonLinkType.ATTACHMENT),
                    rest_length=0.8,
                    compliance=1.0e-3,
                    damping=0.0,
                )
        builder.color()
        return builder.finalize(device=device)

    for backend in ("schur", "coupled"):
        with test.subTest(backend=backend):
            options = {
                "iterations": 1,
                "rigid_articulation_solve": sparse_mode,
                "rigid_articulation_elastic_solve": backend,
                "rigid_articulation_relaxation": 1.0,
            }
            reference, candidate = build(None), build((0, 1))
            outputs = []
            for model in (reference, candidate):
                solver = newton.solvers.SolverVBD(model, **options)
                result = model.state()
                solver.step(model.state(), result, model.control(), None, 0.01)
                outputs.append((result.body_q.numpy(), result.joint_q.numpy()))
            layout = solver.rigid_articulation_sparse_layout
            test.assertEqual(layout.articulation_count, 2)
            test.assertEqual(layout.rigid_articulation_count, 2 if backend == "schur" else 1)
            test.assertTrue(np.isfinite(outputs[1][0]).all() and np.isfinite(outputs[1][1]).all())
            np.testing.assert_allclose(outputs[1][0][2:], outputs[0][0][2:], atol=1.0e-6, rtol=0.0)
            modal_start = int(candidate.joint_q_start.numpy()[candidate.elastic_joint.numpy()[0]])
            np.testing.assert_allclose(outputs[1][1][modal_start:], outputs[0][1][modal_start:], atol=1.0e-6, rtol=0.0)
            dx = 0.01**2 * 1000.0 * 0.2 / (1.0 + 2.0 * 0.01**2 * 1000.0)
            np.testing.assert_allclose(outputs[1][0][:2, 0], [dx, 1.0 - dx], atol=2.0e-6, rtol=0.0)
            with test.assertRaisesRegex(ValueError, "Tendon-coupled sparse articulations containing reduced elastic"):
                newton.solvers.SolverVBD(build((0, 3)), **options)
