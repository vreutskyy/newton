# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check interacting roller profiles against independent route mechanics."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.geometry import RollerProfileCircle, RollerProfileEllipse, RollerProfileSector
from newton.solvers import SolverVBD
from newton.tests.test_roller_profile_vbd import _measure
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_chain(device, *, shared_body=False, reverse=False, mu=0.0):
    builder = newton.ModelBuilder(gravity=0.0)
    left = builder.add_body(xform=wp.transform(p=(-0.45, 0.18, 0)), is_kinematic=True)
    right = builder.add_body(xform=wp.transform(p=(0.45, 0.21, 0)), is_kinematic=True)
    roller_l = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1), com=(0.012, -0.008, 0))
    roller_r = roller_l if shared_body else builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    route = [
        (left, (0, 0, 0), None),
        (roller_l, (-0.12, 0, 0), RollerProfileEllipse(0.05, 0.025)),
        (roller_r, (0.12, 0, 0), RollerProfileSector(0.05, 2.2)),
        (right, (0, 0, 0), None),
    ]
    if reverse:
        route.reverse()
    builder.add_tendon()
    for body, offset, profile in route:
        builder.add_tendon_link(
            body,
            newton.TendonLinkType.ATTACHMENT if profile is None else newton.TendonLinkType.ROLLING,
            offset=offset,
            profile=profile,
            orientation=-1 if reverse else 1,
            mu=mu,
            compliance=1e-3,
        )
    builder.color()
    model = builder.finalize(device=device)
    solver = SolverVBD(model, iterations=16, tendon_settle_tol=1e-7)
    return model, solver, route, sorted({roller_l, roller_r})


def _reference_length(route, poses, orientation=1):
    """Integrate a complete route using float64 support roots and quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(96)
    centers, rotations = [], []
    for body, offset, _ in route:
        x, y, angle = poses[body]
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        rotations.append(rotation)
        centers.append(np.array([x, y]) + rotation @ np.asarray(offset[:2]))

    def support(index, normal):
        profile = route[index][2]
        n = rotations[index].T @ normal
        coordinate = 0.0
        local = np.zeros(2)
        if isinstance(profile, RollerProfileEllipse):
            radii = np.array([profile.a, profile.b])
            local = radii**2 * n / np.linalg.norm(radii * n)
            coordinate = math.atan2(local[1] / profile.b, local[0] / profile.a)
        elif isinstance(profile, RollerProfileSector):
            angle = np.clip(math.atan2(n[1], n[0]), -profile.angle / 2, profile.angle / 2)
            local = profile.radius * np.array([math.cos(angle), math.sin(angle)])
            if local @ n <= 0:
                local = np.zeros(2)
            else:
                coordinate = profile.radius * (1 + angle + profile.angle / 2)
        elif isinstance(profile, RollerProfileCircle):
            local = profile.radius * n / np.linalg.norm(n)
            coordinate = math.atan2(local[1], local[0])
        return centers[index] + rotations[index] @ local, coordinate

    entry, exit = np.zeros(len(route)), np.zeros(len(route))
    length = 0.0
    for i in range(len(route) - 1):
        along = centers[i + 1] - centers[i]
        along /= np.linalg.norm(along)
        across = np.array([-along[1], along[0]])
        lo, hi = -math.pi / 2, math.pi / 2
        for _ in range(60):
            a = (lo + hi) / 2
            normal = math.sin(a) * along - math.cos(a) * across
            pl, sl = support(i, orientation * normal)
            pr, sr = support(i + 1, orientation * normal)
            if (pr - pl) @ normal < 0:
                lo = a
            else:
                hi = a
        length += np.linalg.norm(pr - pl)
        exit[i], entry[i + 1] = sl, sr
    for i, (_, _, profile) in enumerate(route):
        if profile is None:
            continue
        if isinstance(profile, RollerProfileSector):
            length += (orientation * (exit[i] - entry[i])) % (profile.radius * (2 + profile.angle))
        else:
            delta = (orientation * (exit[i] - entry[i])) % (2 * math.pi)
            if isinstance(profile, RollerProfileCircle):
                length += profile.radius * delta
            else:
                t = entry[i] + orientation * (nodes + 1) * delta / 2
                length += delta / 2 * np.dot(weights, np.hypot(profile.a * np.sin(t), profile.b * np.cos(t)))
    return length


def test_chain_virtual_work(test, device):
    """Match interacting and shared-body roller loads to the entire route gradient."""
    for shared_body in (False, True):
        for reverse in (False, True):
            model, solver, route, bodies = _build_chain(device, shared_body=shared_body, reverse=reverse)
            state = model.state()
            rest = solver.tendon_seg_length.numpy() - 4.0 * model.tendon_seg_compliance.numpy()
            solver.tendon_seg_rest_length.assign(rest)
            solver._snapshot_tendon_step_state()
            solver._prepare_tendon_route(model, state.body_q, 1e-8)
            wp.copy(solver.body_q_prev, state.body_q)
            solver._update_tendon_routing(state, 1e-3, True)
            poses = np.zeros((model.body_count, 3))
            poses[:, :2] = state.body_q.numpy()[:, :2]
            for body in bodies:
                force, _ = _measure(model, solver, state, body)
                gradient = []
                for axis in range(3):
                    plus, minus = poses.copy(), poses.copy()
                    plus[body, axis] += 1e-5
                    minus[body, axis] -= 1e-5
                    gradient.append(
                        (
                            _reference_length(route, plus, -1 if reverse else 1)
                            - _reference_length(route, minus, -1 if reverse else 1)
                        )
                        / 2e-5
                    )
                expected = -4 * np.array(gradient)
                # The reference rotates about the body origin, while solver
                # torque is about its COM; retain the off-center lever arm.
                com = model.body_com.numpy()[body]
                expected[2] -= com[0] * expected[1] - com[1] * expected[0]
                np.testing.assert_allclose([force[0, 0], force[0, 1], force[1, 2]], expected, atol=2e-4, rtol=2e-4)


def test_chain_material_conservation(test, device):
    """Conserve free-plus-wrapped material through multi-roller corner transitions."""
    for shared_body, reverse, mu in ((False, False, 0.0), (False, True, 0.3), (True, False, 10.0)):
        model, solver, _, bodies = _build_chain(device, shared_body=shared_body, reverse=reverse, mu=mu)
        state = model.state()
        total = float(solver.tendon_total_cable.numpy()[0])
        for angle in np.linspace(0.0, 2 * math.pi, 97):
            solver._snapshot_tendon_step_state()
            wp.copy(solver.body_q_prev, state.body_q)
            poses = model.body_q.numpy()
            # Hold roller centers fixed while rotating their local profiles;
            # the shared-body case instead exercises a small rigid motion.
            for body in bodies:
                rotation = 0.12 * math.sin(angle) if shared_body else angle
                poses[body, 3:] = (0, 0, math.sin(rotation / 2), math.cos(rotation / 2))
                if not shared_body:
                    offset = np.array([-0.12 if body == bodies[0] else 0.12, 0])
                    rot = np.array(
                        [[math.cos(rotation), -math.sin(rotation)], [math.sin(rotation), math.cos(rotation)]]
                    )
                    poses[body, :2] = offset - rot @ offset
            state.body_q.assign(poses)
            solver._prepare_tendon_route(model, state.body_q, 1e-8)
            solver._update_tendon_routing(state, 1e-3, True)
            test.assertFalse(np.any(solver.tendon_profile_tangent_status.numpy()))
            test.assertFalse(np.any(solver.tendon_profile_wrap_status.numpy()))
            measured = np.sum(solver.tendon_seg_rest_length.numpy(), dtype=np.float64)
            measured += np.sum(solver.tendon_profile_wrap_length.numpy(), dtype=np.float64)
            test.assertAlmostEqual(measured, total, delta=3e-6)


class TestRollerProfileRouting(unittest.TestCase):
    pass


for fn in (test_chain_virtual_work, test_chain_material_conservation):
    add_function_test(TestRollerProfileRouting, fn.__name__, fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
