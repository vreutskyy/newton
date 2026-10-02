# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check convex tendon geometry independently of the body/material solver."""

import math
import unittest
from dataclasses import fields

import numpy as np
import warp as wp

from newton._src.geometry.roller_profile import (
    ProfileData,
    RollerProfileCircle,
    RollerProfileEllipse,
    RollerProfileSector,
    pack_profiles,
    profile_arc_length,
    profile_parameter_delta,
    profile_span_tangent,
    profile_support,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _support_queries(
    profiles: wp.array[ProfileData],
    directions: wp.array[wp.vec2],
    points: wp.array[wp.vec2],
    parameters: wp.array[float],
):
    i = wp.tid()
    point, parameter = profile_support(profiles[0], directions[i])
    points[i] = point
    parameters[i] = parameter


@wp.kernel
def _arc_queries(
    profiles: wp.array[ProfileData],
    intervals: wp.array[wp.vec2],
    result: wp.array[float],
):
    i = wp.tid()
    result[i] = profile_arc_length(profiles[0], intervals[i][0], intervals[i][1])


@wp.kernel
def _span_query(
    profiles: wp.array[ProfileData],
    center_l: wp.vec3,
    center_r: wp.vec3,
    rotation_l: float,
    rotation_r: float,
    orientation_l: int,
    orientation_r: int,
    points: wp.array[wp.vec3],
    params: wp.array[float],
    valid: wp.array[bool],
):
    x_l = wp.vec3(wp.cos(rotation_l), wp.sin(rotation_l), 0.0)
    y_l = wp.vec3(-wp.sin(rotation_l), wp.cos(rotation_l), 0.0)
    x_r = wp.vec3(wp.cos(rotation_r), wp.sin(rotation_r), 0.0)
    y_r = wp.vec3(-wp.sin(rotation_r), wp.cos(rotation_r), 0.0)
    left, right, sl, sr, normal, success = profile_span_tangent(
        profiles[0],
        profiles[1],
        center_l,
        center_r,
        x_l,
        y_l,
        x_r,
        y_r,
        wp.vec3(0.0, 0.0, 1.0),
        orientation_l,
        orientation_r,
    )
    points[0] = left
    points[1] = right
    points[2] = normal
    params[0] = sl
    params[1] = sr
    valid[0] = success


@wp.kernel
def _path_measure(
    profiles: wp.array[ProfileData],
    poses: wp.array[wp.vec3],
    result: wp.array2d[float],
):
    i = wp.tid()
    center = wp.vec3(poses[i][0], poses[i][1], 0.0)
    angle = poses[i][2]
    x = wp.vec3(wp.cos(angle), wp.sin(angle), 0.0)
    y = wp.vec3(-wp.sin(angle), wp.cos(angle), 0.0)
    z = wp.vec3(0.0, 0.0, 1.0)
    world_x = wp.vec3(1.0, 0.0, 0.0)
    world_y = wp.vec3(0.0, 1.0, 0.0)
    anchor_l = wp.vec3(-3.0, 1.0, 0.0)
    anchor_r = wp.vec3(2.5, 1.6, 0.0)
    a, b, _sa, sb, _n_in, valid_in = profile_span_tangent(
        profiles[0],
        profiles[1],
        anchor_l,
        center,
        world_x,
        world_y,
        x,
        y,
        z,
        1,
        1,
    )
    c, d, sc, _sd, _n_out, valid_out = profile_span_tangent(
        profiles[1],
        profiles[0],
        center,
        anchor_r,
        x,
        y,
        world_x,
        world_y,
        z,
        1,
        1,
    )
    delta = sc - sb
    if delta < 0.0:
        delta += profiles[1].period
    arc = profile_arc_length(profiles[1], sb, delta)
    result[i, 0] = wp.length(b - a) + arc + wp.length(d - c)
    f_in = wp.normalize(a - b)
    f_out = wp.normalize(d - c)
    force = f_in + f_out
    torque = wp.cross(b - center, f_in) + wp.cross(c - center, f_out)
    result[i, 1] = force[0]
    result[i, 2] = force[1]
    result[i, 3] = torque[2]
    result[i, 4] = float(valid_in and valid_out)
    # This is exactly the rest-length transport required when the entry/exit
    # contact coordinates change; used below to check boundary seam crossings.
    result[i, 5] = sb
    result[i, 6] = sc
    result[i, 7] = arc


@wp.kernel
def _transport_measure(
    profiles: wp.array[ProfileData],
    paths: wp.array2d[float],
    delta_sum: wp.array[float],
):
    i = wp.tid()
    profile = profiles[1]
    delta_in = profile_parameter_delta(profile, paths[i, 5], paths[i + 1, 5])
    delta_out = profile_parameter_delta(profile, paths[i, 6], paths[i + 1, 6])
    # Incoming rest gains material leaving the entry arc; outgoing rest loses
    # material added to the exit arc. Free rest plus wrapped length is constant.
    delta_sum[i] = (
        profile_arc_length(profile, paths[i, 5], delta_in)
        - profile_arc_length(profile, paths[i, 6], delta_out)
        + paths[i + 1, 7]
        - paths[i, 7]
    )


def _span(profile_l, profile_r, *, device, centers=((0, 0, 0), (4, 0, 0)), rotations=(0, 0), signs=(1, 1)):
    arrays = (pack_profiles([profile_l, profile_r], device=device),)
    points = wp.zeros(3, dtype=wp.vec3, device=device)
    params = wp.zeros(2, dtype=float, device=device)
    valid = wp.zeros(1, dtype=bool, device=device)
    wp.launch(
        _span_query,
        dim=1,
        inputs=[*arrays, wp.vec3(*centers[0]), wp.vec3(*centers[1]), *rotations, *signs],
        outputs=[points, params, valid],
        device=device,
    )
    return points.numpy(), params.numpy(), bool(valid.numpy()[0])


def test_circle_tangents(test, device):
    """Recover analytical circle tangents for both winding signs and several scales."""
    for scale in (0.001, 1.0, 100.0):
        for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            r_l, r_r, distance = scale * 0.3, scale * 0.7, scale * 4.0
            points, _, valid = _span(
                RollerProfileCircle(r_l),
                RollerProfileCircle(r_r),
                centers=((0, 0, 0), (distance, 0, 0)),
                signs=signs,
                device=device,
            )
            nx = (signs[0] * r_l - signs[1] * r_r) / distance
            normal = np.array([nx, -math.sqrt(1 - nx * nx), 0.0])
            test.assertTrue(valid)
            np.testing.assert_allclose(points[0], signs[0] * r_l * normal, atol=scale * 2e-6)
            np.testing.assert_allclose(points[1], (distance, 0, 0) + signs[1] * r_r * normal, atol=scale * 2e-6)


def test_ellipse_tangents(test, device):
    """Check two rotated ellipses against their implicit equations and surface normals."""
    ellipses = (RollerProfileEllipse(0.8, 0.2), RollerProfileEllipse(0.3, 0.9))
    rotations = (0.63, -0.37)
    for signs in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        points, _, valid = _span(*ellipses, rotations=rotations, signs=signs, device=device)
        test.assertTrue(valid)
        for index, (profile, angle) in enumerate(zip(ellipses, rotations, strict=True)):
            rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            local = rot.T @ (points[index, :2] - np.array([4.0 * index, 0.0]))
            radii = np.asarray((profile.a, profile.b))
            test.assertAlmostEqual(float(np.sum((local / radii) ** 2)), 1.0, delta=2e-6)
            normal = rot @ (local / radii**2)
            normal /= np.linalg.norm(normal)
            # Reconstructing a normal from a world-space float32 point amplifies
            # coordinate roundoff by inverse curvature radius. Check that error
            # separately from the directly returned common support normal.
            test.assertAlmostEqual(float(np.dot(normal, points[1, :2] - points[0, :2])), 0.0, delta=1e-5)
        test.assertAlmostEqual(float(np.dot(points[2], points[1] - points[0])), 0.0, delta=1e-6)


def test_ellipse_arc(test, device):
    """Compare ellipse arc integration to independent high-order NumPy quadrature."""
    intervals = [(0.1, 0.003), (6.28, 0.015), (-0.4, 1.7), (0.0, 2 * math.pi), (1.3, -2.1)]
    nodes, weights = np.polynomial.legendre.leggauss(512)
    for radii in ((0.2, 0.2), (0.8, 0.2), (0.01, 0.001), (1.0, 0.01)):
        arrays = (pack_profiles([RollerProfileEllipse(*radii)], device=device),)
        result = wp.zeros(len(intervals), dtype=float, device=device)
        wp.launch(
            _arc_queries,
            dim=len(intervals),
            inputs=[arrays[0], wp.array(intervals, dtype=wp.vec2, device=device)],
            outputs=[result],
            device=device,
        )
        expected = []
        for start, delta in intervals:
            # Split the reference into 64 panels; independent of the kernel's
            # rule and robust even for an aspect ratio of 100.
            ts = start + (np.arange(64)[:, None] + (nodes[None, :] + 1) / 2) * delta / 64
            speeds = np.hypot(radii[0] * np.sin(ts), radii[1] * np.cos(ts))
            expected.append(float(np.sum(speeds @ weights) * delta / 128))
        np.testing.assert_allclose(result.numpy(), expected, rtol=3e-5, atol=2e-8 * max(radii))


def test_invalid_tangent(test, device):
    """Report impossible inner tangents and off-plane spans explicitly."""
    circle = RollerProfileCircle(1.0)
    _, _, valid = _span(circle, circle, centers=((0, 0, 0), (1, 0, 0)), signs=(1, -1), device=device)
    test.assertFalse(valid)
    _, _, valid = _span(circle, circle, centers=((0, 0, 0), (4, 0, 0.2)), device=device)
    test.assertFalse(valid)


def test_tangent_reference(test, device):
    """Match accelerated tangents to independent double-precision bisection."""
    rng = np.random.default_rng(17)
    profiles = [None, RollerProfileCircle(0.3), RollerProfileEllipse(0.8, 0.04), RollerProfileSector(0.7, 2.1)]

    def support(profile, normal):
        if profile is None:
            return np.zeros(2)
        if isinstance(profile, RollerProfileCircle):
            return profile.radius * normal / np.linalg.norm(normal)
        if isinstance(profile, RollerProfileEllipse):
            radii = np.array([profile.a, profile.b])
            return radii**2 * normal / np.linalg.norm(radii * normal)
        angle = np.clip(np.arctan2(normal[1], normal[0]), -profile.angle / 2, profile.angle / 2)
        point = profile.radius * np.array([np.cos(angle), np.sin(angle)])
        return point if np.dot(point, normal) > 0 else np.zeros(2)

    for case in range(96):
        shapes = [profiles[case % 4], profiles[(case // 4) % 4]]
        rotations = rng.uniform(-math.pi, math.pi, 2)
        signs = tuple(rng.choice([-1, 1], 2).tolist())
        scale = 10.0 ** rng.uniform(-3, 2)
        # Include close inner tangents without overlapping the bounding circles.
        distance = 1.6 + rng.uniform(0, 3)
        matrices = [np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in rotations]

        def contacts(angle, matrices=matrices, shapes=shapes, signs=signs, distance=distance):
            normal = np.array([np.sin(angle), -np.cos(angle)])
            points = [matrices[i] @ support(shapes[i], signs[i] * matrices[i].T @ normal) for i in range(2)]
            points[1] += (distance, 0)
            return np.asarray(points), normal

        low, high = -math.pi / 2, math.pi / 2
        for _ in range(60):
            middle = (low + high) / 2
            expected, normal = contacts(middle)
            if np.dot(expected[1] - expected[0], normal) < 0:
                low = middle
            else:
                high = middle
        # Scale the query geometry, not the independent reference arithmetic.
        scaled = []
        for p in shapes:
            scaled_profile = p
            if isinstance(p, RollerProfileCircle):
                scaled_profile = RollerProfileCircle(p.radius * scale)
            elif isinstance(p, RollerProfileEllipse):
                scaled_profile = RollerProfileEllipse(p.a * scale, p.b * scale)
            elif isinstance(p, RollerProfileSector):
                scaled_profile = RollerProfileSector(p.radius * scale, p.angle)
            scaled.append(scaled_profile)
        actual, _, valid = _span(
            *scaled, device=device, centers=((0, 0, 0), (distance * scale, 0, 0)), rotations=rotations, signs=signs
        )
        test.assertTrue(valid, f"case {case}")
        np.testing.assert_allclose(actual[:2, :2], expected * scale, rtol=0, atol=scale * 2e-6)


def test_virtual_work(test, device):
    """Match unit-tension forces and torque to full routed-length finite differences."""
    profiles = [
        RollerProfileCircle(0.5),
        RollerProfileEllipse(0.8, 0.2),
        RollerProfileSector(0.8, math.pi * 0.7),
    ]
    epsilon = 0.002
    for profile in profiles:
        center_pose = np.array([0.17, -0.11, 0.43])
        poses = [center_pose]
        for axis in range(3):
            perturbation = np.eye(3)[axis] * epsilon
            poses.extend([center_pose + perturbation, center_pose - perturbation])
        arrays = (pack_profiles([None, profile], device=device),)
        result = wp.zeros((len(poses), 8), dtype=float, device=device)
        wp.launch(
            _path_measure,
            dim=len(poses),
            inputs=[*arrays, wp.array(poses, dtype=wp.vec3, device=device)],
            outputs=[result],
            device=device,
        )
        values = result.numpy()
        test.assertTrue(np.all(values[:, 4] == 1))
        gradient = (values[1::2, 0] - values[2::2, 0]) / (2 * epsilon)
        np.testing.assert_allclose(-gradient, values[0, 1:4], atol=6e-4, rtol=2e-3)
        if isinstance(profile, RollerProfileCircle):
            test.assertAlmostEqual(float(values[0, 3]), 0.0, delta=1e-6)
        else:
            test.assertGreater(abs(float(values[0, 3])), 0.04)


def test_boundary_transport(test, device):
    """Conserve free material plus wrapped length across rotating profile seams and corners."""
    angles = np.linspace(-math.pi, math.pi, 501)
    poses = np.stack([np.full_like(angles, 0.17), np.full_like(angles, -0.11), angles], axis=1)
    profiles = [
        RollerProfileEllipse(0.8, 0.2),
        RollerProfileSector(0.8, math.pi * 0.7),
    ]
    for profile in profiles:
        arrays = (pack_profiles([None, profile], device=device),)
        paths = wp.zeros((len(poses), 8), dtype=float, device=device)
        wp.launch(
            _path_measure,
            dim=len(poses),
            inputs=[*arrays, wp.array(poses, dtype=wp.vec3, device=device)],
            outputs=[paths],
            device=device,
        )
        result = wp.zeros(len(poses) - 1, dtype=float, device=device)
        wp.launch(_transport_measure, dim=len(poses) - 1, inputs=[arrays[0], paths], outputs=[result], device=device)
        test.assertTrue(np.all(paths.numpy()[:, 4] == 1))
        np.testing.assert_allclose(result.numpy(), 0.0, atol=3e-6)


def test_sector_support(test, device):
    """Verify exact sector support on its curved arc, both corners, and apex."""
    radius, half = 0.7, 0.6
    profile = RollerProfileSector(radius, 2 * half)
    angles = np.array([-math.pi, -2.6, -1.0, -0.4, 0.0, 0.2, 1.1, 2.6, math.pi])
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    arrays = (pack_profiles([profile], device=device),)
    points = wp.zeros(len(normals), dtype=wp.vec2, device=device)
    params = wp.zeros(len(normals), dtype=float, device=device)
    wp.launch(
        _support_queries,
        dim=len(normals),
        inputs=[*arrays, wp.array(normals, dtype=wp.vec2, device=device)],
        outputs=[points, params],
        device=device,
    )
    arc_angles = np.clip(angles, -half, half)
    expected = radius * np.stack([np.cos(arc_angles), np.sin(arc_angles)], axis=1)
    apex = np.sum(expected * normals, axis=1) <= 0
    expected[apex] = 0
    expected_s = radius * (1 + arc_angles + half)
    expected_s[apex] = 0
    np.testing.assert_allclose(points.numpy(), expected, atol=2e-7)
    np.testing.assert_allclose(params.numpy(), expected_s, atol=3e-7)
    intervals = [
        (0, radius),
        (radius, 2 * half * radius),
        (radius * (1 + 2 * half), radius),
        (0, radius * (2 + 2 * half)),
    ]
    lengths = wp.zeros(len(intervals), dtype=float, device=device)
    wp.launch(
        _arc_queries,
        dim=len(intervals),
        inputs=[arrays[0], wp.array(intervals, dtype=wp.vec2, device=device)],
        outputs=[lengths],
        device=device,
    )
    np.testing.assert_allclose(lengths.numpy(), np.array(intervals)[:, 1], atol=2e-7)


class TestRollerProfile(unittest.TestCase):
    def test_shape_parameters(self):
        """Each profile exposes only its own parameters."""
        self.assertEqual([f.name for f in fields(RollerProfileCircle)], ["radius"])
        self.assertEqual([f.name for f in fields(RollerProfileEllipse)], ["a", "b"])
        self.assertEqual([f.name for f in fields(RollerProfileSector)], ["radius", "angle"])
        for factory in (
            lambda: RollerProfileCircle(-1),
            lambda: RollerProfileEllipse(1, 0),
            lambda: RollerProfileSector(1, 4),
            lambda: RollerProfileCircle(float("nan")),
        ):
            with self.assertRaises(ValueError):
                factory()


for fn in (
    test_circle_tangents,
    test_ellipse_tangents,
    test_sector_support,
    test_ellipse_arc,
    test_invalid_tangent,
    test_tangent_reference,
    test_virtual_work,
    test_boundary_transport,
):
    add_function_test(TestRollerProfile, fn.__name__, fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
