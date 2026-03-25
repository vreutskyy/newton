# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the xcol API: Builder, Model, Collider."""

import unittest

import numpy as np

import newton._src.xcol as xc

_collider = xc.create_collider()

# Identity quaternion as (qx, qy, qz, qw)
_QUAT_ID = (0.0, 0.0, 0.0, 1.0)


def _transforms(*positions):
    """Build a (n, 7) numpy array of transforms from positions."""
    out = np.zeros((len(positions), 7), dtype=np.float32)
    for i, p in enumerate(positions):
        out[i, :3] = p
        out[i, 3:] = _QUAT_ID
    return out


def _contact_count(model):
    return model.contact_count.numpy()[0]


class TestBuilder(unittest.TestCase):
    def test_add_shapes(self):
        b = xc.Builder()
        s0 = b.add_shape(xc.SHAPE_BOX, params=(10, 1, 10))
        s1 = b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        self.assertEqual(s0, 0)
        self.assertEqual(s1, 1)
        self.assertEqual(b.shape_count, 2)

    def test_finalize_creates_model(self):
        b = xc.Builder()
        b.add_shape(xc.SHAPE_BOX, params=(1, 2, 3), margin=0.1, world=2)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=0.5, world=-1)
        model = b.finalize()
        self.assertEqual(model.shape_count, 2)
        self.assertEqual(model.shape_types.numpy()[0], xc.SHAPE_BOX)
        self.assertEqual(model.shape_types.numpy()[1], xc.SHAPE_POINT)
        np.testing.assert_allclose(model.shape_params.numpy()[0], [1, 2, 3])
        self.assertAlmostEqual(model.shape_margins.numpy()[0], 0.1)
        self.assertEqual(model.shape_worlds.numpy()[0], 2)
        self.assertEqual(model.shape_worlds.numpy()[1], -1)

    def test_set_transforms(self):
        b = xc.Builder()
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        model = b.finalize()
        model.shape_transforms.assign(_transforms((1, 2, 3)))
        t = model.shape_transforms.numpy()[0]
        np.testing.assert_allclose(t[:3], [1, 2, 3])


class TestCollide(unittest.TestCase):
    def test_two_overlapping_spheres(self):
        b = xc.Builder()
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        model = b.finalize()
        model.shape_transforms.assign(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        _collider.collide(model)
        self.assertGreater(_contact_count(model), 0)

    def test_separated_no_contacts(self):
        b = xc.Builder()
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        model = b.finalize()
        model.shape_transforms.assign(_transforms((-5, 0, 0), (5, 0, 0)))
        _collider.collide(model)
        self.assertEqual(_contact_count(model), 0)

    def test_box_on_box_4_contacts(self):
        b = xc.Builder()
        b.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        b.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        model = b.finalize()
        model.shape_transforms.assign(_transforms((0, 0, 0), (1.5, 0, 0)))
        _collider.collide(model)
        self.assertEqual(_contact_count(model), 4)

    def test_contact_arrays_readable(self):
        b = xc.Builder()
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        model = b.finalize()
        model.shape_transforms.assign(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        _collider.collide(model)
        n = _contact_count(model)
        self.assertEqual(n, 1)
        self.assertEqual(model.contact_shape_a.numpy()[0], 0)
        self.assertEqual(model.contact_shape_b.numpy()[0], 1)
        self.assertLess(model.contact_depth.numpy()[0], 0.0)  # negative = penetrating
        self.assertAlmostEqual(np.linalg.norm(model.contact_normal.numpy()[0]), 1.0, places=3)

    def test_contacts_reset_between_calls(self):
        """Calling collide() resets contacts from the previous call."""
        b = xc.Builder()
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        model = b.finalize()

        # First call: overlapping
        model.shape_transforms.assign(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        _collider.collide(model)
        self.assertGreater(_contact_count(model), 0)

        # Second call: separated — contacts should be 0
        model.shape_transforms.assign(_transforms((-5, 0, 0), (5, 0, 0)))
        _collider.collide(model)
        self.assertEqual(_contact_count(model), 0)


class TestWorldFiltering(unittest.TestCase):
    def test_different_worlds_no_collision(self):
        b = xc.Builder()
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=1)
        model = b.finalize()
        model.shape_transforms.assign(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        _collider.collide(model)
        self.assertEqual(_contact_count(model), 0)

    def test_same_world_collides(self):
        b = xc.Builder()
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        model = b.finalize()
        model.shape_transforms.assign(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        _collider.collide(model)
        self.assertGreater(_contact_count(model), 0)

    def test_global_world_collides_with_any(self):
        b = xc.Builder()
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=-1)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=42)
        model = b.finalize()
        model.shape_transforms.assign(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        _collider.collide(model)
        self.assertGreater(_contact_count(model), 0)

    def test_three_shapes_world_filtering(self):
        b = xc.Builder()
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=1)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=-1)
        model = b.finalize()
        model.shape_transforms.assign(_transforms((0, 0, 0), (0.1, 0, 0), (0.2, 0, 0)))
        _collider.collide(model)
        # Pairs: 0-2 global (1 contact), 1-2 global (1 contact) = 2
        self.assertEqual(_contact_count(model), 2)


class TestContactRotationSweep(unittest.TestCase):
    """Rotate one box through a full circle and validate contacts at each step."""

    @staticmethod
    def _quat_from_axis_angle(ax, ay, az, angle):
        """Build quaternion (qx, qy, qz, qw) from axis + angle."""
        s = np.sin(angle / 2.0)
        c = np.cos(angle / 2.0)
        return (ax * s, ay * s, az * s, c)

    @staticmethod
    def _validate_contacts(model, expected_depth, msg):
        """Validate contacts: unit normal, depth matches expected, points finite."""
        count = model.contact_count.numpy()[0]
        normals = model.contact_normal.numpy()[:count]
        depths = model.contact_depth.numpy()[:count]
        points = model.contact_point.numpy()[:count]

        for i in range(count):
            n = normals[i]
            n_len = np.linalg.norm(n)
            assert 0.99 < n_len < 1.01, f"{msg}: contact {i} normal not unit (len={n_len:.4f})"

            d = depths[i]
            # Depth should be negative (penetrating) and close to expected
            assert d < 0.0, f"{msg}: contact {i} depth not negative ({d:.4f})"
            assert abs(d - expected_depth) < 0.05, (
                f"{msg}: contact {i} depth {d:.4f} doesn't match expected {expected_depth:.4f}"
            )

            p = points[i]
            assert np.all(np.isfinite(p)), f"{msg}: contact {i} non-finite point {p}"
            # Contact point must be near the smaller shape (within 2 units of its center)
            assert np.linalg.norm(p[:2]) < 2.0, f"{msg}: contact {i} XY too far from small box center: {p}"

    def test_face_face_rotation_sweep(self):
        """Large static box + small rotating box, face-to-face with 0.1 overlap."""
        b = xc.Builder()
        b.add_shape(xc.SHAPE_BOX, params=(5, 5, 5))  # large static
        b.add_shape(xc.SHAPE_BOX, params=(0.5, 0.5, 0.5))  # small rotating
        model = b.finalize()

        overlap = 0.1
        small_half = 0.5
        large_half = 5.0
        # Small box center: just above the large box top face minus overlap
        z_pos = large_half + small_half - overlap
        steps = 72  # 5-degree increments
        for step in range(steps):
            angle = 2.0 * np.pi * step / steps
            q = self._quat_from_axis_angle(0, 0, 1, angle)
            transforms = np.zeros((2, 7), dtype=np.float32)
            transforms[0, 3:] = _QUAT_ID
            transforms[1, :3] = [0, 0, z_pos]
            transforms[1, 3:] = q
            model.shape_transforms.assign(transforms)
            _collider.collide(model)

            count = _contact_count(model)
            msg = f"step={step}, angle={np.degrees(angle):.1f}°"
            self.assertGreater(count, 0, f"{msg}: no contacts generated")
            self.assertLessEqual(count, 4, f"{msg}: too many contacts ({count})")
            self._validate_contacts(model, -overlap, msg)

    def test_face_edge_rotation_sweep(self):
        """Large static box + small tilted box (edge-on), rotated around Z.

        Small box is tilted 45° around Y, presenting an edge to the large
        box's top face. Rotated around Z through a full circle.
        """
        b = xc.Builder()
        b.add_shape(xc.SHAPE_BOX, params=(5, 5, 5))  # large static
        b.add_shape(xc.SHAPE_BOX, params=(0.5, 0.5, 0.5))  # small tilted
        model = b.finalize()

        overlap = 0.1
        small_half = 0.5
        large_half = 5.0
        steps = 72
        for step in range(steps):
            angle_z = 2.0 * np.pi * step / steps
            tilt = np.pi / 4.0
            qy = np.array(self._quat_from_axis_angle(0, 1, 0, tilt))
            qz = np.array(self._quat_from_axis_angle(0, 0, 1, angle_z))
            w1, x1, y1, z1 = qz[3], qz[0], qz[1], qz[2]
            w2, x2, y2, z2 = qy[3], qy[0], qy[1], qy[2]
            qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            qy_val = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            qz_val = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            q = (qx, qy_val, qz_val, qw)

            z_pos = large_half + np.sqrt(2.0) * small_half - overlap
            transforms = np.zeros((2, 7), dtype=np.float32)
            transforms[0, 3:] = _QUAT_ID
            transforms[1, :3] = [0, 0, z_pos]
            transforms[1, 3:] = q
            model.shape_transforms.assign(transforms)
            _collider.collide(model)

            count = _contact_count(model)
            msg = f"step={step}, angle_z={np.degrees(angle_z):.1f}°"
            self.assertGreater(count, 0, f"{msg}: no contacts generated")
            self.assertLessEqual(count, 4, f"{msg}: too many contacts ({count})")
            depths = model.contact_depth.numpy()[:count]
            points = model.contact_point.numpy()[:count]
            for di in range(count):
                d = depths[di]
                self.assertLess(d, 0.0, f"{msg}: contact {di} depth not negative ({d:.4f})")
                self.assertGreater(d, -1.0, f"{msg}: contact {di} depth too large ({d:.4f})")
                p = points[di]
                self.assertLess(np.linalg.norm(p[:2]), 2.0, f"{msg}: contact {di} XY too far from small box: {p}")


class TestEdgeEdgeContact(unittest.TestCase):
    """Edge-edge contact: box A tilted 45° (edge up), box B tilted 45° and
    rotated around Z through a full circle."""

    def test_edge_edge_touching_sweep(self):
        """Touching edge-edge, rotate box B around Z axis."""
        b = xc.Builder()
        b.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        b.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        model = b.finalize()

        quat = TestContactRotationSweep._quat_from_axis_angle
        tilt = np.pi / 4.0
        qa = quat(0, 1, 0, tilt)
        sep = np.sqrt(2.0) + np.sqrt(2.0)  # exact touching

        steps = 72
        for step in range(steps):
            angle_z = 2.0 * np.pi * step / steps
            # Box B: tilt 45° around X, then rotate around Z
            qx = np.array(quat(1, 0, 0, tilt))
            qz = np.array(quat(0, 0, 1, angle_z))
            w1, x1, y1, z1 = qz[3], qz[0], qz[1], qz[2]
            w2, x2, y2, z2 = qx[3], qx[0], qx[1], qx[2]
            qb = (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            )

            transforms = np.zeros((2, 7), dtype=np.float32)
            transforms[0, 3:] = qa
            transforms[1, :3] = [0, 0, sep]
            transforms[1, 3:] = qb
            model.shape_transforms.assign(transforms)
            _collider.collide(model, contact_distance=0.2)

            count = _contact_count(model)
            msg = f"step={step}, angle_z={np.degrees(angle_z):.1f}°"
            self.assertGreater(count, 0, f"{msg}: no contacts")
            depths = model.contact_depth.numpy()[:count]
            for i in range(count):
                self.assertAlmostEqual(depths[i], 0.0, delta=0.05, msg=f"{msg}: contact {i} depth {depths[i]:.4f}")

    def test_edge_edge_small_penetration_sweep(self):
        """Edge-edge with 0.01 penetration, rotate box B around Z axis."""
        b = xc.Builder()
        b.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        b.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        model = b.finalize()

        quat = TestContactRotationSweep._quat_from_axis_angle
        tilt = np.pi / 4.0
        qa = quat(0, 1, 0, tilt)
        pen = 0.01
        sep = np.sqrt(2.0) + np.sqrt(2.0) - pen

        steps = 72
        for step in range(steps):
            angle_z = 2.0 * np.pi * step / steps
            qx = np.array(quat(1, 0, 0, tilt))
            qz = np.array(quat(0, 0, 1, angle_z))
            w1, x1, y1, z1 = qz[3], qz[0], qz[1], qz[2]
            w2, x2, y2, z2 = qx[3], qx[0], qx[1], qx[2]
            qb = (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            )

            transforms = np.zeros((2, 7), dtype=np.float32)
            transforms[0, 3:] = qa
            transforms[1, :3] = [0, 0, sep]
            transforms[1, 3:] = qb
            model.shape_transforms.assign(transforms)
            _collider.collide(model)

            count = _contact_count(model)
            msg = f"step={step}, angle_z={np.degrees(angle_z):.1f}°"
            self.assertGreater(count, 0, f"{msg}: no contacts")
            depths = model.contact_depth.numpy()[:count]
            points = model.contact_point.numpy()[:count]
            normals = model.contact_normal.numpy()[:count]
            for i in range(count):
                d = depths[i]
                self.assertLess(abs(d), 0.1, f"{msg}: depth {d:.4f} too large")
                n_len = np.linalg.norm(normals[i])
                self.assertAlmostEqual(n_len, 1.0, delta=0.01, msg=f"{msg}: normal not unit ({n_len:.4f})")
                self.assertTrue(np.all(np.isfinite(points[i])), f"{msg}: non-finite point {points[i]}")


class TestBothBoxesRotating(unittest.TestCase):
    """Both boxes rotate independently — stress test for all contact types."""

    @staticmethod
    def _quat_from_axis_angle(ax, ay, az, angle):
        s = np.sin(angle / 2.0)
        c = np.cos(angle / 2.0)
        return (ax * s, ay * s, az * s, c)

    def test_both_rotating_sweep(self):
        """Box A rotates around Y, box B rotates around X. Sweep both."""
        b = xc.Builder()
        b.add_shape(xc.SHAPE_BOX, params=(0.5, 0.5, 0.5))
        b.add_shape(xc.SHAPE_BOX, params=(0.5, 0.5, 0.5))
        model = b.finalize()

        overlap = 0.05
        steps = 72
        for step_a in range(steps):
            angle_a = 2.0 * np.pi * step_a / steps
            # Box B rotates 3x faster (different period to cover more combos)
            angle_b = 3.0 * angle_a

            qa = self._quat_from_axis_angle(0, 1, 0, angle_a)
            qb = self._quat_from_axis_angle(1, 0, 0, angle_b)

            # Separation: worst case is corner-corner = sqrt(3)*half.
            # Use a conservative distance that always overlaps a bit.
            sep = 2.0 * 0.5 - overlap  # face-face distance minus overlap

            transforms = np.zeros((2, 7), dtype=np.float32)
            transforms[0, 3:] = qa
            transforms[1, :3] = [0, 0, sep]
            transforms[1, 3:] = qb
            model.shape_transforms.assign(transforms)
            _collider.collide(model)

            count = _contact_count(model)
            msg = f"step={step_a}, angle_a={np.degrees(angle_a):.1f}° angle_b={np.degrees(angle_b):.1f}°"

            # At some orientations the boxes may not overlap (corner-corner
            # is further than face-face). That's fine — skip validation.
            if count == 0:
                continue

            self.assertLessEqual(count, 4, f"{msg}: too many contacts ({count})")

            depths = model.contact_depth.numpy()[:count]
            points = model.contact_point.numpy()[:count]
            normals = model.contact_normal.numpy()[:count]
            for i in range(count):
                d = depths[i]
                self.assertLess(d, 0.5, f"{msg}: contact {i} depth too positive ({d:.4f})")
                self.assertGreater(d, -2.0, f"{msg}: contact {i} depth too large ({d:.4f})")

                p = points[i]
                self.assertTrue(np.all(np.isfinite(p)), f"{msg}: contact {i} non-finite point {p}")
                # Both boxes are near the origin, contacts should be close
                self.assertLess(np.linalg.norm(p), 2.0, f"{msg}: contact {i} point too far: {p}")

                n = normals[i]
                n_len = np.linalg.norm(n)
                self.assertGreater(n_len, 0.9, f"{msg}: contact {i} degenerate normal (len={n_len:.4f})")


class TestEdgeDepthSign(unittest.TestCase):
    """Verify contact depth sign is correct for edge-edge contacts."""

    @staticmethod
    def _quat_y(angle):
        return (0.0, np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0))

    def test_separated_parallel_edges_positive_depth(self):
        """Two boxes tilted 45°, edges parallel, separated by gap.
        All contact depths must be positive (= separation)."""
        b = xc.Builder()
        b.add_shape(xc.SHAPE_BOX, params=(0.25, 0.25, 0.25))
        b.add_shape(xc.SHAPE_BOX, params=(0.25, 0.25, 0.25))
        model = b.finalize()

        tilt = np.pi / 4.0
        q = self._quat_y(tilt)
        gap = 0.05
        sep = np.sqrt(2.0) * 0.25 * 2.0 + gap

        transforms = np.zeros((2, 7), dtype=np.float32)
        transforms[0, 3:] = q
        transforms[1, :3] = [0, 0, sep]
        transforms[1, 3:] = q
        model.shape_transforms.assign(transforms)
        _collider.collide(model, contact_distance=0.2)

        count = _contact_count(model)
        self.assertGreater(count, 0, "Should detect separated contact within contact_distance")
        depths = model.contact_depth.numpy()[:count]
        for i in range(count):
            self.assertGreater(depths[i], 0, f"Separated contact {i} should have positive depth, got {depths[i]:.4f}")
            self.assertAlmostEqual(
                depths[i], gap, delta=0.01, msg=f"Contact {i} depth {depths[i]:.4f} != expected gap {gap}"
            )

    def test_penetrating_parallel_edges_negative_depth(self):
        """Two boxes tilted 45°, edges parallel, overlapping.
        All contact depths must be negative (= penetration)."""
        b = xc.Builder()
        b.add_shape(xc.SHAPE_BOX, params=(0.25, 0.25, 0.25))
        b.add_shape(xc.SHAPE_BOX, params=(0.25, 0.25, 0.25))
        model = b.finalize()

        tilt = np.pi / 4.0
        q = self._quat_y(tilt)
        pen = 0.02
        sep = np.sqrt(2.0) * 0.25 * 2.0 - pen

        transforms = np.zeros((2, 7), dtype=np.float32)
        transforms[0, 3:] = q
        transforms[1, :3] = [0, 0, sep]
        transforms[1, 3:] = q
        model.shape_transforms.assign(transforms)
        _collider.collide(model)

        count = _contact_count(model)
        self.assertGreater(count, 0, "Should detect penetrating contact")
        depths = model.contact_depth.numpy()[:count]
        for i in range(count):
            self.assertLess(depths[i], 0, f"Penetrating contact {i} should have negative depth, got {depths[i]:.4f}")

    def test_separated_nonparallel_edges_positive_depth(self):
        """Two boxes at different tilts, edges not parallel, separated.
        Contacts should have positive depth."""
        b = xc.Builder()
        b.add_shape(xc.SHAPE_BOX, params=(0.25, 0.25, 0.25))
        b.add_shape(xc.SHAPE_BOX, params=(0.25, 0.25, 0.25))
        model = b.finalize()

        qa = self._quat_y(np.pi / 4.0)
        # Box B tilted 45° around X instead of Y — edges cross
        qb = (np.sin(np.pi / 8.0), 0.0, 0.0, np.cos(np.pi / 8.0))

        gap = 0.05
        # Conservative separation
        sep = np.sqrt(3.0) * 0.25 + np.sqrt(3.0) * 0.25 + gap

        for angle_z_deg in range(0, 360, 30):
            angle_z = np.radians(angle_z_deg)
            sz = np.sin(angle_z / 2.0)
            cz = np.cos(angle_z / 2.0)
            # Compose qb_rot = qz * qb
            w1, x1, y1, z1 = cz, 0.0, 0.0, sz
            w2, x2, y2, z2 = qb[3], qb[0], qb[1], qb[2]
            qb_rot = (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            )

            transforms = np.zeros((2, 7), dtype=np.float32)
            transforms[0, 3:] = qa
            transforms[1, :3] = [0, 0, sep]
            transforms[1, 3:] = qb_rot
            model.shape_transforms.assign(transforms)
            _collider.collide(model, contact_distance=0.2)

            count = _contact_count(model)
            if count == 0:
                continue
            depths = model.contact_depth.numpy()[:count]
            msg = f"angle_z={angle_z_deg}°"
            for i in range(count):
                self.assertGreater(depths[i], -0.01, f"{msg}: separated contact {i} has wrong depth {depths[i]:.4f}")


if __name__ == "__main__":
    unittest.main()
