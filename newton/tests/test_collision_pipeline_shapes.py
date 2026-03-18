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

"""Tests for shapes: support functions, contact faces, AABBs (Warp).

Shapes are core + margin.  A sphere is a point core with margin = radius.
A capsule is a segment core with margin = radius.
"""

import unittest

import numpy as np
import warp as wp

import newton._src.xcol as xc

_pipeline = xc.create_pipeline()


def _make_shape(shape_type, pos, params, margin=0.0, rot=None):
    s = xc.ShapeData()
    s.shape_type = int(shape_type)
    s.pos = wp.vec3(*pos)
    s.params = wp.vec3(*params)
    s.margin = float(margin)
    s.rot = wp.quat(*rot) if rot else wp.quat_identity()
    return s


def _run_support(shape, direction):
    shapes = wp.array([shape], dtype=xc.ShapeData)
    dirs = wp.array([wp.vec3(*direction)], dtype=wp.vec3)
    out = wp.zeros(1, dtype=wp.vec3)
    wp.launch(_pipeline.support_kernel, dim=1, inputs=[shapes, dirs], outputs=[out])
    return out.numpy()[0]


def _run_contact_face(shape, direction):
    shapes = wp.array([shape], dtype=xc.ShapeData)
    dirs = wp.array([wp.vec3(*direction)], dtype=wp.vec3)
    out_p0 = wp.zeros(1, dtype=wp.vec3)
    out_normal = wp.zeros(1, dtype=wp.vec3)
    out_count = wp.zeros(1, dtype=int)
    wp.launch(_pipeline.contact_face_kernel, dim=1, inputs=[shapes, dirs], outputs=[out_p0, out_normal, out_count])
    return out_p0.numpy()[0], out_normal.numpy()[0], out_count.numpy()[0]


def _run_aabb(shape):
    shapes = wp.array([shape], dtype=xc.ShapeData)
    out_min = wp.zeros(1, dtype=wp.vec3)
    out_max = wp.zeros(1, dtype=wp.vec3)
    wp.launch(_pipeline.aabb_kernel, dim=1, inputs=[shapes], outputs=[out_min, out_max])
    return out_min.numpy()[0], out_max.numpy()[0]


# ---------------------------------------------------------------------------
# Point core (sphere = point + margin)
# ---------------------------------------------------------------------------


class TestPointSupport(unittest.TestCase):
    def test_support_returns_center(self):
        """Point core support always returns the center (core = single point)."""
        s = _make_shape(xc.SHAPE_POINT, (3, 4, 5), (0, 0, 0), margin=1.0)
        pt = _run_support(s, (1, 0, 0))
        # Core support returns center (margin not included in support)
        np.testing.assert_allclose(pt, [3, 4, 5], atol=1e-5)

    def test_support_independent_of_direction(self):
        s = _make_shape(xc.SHAPE_POINT, (0, 0, 0), (0, 0, 0), margin=2.0)
        pt1 = _run_support(s, (1, 0, 0))
        pt2 = _run_support(s, (0, -1, 0))
        pt3 = _run_support(s, (1, 1, 1))
        np.testing.assert_allclose(pt1, [0, 0, 0], atol=1e-5)
        np.testing.assert_allclose(pt2, [0, 0, 0], atol=1e-5)
        np.testing.assert_allclose(pt3, [0, 0, 0], atol=1e-5)


class TestPointContactFace(unittest.TestCase):
    def test_single_point_with_margin(self):
        """Contact face for point+margin is 1 point offset by margin."""
        s = _make_shape(xc.SHAPE_POINT, (0, 0, 0), (0, 0, 0), margin=1.0)
        p0, normal, count = _run_contact_face(s, (1, 0, 0))
        self.assertEqual(count, 1)
        # Contact face includes margin offset along normal
        np.testing.assert_allclose(p0, [1, 0, 0], atol=1e-5)
        np.testing.assert_allclose(normal, [1, 0, 0], atol=1e-5)


class TestPointAABB(unittest.TestCase):
    def test_aabb_is_sphere(self):
        """Point + margin AABB is a cube of side 2*margin centered at pos."""
        s = _make_shape(xc.SHAPE_POINT, (1, 2, 3), (0, 0, 0), margin=0.5)
        mn, mx = _run_aabb(s)
        np.testing.assert_allclose(mn, [0.5, 1.5, 2.5], atol=1e-5)
        np.testing.assert_allclose(mx, [1.5, 2.5, 3.5], atol=1e-5)


# ---------------------------------------------------------------------------
# Segment core (capsule = segment + margin)
# ---------------------------------------------------------------------------


class TestSegmentSupport(unittest.TestCase):
    def test_support_positive_z(self):
        """Segment support in +z returns top endpoint."""
        s = _make_shape(xc.SHAPE_SEGMENT, (0, 0, 0), (2, 0, 0), margin=0.5)
        pt = _run_support(s, (0, 0, 1))
        np.testing.assert_allclose(pt, [0, 0, 2], atol=1e-5)

    def test_support_negative_z(self):
        s = _make_shape(xc.SHAPE_SEGMENT, (0, 0, 0), (2, 0, 0), margin=0.5)
        pt = _run_support(s, (0, 0, -1))
        np.testing.assert_allclose(pt, [0, 0, -2], atol=1e-5)

    def test_support_lateral(self):
        """Lateral direction: segment core returns same z endpoint."""
        s = _make_shape(xc.SHAPE_SEGMENT, (0, 0, 0), (2, 0, 0), margin=0.5)
        pt = _run_support(s, (1, 0, 0.1))
        # z >= 0 so returns +half_height endpoint
        np.testing.assert_allclose(pt, [0, 0, 2], atol=1e-5)


class TestSegmentAABB(unittest.TestCase):
    def test_aabb_capsule(self):
        """Segment + margin AABB is the capsule bounding box."""
        s = _make_shape(xc.SHAPE_SEGMENT, (0, 0, 0), (1, 0, 0), margin=0.5)
        mn, mx = _run_aabb(s)
        # Segment along Z: half_height=1, margin=0.5
        np.testing.assert_allclose(mn, [-0.5, -0.5, -1.5], atol=1e-5)
        np.testing.assert_allclose(mx, [0.5, 0.5, 1.5], atol=1e-5)


# ---------------------------------------------------------------------------
# Box core (rounded box = box + margin)
# ---------------------------------------------------------------------------


class TestBoxSupport(unittest.TestCase):
    def test_support_positive_octant(self):
        s = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 2, 3), margin=0.0)
        pt = _run_support(s, (1, 1, 1))
        np.testing.assert_allclose(pt, [1, 2, 3], atol=1e-5)

    def test_support_with_offset(self):
        s = _make_shape(xc.SHAPE_BOX, (10, 0, 0), (1, 1, 1), margin=0.0)
        pt = _run_support(s, (1, 0, 0))
        np.testing.assert_allclose(pt, [11, 1, 1], atol=1e-5)


class TestBoxAABB(unittest.TestCase):
    def test_axis_aligned_no_margin(self):
        s = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 2, 3), margin=0.0)
        mn, mx = _run_aabb(s)
        np.testing.assert_allclose(mn, [-1, -2, -3], atol=1e-5)
        np.testing.assert_allclose(mx, [1, 2, 3], atol=1e-5)

    def test_axis_aligned_with_margin(self):
        s = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 2, 3), margin=0.5)
        mn, mx = _run_aabb(s)
        np.testing.assert_allclose(mn, [-1.5, -2.5, -3.5], atol=1e-5)
        np.testing.assert_allclose(mx, [1.5, 2.5, 3.5], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
