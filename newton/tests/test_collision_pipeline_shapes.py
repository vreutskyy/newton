# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

"""Tests for Phase 0: Shape support functions and contact faces (Warp)."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.collision_pipeline.core import (
    SHAPE_BOX,
    SHAPE_SPHERE,
    ShapeData,
    create_pipeline,
)

# Create pipeline once — includes all built-in shapes
_pipeline = create_pipeline()


def _make_shape(shape_type, pos, params, rot=None):
    s = ShapeData()
    s.shape_type = int(shape_type)
    s.pos = wp.vec3(*pos)
    s.params = wp.vec3(*params)
    s.rot = wp.quat(*rot) if rot else wp.quat_identity()
    return s


def _run_support(shape, direction):
    shapes = wp.array([shape], dtype=ShapeData)
    dirs = wp.array([wp.vec3(*direction)], dtype=wp.vec3)
    out = wp.zeros(1, dtype=wp.vec3)
    wp.launch(_pipeline.support_kernel, dim=1, inputs=[shapes, dirs], outputs=[out])
    return out.numpy()[0]


def _run_contact_face(shape, direction):
    shapes = wp.array([shape], dtype=ShapeData)
    dirs = wp.array([wp.vec3(*direction)], dtype=wp.vec3)
    out_p0 = wp.zeros(1, dtype=wp.vec3)
    out_normal = wp.zeros(1, dtype=wp.vec3)
    out_count = wp.zeros(1, dtype=int)
    wp.launch(_pipeline.contact_face_kernel, dim=1, inputs=[shapes, dirs], outputs=[out_p0, out_normal, out_count])
    return out_p0.numpy()[0], out_normal.numpy()[0], out_count.numpy()[0]


def _run_aabb(shape):
    shapes = wp.array([shape], dtype=ShapeData)
    out_min = wp.zeros(1, dtype=wp.vec3)
    out_max = wp.zeros(1, dtype=wp.vec3)
    wp.launch(_pipeline.aabb_kernel, dim=1, inputs=[shapes], outputs=[out_min, out_max])
    return out_min.numpy()[0], out_max.numpy()[0]


class TestSphereSupport(unittest.TestCase):
    def test_support_positive_x(self):
        s = _make_shape(SHAPE_SPHERE, (0, 0, 0), (1, 0, 0))
        pt = _run_support(s, (1, 0, 0))
        np.testing.assert_allclose(pt, [1, 0, 0], atol=1e-5)

    def test_support_negative_y(self):
        s = _make_shape(SHAPE_SPHERE, (0, 0, 0), (1, 0, 0))
        pt = _run_support(s, (0, -1, 0))
        np.testing.assert_allclose(pt, [0, -1, 0], atol=1e-5)

    def test_support_diagonal(self):
        s = _make_shape(SHAPE_SPHERE, (0, 0, 0), (2, 0, 0))
        pt = _run_support(s, (1, 1, 1))
        d = np.array([1, 1, 1])
        expected = d / np.linalg.norm(d) * 2.0
        np.testing.assert_allclose(pt, expected, atol=1e-5)

    def test_support_with_offset(self):
        s = _make_shape(SHAPE_SPHERE, (3, 4, 5), (1, 0, 0))
        pt = _run_support(s, (1, 0, 0))
        np.testing.assert_allclose(pt, [4, 4, 5], atol=1e-5)

    def test_support_different_radius(self):
        s = _make_shape(SHAPE_SPHERE, (0, 0, 0), (5, 0, 0))
        pt = _run_support(s, (0, 0, 1))
        np.testing.assert_allclose(pt, [0, 0, 5], atol=1e-5)


class TestSphereContactFace(unittest.TestCase):
    def test_single_point(self):
        s = _make_shape(SHAPE_SPHERE, (0, 0, 0), (1, 0, 0))
        p0, normal, count = _run_contact_face(s, (1, 0, 0))
        self.assertEqual(count, 1)
        np.testing.assert_allclose(p0, [1, 0, 0], atol=1e-5)
        np.testing.assert_allclose(normal, [1, 0, 0], atol=1e-5)

    def test_normal_is_unit(self):
        s = _make_shape(SHAPE_SPHERE, (0, 0, 0), (1, 0, 0))
        _, normal, _ = _run_contact_face(s, (1, 1, 0))
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=5)


class TestSphereAABB(unittest.TestCase):
    def test_aabb(self):
        s = _make_shape(SHAPE_SPHERE, (1, 2, 3), (0.5, 0, 0))
        mn, mx = _run_aabb(s)
        np.testing.assert_allclose(mn, [0.5, 1.5, 2.5], atol=1e-5)
        np.testing.assert_allclose(mx, [1.5, 2.5, 3.5], atol=1e-5)


class TestBoxSupport(unittest.TestCase):
    def test_support_positive_octant(self):
        s = _make_shape(SHAPE_BOX, (0, 0, 0), (1, 2, 3))
        pt = _run_support(s, (1, 1, 1))
        np.testing.assert_allclose(pt, [1, 2, 3], atol=1e-5)

    def test_support_negative_octant(self):
        s = _make_shape(SHAPE_BOX, (0, 0, 0), (1, 2, 3))
        pt = _run_support(s, (-1, -1, -1))
        np.testing.assert_allclose(pt, [-1, -2, -3], atol=1e-5)

    def test_support_with_offset(self):
        s = _make_shape(SHAPE_BOX, (10, 0, 0), (1, 1, 1))
        pt = _run_support(s, (1, 0, 0))
        np.testing.assert_allclose(pt, [11, 1, 1], atol=1e-5)


class TestBoxAABB(unittest.TestCase):
    def test_axis_aligned(self):
        s = _make_shape(SHAPE_BOX, (0, 0, 0), (1, 2, 3))
        mn, mx = _run_aabb(s)
        np.testing.assert_allclose(mn, [-1, -2, -3], atol=1e-5)
        np.testing.assert_allclose(mx, [1, 2, 3], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
