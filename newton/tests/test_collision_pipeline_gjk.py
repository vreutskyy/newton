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

"""Tests for GJK distance query with core + margin shapes (Warp)."""

import unittest

import numpy as np
import warp as wp

import xcol as xc

_collider = xc.create_collider()


def _make_shape(shape_type, pos, params, margin=0.0, rot=None):
    s = xc.ShapeData()
    s.shape_type = int(shape_type)
    s.pos = wp.vec3(*pos)
    s.params = wp.vec3(*params)
    s.margin = float(margin)
    s.rot = wp.quat(*rot) if rot else wp.quat_identity()
    return s


def _run_gjk(shape_a, shape_b):
    sa = wp.array([shape_a], dtype=xc.ShapeData)
    sb = wp.array([shape_b], dtype=xc.ShapeData)
    dist = wp.zeros(1, dtype=float)
    pa = wp.zeros(1, dtype=wp.vec3)
    pb = wp.zeros(1, dtype=wp.vec3)
    normal = wp.zeros(1, dtype=wp.vec3)
    overlap = wp.zeros(1, dtype=int)
    wp.launch(_collider.gjk_kernel, dim=1, inputs=[sa, sb], outputs=[dist, pa, pb, normal, overlap])
    return {
        "distance": dist.numpy()[0],
        "point_a": pa.numpy()[0],
        "point_b": pb.numpy()[0],
        "normal": normal.numpy()[0],
        "overlap": overlap.numpy()[0],
    }


class TestGJKSphereSphere(unittest.TestCase):
    """Sphere = point core + margin."""

    def test_separated(self):
        # Two spheres r=1 centered at ±2 → distance = 2
        a = _make_shape(xc.SHAPE_POINT, (-2, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (2, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk(a, b)
        self.assertAlmostEqual(r["distance"], 2.0, places=3)
        self.assertEqual(r["overlap"], 0)

    def test_touching(self):
        a = _make_shape(xc.SHAPE_POINT, (-1, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (1, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk(a, b)
        self.assertAlmostEqual(r["distance"], 0.0, places=2)

    def test_overlapping(self):
        # margin-only overlap: cores (points) separated but margins overlap
        a = _make_shape(xc.SHAPE_POINT, (-0.5, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (0.5, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk(a, b)
        self.assertGreater(r["overlap"], 0)
        # Penetration = 2*margin - distance_between_cores = 2 - 1 = 1
        self.assertAlmostEqual(r["distance"], 1.0, places=2)

    def test_normal_direction(self):
        a = _make_shape(xc.SHAPE_POINT, (0, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (5, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk(a, b)
        np.testing.assert_allclose(r["normal"], [1, 0, 0], atol=0.05)

    def test_different_radii(self):
        a = _make_shape(xc.SHAPE_POINT, (0, 0, 0), (0, 0, 0), margin=2.0)
        b = _make_shape(xc.SHAPE_POINT, (5, 0, 0), (0, 0, 0), margin=0.5)
        r = _run_gjk(a, b)
        self.assertAlmostEqual(r["distance"], 2.5, places=3)

    def test_witness_points(self):
        a = _make_shape(xc.SHAPE_POINT, (-2, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (2, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk(a, b)
        # Witness on A: center + margin along normal = (-2,0,0) + 1*(1,0,0) = (-1,0,0)
        np.testing.assert_allclose(r["point_a"], [-1, 0, 0], atol=0.05)
        np.testing.assert_allclose(r["point_b"], [1, 0, 0], atol=0.05)


class TestGJKBoxSphere(unittest.TestCase):
    """Box core (no margin) vs point core (with margin = sphere)."""

    def test_separated(self):
        box = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.0)
        sphere = _make_shape(xc.SHAPE_POINT, (3, 0, 0), (0, 0, 0), margin=0.5)
        r = _run_gjk(box, sphere)
        # Box face at x=1, sphere at x=2.5 → distance = 1.5
        self.assertAlmostEqual(r["distance"], 1.5, places=2)
        self.assertEqual(r["overlap"], 0)

    def test_overlapping(self):
        box = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.0)
        sphere = _make_shape(xc.SHAPE_POINT, (1, 0, 0), (0, 0, 0), margin=0.5)
        r = _run_gjk(box, sphere)
        self.assertGreater(r["overlap"], 0)


class TestGJKBoxBox(unittest.TestCase):
    def test_separated(self):
        a = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.0)
        b = _make_shape(xc.SHAPE_BOX, (4, 0, 0), (1, 1, 1), margin=0.0)
        r = _run_gjk(a, b)
        self.assertAlmostEqual(r["distance"], 2.0, places=2)
        self.assertEqual(r["overlap"], 0)

    def test_margin_overlap(self):
        """Boxes with margin: cores separated but margins overlap."""
        a = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.3)
        b = _make_shape(xc.SHAPE_BOX, (2.4, 0, 0), (1, 1, 1), margin=0.3)
        r = _run_gjk(a, b)
        # Core distance = 2.4 - 2 = 0.4, total margin = 0.6 → penetration = 0.2
        self.assertGreater(r["overlap"], 0)
        self.assertAlmostEqual(r["distance"], 0.2, places=1)


class TestGJKZeroMargin(unittest.TestCase):
    """Zero margin should work like the old behavior."""

    def test_boxes_no_margin(self):
        a = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.0)
        b = _make_shape(xc.SHAPE_BOX, (4, 0, 0), (1, 1, 1), margin=0.0)
        r = _run_gjk(a, b)
        self.assertAlmostEqual(r["distance"], 2.0, places=2)
        self.assertEqual(r["overlap"], 0)


if __name__ == "__main__":
    unittest.main()
