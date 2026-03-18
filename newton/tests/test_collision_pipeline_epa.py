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

"""Tests for GJK+EPA with core + margin shapes (Warp)."""

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


def _run_gjk_epa(shape_a, shape_b):
    sa = wp.array([shape_a], dtype=xc.ShapeData)
    sb = wp.array([shape_b], dtype=xc.ShapeData)
    dist = wp.zeros(1, dtype=float)
    pa = wp.zeros(1, dtype=wp.vec3)
    pb = wp.zeros(1, dtype=wp.vec3)
    normal = wp.zeros(1, dtype=wp.vec3)
    overlap = wp.zeros(1, dtype=int)
    wp.launch(_pipeline.gjk_epa_kernel, dim=1, inputs=[sa, sb], outputs=[dist, pa, pb, normal, overlap])
    return {
        "distance": dist.numpy()[0],
        "point_a": pa.numpy()[0],
        "point_b": pb.numpy()[0],
        "normal": normal.numpy()[0],
        "overlap": overlap.numpy()[0],
    }


class TestEPASphereSphere(unittest.TestCase):
    """EPA tests for spheres (point + margin)."""

    def test_margin_only_overlap(self):
        """Margin-only overlap: GJK handles it, no EPA needed."""
        a = _make_shape(xc.SHAPE_POINT, (-0.5, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (0.5, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk_epa(a, b)
        self.assertGreater(r["overlap"], 0)
        self.assertAlmostEqual(r["distance"], 1.0, places=2)

    def test_normal_direction(self):
        a = _make_shape(xc.SHAPE_POINT, (-0.5, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (0.5, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk_epa(a, b)
        np.testing.assert_allclose(np.abs(r["normal"]), [1, 0, 0], atol=0.1)

    def test_barely_overlapping(self):
        a = _make_shape(xc.SHAPE_POINT, (-0.95, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (0.95, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk_epa(a, b)
        self.assertGreater(r["overlap"], 0)
        self.assertAlmostEqual(r["distance"], 0.1, places=1)

    def test_concentric(self):
        """Concentric spheres: core overlap, needs EPA."""
        a = _make_shape(xc.SHAPE_POINT, (0, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (0.1, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk_epa(a, b)
        self.assertGreater(r["overlap"], 0)
        # Penetration ≈ 1.9 (core dist=0.1, margins=2, pen=1.9)
        # But cores are points → they don't overlap → margin-only
        self.assertAlmostEqual(r["distance"], 1.9, places=1)

    def test_symmetry(self):
        a = _make_shape(xc.SHAPE_POINT, (-0.3, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (0.3, 0, 0), (0, 0, 0), margin=1.0)
        r_ab = _run_gjk_epa(a, b)
        r_ba = _run_gjk_epa(b, a)
        self.assertAlmostEqual(r_ab["distance"], r_ba["distance"], places=2)

    def test_y_axis(self):
        a = _make_shape(xc.SHAPE_POINT, (0, -0.25, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (0, 0.25, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk_epa(a, b)
        self.assertGreater(r["overlap"], 0)
        self.assertAlmostEqual(r["distance"], 1.5, places=1)
        self.assertGreater(np.abs(r["normal"][1]), 0.8)

    def test_separated(self):
        a = _make_shape(xc.SHAPE_POINT, (-2, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (2, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk_epa(a, b)
        self.assertEqual(r["overlap"], 0)
        self.assertGreater(r["distance"], 0.0)


class TestEPABoxSphere(unittest.TestCase):
    def test_sphere_inside_box(self):
        box = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (2, 2, 2), margin=0.0)
        sphere = _make_shape(xc.SHAPE_POINT, (0, 0, 0), (0, 0, 0), margin=0.5)
        r = _run_gjk_epa(box, sphere)
        self.assertGreater(r["overlap"], 0)
        self.assertGreater(r["distance"], 0.0)

    def test_sphere_penetrating_box_face(self):
        box = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.0)
        sphere = _make_shape(xc.SHAPE_POINT, (1.5, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_gjk_epa(box, sphere)
        self.assertGreater(r["overlap"], 0)
        # Box face x=1, sphere surface x=0.5 → penetration=0.5
        self.assertAlmostEqual(r["distance"], 0.5, places=1)


class TestEPABoxBox(unittest.TestCase):
    def test_overlapping_boxes(self):
        a = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.0)
        b = _make_shape(xc.SHAPE_BOX, (1.5, 0, 0), (1, 1, 1), margin=0.0)
        r = _run_gjk_epa(a, b)
        self.assertGreater(r["overlap"], 0)
        self.assertAlmostEqual(r["distance"], 0.5, places=1)

    def test_rounded_boxes(self):
        """Boxes with margin: rounded box overlap."""
        a = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.2)
        b = _make_shape(xc.SHAPE_BOX, (2.3, 0, 0), (1, 1, 1), margin=0.2)
        r = _run_gjk_epa(a, b)
        # Core dist = 2.3 - 2 = 0.3, total margin = 0.4 → pen = 0.1
        self.assertGreater(r["overlap"], 0)
        self.assertAlmostEqual(r["distance"], 0.1, places=1)


if __name__ == "__main__":
    unittest.main()
