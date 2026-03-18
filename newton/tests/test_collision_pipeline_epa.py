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

"""Tests for Phase 2: EPA penetration depth (Warp)."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.collision_pipeline.core import (
    SHAPE_BOX,
    SHAPE_SPHERE,
    ShapeData,
    create_pipeline,
)

_pipeline = create_pipeline()


def _make_shape(shape_type, pos, params, rot=None):
    s = ShapeData()
    s.shape_type = int(shape_type)
    s.pos = wp.vec3(*pos)
    s.params = wp.vec3(*params)
    s.rot = wp.quat(*rot) if rot else wp.quat_identity()
    return s


def _run_gjk_epa(shape_a, shape_b):
    sa = wp.array([shape_a], dtype=ShapeData)
    sb = wp.array([shape_b], dtype=ShapeData)
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
    """EPA penetration tests using sphere pairs."""

    def test_overlapping_spheres_depth(self):
        """Two overlapping spheres: correct penetration depth."""
        # Centers 1 apart, radii 1 each -> penetration = 2 - 1 = 1
        a = _make_shape(SHAPE_SPHERE, (-0.5, 0, 0), (1, 0, 0))
        b = _make_shape(SHAPE_SPHERE, (0.5, 0, 0), (1, 0, 0))
        r = _run_gjk_epa(a, b)
        self.assertEqual(r["overlap"], 1)
        self.assertAlmostEqual(r["distance"], 1.0, places=1)

    def test_overlapping_spheres_normal(self):
        """Penetration normal along center-center axis."""
        a = _make_shape(SHAPE_SPHERE, (-0.5, 0, 0), (1, 0, 0))
        b = _make_shape(SHAPE_SPHERE, (0.5, 0, 0), (1, 0, 0))
        r = _run_gjk_epa(a, b)
        # Normal should be along x axis (A→B)
        np.testing.assert_allclose(np.abs(r["normal"]), [1, 0, 0], atol=0.1)

    def test_barely_overlapping(self):
        """Barely overlapping: small penetration depth."""
        # Centers 1.9 apart, radii 1 each -> penetration = 0.1
        a = _make_shape(SHAPE_SPHERE, (-0.95, 0, 0), (1, 0, 0))
        b = _make_shape(SHAPE_SPHERE, (0.95, 0, 0), (1, 0, 0))
        r = _run_gjk_epa(a, b)
        self.assertEqual(r["overlap"], 1)
        self.assertAlmostEqual(r["distance"], 0.1, places=1)

    def test_deeply_overlapping(self):
        """Deeply overlapping: large penetration depth."""
        # Centers 0.1 apart, radii 1 each -> penetration = 1.9
        a = _make_shape(SHAPE_SPHERE, (0, 0, 0), (1, 0, 0))
        b = _make_shape(SHAPE_SPHERE, (0.1, 0, 0), (1, 0, 0))
        r = _run_gjk_epa(a, b)
        self.assertEqual(r["overlap"], 1)
        # EPA approximates sphere with polytope — relax tolerance
        self.assertAlmostEqual(r["distance"], 1.9, delta=0.3)

    def test_symmetry(self):
        """epa(A,B) and epa(B,A) give consistent penetration depth."""
        a = _make_shape(SHAPE_SPHERE, (-0.3, 0, 0), (1, 0, 0))
        b = _make_shape(SHAPE_SPHERE, (0.3, 0, 0), (1, 0, 0))
        r_ab = _run_gjk_epa(a, b)
        r_ba = _run_gjk_epa(b, a)
        self.assertAlmostEqual(r_ab["distance"], r_ba["distance"], delta=0.2)
        # Normal dominant component should flip sign
        self.assertAlmostEqual(r_ab["normal"][0], -r_ba["normal"][0], delta=0.2)

    def test_overlapping_y_axis(self):
        """Overlap along y axis."""
        a = _make_shape(SHAPE_SPHERE, (0, -0.25, 0), (1, 0, 0))
        b = _make_shape(SHAPE_SPHERE, (0, 0.25, 0), (1, 0, 0))
        r = _run_gjk_epa(a, b)
        self.assertEqual(r["overlap"], 1)
        self.assertAlmostEqual(r["distance"], 1.5, delta=0.2)
        # Normal should be predominantly along y (EPA polytope approximation)
        self.assertGreater(np.abs(r["normal"][1]), 0.8)

    def test_separated_returns_no_penetration(self):
        """Separated shapes: EPA not needed, distance > 0, no overlap."""
        a = _make_shape(SHAPE_SPHERE, (-2, 0, 0), (1, 0, 0))
        b = _make_shape(SHAPE_SPHERE, (2, 0, 0), (1, 0, 0))
        r = _run_gjk_epa(a, b)
        self.assertEqual(r["overlap"], 0)
        self.assertGreater(r["distance"], 0.0)


class TestEPABoxSphere(unittest.TestCase):
    """EPA tests for box-sphere pairs."""

    def test_sphere_inside_box(self):
        """Sphere center inside box."""
        box = _make_shape(SHAPE_BOX, (0, 0, 0), (2, 2, 2))
        sphere = _make_shape(SHAPE_SPHERE, (0, 0, 0), (0.5, 0, 0))
        r = _run_gjk_epa(box, sphere)
        self.assertEqual(r["overlap"], 1)
        self.assertGreater(r["distance"], 0.0)

    def test_sphere_penetrating_box_face(self):
        """Sphere penetrating box along x face."""
        box = _make_shape(SHAPE_BOX, (0, 0, 0), (1, 1, 1))
        sphere = _make_shape(SHAPE_SPHERE, (1.5, 0, 0), (1, 0, 0))
        r = _run_gjk_epa(box, sphere)
        self.assertEqual(r["overlap"], 1)
        # Box face at x=1, sphere surface at x=0.5 -> penetration = 0.5
        self.assertAlmostEqual(r["distance"], 0.5, places=1)


class TestEPABoxBox(unittest.TestCase):
    """EPA tests for box-box pairs."""

    def test_overlapping_boxes(self):
        """Two overlapping axis-aligned boxes."""
        a = _make_shape(SHAPE_BOX, (0, 0, 0), (1, 1, 1))
        b = _make_shape(SHAPE_BOX, (1.5, 0, 0), (1, 1, 1))
        r = _run_gjk_epa(a, b)
        self.assertEqual(r["overlap"], 1)
        # Box A face at x=1, box B face at x=0.5 -> penetration = 0.5
        self.assertAlmostEqual(r["distance"], 0.5, places=1)


if __name__ == "__main__":
    unittest.main()
