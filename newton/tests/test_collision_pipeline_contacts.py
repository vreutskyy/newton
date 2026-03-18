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

"""Tests for Phase 3: Contact patch generation (Warp).

Tests Sutherland-Hodgman polygon clipping, polygon reduction to 4 points,
and the full generate_contacts pipeline.
"""

import unittest

import numpy as np
import warp as wp

import newton._src.xcol as xc

_collider = xc.create_collider()


def _make_shape(shape_type, pos, params, margin=0.0, rot=None):
    s = xc.ShapeData()
    s.shape_type = int(shape_type)
    s.pos = wp.vec3(*pos)
    s.params = wp.vec3(*params)
    s.margin = float(margin)
    s.rot = wp.quat(*rot) if rot else wp.quat_identity()
    return s


def _run_generate_contacts(shape_a, shape_b):
    sa = wp.array([shape_a], dtype=xc.ShapeData)
    sb = wp.array([shape_b], dtype=xc.ShapeData)
    out = wp.zeros(1, dtype=xc.ContactResult)
    wp.launch(_collider.generate_contacts_kernel, dim=1, inputs=[sa, sb], outputs=[out])
    r = out.numpy()[0]
    # Warp struct numpy returns named fields
    return {
        "normal": np.array(r["normal"]),
        "p0": np.array(r["p0"]),
        "p1": np.array(r["p1"]),
        "p2": np.array(r["p2"]),
        "p3": np.array(r["p3"]),
        "d0": float(r["d0"]),
        "d1": float(r["d1"]),
        "d2": float(r["d2"]),
        "d3": float(r["d3"]),
        "count": int(r["count"]),
    }


# ---------------------------------------------------------------------------
# Sphere-sphere (point + margin): always 1 contact point
# ---------------------------------------------------------------------------


class TestContactSphereSphere(unittest.TestCase):
    def test_single_contact_point(self):
        """Overlapping spheres produce exactly 1 contact point."""
        a = _make_shape(xc.SHAPE_POINT, (-0.5, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (0.5, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_generate_contacts(a, b)
        self.assertEqual(r["count"], 1)

    def test_contact_depth(self):
        """Contact depth matches penetration."""
        a = _make_shape(xc.SHAPE_POINT, (-0.5, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (0.5, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_generate_contacts(a, b)
        self.assertAlmostEqual(r["d0"], 1.0, places=2)

    def test_separated_no_contacts(self):
        """Separated shapes produce 0 contacts."""
        a = _make_shape(xc.SHAPE_POINT, (-5, 0, 0), (0, 0, 0), margin=1.0)
        b = _make_shape(xc.SHAPE_POINT, (5, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_generate_contacts(a, b)
        self.assertEqual(r["count"], 0)


# ---------------------------------------------------------------------------
# Box-box face-face: should produce 4 contact points
# ---------------------------------------------------------------------------


class TestContactBoxBox(unittest.TestCase):
    def test_face_face_4_contacts(self):
        """Two boxes overlapping face-on produce 4 contact points."""
        a = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.0)
        b = _make_shape(xc.SHAPE_BOX, (1.5, 0, 0), (1, 1, 1), margin=0.0)
        r = _run_generate_contacts(a, b)
        self.assertEqual(r["count"], 4)

    def test_face_face_depth(self):
        """Face-face contacts have correct depth."""
        a = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.0)
        b = _make_shape(xc.SHAPE_BOX, (1.5, 0, 0), (1, 1, 1), margin=0.0)
        r = _run_generate_contacts(a, b)
        # Penetration = 0.5
        for i in range(r["count"]):
            self.assertAlmostEqual(r[f"d{i}"], 0.5, delta=0.15)

    def test_face_face_contacts_on_surface(self):
        """Contact points lie on the penetrating face."""
        a = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.0)
        b = _make_shape(xc.SHAPE_BOX, (1.5, 0, 0), (1, 1, 1), margin=0.0)
        r = _run_generate_contacts(a, b)
        for i in range(r["count"]):
            pt = r[f"p{i}"]
            # All contact points should have y,z within [-1, 1]
            self.assertLessEqual(abs(pt[1]), 1.05)
            self.assertLessEqual(abs(pt[2]), 1.05)


# ---------------------------------------------------------------------------
# Box-sphere: 1 contact point
# ---------------------------------------------------------------------------


class TestContactBoxSphere(unittest.TestCase):
    def test_single_contact(self):
        box = _make_shape(xc.SHAPE_BOX, (0, 0, 0), (1, 1, 1), margin=0.0)
        sphere = _make_shape(xc.SHAPE_POINT, (1.5, 0, 0), (0, 0, 0), margin=1.0)
        r = _run_generate_contacts(box, sphere)
        self.assertEqual(r["count"], 1)


# ---------------------------------------------------------------------------
# Capsule-capsule parallel: 2 contact points
# ---------------------------------------------------------------------------


class TestContactCapsuleCapsule(unittest.TestCase):
    def test_parallel_2_contacts(self):
        """Two parallel capsules produce 2 contact points (line contact)."""
        a = _make_shape(xc.SHAPE_SEGMENT, (0, -0.5, 0), (1, 0, 0), margin=0.5)
        b = _make_shape(xc.SHAPE_SEGMENT, (0, 0.5, 0), (1, 0, 0), margin=0.5)
        r = _run_generate_contacts(a, b)
        self.assertEqual(r["count"], 2)


if __name__ == "__main__":
    unittest.main()
