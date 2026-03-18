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

"""Tests for the xcol user-facing API: add_shape, collide, contacts."""

import unittest

import numpy as np

import newton._src.xcol as xc

# Identity quaternion as (qx, qy, qz, qw)
_QUAT_ID = (0.0, 0.0, 0.0, 1.0)


def _transforms(*positions):
    """Build a (n, 7) numpy array of transforms from positions."""
    out = np.zeros((len(positions), 7), dtype=np.float32)
    for i, p in enumerate(positions):
        out[i, :3] = p
        out[i, 3:] = _QUAT_ID
    return out


def _count(contacts):
    """Read contact count from GPU."""
    return contacts.count.numpy()[0]


class TestAddShape(unittest.TestCase):
    def test_add_shapes(self):
        pipeline = xc.create_pipeline()
        s0 = pipeline.add_shape(xc.SHAPE_BOX, params=(10, 1, 10))
        s1 = pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        self.assertEqual(s0, 0)
        self.assertEqual(s1, 1)
        self.assertEqual(pipeline.shape_count, 2)

    def test_shape_arrays_populated(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_BOX, params=(1, 2, 3), margin=0.1, world=2)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=0.5, world=-1)
        self.assertEqual(pipeline.shape_types.numpy()[0], xc.SHAPE_BOX)
        self.assertEqual(pipeline.shape_types.numpy()[1], xc.SHAPE_POINT)
        np.testing.assert_allclose(pipeline.shape_params.numpy()[0], [1, 2, 3])
        self.assertAlmostEqual(pipeline.shape_margins.numpy()[0], 0.1)
        self.assertAlmostEqual(pipeline.shape_margins.numpy()[1], 0.5)
        self.assertEqual(pipeline.shape_worlds.numpy()[0], 2)
        self.assertEqual(pipeline.shape_worlds.numpy()[1], -1)

    def test_set_transforms(self):
        """User can upload transforms via set_transforms()."""
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.set_transforms(_transforms((1, 2, 3)))
        t = pipeline.shape_transforms.numpy()[0]
        np.testing.assert_allclose(t[:3], [1, 2, 3])


class TestCollide(unittest.TestCase):
    def test_two_overlapping_spheres(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.set_transforms(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        contacts = pipeline.collide()
        self.assertGreater(_count(contacts), 0)

    def test_separated_no_contacts(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.set_transforms(_transforms((-5, 0, 0), (5, 0, 0)))
        contacts = pipeline.collide()
        self.assertEqual(_count(contacts), 0)

    def test_box_on_box_4_contacts(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        pipeline.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        pipeline.set_transforms(_transforms((0, 0, 0), (1.5, 0, 0)))
        contacts = pipeline.collide()
        self.assertEqual(_count(contacts), 4)

    def test_contact_arrays_accessible(self):
        """Contact SoA arrays are readable from GPU."""
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.set_transforms(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        contacts = pipeline.collide()
        n = _count(contacts)
        self.assertEqual(n, 1)
        self.assertEqual(contacts.shape_a.numpy()[0], 0)
        self.assertEqual(contacts.shape_b.numpy()[0], 1)
        self.assertGreater(contacts.depth.numpy()[0], 0.0)
        self.assertAlmostEqual(np.linalg.norm(contacts.normal.numpy()[0]), 1.0, places=3)


class TestWorldFiltering(unittest.TestCase):
    def test_different_worlds_no_collision(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=1)
        pipeline.set_transforms(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        contacts = pipeline.collide()
        self.assertEqual(_count(contacts), 0)

    def test_same_world_collides(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        pipeline.set_transforms(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        contacts = pipeline.collide()
        self.assertGreater(_count(contacts), 0)

    def test_global_world_collides_with_any(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=-1)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=42)
        pipeline.set_transforms(_transforms((-0.5, 0, 0), (0.5, 0, 0)))
        contacts = pipeline.collide()
        self.assertGreater(_count(contacts), 0)

    def test_three_shapes_world_filtering(self):
        """Only same-world pairs collide, global collides with all."""
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=1)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=-1)
        pipeline.set_transforms(_transforms((0, 0, 0), (0.1, 0, 0), (0.2, 0, 0)))
        contacts = pipeline.collide()
        # Pairs: 0-2 global (1 contact), 1-2 global (1 contact) = 2 contacts
        self.assertEqual(_count(contacts), 2)


if __name__ == "__main__":
    unittest.main()
