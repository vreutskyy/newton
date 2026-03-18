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

"""Tests for the xcol user-facing API: add_shape, collide, contacts."""

import unittest

import numpy as np
import warp as wp

import newton._src.xcol as xc


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

    def test_transform_writable(self):
        """User can write transforms before collide()."""
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        # Write a transform: position (1,2,3), identity rotation
        pipeline.shape_transforms.numpy()[0] = (1, 2, 3, 0, 0, 0, 1)
        t = pipeline.shape_transforms.numpy()[0]
        np.testing.assert_allclose(t[:3], [1, 2, 3])


class TestCollide(unittest.TestCase):
    def test_two_overlapping_spheres(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.shape_transforms.numpy()[0] = (-0.5, 0, 0, 0, 0, 0, 1)
        pipeline.shape_transforms.numpy()[1] = (0.5, 0, 0, 0, 0, 0, 1)
        contacts = pipeline.collide()
        self.assertGreater(contacts.count, 0)

    def test_separated_no_contacts(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0)
        pipeline.shape_transforms.numpy()[0] = (-5, 0, 0, 0, 0, 0, 1)
        pipeline.shape_transforms.numpy()[1] = (5, 0, 0, 0, 0, 0, 1)
        contacts = pipeline.collide()
        self.assertEqual(contacts.count, 0)

    def test_box_on_box_4_contacts(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        pipeline.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        pipeline.shape_transforms.numpy()[0] = (0, 0, 0, 0, 0, 0, 1)
        pipeline.shape_transforms.numpy()[1] = (1.5, 0, 0, 0, 0, 0, 1)
        contacts = pipeline.collide()
        self.assertGreater(contacts.count, 0)
        # First pair should have 4 contact points
        self.assertEqual(contacts.point_count[0], 4)


class TestWorldFiltering(unittest.TestCase):
    def test_different_worlds_no_collision(self):
        """Shapes in different worlds don't collide."""
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=1)
        pipeline.shape_transforms.numpy()[0] = (-0.5, 0, 0, 0, 0, 0, 1)
        pipeline.shape_transforms.numpy()[1] = (0.5, 0, 0, 0, 0, 0, 1)
        contacts = pipeline.collide()
        self.assertEqual(contacts.count, 0)

    def test_same_world_collides(self):
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)
        pipeline.shape_transforms.numpy()[0] = (-0.5, 0, 0, 0, 0, 0, 1)
        pipeline.shape_transforms.numpy()[1] = (0.5, 0, 0, 0, 0, 0, 1)
        contacts = pipeline.collide()
        self.assertGreater(contacts.count, 0)

    def test_global_world_collides_with_any(self):
        """World -1 shapes collide with everything."""
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=-1)
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=42)
        pipeline.shape_transforms.numpy()[0] = (-0.5, 0, 0, 0, 0, 0, 1)
        pipeline.shape_transforms.numpy()[1] = (0.5, 0, 0, 0, 0, 0, 1)
        contacts = pipeline.collide()
        self.assertGreater(contacts.count, 0)

    def test_three_shapes_world_filtering(self):
        """Only same-world pairs collide, global collides with all."""
        pipeline = xc.create_pipeline()
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=0)   # 0
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=1)   # 1
        pipeline.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=1.0, world=-1)  # 2
        # All overlapping at origin
        pipeline.shape_transforms.numpy()[0] = (0, 0, 0, 0, 0, 0, 1)
        pipeline.shape_transforms.numpy()[1] = (0.1, 0, 0, 0, 0, 0, 1)
        pipeline.shape_transforms.numpy()[2] = (0.2, 0, 0, 0, 0, 0, 1)
        contacts = pipeline.collide()
        # Pairs: 0-1 different worlds (no), 0-2 global (yes), 1-2 global (yes)
        self.assertEqual(contacts.count, 2)


if __name__ == "__main__":
    unittest.main()
