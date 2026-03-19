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


if __name__ == "__main__":
    unittest.main()
