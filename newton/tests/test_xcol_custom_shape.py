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

"""Test registering a custom shape type and using it in the full pipeline.

Registers a *disk* shape (flat circle in XY plane, axis along Z).
Core support is ``radius * normalize(direction_xy)``.
Disk + margin = wheel / hockey puck.
"""

import unittest

import numpy as np
import warp as wp

import newton._src.xcol as xc
from newton._src.xcol.types import ContactFaceResult, ShapeData

# ---------------------------------------------------------------------------
# Custom disk shape functions
# ---------------------------------------------------------------------------


@wp.func
def _support_disk(params: wp.vec3, direction: wp.vec3) -> wp.vec3:
    """Core support for a disk (circle in XY plane).

    params: (radius, 0, 0).
    """
    radius = params[0]
    dx = direction[0]
    dy = direction[1]
    lateral_len = wp.sqrt(dx * dx + dy * dy)
    if lateral_len > 1.0e-12:
        return wp.vec3(radius * dx / lateral_len, radius * dy / lateral_len, 0.0)
    return wp.vec3(radius, 0.0, 0.0)


@wp.func
def _contact_face_disk(params: wp.vec3, direction: wp.vec3) -> ContactFaceResult:
    """Contact face for a disk — single point (same as support)."""
    result = ContactFaceResult()
    pt = _support_disk(params, direction)
    result.p0 = pt
    result.p1 = pt
    result.p2 = pt
    result.p3 = pt
    d_len = wp.length(direction)
    if d_len > 1.0e-12:
        result.normal = direction / d_len
    else:
        result.normal = wp.vec3(0.0, 0.0, 1.0)
    result.count = 1
    return result


@wp.func
def _aabb_disk(shape: ShapeData) -> tuple[wp.vec3, wp.vec3]:
    """AABB for a disk core + margin."""
    r = shape.params[0] + shape.margin
    rv = wp.vec3(r, r, shape.margin)
    return shape.pos - rv, shape.pos + rv


# Register the custom disk shape — same path as built-ins
SHAPE_DISK = xc.register_shape(
    "disk",
    support_fn=_support_disk,
    contact_face_fn=_contact_face_disk,
    aabb_fn=_aabb_disk,
)

# Create collider AFTER registering the custom shape
_collider = xc.create_collider()

_QUAT_ID = (0.0, 0.0, 0.0, 1.0)


def _transforms(*positions):
    out = np.zeros((len(positions), 7), dtype=np.float32)
    for i, p in enumerate(positions):
        out[i, :3] = p
        out[i, 3:] = _QUAT_ID
    return out


def _contact_count(model):
    return model.contact_count.numpy()[0]


class TestCustomDiskShape(unittest.TestCase):
    """Test that a user-registered disk shape works through the full pipeline."""

    def test_disk_registered(self):
        """Disk shape gets a valid type id."""
        self.assertIsInstance(SHAPE_DISK, int)
        self.assertGreater(SHAPE_DISK, 0)  # after built-in shapes

    def test_disk_vs_sphere_separated(self):
        """Disk and sphere far apart: no contacts."""
        b = xc.Builder()
        b.add_shape(SHAPE_DISK, params=(1, 0, 0), margin=0.1)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=0.5)
        model = b.finalize()
        model.shape_transforms.assign(_transforms((0, 0, 0), (5, 0, 0)))
        _collider.collide(model)
        self.assertEqual(_contact_count(model), 0)

    def test_disk_vs_sphere_overlapping(self):
        """Disk and sphere overlapping: produces contacts."""
        b = xc.Builder()
        b.add_shape(SHAPE_DISK, params=(1, 0, 0), margin=0.1)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=0.5)
        model = b.finalize()
        # Disk radius=1 + margin=0.1, sphere margin=0.5, along x
        # Disk support at x=1, total reach = 1.1.  Sphere reach = 0.5.
        # Place sphere at x=1.4 -> gap = 1.4 - 1.1 - 0.5 = -0.2 -> overlap
        model.shape_transforms.assign(_transforms((0, 0, 0), (1.4, 0, 0)))
        _collider.collide(model)
        self.assertGreater(_contact_count(model), 0)

    def test_disk_vs_disk(self):
        """Two disks colliding."""
        b = xc.Builder()
        b.add_shape(SHAPE_DISK, params=(1, 0, 0), margin=0.2)
        b.add_shape(SHAPE_DISK, params=(1, 0, 0), margin=0.2)
        model = b.finalize()
        # Both disks reach 1.2 along x. Place at x=+-1.0 -> gap = 2.0 - 2.4 = -0.4
        model.shape_transforms.assign(_transforms((-1, 0, 0), (1, 0, 0)))
        _collider.collide(model)
        self.assertGreater(_contact_count(model), 0)

    def test_disk_vs_box(self):
        """Disk colliding with built-in box."""
        b = xc.Builder()
        b.add_shape(SHAPE_DISK, params=(0.5, 0, 0), margin=0.1)
        b.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        model = b.finalize()
        # Disk at origin, box at x=1.2: box face at x=0.2, disk reaches 0.6
        model.shape_transforms.assign(_transforms((0, 0, 0), (1.2, 0, 0)))
        _collider.collide(model)
        self.assertGreater(_contact_count(model), 0)

    def test_disk_contact_normal(self):
        """Contact normal is reasonable for disk-sphere collision along x."""
        b = xc.Builder()
        b.add_shape(SHAPE_DISK, params=(1, 0, 0), margin=0.1)
        b.add_shape(xc.SHAPE_POINT, params=(0, 0, 0), margin=0.5)
        model = b.finalize()
        model.shape_transforms.assign(_transforms((0, 0, 0), (1.4, 0, 0)))
        _collider.collide(model)
        n = _contact_count(model)
        self.assertGreater(n, 0)
        normal = model.contact_normal.numpy()[0]
        # Normal should be predominantly along +x
        self.assertGreater(normal[0], 0.5)


if __name__ == "__main__":
    unittest.main()
