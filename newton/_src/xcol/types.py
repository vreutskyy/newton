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

"""Warp structs, constants, and fixed-size matrix types for xcol."""

import warp as wp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FACE_POINTS = 4
GJK_MAX_ITERATIONS = 64
GJK_EPSILON = 1.0e-6

EPA_MAX_ITERATIONS = 32
EPA_MAX_FACES = 64
EPA_MAX_VERTS = 32
EPA_EPSILON = 1.0e-4

# Maximum intermediate polygon size during Sutherland-Hodgman clipping
CLIP_MAX_POINTS = 8

# ---------------------------------------------------------------------------
# Fixed-size matrix types
# ---------------------------------------------------------------------------

# EPA polytope storage
EPAVerts = wp.types.matrix(shape=(EPA_MAX_VERTS, 3), dtype=wp.float32)
EPAVertsA = wp.types.matrix(shape=(EPA_MAX_VERTS, 3), dtype=wp.float32)
EPAVertsB = wp.types.matrix(shape=(EPA_MAX_VERTS, 3), dtype=wp.float32)
EPAFaces = wp.types.matrix(shape=(EPA_MAX_FACES, 3), dtype=wp.float32)

# Clipped polygon storage
ClipPoly = wp.types.matrix(shape=(CLIP_MAX_POINTS, 3), dtype=wp.float32)


# ---------------------------------------------------------------------------
# Warp structs
# ---------------------------------------------------------------------------


@wp.struct
class ShapeData:
    """Uniform shape descriptor for all primitive types.

    Each shape is a *core* geometry inflated by a uniform *margin*.
    The actual shape is the Minkowski sum: ``core ⊕ sphere(margin)``.
    GJK/EPA operate on the core; the margin is subtracted from the
    computed distance afterward.

    Fields:
        shape_type: Integer id assigned by :func:`register_shape`.
        pos: World-space position of the shape center [m].
        rot: World-space orientation quaternion.
        params: Core shape parameters (meaning depends on shape type):
            - POINT: ``(0, 0, 0)``  — degenerate core (sphere = point + margin)
            - SEGMENT: ``(half_height, 0, 0)``  — axis along local +Z (capsule = segment + margin)
            - BOX: ``(half_x, half_y, half_z)``  — core box (rounded box = box + margin)
        margin: Uniform inflation distance [m].
    """

    shape_type: int
    pos: wp.vec3
    rot: wp.quat
    params: wp.vec3
    margin: float


@wp.struct
class ContactFaceResult:
    """Result of a contact face query (1-4 polygon vertices + normal)."""

    p0: wp.vec3
    p1: wp.vec3
    p2: wp.vec3
    p3: wp.vec3
    normal: wp.vec3
    count: int


@wp.struct
class ContactResult:
    """Result of contact generation between two shapes.

    Attributes:
        normal: Contact normal (A->B direction, unit length).
        p0..p3: Up to 4 contact points (world space, on shape A surface).
        d0..d3: Penetration depth at each contact point [m].
        count: Number of valid contacts (0..4).
    """

    normal: wp.vec3
    p0: wp.vec3
    p1: wp.vec3
    p2: wp.vec3
    p3: wp.vec3
    d0: float
    d1: float
    d2: float
    d3: float
    count: int


@wp.struct
class GJKResult:
    """Result of a GJK distance query."""

    distance: float
    point_a: wp.vec3
    point_b: wp.vec3
    normal: wp.vec3
    overlap: int  # 0 = separated, 1 = margin overlap, 2 = core overlap
    # Terminal simplex (valid when overlap==2, used by EPA)
    sw0: wp.vec3
    sw1: wp.vec3
    sw2: wp.vec3
    sw3: wp.vec3
    spa0: wp.vec3
    spa1: wp.vec3
    spa2: wp.vec3
    spa3: wp.vec3
    spb0: wp.vec3
    spb1: wp.vec3
    spb2: wp.vec3
    spb3: wp.vec3
