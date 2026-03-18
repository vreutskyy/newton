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

"""Experimental collision pipeline — shapes, support functions, and GJK.

Shape types are registered via :func:`register_shape` before calling
:func:`create_pipeline`, which generates all dispatch functions and kernels
from the registry.  Built-in primitives (sphere, box, capsule) are registered
at module load time through the same path as user-defined shapes.

All Warp structs, ``@wp.func`` functions, and ``@wp.kernel`` wrappers live in
one module so that CUDA/CPU code generation resolves all symbols correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

# Fixed-size matrix types for EPA polytope storage
# Vertices: each row is a vec3 (Minkowski diff point)
EPAVerts = wp.types.matrix(shape=(EPA_MAX_VERTS, 3), dtype=wp.float32)
# Same for witness points on A and B
EPAVertsA = wp.types.matrix(shape=(EPA_MAX_VERTS, 3), dtype=wp.float32)
EPAVertsB = wp.types.matrix(shape=(EPA_MAX_VERTS, 3), dtype=wp.float32)
# Faces: each row stores 3 vertex indices as floats (Warp matrices are float-only)
EPAFaces = wp.types.matrix(shape=(EPA_MAX_FACES, 3), dtype=wp.float32)


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
        normal: Contact normal (A→B direction, unit length).
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


# Maximum intermediate polygon size during clipping (face_a clipped against face_b edges)
CLIP_MAX_POINTS = 8

# Fixed-size matrix for clipped polygon storage: rows = points, cols = xyz
ClipPoly = wp.types.matrix(shape=(CLIP_MAX_POINTS, 3), dtype=wp.float32)


@wp.struct
class GJKResult:
    """Result of a GJK distance query."""

    distance: float
    point_a: wp.vec3
    point_b: wp.vec3
    normal: wp.vec3
    overlap: int  # 1 = overlap, 0 = separated
    # Terminal simplex (valid when overlap=1, used by EPA)
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


# ===================================================================
# Shape registry
# ===================================================================


@dataclass
class ShapeEntry:
    """Registration entry for a shape type.

    Attributes:
        type_id: Unique integer id for this shape type.
        name: Human-readable name.
        support_fn: ``@wp.func(params: wp.vec3, direction: wp.vec3) -> wp.vec3``
        contact_face_fn: ``@wp.func(params: wp.vec3, direction: wp.vec3) -> ContactFaceResult``
        aabb_fn: ``@wp.func(shape: ShapeData) -> tuple[wp.vec3, wp.vec3]``
    """

    type_id: int
    name: str
    support_fn: Any
    contact_face_fn: Any
    aabb_fn: Any


_shape_registry: dict[int, ShapeEntry] = {}
_next_type_id: list[int] = [0]  # mutable container to avoid global statement


def register_shape(
    name: str,
    *,
    support_fn: Any,
    contact_face_fn: Any,
    aabb_fn: Any,
) -> int:
    """Register a shape type with the collision pipeline.

    All shape types — built-in and custom — must be registered before
    :func:`create_pipeline` is called.

    Args:
        name: Human-readable name (e.g. ``"sphere"``).
        support_fn: Warp function ``(params, direction) -> wp.vec3`` in local space.
        contact_face_fn: Warp function ``(params, direction) -> ContactFaceResult`` in local space.
        aabb_fn: Warp function ``(shape: ShapeData) -> (wp.vec3, wp.vec3)``.

    Returns:
        Integer type id to use in :attr:`ShapeData.shape_type`.
    """
    type_id = _next_type_id[0]
    _next_type_id[0] += 1
    _shape_registry[type_id] = ShapeEntry(
        type_id=type_id,
        name=name,
        support_fn=support_fn,
        contact_face_fn=contact_face_fn,
        aabb_fn=aabb_fn,
    )
    return type_id


# ===================================================================
# Built-in shape functions (local space)
# ===================================================================


@wp.func
def _support_point(params: wp.vec3, direction: wp.vec3) -> wp.vec3:
    """Core support for a point shape — always returns the origin."""
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _support_segment(params: wp.vec3, direction: wp.vec3) -> wp.vec3:
    """Core support for a line segment along local +Z.

    Args:
        params: ``(half_height, 0, 0)``.
    """
    half_height = params[0]
    if direction[2] >= 0.0:
        return wp.vec3(0.0, 0.0, half_height)
    return wp.vec3(0.0, 0.0, -half_height)


@wp.func
def _support_box(params: wp.vec3, direction: wp.vec3) -> wp.vec3:
    """Core support for an axis-aligned box.

    Args:
        params: Half-extents ``(hx, hy, hz)``.
    """
    sx = 1.0
    if direction[0] < 0.0:
        sx = -1.0
    sy = 1.0
    if direction[1] < 0.0:
        sy = -1.0
    sz = 1.0
    if direction[2] < 0.0:
        sz = -1.0
    return wp.vec3(sx * params[0], sy * params[1], sz * params[2])


@wp.func
def _contact_face_point(params: wp.vec3, direction: wp.vec3) -> ContactFaceResult:
    """Contact face for a point core — always 1 point at origin."""
    d_len = wp.length(direction)
    if d_len > 1.0e-12:
        n = wp.normalize(direction)
    else:
        n = wp.vec3(0.0, 0.0, 1.0)
    result = ContactFaceResult()
    result.p0 = wp.vec3(0.0, 0.0, 0.0)
    result.p1 = wp.vec3(0.0, 0.0, 0.0)
    result.p2 = wp.vec3(0.0, 0.0, 0.0)
    result.p3 = wp.vec3(0.0, 0.0, 0.0)
    result.normal = n
    result.count = 1
    return result


@wp.func
def _contact_face_box(params: wp.vec3, direction: wp.vec3) -> ContactFaceResult:
    hx = params[0]
    hy = params[1]
    hz = params[2]
    ax = wp.abs(direction[0])
    ay = wp.abs(direction[1])
    az = wp.abs(direction[2])
    result = ContactFaceResult()
    result.count = 4
    if ax >= ay and ax >= az:
        s = 1.0
        if direction[0] < 0.0:
            s = -1.0
        result.normal = wp.vec3(s, 0.0, 0.0)
        result.p0 = wp.vec3(s * hx, -hy, -hz)
        result.p1 = wp.vec3(s * hx, hy, -hz)
        result.p2 = wp.vec3(s * hx, hy, hz)
        result.p3 = wp.vec3(s * hx, -hy, hz)
    elif ay >= az:
        s = 1.0
        if direction[1] < 0.0:
            s = -1.0
        result.normal = wp.vec3(0.0, s, 0.0)
        result.p0 = wp.vec3(-hx, s * hy, -hz)
        result.p1 = wp.vec3(hx, s * hy, -hz)
        result.p2 = wp.vec3(hx, s * hy, hz)
        result.p3 = wp.vec3(-hx, s * hy, hz)
    else:
        s = 1.0
        if direction[2] < 0.0:
            s = -1.0
        result.normal = wp.vec3(0.0, 0.0, s)
        result.p0 = wp.vec3(-hx, -hy, s * hz)
        result.p1 = wp.vec3(hx, -hy, s * hz)
        result.p2 = wp.vec3(hx, hy, s * hz)
        result.p3 = wp.vec3(-hx, hy, s * hz)
    return result


@wp.func
def _contact_face_segment(params: wp.vec3, direction: wp.vec3) -> ContactFaceResult:
    """Contact face for a segment core along +Z.

    Lateral: 2 endpoints.  End-on: 1 endpoint.
    Margin is NOT included here — it is applied later by the pipeline.
    """
    half_height = params[0]
    d_len = wp.length(direction)
    if d_len > 1.0e-12:
        n = wp.normalize(direction)
    else:
        n = wp.vec3(0.0, 0.0, 1.0)
    az = wp.abs(n[2])
    result = ContactFaceResult()
    if az > 0.9:
        # End-on: single endpoint
        s = 1.0
        if n[2] < 0.0:
            s = -1.0
        pt = wp.vec3(0.0, 0.0, s * half_height)
        result.p0 = pt
        result.p1 = pt
        result.p2 = pt
        result.p3 = pt
        result.normal = wp.vec3(0.0, 0.0, s)
        result.count = 1
    else:
        # Lateral: both endpoints
        result.p0 = wp.vec3(0.0, 0.0, half_height)
        result.p1 = wp.vec3(0.0, 0.0, -half_height)
        result.p2 = result.p1
        result.p3 = result.p1
        lateral = wp.vec3(n[0], n[1], 0.0)
        lat_len = wp.length(lateral)
        if lat_len > 1.0e-12:
            result.normal = lateral / lat_len
        else:
            result.normal = wp.vec3(1.0, 0.0, 0.0)
        result.count = 2
    return result


@wp.func
def _aabb_point(shape: ShapeData) -> tuple[wp.vec3, wp.vec3]:
    """AABB for a point core + margin = sphere."""
    r = shape.margin
    rv = wp.vec3(r, r, r)
    return shape.pos - rv, shape.pos + rv


@wp.func
def _aabb_segment(shape: ShapeData) -> tuple[wp.vec3, wp.vec3]:
    """AABB for a segment core + margin = capsule."""
    r = shape.margin
    hh = shape.params[0]
    top = wp.quat_rotate(shape.rot, wp.vec3(0.0, 0.0, hh))
    bot = wp.quat_rotate(shape.rot, wp.vec3(0.0, 0.0, -hh))
    mn = wp.vec3(
        wp.min(top[0], bot[0]) - r,
        wp.min(top[1], bot[1]) - r,
        wp.min(top[2], bot[2]) - r,
    )
    mx = wp.vec3(
        wp.max(top[0], bot[0]) + r,
        wp.max(top[1], bot[1]) + r,
        wp.max(top[2], bot[2]) + r,
    )
    return shape.pos + mn, shape.pos + mx


@wp.func
def _aabb_box(shape: ShapeData) -> tuple[wp.vec3, wp.vec3]:
    """AABB for a box core + margin = rounded box."""
    hx = shape.params[0]
    hy = shape.params[1]
    hz = shape.params[2]
    r = shape.margin
    ax = wp.quat_rotate(shape.rot, wp.vec3(hx, 0.0, 0.0))
    ay = wp.quat_rotate(shape.rot, wp.vec3(0.0, hy, 0.0))
    aaz = wp.quat_rotate(shape.rot, wp.vec3(0.0, 0.0, hz))
    extent = wp.vec3(
        wp.abs(ax[0]) + wp.abs(ay[0]) + wp.abs(aaz[0]) + r,
        wp.abs(ax[1]) + wp.abs(ay[1]) + wp.abs(aaz[1]) + r,
        wp.abs(ax[2]) + wp.abs(ay[2]) + wp.abs(aaz[2]) + r,
    )
    return shape.pos - extent, shape.pos + extent


# ===================================================================
# Register built-in shapes (same path as custom shapes)
# ===================================================================

SHAPE_POINT = register_shape(
    "point",
    support_fn=_support_point,
    contact_face_fn=_contact_face_point,
    aabb_fn=_aabb_point,
)
SHAPE_SEGMENT = register_shape(
    "segment",
    support_fn=_support_segment,
    contact_face_fn=_contact_face_segment,
    aabb_fn=_aabb_segment,
)
SHAPE_BOX = register_shape(
    "box",
    support_fn=_support_box,
    contact_face_fn=_contact_face_box,
    aabb_fn=_aabb_box,
)


# ===================================================================
# GJK — closest-point subalgorithm (shape-independent)
# ===================================================================


@wp.func
def closest_segment(a: wp.vec3, b: wp.vec3) -> tuple[wp.vec3, float, float]:
    ab = b - a
    denom = wp.dot(ab, ab)
    if denom < GJK_EPSILON:
        return a, 1.0, 0.0
    t = wp.clamp(-wp.dot(a, ab) / denom, 0.0, 1.0)
    return a + t * ab, 1.0 - t, t


@wp.func
def closest_triangle(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> tuple[wp.vec3, float, float, float]:
    ab = b - a
    ac = c - a
    ao = -a

    d1 = wp.dot(ab, ao)
    d2 = wp.dot(ac, ao)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, 1.0, 0.0, 0.0

    bo = -b
    d3 = wp.dot(ab, bo)
    d4 = wp.dot(ac, bo)
    if d3 >= 0.0 and d4 <= d3:
        return b, 0.0, 1.0, 0.0

    co = -c
    d5 = wp.dot(ab, co)
    d6 = wp.dot(ac, co)
    if d6 >= 0.0 and d5 <= d6:
        return c, 0.0, 0.0, 1.0

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        denom = d1 - d3
        if wp.abs(denom) < GJK_EPSILON:
            denom = GJK_EPSILON
        bv = d1 / denom
        return a + bv * ab, 1.0 - bv, bv, 0.0

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        denom = d2 - d6
        if wp.abs(denom) < GJK_EPSILON:
            denom = GJK_EPSILON
        bw = d2 / denom
        return a + bw * ac, 1.0 - bw, 0.0, bw

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        if wp.abs(denom) < GJK_EPSILON:
            denom = GJK_EPSILON
        bw = (d4 - d3) / denom
        return b + bw * (c - b), 0.0, 1.0 - bw, bw

    denom = va + vb + vc
    if wp.abs(denom) < GJK_EPSILON:
        denom = GJK_EPSILON
    inv = 1.0 / denom
    bv = vb * inv
    bw = vc * inv
    return a + bv * ab + bw * ac, 1.0 - bv - bw, bv, bw


# ===================================================================
# Pipeline factory
# ===================================================================


def create_pipeline(shape_entries: list[ShapeEntry] | None = None):
    """Create dispatch functions and kernels from registered shape types.

    Call this after all :func:`register_shape` calls are done (including
    custom shapes).  If *shape_entries* is ``None``, uses the global
    registry.

    Returns:
        A :class:`Pipeline` object with the generated dispatch functions
        and kernels.
    """
    if shape_entries is None:
        shape_entries = list(_shape_registry.values())

    # Build simple lists for wp.static closure capture
    type_ids = [e.type_id for e in shape_entries]
    support_fns = [e.support_fn for e in shape_entries]
    contact_face_fns = [e.contact_face_fn for e in shape_entries]
    aabb_fns = [e.aabb_fn for e in shape_entries]
    n_types = len(type_ids)

    # -- Dispatch: support (local space) --------------------------------

    @wp.func
    def support_local(shape_type: int, params: wp.vec3, direction: wp.vec3) -> wp.vec3:
        for i in range(wp.static(n_types)):
            if shape_type == wp.static(type_ids[i]):
                return wp.static(support_fns[i])(params, direction)
        return wp.vec3(0.0, 0.0, 0.0)

    # -- Dispatch: support (world space) --------------------------------

    @wp.func
    def support_world(shape: ShapeData, direction: wp.vec3) -> wp.vec3:
        local_dir = wp.quat_rotate_inv(shape.rot, direction)
        local_pt = support_local(shape.shape_type, shape.params, local_dir)
        return wp.quat_rotate(shape.rot, local_pt) + shape.pos

    # -- Dispatch: contact face (local space) ---------------------------

    @wp.func
    def contact_face_local(shape_type: int, params: wp.vec3, direction: wp.vec3) -> ContactFaceResult:
        for i in range(wp.static(n_types)):
            if shape_type == wp.static(type_ids[i]):
                return wp.static(contact_face_fns[i])(params, direction)
        result = ContactFaceResult()
        pt = support_local(shape_type, params, direction)
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

    # -- Dispatch: contact face (world space) ---------------------------

    @wp.func
    def contact_face_world(shape: ShapeData, direction: wp.vec3, point: wp.vec3) -> ContactFaceResult:
        local_dir = wp.quat_rotate_inv(shape.rot, direction)
        result = contact_face_local(shape.shape_type, shape.params, local_dir)
        # Inflate core points by margin along the face normal
        margin_offset = result.normal * shape.margin
        result.p0 = wp.quat_rotate(shape.rot, result.p0 + margin_offset) + shape.pos
        result.p1 = wp.quat_rotate(shape.rot, result.p1 + margin_offset) + shape.pos
        result.p2 = wp.quat_rotate(shape.rot, result.p2 + margin_offset) + shape.pos
        result.p3 = wp.quat_rotate(shape.rot, result.p3 + margin_offset) + shape.pos
        result.normal = wp.quat_rotate(shape.rot, result.normal)
        return result

    # -- Dispatch: AABB -------------------------------------------------

    @wp.func
    def get_aabb(shape: ShapeData) -> tuple[wp.vec3, wp.vec3]:
        for i in range(wp.static(n_types)):
            if shape.shape_type == wp.static(type_ids[i]):
                return wp.static(aabb_fns[i])(shape)
        return shape.pos, shape.pos

    # -- GJK distance ---------------------------------------------------

    @wp.func
    def gjk_distance(shape_a: ShapeData, shape_b: ShapeData) -> GJKResult:
        result = GJKResult()
        result.distance = 0.0
        result.point_a = wp.vec3(0.0, 0.0, 0.0)
        result.point_b = wp.vec3(0.0, 0.0, 0.0)
        result.normal = wp.vec3(0.0, 0.0, 0.0)
        result.overlap = 0

        direction = shape_b.pos - shape_a.pos
        if wp.length_sq(direction) < GJK_EPSILON:
            direction = wp.vec3(1.0, 0.0, 0.0)

        w0 = wp.vec3(0.0, 0.0, 0.0)
        w1 = wp.vec3(0.0, 0.0, 0.0)
        w2 = wp.vec3(0.0, 0.0, 0.0)
        pa0 = wp.vec3(0.0, 0.0, 0.0)
        pa1 = wp.vec3(0.0, 0.0, 0.0)
        pa2 = wp.vec3(0.0, 0.0, 0.0)
        pb0 = wp.vec3(0.0, 0.0, 0.0)
        pb1 = wp.vec3(0.0, 0.0, 0.0)
        pb2 = wp.vec3(0.0, 0.0, 0.0)

        sa = support_world(shape_a, direction)
        sb = support_world(shape_b, -direction)
        w0 = sa - sb
        pa0 = sa
        pb0 = sb
        num_verts = int(1)

        v = w0
        dist_sq = wp.length_sq(v)

        for _iter in range(GJK_MAX_ITERATIONS):
            if dist_sq < GJK_EPSILON * GJK_EPSILON:
                result.overlap = 2  # core overlap — needs EPA
                result.distance = 0.0
                result.point_a = pa0
                result.point_b = pb0
                return result

            direction = -v

            sa = support_world(shape_a, direction)
            sb = support_world(shape_b, -direction)
            w_new = sa - sb

            progress = wp.dot(v, v - w_new)
            if progress < GJK_EPSILON * wp.sqrt(dist_sq):
                break

            is_dup = int(0)
            if wp.length_sq(w_new - w0) < GJK_EPSILON * GJK_EPSILON:
                is_dup = int(1)
            if num_verts >= 2:
                if wp.length_sq(w_new - w1) < GJK_EPSILON * GJK_EPSILON:
                    is_dup = int(1)
            if num_verts >= 3:
                if wp.length_sq(w_new - w2) < GJK_EPSILON * GJK_EPSILON:
                    is_dup = int(1)
            if is_dup == 1:
                break

            if num_verts == 1:
                w1 = w_new
                pa1 = sa
                pb1 = sb
                pt, la, lb = closest_segment(w0, w1)
                v = pt
                dist_sq = wp.length_sq(v)
                if la < GJK_EPSILON:
                    w0 = w1
                    pa0 = pa1
                    pb0 = pb1
                    num_verts = int(1)
                elif lb < GJK_EPSILON:
                    num_verts = int(1)
                else:
                    num_verts = int(2)

            elif num_verts == 2:
                w2 = w_new
                pa2 = sa
                pb2 = sb
                pt, u, bv, bw = closest_triangle(w0, w1, w2)
                v = pt
                dist_sq = wp.length_sq(v)

                keep0 = u >= GJK_EPSILON
                keep1 = bv >= GJK_EPSILON
                keep2 = bw >= GJK_EPSILON
                alive = int(0)
                if keep0:
                    alive = alive + 1
                if keep1:
                    alive = alive + 1
                if keep2:
                    alive = alive + 1

                if alive <= 1:
                    if keep1:
                        w0 = w1
                        pa0 = pa1
                        pb0 = pb1
                    elif keep2:
                        w0 = w2
                        pa0 = pa2
                        pb0 = pb2
                    num_verts = int(1)
                elif alive == 2:
                    if not keep0:
                        w0 = w1
                        pa0 = pa1
                        pb0 = pb1
                        w1 = w2
                        pa1 = pa2
                        pb1 = pb2
                    elif not keep1:
                        w1 = w2
                        pa1 = pa2
                        pb1 = pb2
                    num_verts = int(2)
                else:
                    num_verts = int(3)

            elif num_verts == 3:
                w3 = w_new
                pa3 = sa
                pb3 = sb

                d0 = w1 - w0
                d1 = w2 - w0
                d2 = w3 - w0
                det = wp.dot(d0, wp.cross(d1, d2))

                inside = int(0)
                if wp.abs(det) > GJK_EPSILON:
                    inv = 1.0 / det
                    ao = -w0
                    lam1 = wp.dot(ao, wp.cross(d1, d2)) * inv
                    lam2 = wp.dot(d0, wp.cross(ao, d2)) * inv
                    lam3 = wp.dot(d0, wp.cross(d1, ao)) * inv
                    lam0 = 1.0 - lam1 - lam2 - lam3
                    if lam0 >= 0.0 and lam1 >= 0.0 and lam2 >= 0.0 and lam3 >= 0.0:
                        inside = int(1)

                if inside == 1:
                    result.overlap = 2  # core overlap — needs EPA
                    result.distance = 0.0
                    result.point_a = lam0 * pa0 + lam1 * pa1 + lam2 * pa2 + lam3 * pa3
                    result.point_b = lam0 * pb0 + lam1 * pb1 + lam2 * pb2 + lam3 * pb3
                    # Store tetrahedron for EPA
                    result.sw0 = w0
                    result.sw1 = w1
                    result.sw2 = w2
                    result.sw3 = w3
                    result.spa0 = pa0
                    result.spa1 = pa1
                    result.spa2 = pa2
                    result.spa3 = pa3
                    result.spb0 = pb0
                    result.spb1 = pb1
                    result.spb2 = pb2
                    result.spb3 = pb3
                    return result

                best_dist = float(1.0e30)
                best_v = wp.vec3(0.0, 0.0, 0.0)
                best_face = int(0)

                pt, u, bv, bw = closest_triangle(w1, w2, w3)
                d = wp.length_sq(pt)
                if d < best_dist:
                    best_dist = d
                    best_v = pt
                    best_face = int(0)

                pt, u, bv, bw = closest_triangle(w0, w2, w3)
                d = wp.length_sq(pt)
                if d < best_dist:
                    best_dist = d
                    best_v = pt
                    best_face = int(1)

                pt, u, bv, bw = closest_triangle(w0, w1, w3)
                d = wp.length_sq(pt)
                if d < best_dist:
                    best_dist = d
                    best_v = pt
                    best_face = int(2)

                pt, u, bv, bw = closest_triangle(w0, w1, w2)
                d = wp.length_sq(pt)
                if d < best_dist:
                    best_dist = d
                    best_v = pt
                    best_face = int(3)

                v = best_v
                dist_sq = best_dist

                if best_face == 0:
                    w0 = w1
                    pa0 = pa1
                    pb0 = pb1
                    w1 = w2
                    pa1 = pa2
                    pb1 = pb2
                    w2 = w3
                    pa2 = pa3
                    pb2 = pb3
                elif best_face == 1:
                    w1 = w2
                    pa1 = pa2
                    pb1 = pb2
                    w2 = w3
                    pa2 = pa3
                    pb2 = pb3
                elif best_face == 2:
                    w2 = w3
                    pa2 = pa3
                    pb2 = pb3
                num_verts = int(3)

        # Final witness points
        if num_verts == 1:
            result.point_a = pa0
            result.point_b = pb0
        elif num_verts == 2:
            pt, la, lb = closest_segment(w0, w1)
            result.point_a = la * pa0 + lb * pa1
            result.point_b = la * pb0 + lb * pb1
            v = pt
        else:
            pt, u, bv, bw = closest_triangle(w0, w1, w2)
            result.point_a = u * pa0 + bv * pa1 + bw * pa2
            result.point_b = u * pb0 + bv * pb1 + bw * pb2
            v = pt

        core_dist = wp.length(v)
        total_margin = shape_a.margin + shape_b.margin
        real_dist = core_dist - total_margin

        if core_dist > GJK_EPSILON:
            result.normal = -v / core_dist
            # Move witness points from core surfaces to inflated surfaces
            result.point_a = result.point_a + result.normal * shape_a.margin
            result.point_b = result.point_b - result.normal * shape_b.margin

        if real_dist <= 0.0:
            result.overlap = 1  # margin-only overlap (GJK has valid answer)
            result.distance = -real_dist  # penetration depth (positive)
        else:
            result.overlap = 0
            result.distance = real_dist
        return result

    # -- EPA (Expanding Polytope Algorithm) ------------------------------

    @wp.func
    def epa(
        shape_a: ShapeData,
        shape_b: ShapeData,
        s0: wp.vec3,
        s1: wp.vec3,
        s2: wp.vec3,
        s3: wp.vec3,
        pa0_in: wp.vec3,
        pa1_in: wp.vec3,
        pa2_in: wp.vec3,
        pa3_in: wp.vec3,
        pb0_in: wp.vec3,
        pb1_in: wp.vec3,
        pb2_in: wp.vec3,
        pb3_in: wp.vec3,
    ) -> GJKResult:
        """Expand polytope from GJK tetrahedron to find penetration depth."""
        result = GJKResult()
        result.distance = 0.0
        result.point_a = wp.vec3(0.0, 0.0, 0.0)
        result.point_b = wp.vec3(0.0, 0.0, 0.0)
        result.normal = wp.vec3(0.0, 0.0, 0.0)
        result.overlap = 1

        # Initialize polytope vertices
        verts = EPAVerts()
        verts_a = EPAVertsA()
        verts_b = EPAVertsB()
        verts[0] = s0
        verts[1] = s1
        verts[2] = s2
        verts[3] = s3
        verts_a[0] = pa0_in
        verts_a[1] = pa1_in
        verts_a[2] = pa2_in
        verts_a[3] = pa3_in
        verts_b[0] = pb0_in
        verts_b[1] = pb1_in
        verts_b[2] = pb2_in
        verts_b[3] = pb3_in
        num_verts = int(4)

        # Initialize faces (4 faces of tetrahedron)
        # Ensure outward-facing normals by checking winding
        faces = EPAFaces()

        # Check if face 0-1-2 normal points away from vertex 3
        n012 = wp.cross(s1 - s0, s2 - s0)
        if wp.dot(n012, s3 - s0) > 0.0:
            # Vertex 3 is on the positive side, so 0-1-2 points toward 3 — flip
            faces[0] = wp.vec3(0.0, 2.0, 1.0)
            faces[1] = wp.vec3(0.0, 1.0, 3.0)
            faces[2] = wp.vec3(1.0, 2.0, 3.0)
            faces[3] = wp.vec3(0.0, 3.0, 2.0)
        else:
            faces[0] = wp.vec3(0.0, 1.0, 2.0)
            faces[1] = wp.vec3(0.0, 3.0, 1.0)
            faces[2] = wp.vec3(1.0, 3.0, 2.0)
            faces[3] = wp.vec3(0.0, 2.0, 3.0)
        num_faces = int(4)

        for _epa_iter in range(EPA_MAX_ITERATIONS):
            # Find closest face to origin
            best_face = int(0)
            best_dist = float(1.0e30)
            best_normal = wp.vec3(0.0, 0.0, 0.0)

            for fi in range(EPA_MAX_FACES):
                if fi >= num_faces:
                    break
                i0 = int(faces[fi][0])
                i1 = int(faces[fi][1])
                i2 = int(faces[fi][2])
                va = wp.vec3(verts[i0][0], verts[i0][1], verts[i0][2])
                vb = wp.vec3(verts[i1][0], verts[i1][1], verts[i1][2])
                vc = wp.vec3(verts[i2][0], verts[i2][1], verts[i2][2])
                n = wp.cross(vb - va, vc - va)
                n_len = wp.length(n)
                if n_len < EPA_EPSILON:
                    continue
                n = n / n_len
                d = wp.dot(n, va)
                if d < 0.0:
                    n = -n
                    d = -d
                if d < best_dist:
                    best_dist = d
                    best_normal = n
                    best_face = fi

            # Get new support point along best normal
            sa = support_world(shape_a, best_normal)
            sb = support_world(shape_b, -best_normal)
            w_new = sa - sb

            # Check convergence
            new_dist = wp.dot(w_new, best_normal)
            if new_dist - best_dist < EPA_EPSILON:
                # Converged — compute witness points
                result.distance = best_dist
                result.normal = best_normal

                # Barycentric coords of origin projection on best face
                bi0 = int(faces[best_face][0])
                bi1 = int(faces[best_face][1])
                bi2 = int(faces[best_face][2])
                fa = wp.vec3(verts[bi0][0], verts[bi0][1], verts[bi0][2])
                fb = wp.vec3(verts[bi1][0], verts[bi1][1], verts[bi1][2])
                fc = wp.vec3(verts[bi2][0], verts[bi2][1], verts[bi2][2])

                # Project origin onto face plane
                proj = best_normal * best_dist
                v0 = fb - fa
                v1 = fc - fa
                v2 = proj - fa
                d00 = wp.dot(v0, v0)
                d01 = wp.dot(v0, v1)
                d11 = wp.dot(v1, v1)
                d20 = wp.dot(v2, v0)
                d21 = wp.dot(v2, v1)
                denom = d00 * d11 - d01 * d01
                if wp.abs(denom) < EPA_EPSILON:
                    denom = EPA_EPSILON
                bv = (d11 * d20 - d01 * d21) / denom
                bw = (d00 * d21 - d01 * d20) / denom
                bu = 1.0 - bv - bw

                pa_a = wp.vec3(verts_a[bi0][0], verts_a[bi0][1], verts_a[bi0][2])
                pa_b = wp.vec3(verts_a[bi1][0], verts_a[bi1][1], verts_a[bi1][2])
                pa_c = wp.vec3(verts_a[bi2][0], verts_a[bi2][1], verts_a[bi2][2])
                pb_a = wp.vec3(verts_b[bi0][0], verts_b[bi0][1], verts_b[bi0][2])
                pb_b = wp.vec3(verts_b[bi1][0], verts_b[bi1][1], verts_b[bi1][2])
                pb_c = wp.vec3(verts_b[bi2][0], verts_b[bi2][1], verts_b[bi2][2])

                result.point_a = bu * pa_a + bv * pa_b + bw * pa_c
                result.point_b = bu * pb_a + bv * pb_b + bw * pb_c
                return result

            if num_verts >= EPA_MAX_VERTS:
                break

            # Add new vertex
            new_vi = num_verts
            verts[new_vi] = w_new
            verts_a[new_vi] = sa
            verts_b[new_vi] = sb
            num_verts = num_verts + 1

            # Remove faces visible from new point and collect horizon edges
            # Use a simple edge buffer (max 64 edges)
            # Edge stored as vec3(v0, v1, 0) — we reuse float storage
            edges = EPAFaces()  # reuse same matrix type for edge storage
            num_edges = int(0)
            new_num_faces = int(0)

            for fi in range(EPA_MAX_FACES):
                if fi >= num_faces:
                    break
                i0 = int(faces[fi][0])
                i1 = int(faces[fi][1])
                i2 = int(faces[fi][2])
                va = wp.vec3(verts[i0][0], verts[i0][1], verts[i0][2])
                vb = wp.vec3(verts[i1][0], verts[i1][1], verts[i1][2])
                vc = wp.vec3(verts[i2][0], verts[i2][1], verts[i2][2])
                fn = wp.cross(vb - va, vc - va)
                fn_len = wp.length(fn)
                if fn_len > EPA_EPSILON:
                    fn = fn / fn_len
                    if wp.dot(fn, va) < 0.0:
                        fn = -fn

                # Check if face is visible from new point
                visible = int(0)
                if fn_len > EPA_EPSILON:
                    if wp.dot(fn, w_new - va) > EPA_EPSILON:
                        visible = int(1)

                if visible == 1:
                    # Add edges to horizon (remove shared edges)
                    e0 = wp.vec3(float(i0), float(i1), 0.0)
                    e1 = wp.vec3(float(i1), float(i2), 0.0)
                    e2 = wp.vec3(float(i2), float(i0), 0.0)

                    # For each edge, check if its reverse already exists
                    for ei_new in range(3):
                        if ei_new == 0:
                            edge = e0
                        elif ei_new == 1:
                            edge = e1
                        else:
                            edge = e2
                        rev = wp.vec3(edge[1], edge[0], 0.0)
                        found = int(-1)
                        for k in range(EPA_MAX_FACES):
                            if k >= num_edges:
                                break
                            if edges[k][0] == rev[0] and edges[k][1] == rev[1]:
                                found = k
                        if found >= 0:
                            # Remove by swapping with last
                            num_edges = num_edges - 1
                            edges[found] = edges[num_edges]
                        else:
                            if num_edges < EPA_MAX_FACES:
                                edges[num_edges] = edge
                                num_edges = num_edges + 1
                else:
                    # Keep face — compact into new_num_faces
                    if new_num_faces != fi:
                        faces[new_num_faces] = faces[fi]
                    new_num_faces = new_num_faces + 1

            num_faces = new_num_faces

            # Create new faces from horizon edges to new vertex
            for ei in range(EPA_MAX_FACES):
                if ei >= num_edges:
                    break
                if num_faces < EPA_MAX_FACES:
                    faces[num_faces] = wp.vec3(edges[ei][0], edges[ei][1], float(new_vi))
                    num_faces = num_faces + 1

        # Max iterations — return best so far
        result.distance = best_dist
        result.normal = best_normal
        return result

    # -- Combined GJK + EPA ---------------------------------------------

    @wp.func
    def build_initial_tetrahedron(shape_a: ShapeData, shape_b: ShapeData) -> GJKResult:
        """Build an initial tetrahedron enclosing the origin for EPA.

        For overlapping convex shapes, the origin lies inside their
        Minkowski difference.  This function finds 4 support points
        that form a tetrahedron containing the origin.
        """
        result = GJKResult()
        result.overlap = 1
        result.distance = 0.0

        # Point 0: support in +x (or center-to-center) direction
        d0 = shape_b.pos - shape_a.pos
        if wp.length(d0) < GJK_EPSILON:
            d0 = wp.vec3(1.0, 0.0, 0.0)
        d0 = wp.normalize(d0)
        sa = support_world(shape_a, d0)
        sb = support_world(shape_b, -d0)
        result.sw0 = sa - sb
        result.spa0 = sa
        result.spb0 = sb

        # Point 1: support in opposite direction
        d1 = -d0
        sa = support_world(shape_a, d1)
        sb = support_world(shape_b, -d1)
        result.sw1 = sa - sb
        result.spa1 = sa
        result.spb1 = sb

        # Point 2: perpendicular to line 0→1, toward origin
        line = result.sw1 - result.sw0
        # Project origin onto line to find closest point
        t = -wp.dot(result.sw0, line) / wp.max(wp.dot(line, line), GJK_EPSILON)
        closest = result.sw0 + t * line
        # Direction from closest point to origin
        d2 = -closest
        if wp.length(d2) < GJK_EPSILON:
            # Origin is on the line — pick any perpendicular
            d2 = wp.cross(line, wp.vec3(1.0, 0.0, 0.0))
            if wp.length(d2) < GJK_EPSILON:
                d2 = wp.cross(line, wp.vec3(0.0, 1.0, 0.0))
        d2 = wp.normalize(d2)
        sa = support_world(shape_a, d2)
        sb = support_world(shape_b, -d2)
        result.sw2 = sa - sb
        result.spa2 = sa
        result.spb2 = sb

        # Point 3: perpendicular to triangle, on the side of the origin
        tri_n = wp.cross(result.sw1 - result.sw0, result.sw2 - result.sw0)
        tri_n_len = wp.length(tri_n)
        if tri_n_len < GJK_EPSILON:
            tri_n = wp.vec3(0.0, 0.0, 1.0)
        else:
            tri_n = tri_n / tri_n_len
        # Check which side of the triangle the origin is on
        if wp.dot(tri_n, -result.sw0) < 0.0:
            tri_n = -tri_n
        sa = support_world(shape_a, tri_n)
        sb = support_world(shape_b, -tri_n)
        result.sw3 = sa - sb
        result.spa3 = sa
        result.spb3 = sb

        return result

    @wp.func
    def gjk_epa(shape_a: ShapeData, shape_b: ShapeData) -> GJKResult:
        """Run GJK. If overlap detected, run EPA for penetration depth."""
        gjk_result = gjk_distance(shape_a, shape_b)
        if gjk_result.overlap == 0:
            return gjk_result
        if gjk_result.overlap == 1:
            # Margin-only overlap — GJK already has correct answer
            gjk_result.overlap = 1
            return gjk_result

        # Core overlap (overlap == 2) — need EPA
        total_margin = shape_a.margin + shape_b.margin

        # Check if GJK gave us a valid tetrahedron (non-zero volume)
        d0 = gjk_result.sw1 - gjk_result.sw0
        d1 = gjk_result.sw2 - gjk_result.sw0
        d2 = gjk_result.sw3 - gjk_result.sw0
        vol = wp.abs(wp.dot(d0, wp.cross(d1, d2)))

        if vol < GJK_EPSILON:
            # GJK exited early without a full tetrahedron — build one
            gjk_result = build_initial_tetrahedron(shape_a, shape_b)

        epa_result = epa(
            shape_a,
            shape_b,
            gjk_result.sw0,
            gjk_result.sw1,
            gjk_result.sw2,
            gjk_result.sw3,
            gjk_result.spa0,
            gjk_result.spa1,
            gjk_result.spa2,
            gjk_result.spa3,
            gjk_result.spb0,
            gjk_result.spb1,
            gjk_result.spb2,
            gjk_result.spb3,
        )

        # EPA returns core penetration depth; add margins
        epa_result.distance = epa_result.distance + total_margin
        # Move witness points from core to inflated surfaces
        epa_result.point_a = epa_result.point_a + epa_result.normal * shape_a.margin
        epa_result.point_b = epa_result.point_b - epa_result.normal * shape_b.margin
        return epa_result

    # -- Polygon clipping (Sutherland-Hodgman) --------------------------

    @wp.func
    def clip_poly_against_plane(
        poly: ClipPoly,
        num_pts: int,
        plane_n: wp.vec3,
        plane_d: float,
    ) -> tuple[ClipPoly, int]:
        """Clip a polygon against a single plane (keep positive side).

        The plane is defined as ``dot(plane_n, x) <= plane_d``.
        """
        out = ClipPoly()
        out_count = int(0)

        for i in range(CLIP_MAX_POINTS):
            if i >= num_pts:
                break
            j = i + 1
            if j >= num_pts:
                j = int(0)

            pi = wp.vec3(poly[i][0], poly[i][1], poly[i][2])
            pj = wp.vec3(poly[j][0], poly[j][1], poly[j][2])
            di = wp.dot(plane_n, pi) - plane_d
            dj = wp.dot(plane_n, pj) - plane_d

            if di <= 0.0:
                # pi is inside
                if out_count < CLIP_MAX_POINTS:
                    out[out_count] = pi
                    out_count = out_count + 1
                if dj > 0.0:
                    # pj is outside — add intersection
                    denom = di - dj
                    if wp.abs(denom) < 1.0e-10:
                        denom = 1.0e-10
                    t = di / denom
                    intersection = pi + t * (pj - pi)
                    if out_count < CLIP_MAX_POINTS:
                        out[out_count] = intersection
                        out_count = out_count + 1
            elif dj <= 0.0:
                # pi is outside, pj is inside — add intersection
                denom = di - dj
                if wp.abs(denom) < 1.0e-10:
                    denom = 1.0e-10
                t = di / denom
                intersection = pi + t * (pj - pi)
                if out_count < CLIP_MAX_POINTS:
                    out[out_count] = intersection
                    out_count = out_count + 1

        return out, out_count

    @wp.func
    def clip_face_against_face(
        face_a: ContactFaceResult,
        face_b: ContactFaceResult,
        normal: wp.vec3,
    ) -> tuple[ClipPoly, int]:
        """Clip face_a polygon against face_b edges using Sutherland-Hodgman.

        Returns clipped polygon points (on face_a side) and count.
        """
        # Initialize polygon from face_a points
        poly = ClipPoly()
        poly[0] = face_a.p0
        poly[1] = face_a.p1
        poly[2] = face_a.p2
        poly[3] = face_a.p3
        num_pts = face_a.count

        # Get face_b points into an array
        fb0 = face_b.p0
        fb1 = face_b.p1
        fb2 = face_b.p2
        fb3 = face_b.p3
        fb_count = face_b.count

        # Clip against each edge of face_b
        # Edge normal = cross(edge_dir, contact_normal) pointing inward
        for ei in range(4):
            if ei >= fb_count:
                break
            ej = ei + 1
            if ej >= fb_count:
                ej = int(0)

            if ei == 0:
                ea = fb0
            elif ei == 1:
                ea = fb1
            elif ei == 2:
                ea = fb2
            else:
                ea = fb3

            if ej == 0:
                eb = fb0
            elif ej == 1:
                eb = fb1
            elif ej == 2:
                eb = fb2
            else:
                eb = fb3

            edge = eb - ea
            # Clip plane normal: inward-facing perpendicular to edge
            clip_n = wp.cross(edge, normal)
            clip_n_len = wp.length(clip_n)
            if clip_n_len > 1.0e-10:
                clip_n = clip_n / clip_n_len
                clip_d = wp.dot(clip_n, ea)
                poly, num_pts = clip_poly_against_plane(poly, num_pts, clip_n, clip_d)

        return poly, num_pts

    # -- Polygon reduction -----------------------------------------------

    @wp.func
    def reduce_polygon(
        poly: ClipPoly,
        num_pts: int,
        normal: wp.vec3,
    ) -> tuple[ClipPoly, int]:
        """Reduce a polygon to at most 4 points forming the largest quad.

        Strategy:
        1. Find the two most distant points (diameter).
        2. Find the point most distant from this line on each side.
        """
        if num_pts <= 4:
            return poly, num_pts

        # Step 1: find two most distant points
        best_dist = float(-1.0)
        best_i = int(0)
        best_j = int(1)
        for i in range(CLIP_MAX_POINTS):
            if i >= num_pts:
                break
            pi = wp.vec3(poly[i][0], poly[i][1], poly[i][2])
            for j in range(CLIP_MAX_POINTS):
                if j >= num_pts:
                    break
                if j <= i:
                    continue
                pj = wp.vec3(poly[j][0], poly[j][1], poly[j][2])
                d = wp.length_sq(pj - pi)
                if d > best_dist:
                    best_dist = d
                    best_i = i
                    best_j = j

        p0 = wp.vec3(poly[best_i][0], poly[best_i][1], poly[best_i][2])
        p1 = wp.vec3(poly[best_j][0], poly[best_j][1], poly[best_j][2])

        # Step 2: find point most distant from line p0-p1 on each side
        line = p1 - p0
        perp = wp.cross(normal, line)

        best_pos = float(-1.0e30)
        best_neg = float(1.0e30)
        idx_pos = int(-1)
        idx_neg = int(-1)

        for k in range(CLIP_MAX_POINTS):
            if k >= num_pts:
                break
            if k == best_i or k == best_j:
                continue
            pk = wp.vec3(poly[k][0], poly[k][1], poly[k][2])
            d = wp.dot(pk - p0, perp)
            if d > best_pos:
                best_pos = d
                idx_pos = k
            if d < best_neg:
                best_neg = d
                idx_neg = k

        out = ClipPoly()
        out_count = int(2)
        out[0] = p0
        out[1] = p1
        if idx_pos >= 0:
            out[out_count] = wp.vec3(poly[idx_pos][0], poly[idx_pos][1], poly[idx_pos][2])
            out_count = out_count + 1
        if idx_neg >= 0 and idx_neg != idx_pos:
            out[out_count] = wp.vec3(poly[idx_neg][0], poly[idx_neg][1], poly[idx_neg][2])
            out_count = out_count + 1

        return out, out_count

    # -- Full contact generation -----------------------------------------

    @wp.func
    def generate_contacts(shape_a: ShapeData, shape_b: ShapeData) -> ContactResult:
        """Generate contact patch between two shapes.

        Runs GJK/EPA to find penetration, then clips contact faces to
        produce up to 4 contact points with depths.
        """
        result = ContactResult()
        result.count = 0

        # Step 1: GJK + EPA
        gjk_result = gjk_epa(shape_a, shape_b)
        if gjk_result.overlap == 0:
            return result

        n = gjk_result.normal
        depth = gjk_result.distance
        result.normal = n

        # Step 2: Get contact faces from both shapes
        face_a = contact_face_world(shape_a, n, gjk_result.point_a)
        face_b = contact_face_world(shape_b, -n, gjk_result.point_b)

        # Step 3: Determine reference/incident faces
        # Reference = face with more points (or face_a if equal).
        # Clip incident face against reference face edges.
        # The contact normal n points A→B. For the reference face on A,
        # clip planes use n directly. For reference face on B, we negate
        # because the face normal of B points toward A (opposite to n).
        if face_b.count > face_a.count:
            # Reference = B, incident = A
            clipped, num_clipped = clip_face_against_face(face_a, face_b, -n)
        else:
            # Reference = A, incident = B
            clipped, num_clipped = clip_face_against_face(face_b, face_a, n)

        if num_clipped == 0:
            # Clipping produced no points — fall back to single contact
            mid = (gjk_result.point_a + gjk_result.point_b) * 0.5
            result.p0 = mid
            result.d0 = depth
            result.count = 1
            return result

        # Step 4: Reduce to max 4 points
        clipped, num_clipped = reduce_polygon(clipped, num_clipped, n)

        # Step 5: Compute per-point depths and store
        count = int(0)
        for ci in range(CLIP_MAX_POINTS):
            if ci >= num_clipped:
                break
            if count >= 4:
                break
            pt = wp.vec3(clipped[ci][0], clipped[ci][1], clipped[ci][2])
            # Depth = how far below the reference face this point is
            pt_depth = depth  # approximate: use overall penetration depth
            if count == 0:
                result.p0 = pt
                result.d0 = pt_depth
            elif count == 1:
                result.p1 = pt
                result.d1 = pt_depth
            elif count == 2:
                result.p2 = pt
                result.d2 = pt_depth
            elif count == 3:
                result.p3 = pt
                result.d3 = pt_depth
            count = count + 1

        result.count = count
        return result

    # -- Kernels --------------------------------------------------------

    @wp.kernel(enable_backward=False)
    def support_kernel(
        shapes: wp.array(dtype=ShapeData),
        directions: wp.array(dtype=wp.vec3),
        out_points: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        out_points[tid] = support_world(shapes[tid], directions[tid])

    @wp.kernel(enable_backward=False)
    def contact_face_kernel(
        shapes: wp.array(dtype=ShapeData),
        directions: wp.array(dtype=wp.vec3),
        out_p0: wp.array(dtype=wp.vec3),
        out_normal: wp.array(dtype=wp.vec3),
        out_count: wp.array(dtype=int),
    ):
        tid = wp.tid()
        r = contact_face_world(shapes[tid], directions[tid], wp.vec3(0.0, 0.0, 0.0))
        out_p0[tid] = r.p0
        out_normal[tid] = r.normal
        out_count[tid] = r.count

    @wp.kernel(enable_backward=False)
    def aabb_kernel(
        shapes: wp.array(dtype=ShapeData),
        out_min: wp.array(dtype=wp.vec3),
        out_max: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        mn, mx = get_aabb(shapes[tid])
        out_min[tid] = mn
        out_max[tid] = mx

    @wp.kernel(enable_backward=False)
    def gjk_kernel(
        shapes_a: wp.array(dtype=ShapeData),
        shapes_b: wp.array(dtype=ShapeData),
        results_dist: wp.array(dtype=float),
        results_point_a: wp.array(dtype=wp.vec3),
        results_point_b: wp.array(dtype=wp.vec3),
        results_normal: wp.array(dtype=wp.vec3),
        results_overlap: wp.array(dtype=int),
    ):
        tid = wp.tid()
        r = gjk_distance(shapes_a[tid], shapes_b[tid])
        results_dist[tid] = r.distance
        results_point_a[tid] = r.point_a
        results_point_b[tid] = r.point_b
        results_normal[tid] = r.normal
        results_overlap[tid] = r.overlap

    @wp.kernel(enable_backward=False)
    def gjk_epa_kernel(
        shapes_a: wp.array(dtype=ShapeData),
        shapes_b: wp.array(dtype=ShapeData),
        results_dist: wp.array(dtype=float),
        results_point_a: wp.array(dtype=wp.vec3),
        results_point_b: wp.array(dtype=wp.vec3),
        results_normal: wp.array(dtype=wp.vec3),
        results_overlap: wp.array(dtype=int),
    ):
        tid = wp.tid()
        r = gjk_epa(shapes_a[tid], shapes_b[tid])
        results_dist[tid] = r.distance
        results_point_a[tid] = r.point_a
        results_point_b[tid] = r.point_b
        results_normal[tid] = r.normal
        results_overlap[tid] = r.overlap

    @wp.kernel(enable_backward=False)
    def generate_contacts_kernel(
        shapes_a: wp.array(dtype=ShapeData),
        shapes_b: wp.array(dtype=ShapeData),
        out: wp.array(dtype=ContactResult),
    ):
        tid = wp.tid()
        out[tid] = generate_contacts(shapes_a[tid], shapes_b[tid])

    return Pipeline(
        support_local=support_local,
        support_world=support_world,
        contact_face_local=contact_face_local,
        contact_face_world=contact_face_world,
        get_aabb=get_aabb,
        gjk_distance=gjk_distance,
        gjk_epa=gjk_epa,
        epa=epa,
        generate_contacts=generate_contacts,
        support_kernel=support_kernel,
        contact_face_kernel=contact_face_kernel,
        aabb_kernel=aabb_kernel,
        gjk_kernel=gjk_kernel,
        gjk_epa_kernel=gjk_epa_kernel,
        generate_contacts_kernel=generate_contacts_kernel,
    )


@dataclass
class Pipeline:
    """Compiled collision pipeline with dispatch for all registered shapes.

    Created by :func:`create_pipeline`.  Holds generated ``@wp.func``
    dispatch functions and ``@wp.kernel`` wrappers.
    """

    support_local: Any
    support_world: Any
    contact_face_local: Any
    contact_face_world: Any
    get_aabb: Any
    gjk_distance: Any
    gjk_epa: Any
    epa: Any
    generate_contacts: Any
    support_kernel: Any
    contact_face_kernel: Any
    aabb_kernel: Any
    gjk_kernel: Any
    gjk_epa_kernel: Any
    generate_contacts_kernel: Any
