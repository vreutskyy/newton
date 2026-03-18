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


# ---------------------------------------------------------------------------
# Warp structs
# ---------------------------------------------------------------------------


@wp.struct
class ShapeData:
    """Uniform shape descriptor for all primitive types.

    Fields:
        shape_type: Integer id assigned by :func:`register_shape`.
        pos: World-space position of the shape center [m].
        rot: World-space orientation quaternion.
        params: Shape-specific parameters (meaning depends on shape type).
    """

    shape_type: int
    pos: wp.vec3
    rot: wp.quat
    params: wp.vec3


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
class GJKResult:
    """Result of a GJK distance query."""

    distance: float
    point_a: wp.vec3
    point_b: wp.vec3
    normal: wp.vec3
    overlap: int  # 1 = overlap, 0 = separated


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
_next_type_id: int = 0


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
    global _next_type_id
    type_id = _next_type_id
    _next_type_id += 1
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
def _support_sphere(params: wp.vec3, direction: wp.vec3) -> wp.vec3:
    radius = params[0]
    d_len = wp.length(direction)
    if d_len > 1.0e-12:
        return wp.normalize(direction) * radius
    return wp.vec3(radius, 0.0, 0.0)


@wp.func
def _support_box(params: wp.vec3, direction: wp.vec3) -> wp.vec3:
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
def _support_capsule(params: wp.vec3, direction: wp.vec3) -> wp.vec3:
    radius = params[0]
    half_height = params[1]
    d_len = wp.length(direction)
    if d_len > 1.0e-12:
        n = wp.normalize(direction)
    else:
        n = wp.vec3(1.0, 0.0, 0.0)
    result = n * radius
    if direction[2] >= 0.0:
        result = result + wp.vec3(0.0, 0.0, half_height)
    else:
        result = result - wp.vec3(0.0, 0.0, half_height)
    return result


@wp.func
def _contact_face_sphere(params: wp.vec3, direction: wp.vec3) -> ContactFaceResult:
    radius = params[0]
    d_len = wp.length(direction)
    if d_len > 1.0e-12:
        n = wp.normalize(direction)
    else:
        n = wp.vec3(0.0, 0.0, 1.0)
    pt = n * radius
    result = ContactFaceResult()
    result.p0 = pt
    result.p1 = pt
    result.p2 = pt
    result.p3 = pt
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
def _contact_face_capsule(params: wp.vec3, direction: wp.vec3) -> ContactFaceResult:
    radius = params[0]
    half_height = params[1]
    d_len = wp.length(direction)
    if d_len > 1.0e-12:
        n = wp.normalize(direction)
    else:
        n = wp.vec3(0.0, 0.0, 1.0)
    az = wp.abs(n[2])
    result = ContactFaceResult()
    if az > 0.9:
        s = 1.0
        if n[2] < 0.0:
            s = -1.0
        result.p0 = wp.vec3(0.0, 0.0, s * (half_height + radius))
        result.p1 = result.p0
        result.p2 = result.p0
        result.p3 = result.p0
        result.normal = wp.vec3(0.0, 0.0, s)
        result.count = 1
    else:
        lateral = wp.vec3(n[0], n[1], 0.0)
        lat_len = wp.length(lateral)
        if lat_len > 1.0e-12:
            lateral = lateral / lat_len
        else:
            lateral = wp.vec3(1.0, 0.0, 0.0)
        offset = lateral * radius
        result.p0 = wp.vec3(0.0, 0.0, half_height) + offset
        result.p1 = wp.vec3(0.0, 0.0, -half_height) + offset
        result.p2 = result.p1
        result.p3 = result.p1
        result.normal = lateral
        result.count = 2
    return result


@wp.func
def _aabb_sphere(shape: ShapeData) -> tuple[wp.vec3, wp.vec3]:
    r = shape.params[0]
    rv = wp.vec3(r, r, r)
    return shape.pos - rv, shape.pos + rv


@wp.func
def _aabb_box(shape: ShapeData) -> tuple[wp.vec3, wp.vec3]:
    hx = shape.params[0]
    hy = shape.params[1]
    hz = shape.params[2]
    ax = wp.quat_rotate(shape.rot, wp.vec3(hx, 0.0, 0.0))
    ay = wp.quat_rotate(shape.rot, wp.vec3(0.0, hy, 0.0))
    aaz = wp.quat_rotate(shape.rot, wp.vec3(0.0, 0.0, hz))
    extent = wp.vec3(
        wp.abs(ax[0]) + wp.abs(ay[0]) + wp.abs(aaz[0]),
        wp.abs(ax[1]) + wp.abs(ay[1]) + wp.abs(aaz[1]),
        wp.abs(ax[2]) + wp.abs(ay[2]) + wp.abs(aaz[2]),
    )
    return shape.pos - extent, shape.pos + extent


@wp.func
def _aabb_capsule(shape: ShapeData) -> tuple[wp.vec3, wp.vec3]:
    r = shape.params[0]
    hh = shape.params[1]
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


# ===================================================================
# Register built-in shapes (same path as custom shapes)
# ===================================================================

SHAPE_SPHERE = register_shape(
    "sphere",
    support_fn=_support_sphere,
    contact_face_fn=_contact_face_sphere,
    aabb_fn=_aabb_sphere,
)
SHAPE_BOX = register_shape(
    "box",
    support_fn=_support_box,
    contact_face_fn=_contact_face_box,
    aabb_fn=_aabb_box,
)
SHAPE_CAPSULE = register_shape(
    "capsule",
    support_fn=_support_capsule,
    contact_face_fn=_contact_face_capsule,
    aabb_fn=_aabb_capsule,
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
    def contact_face_local(
        shape_type: int, params: wp.vec3, direction: wp.vec3
    ) -> ContactFaceResult:
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
    def contact_face_world(
        shape: ShapeData, direction: wp.vec3, point: wp.vec3
    ) -> ContactFaceResult:
        local_dir = wp.quat_rotate_inv(shape.rot, direction)
        result = contact_face_local(shape.shape_type, shape.params, local_dir)
        result.p0 = wp.quat_rotate(shape.rot, result.p0) + shape.pos
        result.p1 = wp.quat_rotate(shape.rot, result.p1) + shape.pos
        result.p2 = wp.quat_rotate(shape.rot, result.p2) + shape.pos
        result.p3 = wp.quat_rotate(shape.rot, result.p3) + shape.pos
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
                result.overlap = 1
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
                    result.overlap = 1
                    result.distance = 0.0
                    result.point_a = lam0 * pa0 + lam1 * pa1 + lam2 * pa2 + lam3 * pa3
                    result.point_b = lam0 * pb0 + lam1 * pb1 + lam2 * pb2 + lam3 * pb3
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

        dist = wp.length(v)
        result.distance = dist
        if dist > GJK_EPSILON:
            result.normal = -v / dist
        result.overlap = 0
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

    return Pipeline(
        support_local=support_local,
        support_world=support_world,
        contact_face_local=contact_face_local,
        contact_face_world=contact_face_world,
        get_aabb=get_aabb,
        gjk_distance=gjk_distance,
        support_kernel=support_kernel,
        contact_face_kernel=contact_face_kernel,
        aabb_kernel=aabb_kernel,
        gjk_kernel=gjk_kernel,
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
    support_kernel: Any
    contact_face_kernel: Any
    aabb_kernel: Any
    gjk_kernel: Any
