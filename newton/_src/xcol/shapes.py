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

"""Shape registration and built-in shape functions for xcol.

All shape types — built-in and custom — are registered via
:func:`register_shape` before :func:`~xcol.pipeline.create_pipeline` is
called.  Each shape provides core support, contact face, and AABB
functions as ``@wp.func``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from .types import ContactFaceResult, ShapeData

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
    :func:`~xcol.pipeline.create_pipeline` is called.

    Args:
        name: Human-readable name (e.g. ``"point"``).
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


def get_registered_shapes() -> list[ShapeEntry]:
    """Return all registered shape entries (in registration order)."""
    return list(_shape_registry.values())


# ===================================================================
# Built-in shape functions (local space, core only — no margin)
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
