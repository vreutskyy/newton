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

# Based on the GJK/simplex solver from Jitter Physics 2
# (https://github.com/notgiven688/jitterphysics2)
# Copyright (c) Thorben Linneweber (MIT License)
# Translated from C# and adapted for xcol.

"""GJK simplex solver using bitmask vertex management.

Simplex vertices are stored in a ``Mat83f`` (8×3 matrix).  Each vertex
occupies two consecutive rows:

- ``v[2*i]``     → B  (point on shape B in A-local frame)
- ``v[2*i + 1]`` → BtoA  (Minkowski difference = A − B)

A ``wp.uint32`` usage mask tracks which of the 4 slots are active.
"""

import warp as wp

from .types import GJK_EPSILON, Mat83f

EPSILON = GJK_EPSILON


@wp.func
def closest_segment(v: Mat83f, i0: int, i1: int) -> tuple[wp.vec3, wp.vec4, wp.uint32]:
    """Closest point on segment to origin."""
    a = v[2 * i0 + 1]
    b = v[2 * i1 + 1]

    edge = b - a
    vsq = wp.length_sq(edge)
    degenerate = vsq < EPSILON

    denom = vsq
    if degenerate:
        denom = EPSILON
    t = -wp.dot(a, edge) / denom
    lambda0 = 1.0 - t
    lambda1 = t

    mask = (wp.uint32(1) << wp.uint32(i0)) | (wp.uint32(1) << wp.uint32(i1))
    bc = wp.vec4(0.0, 0.0, 0.0, 0.0)

    if lambda0 < 0.0 or degenerate:
        mask = wp.uint32(1) << wp.uint32(i1)
        lambda0 = 0.0
        lambda1 = 1.0
    elif lambda1 < 0.0:
        mask = wp.uint32(1) << wp.uint32(i0)
        lambda0 = 1.0
        lambda1 = 0.0

    bc[i0] = lambda0
    bc[i1] = lambda1
    return lambda0 * a + lambda1 * b, bc, mask


@wp.func
def closest_triangle(v: Mat83f, i0: int, i1: int, i2: int) -> tuple[wp.vec3, wp.vec4, wp.uint32]:
    """Closest point on triangle to origin."""
    a = v[2 * i0 + 1]
    b = v[2 * i1 + 1]
    c = v[2 * i2 + 1]

    u = a - b
    w = a - c
    normal = wp.cross(u, w)

    t = wp.length_sq(normal)
    degenerate = t < EPSILON
    denom = t
    if degenerate:
        denom = EPSILON
    it = 1.0 / denom

    c1 = wp.cross(u, a)
    c2 = wp.cross(a, w)

    lambda2 = wp.dot(c1, normal) * it
    lambda1 = wp.dot(c2, normal) * it
    lambda0 = 1.0 - lambda2 - lambda1

    best_distance = float(1.0e30)
    closest_pt = wp.vec3(0.0, 0.0, 0.0)
    bc = wp.vec4(0.0, 0.0, 0.0, 0.0)
    mask = wp.uint32(0)

    if lambda0 < 0.0 or degenerate:
        closest, bc_tmp, m = closest_segment(v, i1, i2)
        dist = wp.length_sq(closest)
        if dist < best_distance:
            bc = bc_tmp
            mask = m
            best_distance = dist
            closest_pt = closest

    if lambda1 < 0.0 or degenerate:
        closest, bc_tmp, m = closest_segment(v, i0, i2)
        dist = wp.length_sq(closest)
        if dist < best_distance:
            bc = bc_tmp
            mask = m
            best_distance = dist
            closest_pt = closest

    if lambda2 < 0.0 or degenerate:
        closest, bc_tmp, m = closest_segment(v, i0, i1)
        dist = wp.length_sq(closest)
        if dist < best_distance:
            bc = bc_tmp
            mask = m
            closest_pt = closest

    if mask != wp.uint32(0):
        return closest_pt, bc, mask

    bc[i0] = lambda0
    bc[i1] = lambda1
    bc[i2] = lambda2
    mask = (wp.uint32(1) << wp.uint32(i0)) | (wp.uint32(1) << wp.uint32(i1)) | (wp.uint32(1) << wp.uint32(i2))
    return lambda0 * a + lambda1 * b + lambda2 * c, bc, mask


@wp.func
def determinant(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3) -> float:
    """Signed volume of tetrahedron ABCD."""
    return wp.dot(b - a, wp.cross(c - a, d - a))


@wp.func
def closest_tetrahedron(v: Mat83f) -> tuple[wp.vec3, wp.vec4, wp.uint32]:
    """Closest point on tetrahedron to origin."""
    v0 = v[1]
    v1 = v[3]
    v2 = v[5]
    v3 = v[7]

    det_t = determinant(v0, v1, v2, v3)
    degenerate = wp.abs(det_t) < EPSILON
    denom = det_t
    if degenerate:
        denom = EPSILON
    inverse_det_t = 1.0 / denom

    zero = wp.vec3(0.0, 0.0, 0.0)
    lambda0 = determinant(zero, v1, v2, v3) * inverse_det_t
    lambda1 = determinant(v0, zero, v2, v3) * inverse_det_t
    lambda2 = determinant(v0, v1, zero, v3) * inverse_det_t
    lambda3 = 1.0 - lambda0 - lambda1 - lambda2

    best_distance = float(1.0e30)
    closest_pt = wp.vec3(0.0, 0.0, 0.0)
    bc = wp.vec4(0.0, 0.0, 0.0, 0.0)
    mask = wp.uint32(0)

    if lambda0 < 0.0 or degenerate:
        closest, bc_tmp, m = closest_triangle(v, 1, 2, 3)
        dist = wp.length_sq(closest)
        if dist < best_distance:
            bc = bc_tmp
            mask = m
            best_distance = dist
            closest_pt = closest

    if lambda1 < 0.0 or degenerate:
        closest, bc_tmp, m = closest_triangle(v, 0, 2, 3)
        dist = wp.length_sq(closest)
        if dist < best_distance:
            bc = bc_tmp
            mask = m
            best_distance = dist
            closest_pt = closest

    if lambda2 < 0.0 or degenerate:
        closest, bc_tmp, m = closest_triangle(v, 0, 1, 3)
        dist = wp.length_sq(closest)
        if dist < best_distance:
            bc = bc_tmp
            mask = m
            best_distance = dist
            closest_pt = closest

    if lambda3 < 0.0 or degenerate:
        closest, bc_tmp, m = closest_triangle(v, 0, 1, 2)
        dist = wp.length_sq(closest)
        if dist < best_distance:
            bc = bc_tmp
            mask = m
            closest_pt = closest

    if mask != wp.uint32(0):
        return closest_pt, bc, mask

    bc[0] = lambda0
    bc[1] = lambda1
    bc[2] = lambda2
    bc[3] = lambda3
    mask = wp.uint32(15)  # 0b1111
    return zero, bc, mask


@wp.func
def simplex_get_closest(v: Mat83f, barycentric: wp.vec4, usage_mask: wp.uint32) -> tuple[wp.vec3, wp.vec3]:
    """Reconstruct witness points on shapes A and B from barycentric coords."""
    point_a = wp.vec3(0.0, 0.0, 0.0)
    point_b = wp.vec3(0.0, 0.0, 0.0)

    for i in range(4):
        if (usage_mask & (wp.uint32(1) << wp.uint32(i))) == wp.uint32(0):
            continue
        bc_val = barycentric[i]
        b_pt = v[2 * i]
        btoa = v[2 * i + 1]
        point_a = point_a + bc_val * (b_pt + btoa)
        point_b = point_b + bc_val * b_pt

    return point_a, point_b
