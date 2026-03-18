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

"""GJK closest-point subalgorithm — shape-independent helpers.

These ``@wp.func`` functions are referenced via ``wp.static()`` inside
the pipeline factory and can live in a separate module.
"""

import warp as wp

from .types import GJK_EPSILON


@wp.func
def closest_segment(a: wp.vec3, b: wp.vec3) -> tuple[wp.vec3, float, float]:
    """Closest point on segment AB to origin."""
    ab = b - a
    denom = wp.dot(ab, ab)
    if denom < GJK_EPSILON:
        return a, 1.0, 0.0
    t = wp.clamp(-wp.dot(a, ab) / denom, 0.0, 1.0)
    return a + t * ab, 1.0 - t, t


@wp.func
def closest_triangle(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> tuple[wp.vec3, float, float, float]:
    """Closest point on triangle ABC to origin."""
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
