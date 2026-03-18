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

"""Pipeline factory and compiled pipeline object for xcol.

:func:`create_pipeline` generates all Warp dispatch functions and kernels
from the shape registry.  All ``@wp.func`` and ``@wp.kernel`` definitions
live inside the factory closure so they compile into a single Warp module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .gjk import closest_segment, closest_triangle
from .shapes import ShapeEntry, get_registered_shapes
from .types import (
    CLIP_MAX_POINTS,
    EPA_EPSILON,
    EPA_MAX_FACES,
    EPA_MAX_ITERATIONS,
    EPA_MAX_VERTS,
    GJK_EPSILON,
    GJK_MAX_ITERATIONS,
    ClipPoly,
    ContactFaceResult,
    ContactResult,
    EPAFaces,
    EPAVerts,
    EPAVertsA,
    EPAVertsB,
    GJKResult,
    ShapeData,
)


def create_pipeline(shape_entries: list[ShapeEntry] | None = None):
    """Create dispatch functions and kernels from registered shape types.

    Call this after all :func:`~xcol.shapes.register_shape` calls are done
    (including custom shapes).  If *shape_entries* is ``None``, uses the
    global registry.

    Returns:
        A :class:`Pipeline` object with the generated dispatch functions
        and kernels.
    """
    if shape_entries is None:
        shape_entries = get_registered_shapes()

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
            result.point_a = result.point_a + result.normal * shape_a.margin
            result.point_b = result.point_b - result.normal * shape_b.margin

        if real_dist <= 0.0:
            result.overlap = 1  # margin-only overlap
            result.distance = -real_dist
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

        faces = EPAFaces()
        n012 = wp.cross(s1 - s0, s2 - s0)
        if wp.dot(n012, s3 - s0) > 0.0:
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

            sa = support_world(shape_a, best_normal)
            sb = support_world(shape_b, -best_normal)
            w_new = sa - sb

            new_dist = wp.dot(w_new, best_normal)
            if new_dist - best_dist < EPA_EPSILON:
                result.distance = best_dist
                result.normal = best_normal

                bi0 = int(faces[best_face][0])
                bi1 = int(faces[best_face][1])
                bi2 = int(faces[best_face][2])
                fa = wp.vec3(verts[bi0][0], verts[bi0][1], verts[bi0][2])
                fb = wp.vec3(verts[bi1][0], verts[bi1][1], verts[bi1][2])
                fc = wp.vec3(verts[bi2][0], verts[bi2][1], verts[bi2][2])

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

            new_vi = num_verts
            verts[new_vi] = w_new
            verts_a[new_vi] = sa
            verts_b[new_vi] = sb
            num_verts = num_verts + 1

            edges = EPAFaces()
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

                visible = int(0)
                if fn_len > EPA_EPSILON:
                    if wp.dot(fn, w_new - va) > EPA_EPSILON:
                        visible = int(1)

                if visible == 1:
                    e0 = wp.vec3(float(i0), float(i1), 0.0)
                    e1 = wp.vec3(float(i1), float(i2), 0.0)
                    e2 = wp.vec3(float(i2), float(i0), 0.0)

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
                            num_edges = num_edges - 1
                            edges[found] = edges[num_edges]
                        else:
                            if num_edges < EPA_MAX_FACES:
                                edges[num_edges] = edge
                                num_edges = num_edges + 1
                else:
                    if new_num_faces != fi:
                        faces[new_num_faces] = faces[fi]
                    new_num_faces = new_num_faces + 1

            num_faces = new_num_faces

            for ei in range(EPA_MAX_FACES):
                if ei >= num_edges:
                    break
                if num_faces < EPA_MAX_FACES:
                    faces[num_faces] = wp.vec3(edges[ei][0], edges[ei][1], float(new_vi))
                    num_faces = num_faces + 1

        result.distance = best_dist
        result.normal = best_normal
        return result

    # -- Combined GJK + EPA ---------------------------------------------

    @wp.func
    def build_initial_tetrahedron(shape_a: ShapeData, shape_b: ShapeData) -> GJKResult:
        """Build an initial tetrahedron enclosing the origin for EPA."""
        result = GJKResult()
        result.overlap = 1
        result.distance = 0.0

        d0 = shape_b.pos - shape_a.pos
        if wp.length(d0) < GJK_EPSILON:
            d0 = wp.vec3(1.0, 0.0, 0.0)
        d0 = wp.normalize(d0)
        sa = support_world(shape_a, d0)
        sb = support_world(shape_b, -d0)
        result.sw0 = sa - sb
        result.spa0 = sa
        result.spb0 = sb

        d1 = -d0
        sa = support_world(shape_a, d1)
        sb = support_world(shape_b, -d1)
        result.sw1 = sa - sb
        result.spa1 = sa
        result.spb1 = sb

        line = result.sw1 - result.sw0
        t = -wp.dot(result.sw0, line) / wp.max(wp.dot(line, line), GJK_EPSILON)
        closest_pt = result.sw0 + t * line
        d2 = -closest_pt
        if wp.length(d2) < GJK_EPSILON:
            d2 = wp.cross(line, wp.vec3(1.0, 0.0, 0.0))
            if wp.length(d2) < GJK_EPSILON:
                d2 = wp.cross(line, wp.vec3(0.0, 1.0, 0.0))
        d2 = wp.normalize(d2)
        sa = support_world(shape_a, d2)
        sb = support_world(shape_b, -d2)
        result.sw2 = sa - sb
        result.spa2 = sa
        result.spb2 = sb

        tri_n = wp.cross(result.sw1 - result.sw0, result.sw2 - result.sw0)
        tri_n_len = wp.length(tri_n)
        if tri_n_len < GJK_EPSILON:
            tri_n = wp.vec3(0.0, 0.0, 1.0)
        else:
            tri_n = tri_n / tri_n_len
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
            return gjk_result

        total_margin = shape_a.margin + shape_b.margin

        d0 = gjk_result.sw1 - gjk_result.sw0
        d1 = gjk_result.sw2 - gjk_result.sw0
        d2 = gjk_result.sw3 - gjk_result.sw0
        vol = wp.abs(wp.dot(d0, wp.cross(d1, d2)))

        if vol < GJK_EPSILON:
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

        epa_result.distance = epa_result.distance + total_margin
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
        """Clip a polygon against a single plane (keep negative side)."""
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
                if out_count < CLIP_MAX_POINTS:
                    out[out_count] = pi
                    out_count = out_count + 1
                if dj > 0.0:
                    denom = di - dj
                    if wp.abs(denom) < 1.0e-10:
                        denom = 1.0e-10
                    t = di / denom
                    intersection = pi + t * (pj - pi)
                    if out_count < CLIP_MAX_POINTS:
                        out[out_count] = intersection
                        out_count = out_count + 1
            elif dj <= 0.0:
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
        """Clip face_a polygon against face_b edges."""
        poly = ClipPoly()
        poly[0] = face_a.p0
        poly[1] = face_a.p1
        poly[2] = face_a.p2
        poly[3] = face_a.p3
        num_pts = face_a.count

        fb0 = face_b.p0
        fb1 = face_b.p1
        fb2 = face_b.p2
        fb3 = face_b.p3
        fb_count = face_b.count

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
        ref_offset: float,
    ) -> tuple[ClipPoly, int]:
        """Reduce a polygon to at most 4 points (deepest first)."""
        if num_pts <= 4:
            return poly, num_pts

        best_depth = float(-1.0e30)
        idx0 = int(0)
        for i in range(CLIP_MAX_POINTS):
            if i >= num_pts:
                break
            pi = wp.vec3(poly[i][0], poly[i][1], poly[i][2])
            d = ref_offset - wp.dot(normal, pi)
            if d > best_depth:
                best_depth = d
                idx0 = i

        p0 = wp.vec3(poly[idx0][0], poly[idx0][1], poly[idx0][2])

        best_dist = float(-1.0)
        idx1 = int(0)
        for i in range(CLIP_MAX_POINTS):
            if i >= num_pts:
                break
            if i == idx0:
                continue
            pi = wp.vec3(poly[i][0], poly[i][1], poly[i][2])
            d = wp.length_sq(pi - p0)
            if d > best_dist:
                best_dist = d
                idx1 = i

        p1 = wp.vec3(poly[idx1][0], poly[idx1][1], poly[idx1][2])

        line = p1 - p0
        perp = wp.cross(normal, line)

        best_pos = float(-1.0e30)
        best_neg = float(1.0e30)
        idx2 = int(-1)
        idx3 = int(-1)

        for k in range(CLIP_MAX_POINTS):
            if k >= num_pts:
                break
            if k == idx0 or k == idx1:
                continue
            pk = wp.vec3(poly[k][0], poly[k][1], poly[k][2])
            d = wp.dot(pk - p0, perp)
            if d > best_pos:
                best_pos = d
                idx2 = k
            if d < best_neg:
                best_neg = d
                idx3 = k

        out = ClipPoly()
        out_count = int(2)
        out[0] = p0
        out[1] = p1
        if idx2 >= 0:
            out[out_count] = wp.vec3(poly[idx2][0], poly[idx2][1], poly[idx2][2])
            out_count = out_count + 1
        if idx3 >= 0 and idx3 != idx2:
            out[out_count] = wp.vec3(poly[idx3][0], poly[idx3][1], poly[idx3][2])
            out_count = out_count + 1

        return out, out_count

    # -- Full contact generation -----------------------------------------

    @wp.func
    def generate_contacts(shape_a: ShapeData, shape_b: ShapeData) -> ContactResult:
        """Generate contact patch between two shapes."""
        result = ContactResult()
        result.count = 0

        gjk_result = gjk_epa(shape_a, shape_b)
        if gjk_result.overlap == 0:
            return result

        n = gjk_result.normal
        depth = gjk_result.distance
        result.normal = n

        face_a = contact_face_world(shape_a, n, gjk_result.point_a)
        face_b = contact_face_world(shape_b, -n, gjk_result.point_b)

        if face_b.count > face_a.count:
            clipped, num_clipped = clip_face_against_face(face_a, face_b, -n)
            ref_offset = wp.dot(face_a.p0, n)
        else:
            clipped, num_clipped = clip_face_against_face(face_b, face_a, n)
            ref_offset = wp.dot(face_a.p0, n)

        if num_clipped == 0:
            mid = (gjk_result.point_a + gjk_result.point_b) * 0.5
            result.p0 = mid
            result.d0 = depth
            result.count = 1
            return result

        clipped, num_clipped = reduce_polygon(clipped, num_clipped, n, ref_offset)

        count = int(0)
        for ci in range(CLIP_MAX_POINTS):
            if ci >= num_clipped:
                break
            if count >= 4:
                break
            pt = wp.vec3(clipped[ci][0], clipped[ci][1], clipped[ci][2])
            pt_depth = ref_offset - wp.dot(n, pt)
            if pt_depth < 0.0:
                pt_depth = depth
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
class Contacts:
    """Contact query results from :meth:`Pipeline.collide`.

    Attributes:
        count: Number of contacting pairs.
        pair_shape_a: Shape index of first shape in each pair, length *count*.
        pair_shape_b: Shape index of second shape in each pair, length *count*.
        normal: Contact normal (A->B) for each pair, length *count*.
        point_count: Number of contact points (1-4) for each pair, length *count*.
        points: Contact point positions, shape ``(count, 4, 3)``.
        depths: Penetration depth per point, shape ``(count, 4)``.
    """

    count: int
    pair_shape_a: np.ndarray
    pair_shape_b: np.ndarray
    normal: np.ndarray
    point_count: np.ndarray
    points: np.ndarray
    depths: np.ndarray


class Pipeline:
    """Compiled collision pipeline with dispatch for all registered shapes.

    Created by :func:`create_pipeline`.  Use :meth:`add_shape` to populate,
    then :meth:`collide` to run broadphase + narrowphase.
    """

    def __init__(self, **funcs: Any) -> None:
        # Store compiled Warp functions/kernels
        for k, v in funcs.items():
            setattr(self, k, v)

        # SoA shape storage (Python lists, flushed to Warp arrays on collide)
        self._types: list[int] = []
        self._params: list[tuple[float, float, float]] = []
        self._margins: list[float] = []
        self._worlds: list[int] = []
        self._transforms: list[tuple[float, ...]] = []

        # Warp arrays (lazily built)
        self._shape_types_wp: wp.array | None = None
        self._shape_params_wp: wp.array | None = None
        self._shape_margins_wp: wp.array | None = None
        self._shape_worlds_wp: wp.array | None = None
        self._shape_transforms_wp: wp.array | None = None
        self._dirty = True

    @property
    def shape_count(self) -> int:
        return len(self._types)

    def add_shape(
        self,
        shape_type: int,
        params: tuple[float, float, float] = (0.0, 0.0, 0.0),
        margin: float = 0.0,
        world: int = 0,
        transform: tuple[float, ...] | None = None,
    ) -> int:
        """Add a shape to the pipeline.

        Args:
            shape_type: Shape type id (e.g. ``SHAPE_POINT``, ``SHAPE_BOX``).
            params: Core shape parameters.
            margin: Uniform inflation distance [m].
            world: World index. Shapes in the same world collide.
                ``-1`` = global (collides with all worlds).
            transform: Initial transform as ``(px, py, pz, qx, qy, qz, qw)``.
                Defaults to identity.

        Returns:
            Integer shape handle (index into SoA arrays).
        """
        idx = len(self._types)
        self._types.append(int(shape_type))
        self._params.append((float(params[0]), float(params[1]), float(params[2])))
        self._margins.append(float(margin))
        self._worlds.append(int(world))
        if transform is None:
            transform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self._transforms.append(transform)
        self._dirty = True
        return idx

    def _flush(self) -> None:
        """Rebuild Warp arrays from Python lists."""
        if not self._dirty:
            return
        n = len(self._types)
        if n == 0:
            self._dirty = False
            return

        # Use CPU device so numpy() returns writable views
        device = "cpu"
        self._shape_types_wp = wp.array(self._types, dtype=int, device=device)
        self._shape_params_wp = wp.array([wp.vec3(*p) for p in self._params], dtype=wp.vec3, device=device)
        self._shape_margins_wp = wp.array(self._margins, dtype=float, device=device)
        self._shape_worlds_wp = wp.array(self._worlds, dtype=int, device=device)
        self._shape_transforms_wp = wp.array(
            [wp.transform(*t) for t in self._transforms], dtype=wp.transform, device=device
        )
        self._dirty = False

    # -- SoA array properties (user can read/write transforms) -----------

    @property
    def shape_types(self) -> wp.array:
        self._flush()
        return self._shape_types_wp

    @property
    def shape_params(self) -> wp.array:
        self._flush()
        return self._shape_params_wp

    @property
    def shape_margins(self) -> wp.array:
        self._flush()
        return self._shape_margins_wp

    @property
    def shape_worlds(self) -> wp.array:
        self._flush()
        return self._shape_worlds_wp

    @property
    def shape_transforms(self) -> wp.array:
        self._flush()
        return self._shape_transforms_wp

    # -- Collide ---------------------------------------------------------

    def collide(self) -> Contacts:
        """Run broadphase + narrowphase and return contacts.

        The user should update :attr:`shape_transforms` before calling this.
        Currently uses brute-force N*N broadphase.
        """
        self._flush()
        n = self.shape_count
        if n < 2:
            return Contacts(
                count=0,
                pair_shape_a=np.empty(0, dtype=np.int32),
                pair_shape_b=np.empty(0, dtype=np.int32),
                normal=np.empty((0, 3), dtype=np.float32),
                point_count=np.empty(0, dtype=np.int32),
                points=np.empty((0, 4, 3), dtype=np.float32),
                depths=np.empty((0, 4), dtype=np.float32),
            )

        # Read arrays to numpy for broadphase (CPU for now)
        types_np = self._shape_types_wp.numpy()
        params_np = self._shape_params_wp.numpy()
        margins_np = self._shape_margins_wp.numpy()
        worlds_np = self._shape_worlds_wp.numpy()
        transforms_np = self._shape_transforms_wp.numpy()

        # Brute-force N×N broadphase: collect candidate pairs
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                wi = worlds_np[i]
                wj = worlds_np[j]
                # Same world or either is global (-1)
                if wi == wj or wi == -1 or wj == -1:
                    pairs.append((i, j))

        if len(pairs) == 0:
            return Contacts(
                count=0,
                pair_shape_a=np.empty(0, dtype=np.int32),
                pair_shape_b=np.empty(0, dtype=np.int32),
                normal=np.empty((0, 3), dtype=np.float32),
                point_count=np.empty(0, dtype=np.int32),
                points=np.empty((0, 4, 3), dtype=np.float32),
                depths=np.empty((0, 4), dtype=np.float32),
            )

        # Build ShapeData arrays for each side of the pairs
        num_pairs = len(pairs)
        shapes_a = wp.zeros(num_pairs, dtype=ShapeData)
        shapes_b = wp.zeros(num_pairs, dtype=ShapeData)
        shapes_a_np = shapes_a.numpy()
        shapes_b_np = shapes_b.numpy()

        for pi, (i, j) in enumerate(pairs):
            ti = transforms_np[i]
            tj = transforms_np[j]
            # ShapeData fields: shape_type, pos(3), rot(4), params(3), margin
            shapes_a_np[pi]["shape_type"] = types_np[i]
            shapes_a_np[pi]["pos"] = ti[:3]
            shapes_a_np[pi]["rot"] = ti[3:]
            shapes_a_np[pi]["params"] = params_np[i]
            shapes_a_np[pi]["margin"] = margins_np[i]

            shapes_b_np[pi]["shape_type"] = types_np[j]
            shapes_b_np[pi]["pos"] = tj[:3]
            shapes_b_np[pi]["rot"] = tj[3:]
            shapes_b_np[pi]["params"] = params_np[j]
            shapes_b_np[pi]["margin"] = margins_np[j]

        shapes_a = wp.array(shapes_a_np, dtype=ShapeData)
        shapes_b = wp.array(shapes_b_np, dtype=ShapeData)

        # Run narrowphase
        out = wp.zeros(num_pairs, dtype=ContactResult)
        wp.launch(
            self.generate_contacts_kernel,
            dim=num_pairs,
            inputs=[shapes_a, shapes_b],
            outputs=[out],
        )
        results_np = out.numpy()

        # Collect results — filter out pairs with count == 0
        pair_a_list = []
        pair_b_list = []
        normal_list = []
        point_count_list = []
        points_list = []
        depths_list = []

        for pi in range(num_pairs):
            r = results_np[pi]
            cnt = int(r["count"])
            if cnt == 0:
                continue
            i, j = pairs[pi]
            pair_a_list.append(i)
            pair_b_list.append(j)
            normal_list.append(r["normal"])
            point_count_list.append(cnt)
            pts = np.zeros((4, 3), dtype=np.float32)
            dps = np.zeros(4, dtype=np.float32)
            for k in range(cnt):
                pts[k] = r[f"p{k}"]
                dps[k] = r[f"d{k}"]
            points_list.append(pts)
            depths_list.append(dps)

        count = len(pair_a_list)
        if count == 0:
            return Contacts(
                count=0,
                pair_shape_a=np.empty(0, dtype=np.int32),
                pair_shape_b=np.empty(0, dtype=np.int32),
                normal=np.empty((0, 3), dtype=np.float32),
                point_count=np.empty(0, dtype=np.int32),
                points=np.empty((0, 4, 3), dtype=np.float32),
                depths=np.empty((0, 4), dtype=np.float32),
            )

        return Contacts(
            count=count,
            pair_shape_a=np.array(pair_a_list, dtype=np.int32),
            pair_shape_b=np.array(pair_b_list, dtype=np.int32),
            normal=np.array(normal_list, dtype=np.float32),
            point_count=np.array(point_count_list, dtype=np.int32),
            points=np.array(points_list, dtype=np.float32),
            depths=np.array(depths_list, dtype=np.float32),
        )
