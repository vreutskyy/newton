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

"""Collider factory and compiled Warp kernels for xcol.

:func:`create_collider` generates all Warp dispatch functions and kernels
from the shape registry.  All ``@wp.func`` and ``@wp.kernel`` definitions
live inside the factory closure so they compile into a single Warp module.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from .gjk import closest_segment, closest_triangle
from .shapes import ShapeEntry, get_registered_shapes
from .types import (
    CLIP_MAX_POINTS,
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


def create_collider(shape_entries: list[ShapeEntry] | None = None):
    """Create a :class:`Collider` with dispatch for all registered shape types.

    Call this after all :func:`~xcol.shapes.register_shape` calls are done
    (including custom shapes).  If *shape_entries* is ``None``, uses the
    global registry.

    Returns:
        A stateless :class:`Collider` with compiled kernels.
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

    # ===================================================================
    # GJK distance (PhysX RefGjkEpa pattern)
    # ===================================================================

    @wp.func
    def gjk_distance(shape_a: ShapeData, shape_b: ShapeData) -> GJKResult:
        """Compute distance between two core shapes using GJK.

        Returns core-to-core distance (0 if overlapping).
        Witness points are on core surfaces (before margin adjustment).
        """
        result = GJKResult()
        result.normal = wp.vec3(0.0, 0.0, 1.0)

        # Accuracy threshold scaled to shape size (PhysX pattern)
        accuracy = wp.max(wp.length(shape_a.params) + shape_a.margin, wp.length(shape_b.params) + shape_b.margin) * 0.01
        accuracy = wp.max(accuracy, GJK_EPSILON)
        dist_eps = accuracy * 0.01

        # Simplex vertices (Minkowski diff + individual shape points)
        w0 = wp.vec3(0.0, 0.0, 0.0)
        w1 = wp.vec3(0.0, 0.0, 0.0)
        w2 = wp.vec3(0.0, 0.0, 0.0)
        w3 = wp.vec3(0.0, 0.0, 0.0)
        pa0 = wp.vec3(0.0, 0.0, 0.0)
        pa1 = wp.vec3(0.0, 0.0, 0.0)
        pa2 = wp.vec3(0.0, 0.0, 0.0)
        pa3 = wp.vec3(0.0, 0.0, 0.0)
        pb0 = wp.vec3(0.0, 0.0, 0.0)
        pb1 = wp.vec3(0.0, 0.0, 0.0)
        pb2 = wp.vec3(0.0, 0.0, 0.0)
        pb3 = wp.vec3(0.0, 0.0, 0.0)
        num_verts = int(0)

        # Best simplex snapshot (PhysX mClosest pattern)
        best_v = wp.vec3(1.0e30, 0.0, 0.0)
        best_w0 = wp.vec3(0.0, 0.0, 0.0)
        best_w1 = wp.vec3(0.0, 0.0, 0.0)
        best_w2 = wp.vec3(0.0, 0.0, 0.0)
        best_pa0 = wp.vec3(0.0, 0.0, 0.0)
        best_pa1 = wp.vec3(0.0, 0.0, 0.0)
        best_pa2 = wp.vec3(0.0, 0.0, 0.0)
        best_pb0 = wp.vec3(0.0, 0.0, 0.0)
        best_pb1 = wp.vec3(0.0, 0.0, 0.0)
        best_pb2 = wp.vec3(0.0, 0.0, 0.0)
        best_num_verts = int(0)

        # Add first support point
        direction = shape_b.pos - shape_a.pos
        if wp.length_sq(direction) < GJK_EPSILON:
            direction = wp.vec3(1.0, 0.0, 0.0)
        direction = wp.normalize(direction)
        sa = support_world(shape_a, direction)
        sb = support_world(shape_b, -direction)
        w0 = sa - sb
        pa0 = sa
        pb0 = sb
        num_verts = int(1)

        for _iter in range(GJK_MAX_ITERATIONS):
            # === computeClosest: reduce simplex, compute closest point ===
            v = w0  # default for num_verts==1

            if num_verts == 1:
                pass  # v = w0, nothing to reduce
            elif num_verts == 2:
                pt, la, lb = closest_segment(w0, w1)
                v = pt
                if la < GJK_EPSILON:
                    w0 = w1
                    pa0 = pa1
                    pb0 = pb1
                    num_verts = int(1)
                elif lb < GJK_EPSILON:
                    num_verts = int(1)
                # else: keep 2 verts
            elif num_verts == 3:
                pt, u, bv, bw = closest_triangle(w0, w1, w2)
                v = pt
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
                # else: keep 3 verts
            elif num_verts == 4:
                # Tetrahedron: check if origin is inside
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
                    # Core overlap — return 0 distance
                    result.distance = 0.0
                    result.point_a = lam0 * pa0 + lam1 * pa1 + lam2 * pa2 + lam3 * pa3
                    result.point_b = lam0 * pb0 + lam1 * pb1 + lam2 * pb2 + lam3 * pb3
                    if wp.length(best_v) > GJK_EPSILON:
                        result.normal = -wp.normalize(best_v)
                    return result
                # Not inside: find closest face and reduce to triangle
                face_best_dist = float(1.0e30)
                face_best_v = wp.vec3(0.0, 0.0, 0.0)
                face_best = int(0)
                pt, _u, _bv, _bw = closest_triangle(w1, w2, w3)
                fd = wp.length_sq(pt)
                if fd < face_best_dist:
                    face_best_dist = fd
                    face_best_v = pt
                    face_best = int(0)
                pt, _u, _bv, _bw = closest_triangle(w0, w2, w3)
                fd = wp.length_sq(pt)
                if fd < face_best_dist:
                    face_best_dist = fd
                    face_best_v = pt
                    face_best = int(1)
                pt, _u, _bv, _bw = closest_triangle(w0, w1, w3)
                fd = wp.length_sq(pt)
                if fd < face_best_dist:
                    face_best_dist = fd
                    face_best_v = pt
                    face_best = int(2)
                pt, _u, _bv, _bw = closest_triangle(w0, w1, w2)
                fd = wp.length_sq(pt)
                if fd < face_best_dist:
                    face_best_dist = fd
                    face_best_v = pt
                    face_best = int(3)
                v = face_best_v
                if face_best == 0:
                    w0 = w1
                    pa0 = pa1
                    pb0 = pb1
                    w1 = w2
                    pa1 = pa2
                    pb1 = pb2
                    w2 = w3
                    pa2 = pa3
                    pb2 = pb3
                elif face_best == 1:
                    w1 = w2
                    pa1 = pa2
                    pb1 = pb2
                    w2 = w3
                    pa2 = pa3
                    pb2 = pb3
                elif face_best == 2:
                    w2 = w3
                    pa2 = pa3
                    pb2 = pb3
                num_verts = int(3)

            # === Best-result gate (PhysX mClosest pattern) ===
            v_mag = wp.length(v)
            best_mag = wp.length(best_v)
            if v_mag < best_mag - GJK_EPSILON:
                # Improved — snapshot simplex
                best_v = v
                best_w0 = w0
                best_w1 = w1
                best_w2 = w2
                best_pa0 = pa0
                best_pa1 = pa1
                best_pa2 = pa2
                best_pb0 = pb0
                best_pb1 = pb1
                best_pb2 = pb2
                best_num_verts = num_verts
            else:
                # No improvement — discard last point
                num_verts = num_verts - 1

            # === Convergence check ===
            closest_dist = wp.length(best_v)
            if closest_dist < dist_eps:
                result.distance = 0.0
                result.point_a = best_pa0
                result.point_b = best_pb0
                if closest_dist > GJK_EPSILON:
                    result.normal = -best_v / closest_dist
                return result

            # === Add new support point ===
            search_dir = -best_v / closest_dist
            sa = support_world(shape_a, search_dir)
            sb = support_world(shape_b, -search_dir)
            w_new = sa - sb
            proj = wp.dot(w_new, -search_dir)  # PhysX: p.dot(-dir)

            if proj >= closest_dist - dist_eps:
                break  # Converged — no improvement possible

            # Restore best simplex and add new point
            w0 = best_w0
            w1 = best_w1
            w2 = best_w2
            pa0 = best_pa0
            pa1 = best_pa1
            pa2 = best_pa2
            pb0 = best_pb0
            pb1 = best_pb1
            pb2 = best_pb2
            num_verts = best_num_verts

            if num_verts == 1:
                w1 = w_new
                pa1 = sa
                pb1 = sb
            elif num_verts == 2:
                w2 = w_new
                pa2 = sa
                pb2 = sb
            elif num_verts == 3:
                w3 = w_new
                pa3 = sa
                pb3 = sb
            num_verts = num_verts + 1

        # === Final: compute witness points from best simplex ===
        if best_num_verts == 1:
            result.point_a = best_pa0
            result.point_b = best_pb0
        elif best_num_verts == 2:
            pt, la, lb = closest_segment(best_w0, best_w1)
            result.point_a = la * best_pa0 + lb * best_pa1
            result.point_b = la * best_pb0 + lb * best_pb1
        else:
            pt, u, bv, bw = closest_triangle(best_w0, best_w1, best_w2)
            result.point_a = u * best_pa0 + bv * best_pa1 + bw * best_pa2
            result.point_b = u * best_pb0 + bv * best_pb1 + bw * best_pb2

        core_dist = wp.length(best_v)
        if core_dist > GJK_EPSILON:
            result.normal = -best_v / core_dist
        result.distance = core_dist
        return result

    # ===================================================================
    # EPA depth (PhysX RefGjkEpa pattern)
    # ===================================================================

    @wp.func
    def epa_depth(shape_a: ShapeData, shape_b: ShapeData) -> GJKResult:
        """Find minimum penetration depth for overlapping core shapes.

        Builds polytope from scratch (3 support directions), then expands.
        Tracks best result across iterations (never degrades).
        """
        result = GJKResult()
        result.normal = wp.vec3(0.0, 0.0, 1.0)

        accuracy = wp.max(wp.length(shape_a.params) + shape_a.margin, wp.length(shape_b.params) + shape_b.margin) * 0.01
        accuracy = wp.max(accuracy, GJK_EPSILON)
        dist_eps = accuracy * 0.01

        verts = EPAVerts()
        verts_a = EPAVertsA()
        verts_b = EPAVertsB()
        faces = EPAFaces()
        face_nx = EPAVerts()  # reuse: normal per face
        face_dv = EPAVertsA()  # reuse: d value in column 0
        num_verts = int(0)
        num_faces = int(0)

        # Best result tracking (PhysX mBest/mProj)
        best_proj = float(-1.0e30)
        best_face = int(0)
        best_n = wp.vec3(0.0, 0.0, 1.0)
        best_d = float(-1.0e30)

        # === Build initial polytope from 3 support directions ===
        dir0 = shape_b.pos - shape_a.pos
        if wp.length(dir0) < GJK_EPSILON:
            dir0 = wp.vec3(1.0, 0.0, 0.0)
        dir0 = wp.normalize(dir0)

        # Orthogonal direction
        dir1 = wp.cross(dir0, wp.vec3(0.0, 0.0, 1.0))
        if wp.length(dir1) < GJK_EPSILON:
            dir1 = wp.cross(dir0, wp.vec3(0.0, 1.0, 0.0))
        dir1 = wp.normalize(dir1)

        # Point 0: along dir0
        sa0 = support_world(shape_a, dir0)
        sb0 = support_world(shape_b, -dir0)
        verts[0] = sa0 - sb0
        verts_a[0] = sa0
        verts_b[0] = sb0
        num_verts = int(1)

        # Point 1: along -dir0
        sa1 = support_world(shape_a, -dir0)
        sb1 = support_world(shape_b, dir0)
        p1 = sa1 - sb1
        v0_pos = wp.vec3(verts[0][0], verts[0][1], verts[0][2])
        if wp.length_sq(p1 - v0_pos) > accuracy * accuracy:
            verts[1] = p1
            verts_a[1] = sa1
            verts_b[1] = sb1
            num_verts = int(2)

        # Point 2: along dir1 (or -dir1 if duplicate)
        if num_verts >= 2:
            sa2 = support_world(shape_a, dir1)
            sb2 = support_world(shape_b, -dir1)
            p2 = sa2 - sb2
            dup2 = int(0)
            for vi in range(2):
                vv = wp.vec3(verts[vi][0], verts[vi][1], verts[vi][2])
                if wp.length_sq(p2 - vv) < accuracy * accuracy:
                    dup2 = int(1)
            if dup2 == 0:
                verts[2] = p2
                verts_a[2] = sa2
                verts_b[2] = sb2
                num_verts = int(3)
        if num_verts < 3:
            sa2 = support_world(shape_a, -dir1)
            sb2 = support_world(shape_b, dir1)
            p2 = sa2 - sb2
            dup2 = int(0)
            for vi in range(2):
                vv = wp.vec3(verts[vi][0], verts[vi][1], verts[vi][2])
                if wp.length_sq(p2 - vv) < accuracy * accuracy:
                    dup2 = int(1)
            if dup2 == 0:
                verts[num_verts] = p2
                verts_a[num_verts] = sa2
                verts_b[num_verts] = sb2
                num_verts = num_verts + 1

        if num_verts < 3:
            return result  # Degenerate

        # Create 2 initial faces with planes (triple cross for robust normal)
        va0 = wp.vec3(verts[0][0], verts[0][1], verts[0][2])
        va1 = wp.vec3(verts[1][0], verts[1][1], verts[1][2])
        va2 = wp.vec3(verts[2][0], verts[2][1], verts[2][2])

        abc0 = wp.cross(va1 - va0, va2 - va0) + wp.cross(va2 - va1, va0 - va1) + wp.cross(va0 - va2, va1 - va2)
        len0 = wp.length(abc0)
        if len0 < GJK_EPSILON:
            return result
        n0 = abc0 / len0
        d0 = wp.dot(n0, -va0)
        faces[0] = wp.vec3(0.0, 1.0, 2.0)
        face_nx[0] = n0
        face_dv[0] = wp.vec3(d0, 0.0, 0.0)

        abc1 = wp.cross(va0 - va1, va2 - va1) + wp.cross(va2 - va0, va1 - va0) + wp.cross(va1 - va2, va0 - va2)
        n1 = abc1 / wp.max(wp.length(abc1), GJK_EPSILON)
        d1_val = wp.dot(n1, -va1)
        faces[1] = wp.vec3(1.0, 0.0, 2.0)
        face_nx[1] = n1
        face_dv[1] = wp.vec3(d1_val, 0.0, 0.0)
        num_faces = int(2)

        # Initialize best from first 2 faces
        if d0 > d1_val:
            best_d = d0
            best_n = n0
            best_face = int(0)
        else:
            best_d = d1_val
            best_n = n1
            best_face = int(1)

        # === Main EPA loop (done flag, no break) ===
        done = int(0)
        for _epa_iter in range(EPA_MAX_ITERATIONS):
            if done == 1:
                continue

            # Find closest face (largest d)
            closest_idx = int(0)
            closest_w = float(-1.0e30)
            for fi in range(EPA_MAX_FACES):
                if fi >= num_faces:
                    continue
                w = face_dv[fi][0]
                if w > closest_w:
                    closest_w = w
                    closest_idx = fi

            closest_n = wp.vec3(face_nx[closest_idx][0], face_nx[closest_idx][1], face_nx[closest_idx][2])

            # Get new support point
            sa_new = support_world(shape_a, closest_n)
            sb_new = support_world(shape_b, -closest_n)
            p_new = sa_new - sb_new
            proj = wp.dot(p_new, -closest_n)

            # Track best (PhysX mProj/mBest ratchet)
            if proj > best_proj:
                best_proj = proj
                best_n = closest_n
                best_d = closest_w
                best_face = closest_idx

            # Duplicate check
            is_dup = int(0)
            for vi in range(EPA_MAX_VERTS):
                if vi >= num_verts:
                    continue
                vv = wp.vec3(verts[vi][0], verts[vi][1], verts[vi][2])
                if wp.length_sq(p_new - vv) < accuracy * accuracy:
                    is_dup = int(1)
            if is_dup == 1:
                done = int(1)
                continue

            # Convergence
            if proj >= closest_w - dist_eps:
                done = int(1)
                continue

            if num_verts >= EPA_MAX_VERTS:
                done = int(1)
                continue

            # Add new vertex
            new_vi = num_verts
            verts[new_vi] = p_new
            verts_a[new_vi] = sa_new
            verts_b[new_vi] = sb_new
            num_verts = num_verts + 1

            # Remove visible faces, collect horizon edges
            edges = EPAFaces()
            num_edges = int(0)
            new_num_faces = int(0)

            for fi in range(EPA_MAX_FACES):
                if fi >= num_faces:
                    continue
                fn = wp.vec3(face_nx[fi][0], face_nx[fi][1], face_nx[fi][2])
                fd = face_dv[fi][0]
                plane_dist = wp.dot(fn, p_new) + fd

                if plane_dist > dist_eps:
                    # Visible — remove, collect edges
                    fv = faces[fi]
                    fv0 = int(fv[0])
                    fv1 = int(fv[1])
                    fv2 = int(fv[2])
                    for ei_new in range(3):
                        if ei_new == 0:
                            ev0 = fv0
                            ev1 = fv1
                        elif ei_new == 1:
                            ev0 = fv1
                            ev1 = fv2
                        else:
                            ev0 = fv2
                            ev1 = fv0
                        found = int(-1)
                        for k in range(EPA_MAX_FACES):
                            if k >= num_edges:
                                continue
                            if int(edges[k][0]) == ev1 and int(edges[k][1]) == ev0:
                                found = k
                        if found >= 0:
                            edges[found] = edges[num_edges - 1]
                            num_edges = num_edges - 1
                        else:
                            if num_edges < EPA_MAX_FACES:
                                edges[num_edges] = wp.vec3(float(ev0), float(ev1), 0.0)
                                num_edges = num_edges + 1
                else:
                    if new_num_faces != fi:
                        faces[new_num_faces] = faces[fi]
                        face_nx[new_num_faces] = face_nx[fi]
                        face_dv[new_num_faces] = face_dv[fi]
                    new_num_faces = new_num_faces + 1

            num_faces = new_num_faces

            if num_edges == 0 or num_faces + num_edges > EPA_MAX_FACES:
                done = int(1)
                continue

            # Create new faces from horizon edges + new vertex
            for ei in range(EPA_MAX_FACES):
                if ei >= num_edges:
                    continue
                if num_faces >= EPA_MAX_FACES:
                    continue
                ev0 = int(edges[ei][0])
                ev1 = int(edges[ei][1])
                va_e = wp.vec3(verts[ev0][0], verts[ev0][1], verts[ev0][2])
                vb_e = wp.vec3(verts[ev1][0], verts[ev1][1], verts[ev1][2])
                vc_e = wp.vec3(verts[new_vi][0], verts[new_vi][1], verts[new_vi][2])
                # Triple cross for robust normal
                abc_e = (
                    wp.cross(vb_e - va_e, vc_e - va_e)
                    + wp.cross(vc_e - vb_e, va_e - vb_e)
                    + wp.cross(va_e - vc_e, vb_e - vc_e)
                )
                len_e = wp.length(abc_e)
                if len_e < GJK_EPSILON:
                    done = int(1)
                    continue
                n_e = abc_e / len_e
                d_e = wp.dot(n_e, -va_e)
                faces[num_faces] = wp.vec3(float(ev0), float(ev1), float(new_vi))
                face_nx[num_faces] = n_e
                face_dv[num_faces] = wp.vec3(d_e, 0.0, 0.0)
                num_faces = num_faces + 1

        # === Extract result from best face ===
        result.distance = -best_d  # positive penetration depth
        result.normal = best_n

        bi0 = int(faces[best_face][0])
        bi1 = int(faces[best_face][1])
        bi2 = int(faces[best_face][2])
        fa = wp.vec3(verts[bi0][0], verts[bi0][1], verts[bi0][2])
        fb = wp.vec3(verts[bi1][0], verts[bi1][1], verts[bi1][2])
        fc = wp.vec3(verts[bi2][0], verts[bi2][1], verts[bi2][2])

        abc_best = wp.cross(fb - fa, fc - fa)
        iabc2 = 1.0 / wp.max(wp.dot(abc_best, abc_best), GJK_EPSILON)
        pabc = abc_best * wp.dot(abc_best, fa) * iabc2
        tbc = wp.dot(abc_best, wp.cross(fb - pabc, fc - pabc))
        tca = wp.dot(abc_best, wp.cross(fc - pabc, fa - pabc))
        sa_w = tbc * iabc2
        sb_w = tca * iabc2
        sc_w = 1.0 - sa_w - sb_w

        result.point_a = (
            wp.vec3(verts_a[bi0][0], verts_a[bi0][1], verts_a[bi0][2]) * sa_w
            + wp.vec3(verts_a[bi1][0], verts_a[bi1][1], verts_a[bi1][2]) * sb_w
            + wp.vec3(verts_a[bi2][0], verts_a[bi2][1], verts_a[bi2][2]) * sc_w
        )
        result.point_b = (
            wp.vec3(verts_b[bi0][0], verts_b[bi0][1], verts_b[bi0][2]) * sa_w
            + wp.vec3(verts_b[bi1][0], verts_b[bi1][1], verts_b[bi1][2]) * sb_w
            + wp.vec3(verts_b[bi2][0], verts_b[bi2][1], verts_b[bi2][2]) * sc_w
        )
        return result

    # ===================================================================
    # Combined GJK + EPA
    # ===================================================================

    @wp.func
    def gjk_epa(shape_a: ShapeData, shape_b: ShapeData) -> GJKResult:
        """Compute signed distance between two shapes.

        Returns positive distance (separated) or negative (penetrating).
        Witness points are on shape surfaces (after margin adjustment).
        """
        result = gjk_distance(shape_a, shape_b)
        total_margin = shape_a.margin + shape_b.margin
        core_dist = result.distance

        if core_dist > 0.0:
            # Separated cores — apply margin
            result.distance = core_dist - total_margin
            result.point_a = result.point_a + result.normal * shape_a.margin
            result.point_b = result.point_b - result.normal * shape_b.margin
        else:
            # Core overlap — run EPA for penetration depth
            epa_result = epa_depth(shape_a, shape_b)
            result.distance = -(epa_result.distance + total_margin)
            result.normal = epa_result.normal
            result.point_a = epa_result.point_a + epa_result.normal * shape_a.margin
            result.point_b = epa_result.point_b - epa_result.normal * shape_b.margin

        return result

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
        """Generate contact patch between two shapes (runs GJK/EPA internally)."""
        gjk_result = gjk_epa(shape_a, shape_b)
        if gjk_result.distance >= 0.0:
            result = ContactResult()
            result.count = 0
            return result
        return generate_contacts_from_gjk(shape_a, shape_b, gjk_result)

    @wp.func
    def generate_contacts_from_gjk(shape_a: ShapeData, shape_b: ShapeData, gjk_result: GJKResult) -> ContactResult:
        """Generate contact patch from a pre-computed GJK/EPA result."""
        result = ContactResult()
        result.count = 0

        n = gjk_result.normal
        depth = gjk_result.distance  # negative when penetrating
        result.normal = n

        face_a = contact_face_world(shape_a, n, gjk_result.point_a)
        face_b = contact_face_world(shape_b, -n, gjk_result.point_b)

        # ref_offset: projection of shape A's surface onto the contact normal.
        # Use point_a (on A's surface near the contact) rather than face
        # corners which can be far from the contact area for large shapes.
        ref_offset = wp.dot(gjk_result.point_a, n)

        if face_b.count > face_a.count:
            clipped, num_clipped = clip_face_against_face(face_a, face_b, -n)
        else:
            clipped, num_clipped = clip_face_against_face(face_b, face_a, n)

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
            pt_depth = wp.dot(n, pt) - ref_offset  # negative = penetrating
            if pt_depth > 0.0:
                pt_depth = depth  # fallback to GJK distance
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
    ):
        tid = wp.tid()
        r = gjk_distance(shapes_a[tid], shapes_b[tid])
        results_dist[tid] = r.distance
        results_point_a[tid] = r.point_a
        results_point_b[tid] = r.point_b
        results_normal[tid] = r.normal

    @wp.kernel(enable_backward=False)
    def gjk_epa_kernel(
        shapes_a: wp.array(dtype=ShapeData),
        shapes_b: wp.array(dtype=ShapeData),
        results_dist: wp.array(dtype=float),
        results_point_a: wp.array(dtype=wp.vec3),
        results_point_b: wp.array(dtype=wp.vec3),
        results_normal: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        r = gjk_epa(shapes_a[tid], shapes_b[tid])
        results_dist[tid] = r.distance
        results_point_a[tid] = r.point_a
        results_point_b[tid] = r.point_b
        results_normal[tid] = r.normal

    @wp.kernel(enable_backward=False)
    def generate_contacts_kernel(
        shapes_a: wp.array(dtype=ShapeData),
        shapes_b: wp.array(dtype=ShapeData),
        out: wp.array(dtype=ContactResult),
    ):
        tid = wp.tid()
        out[tid] = generate_contacts(shapes_a[tid], shapes_b[tid])

    @wp.kernel(enable_backward=False)
    def collide_nxn_kernel(
        num_shapes: int,
        contact_distance: float,
        shape_types: wp.array(dtype=int),
        shape_params: wp.array(dtype=wp.vec3),
        shape_margins: wp.array(dtype=float),
        shape_transforms: wp.array(dtype=wp.transform),
        shape_worlds: wp.array(dtype=int),
        # Outputs (flat SoA, atomic counter)
        out_count: wp.array(dtype=int),
        out_shape_a: wp.array(dtype=int),
        out_shape_b: wp.array(dtype=int),
        out_point: wp.array(dtype=wp.vec3),
        out_normal: wp.array(dtype=wp.vec3),
        out_depth: wp.array(dtype=float),
    ):
        """NxN broadphase + narrowphase + flatten in one kernel.

        Thread tid maps to pair (i, j) where i < j, using triangular indexing.
        """
        tid = wp.tid()
        # Map linear tid to (i, j) pair where i < j
        # tid = j*(j-1)/2 + i  =>  j = floor((1+sqrt(1+8*tid))/2), i = tid - j*(j-1)/2
        j = int(wp.floor(0.5 + wp.sqrt(0.25 + 2.0 * float(tid))))
        i = tid - j * (j - 1) / 2
        if i < 0 or i >= j or j >= num_shapes:
            return

        # World filter
        wi = shape_worlds[i]
        wj = shape_worlds[j]
        if wi != wj and wi != -1 and wj != -1:
            return

        # Build ShapeData from SoA
        ta = shape_transforms[i]
        tb = shape_transforms[j]

        sa = ShapeData()
        sa.shape_type = shape_types[i]
        sa.pos = wp.transform_get_translation(ta)
        sa.rot = wp.transform_get_rotation(ta)
        sa.params = shape_params[i]
        sa.margin = shape_margins[i]

        sb = ShapeData()
        sb.shape_type = shape_types[j]
        sb.pos = wp.transform_get_translation(tb)
        sb.rot = wp.transform_get_rotation(tb)
        sb.params = shape_params[j]
        sb.margin = shape_margins[j]

        # Narrowphase
        gjk_result = gjk_epa(sa, sb)

        # Generate full contact patch if within contact_distance
        if gjk_result.distance < contact_distance:
            r = generate_contacts_from_gjk(sa, sb, gjk_result)
            for ci in range(4):
                if ci >= r.count:
                    break
                idx = wp.atomic_add(out_count, 0, 1)
                if idx < out_shape_a.shape[0]:
                    out_shape_a[idx] = i
                    out_shape_b[idx] = j
                    out_normal[idx] = r.normal
                    if ci == 0:
                        out_point[idx] = r.p0
                        out_depth[idx] = r.d0
                    elif ci == 1:
                        out_point[idx] = r.p1
                        out_depth[idx] = r.d1
                    elif ci == 2:
                        out_point[idx] = r.p2
                        out_depth[idx] = r.d2
                    elif ci == 3:
                        out_point[idx] = r.p3
                        out_depth[idx] = r.d3

    return Collider(
        support_local=support_local,
        support_world=support_world,
        contact_face_local=contact_face_local,
        contact_face_world=contact_face_world,
        get_aabb=get_aabb,
        gjk_distance=gjk_distance,
        gjk_epa=gjk_epa,
        epa_depth=epa_depth,
        generate_contacts=generate_contacts,
        support_kernel=support_kernel,
        contact_face_kernel=contact_face_kernel,
        aabb_kernel=aabb_kernel,
        gjk_kernel=gjk_kernel,
        gjk_epa_kernel=gjk_epa_kernel,
        generate_contacts_kernel=generate_contacts_kernel,
        collide_nxn_kernel=collide_nxn_kernel,
    )


class Collider:
    """Compiled collision kernels for all registered shape types.

    Created by :func:`create_collider`.  Stateless — call
    :meth:`collide` with a :class:`~xcol.model.Model` to run
    broadphase + narrowphase.  Reusable across multiple models.
    """

    def __init__(self, **funcs: Any) -> None:
        for k, v in funcs.items():
            setattr(self, k, v)

    def collide(self, model: Any, contact_distance: float = 0.0) -> None:
        """Run N*N broadphase + narrowphase and write contacts into *model*.

        The user should update ``model.shape_transforms`` before calling this.
        Results are written to ``model.contact_*`` arrays.  The contact count
        is reset to 0 at the start of each call.

        Args:
            model: A :class:`~xcol.model.Model` instance.
            contact_distance: Report contacts for shapes separated by less
                than this distance [m]. Separated contacts have negative depth.
        """
        n = model.shape_count
        num_pairs = n * (n - 1) // 2
        if num_pairs == 0:
            model.contact_count.zero_()
            return

        model.contact_count.zero_()

        wp.launch(
            self.collide_nxn_kernel,
            dim=num_pairs,
            inputs=[
                n,
                contact_distance,
                model.shape_types,
                model.shape_params,
                model.shape_margins,
                model.shape_transforms,
                model.shape_worlds,
            ],
            outputs=[
                model.contact_count,
                model.contact_shape_a,
                model.contact_shape_b,
                model.contact_point,
                model.contact_normal,
                model.contact_depth,
            ],
        )
