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

from .gjk import (
    closest_segment as gjk_closest_segment,
)
from .gjk import (
    closest_tetrahedron as gjk_closest_tetrahedron,
)
from .gjk import (
    closest_triangle as gjk_closest_triangle,
)
from .gjk import (
    simplex_get_closest,
)
from .shapes import ShapeEntry, get_registered_shapes
from .types import (
    CLIP_MAX_POINTS,
    GJK_COLLIDE_EPSILON,
    GJK_EPSILON,
    GJK_MAX_ITERATIONS,
    MPR_COLLIDE_EPSILON,
    MPR_MAX_ITERATIONS,
    ClipPlanes,
    ClipPoly,
    ContactFaceResult,
    ContactResult,
    GJKResult,
    Mat83f,
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
    # GJK distance (Jitter Physics 2 / Newton pattern)
    # ===================================================================

    @wp.func
    def minkowski_support(shape_a: ShapeData, shape_b: ShapeData, direction: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
        """Support point on Minkowski difference (A - B) in A-local frame.

        Returns (B_point, BtoA_vector) both in A-local frame.
        """
        # Support on A in local frame (A is at origin, no rotation)
        a_pt = support_local(shape_a.shape_type, shape_a.params, direction)
        # Support on B in A-local frame
        local_dir_b = wp.quat_rotate_inv(shape_b.rot, -direction)
        b_local = support_local(shape_b.shape_type, shape_b.params, local_dir_b)
        b_pt = wp.quat_rotate(shape_b.rot, b_local) + shape_b.pos
        return b_pt, a_pt - b_pt

    @wp.func
    def gjk_distance(shape_a: ShapeData, shape_b: ShapeData) -> GJKResult:
        """Compute distance between two core shapes using GJK.

        All computation in shape A's local frame (A at origin, unrotated).
        Shape B is pre-transformed to this frame by the caller.

        Returns core-to-core distance (0 if overlapping).
        Witness points are in A-local frame.
        """
        result = GJKResult()
        result.normal = wp.vec3(0.0, 0.0, 1.0)

        # Transform shape B into A-local frame
        inv_rot_a = wp.quat_inverse(shape_a.rot)
        shape_b_local = ShapeData()
        shape_b_local.shape_type = shape_b.shape_type
        shape_b_local.params = shape_b.params
        shape_b_local.margin = shape_b.margin
        shape_b_local.pos = wp.quat_rotate(inv_rot_a, shape_b.pos - shape_a.pos)
        shape_b_local.rot = inv_rot_a * shape_b.rot

        shape_a_local = ShapeData()
        shape_a_local.shape_type = shape_a.shape_type
        shape_a_local.params = shape_a.params
        shape_a_local.margin = shape_a.margin
        shape_a_local.pos = wp.vec3(0.0, 0.0, 0.0)
        shape_a_local.rot = wp.quat_identity()

        # GJK with Nesterov acceleration (Jitter Physics 2 / Newton pattern)
        simplex_v = Mat83f()
        simplex_bc = wp.vec4(0.0, 0.0, 0.0, 0.0)
        simplex_mask = wp.uint32(0)

        # Initial direction: B center - A center in A-local frame
        v = shape_b_local.pos
        if wp.length_sq(v) < GJK_EPSILON:
            v = wp.vec3(1.0, 0.0, 0.0)
        dist_sq = wp.length_sq(v)

        last_search_dir = wp.vec3(1.0, 0.0, 0.0)

        # Nesterov acceleration state
        nesterov_dir = v
        w_prev = v
        use_nesterov = bool(False)  # TODO: investigate Nesterov for asymmetric shapes
        iteration = int(0)
        iter_count = int(GJK_MAX_ITERATIONS)

        while iter_count > 0:
            iter_count -= 1

            if dist_sq < GJK_COLLIDE_EPSILON * GJK_COLLIDE_EPSILON:
                # Core overlap
                result.distance = 0.0
                result.normal = wp.vec3(0.0, 0.0, 0.0)
                pa, pb = simplex_get_closest(simplex_v, simplex_bc, simplex_mask)
                result.point_a = wp.quat_rotate(shape_a.rot, pa) + shape_a.pos
                result.point_b = wp.quat_rotate(shape_a.rot, pb) + shape_a.pos
                return result

            # Search direction with Nesterov acceleration
            used_fallback = bool(False)
            if use_nesterov and iteration >= 3:
                momentum = float(iteration + 2) / float(iteration + 3)
                y = momentum * v + (1.0 - momentum) * w_prev
                y_len = wp.length(y)
                ndir_len = wp.length(nesterov_dir)
                if y_len > GJK_EPSILON and ndir_len > GJK_EPSILON:
                    nesterov_dir = momentum * (nesterov_dir / ndir_len) + (1.0 - momentum) * (y / y_len)
                    search_dir = -nesterov_dir
                else:
                    search_dir = -v
            else:
                nesterov_dir = v
                search_dir = -v
            if wp.length_sq(search_dir) < 1.0e-12:
                search_dir = wp.vec3(1.0, 0.0, 0.0)
                used_fallback = bool(True)
            last_search_dir = search_dir

            # Get support point
            b_pt, w_btoa = minkowski_support(shape_a_local, shape_b_local, search_dir)
            w_v = w_btoa

            # Nesterov deactivation
            if use_nesterov and iteration >= 3:
                duality_gap = 2.0 * wp.dot(v, v - w_v)
                if duality_gap <= GJK_COLLIDE_EPSILON * wp.sqrt(dist_sq):
                    use_nesterov = bool(False)

            # Frank-Wolfe convergence check
            if not used_fallback:
                delta_dist = wp.dot(v, v - w_v)
                if delta_dist < GJK_COLLIDE_EPSILON * wp.sqrt(dist_sq):
                    iteration = -1  # signal: converged

            # Duplicate vertex check
            if iteration >= 0:
                is_duplicate = bool(False)
                for i in range(4):
                    if (simplex_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0):
                        if wp.length_sq(simplex_v[2 * i + 1] - w_v) < GJK_COLLIDE_EPSILON * GJK_COLLIDE_EPSILON:
                            is_duplicate = bool(True)
                if is_duplicate:
                    iteration = -1  # signal: stalled

            if iteration < 0:
                # Converged or stalled — break
                iteration = GJK_MAX_ITERATIONS  # prevent further iterations
                iter_count = 0
            else:
                # Add vertex to simplex
                use_count = int(0)
                free_slot = int(0)
                for i in range(4):
                    if (simplex_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0):
                        use_count += 1
                    else:
                        free_slot = i
                use_count += 1
                simplex_v[2 * free_slot] = b_pt
                simplex_v[2 * free_slot + 1] = w_btoa

                closest = wp.vec3(0.0, 0.0, 0.0)
                success = bool(True)

                if use_count == 1:
                    closest = simplex_v[2 * free_slot + 1]
                    simplex_mask = wp.uint32(1) << wp.uint32(free_slot)
                    simplex_bc[free_slot] = 1.0
                elif use_count == 2:
                    i0 = int(0)
                    i1 = int(0)
                    ci = int(0)
                    for i in range(4):
                        if (simplex_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0) or i == free_slot:
                            if ci == 0:
                                i0 = i
                            else:
                                i1 = i
                            ci += 1
                    closest, bc, mask = gjk_closest_segment(simplex_v, i0, i1)
                    simplex_bc = bc
                    simplex_mask = mask
                elif use_count == 3:
                    i0 = int(0)
                    i1 = int(0)
                    i2 = int(0)
                    ci = int(0)
                    for i in range(4):
                        if (simplex_mask & (wp.uint32(1) << wp.uint32(i))) != wp.uint32(0) or i == free_slot:
                            if ci == 0:
                                i0 = i
                            elif ci == 1:
                                i1 = i
                            else:
                                i2 = i
                            ci += 1
                    closest, bc, mask = gjk_closest_triangle(simplex_v, i0, i1, i2)
                    simplex_bc = bc
                    simplex_mask = mask
                elif use_count == 4:
                    closest, bc, mask = gjk_closest_tetrahedron(simplex_v)
                    simplex_bc = bc
                    simplex_mask = mask
                    inside_tetrahedron = mask == wp.uint32(15)
                    success = not inside_tetrahedron
                else:
                    success = bool(False)

                if not success:
                    # Core overlap
                    result.distance = 0.0
                    result.normal = wp.vec3(0.0, 0.0, 0.0)
                    pa, pb = simplex_get_closest(simplex_v, simplex_bc, simplex_mask)
                    result.point_a = wp.quat_rotate(shape_a.rot, pa) + shape_a.pos
                    result.point_b = wp.quat_rotate(shape_a.rot, pb) + shape_a.pos
                    return result

                v = closest
                dist_sq = wp.length_sq(v)
                w_prev = w_v
                iteration += 1

        # Converged: compute final result
        distance = wp.sqrt(dist_sq)
        pa, pb = simplex_get_closest(simplex_v, simplex_bc, simplex_mask)

        # Normal: prefer A→B vector, fallback to -v, then last_search_dir
        delta = pb - pa
        delta_len_sq = wp.length_sq(delta)
        if delta_len_sq > GJK_EPSILON * GJK_EPSILON:
            result.normal = delta * (1.0 / wp.sqrt(delta_len_sq))
        elif distance > GJK_COLLIDE_EPSILON:
            result.normal = v * (-1.0 / distance)
        else:
            nsq = wp.length_sq(last_search_dir)
            if nsq > 0.0:
                result.normal = last_search_dir * (1.0 / wp.sqrt(nsq))
            else:
                result.normal = wp.vec3(1.0, 0.0, 0.0)

        result.distance = distance
        # Transform witness points back to world frame
        result.point_a = wp.quat_rotate(shape_a.rot, pa) + shape_a.pos
        result.point_b = wp.quat_rotate(shape_a.rot, pb) + shape_a.pos
        # Normal to world frame
        result.normal = wp.quat_rotate(shape_a.rot, result.normal)
        return result

    # ===================================================================
    # MPR penetration depth (Jitter Physics 2 / XenoCollide pattern)
    # ===================================================================

    @wp.func
    def mpr_depth(shape_a: ShapeData, shape_b: ShapeData) -> GJKResult:
        """Find penetration depth and normal for overlapping core shapes.

        Uses Minkowski Portal Refinement.  Works in A-local frame.
        Returns penetration depth (positive) and witness points in world frame.
        """
        result = GJKResult()
        result.normal = wp.vec3(0.0, 0.0, 1.0)

        # Transform shape B into A-local frame
        inv_rot_a = wp.quat_inverse(shape_a.rot)
        sb = ShapeData()
        sb.shape_type = shape_b.shape_type
        sb.params = shape_b.params
        sb.margin = shape_b.margin
        sb.pos = wp.quat_rotate(inv_rot_a, shape_b.pos - shape_a.pos)
        sb.rot = inv_rot_a * shape_b.rot

        sa = ShapeData()
        sa.shape_type = shape_a.shape_type
        sa.params = shape_a.params
        sa.margin = shape_a.margin
        sa.pos = wp.vec3(0.0, 0.0, 0.0)
        sa.rot = wp.quat_identity()

        NUMERIC_EPSILON = 1.0e-16

        # v0 = geometric center of Minkowski difference
        v0_btoa = -sb.pos  # center_A(=0) - center_B
        normal = -v0_btoa
        if (
            wp.abs(normal[0]) < NUMERIC_EPSILON
            and wp.abs(normal[1]) < NUMERIC_EPSILON
            and wp.abs(normal[2]) < NUMERIC_EPSILON
        ):
            v0_btoa = wp.vec3(1.0e-5, 0.0, 0.0)
            normal = -v0_btoa

        # v1 = first support
        v1_b, v1_btoa = minkowski_support(sa, sb, normal)
        point_a = v1_b + v1_btoa  # reconstruct A point
        point_b = v1_b
        if wp.dot(v1_btoa, normal) <= 0.0:
            result.distance = 0.0
            result.point_a = wp.quat_rotate(shape_a.rot, point_a) + shape_a.pos
            result.point_b = wp.quat_rotate(shape_a.rot, point_b) + shape_a.pos
            return result

        normal = wp.cross(v1_btoa, v0_btoa)
        if wp.length_sq(normal) < NUMERIC_EPSILON * NUMERIC_EPSILON:
            # Shapes are on the same line
            normal = v1_btoa - v0_btoa
            normal = wp.normalize(normal)
            penetration = wp.dot(v1_btoa, normal)
            result.distance = penetration
            result.normal = wp.quat_rotate(shape_a.rot, normal)
            result.point_a = wp.quat_rotate(shape_a.rot, point_a) + shape_a.pos
            result.point_b = wp.quat_rotate(shape_a.rot, point_b) + shape_a.pos
            return result

        # v2 = second support
        v2_b, v2_btoa = minkowski_support(sa, sb, normal)
        if wp.dot(v2_btoa, normal) <= 0.0:
            result.distance = 0.0
            result.point_a = wp.quat_rotate(shape_a.rot, point_a) + shape_a.pos
            result.point_b = wp.quat_rotate(shape_a.rot, point_b) + shape_a.pos
            return result

        # Orient portal so origin is on negative side
        temp1 = v1_btoa - v0_btoa
        temp2 = v2_btoa - v0_btoa
        normal = wp.cross(temp1, temp2)
        dist = wp.dot(normal, v0_btoa)
        if dist > 0.0:
            # Swap v1 and v2
            tmp_b = v1_b
            tmp_btoa = v1_btoa
            v1_b = v2_b
            v1_btoa = v2_btoa
            v2_b = tmp_b
            v2_btoa = tmp_btoa
            normal = -normal

        # Phase 1: Find initial portal (triangle v1, v2, v3)
        v3_b = wp.vec3(0.0, 0.0, 0.0)
        v3_btoa = wp.vec3(0.0, 0.0, 0.0)
        portal_found = int(0)
        for _p1 in range(MPR_MAX_ITERATIONS):
            if portal_found == 0:
                v3_b, v3_btoa = minkowski_support(sa, sb, normal)
                if wp.dot(v3_btoa, normal) <= 0.0:
                    portal_found = int(-1)  # no collision
                else:
                    # If origin is outside (v1, v0, v3), eliminate v2
                    temp1 = wp.cross(v1_btoa, v3_btoa)
                    if wp.dot(temp1, v0_btoa) < 0.0:
                        v2_b = v3_b
                        v2_btoa = v3_btoa
                        temp1 = v1_btoa - v0_btoa
                        temp2 = v3_btoa - v0_btoa
                        normal = wp.cross(temp1, temp2)
                    else:
                        # If origin is outside (v3, v0, v2), eliminate v1
                        temp1 = wp.cross(v3_btoa, v2_btoa)
                        if wp.dot(temp1, v0_btoa) < 0.0:
                            v1_b = v3_b
                            v1_btoa = v3_btoa
                            temp1 = v3_btoa - v0_btoa
                            temp2 = v2_btoa - v0_btoa
                            normal = wp.cross(temp1, temp2)
                        else:
                            # Portal found
                            portal_found = int(1)

        if portal_found != 1:
            result.distance = 0.0
            result.point_a = wp.quat_rotate(shape_a.rot, point_a) + shape_a.pos
            result.point_b = wp.quat_rotate(shape_a.rot, point_b) + shape_a.pos
            return result

        # Phase 2: Refine the portal
        hit = bool(False)
        penetration = float(0.0)
        v4_b = wp.vec3(0.0, 0.0, 0.0)
        v4_btoa = wp.vec3(0.0, 0.0, 0.0)
        for _p2 in range(MPR_MAX_ITERATIONS):
            # Compute normal of wedge face (v1, v2, v3)
            temp1 = v2_btoa - v1_btoa
            temp2 = v3_btoa - v1_btoa
            normal = wp.cross(temp1, temp2)
            normal_sq = wp.length_sq(normal)

            if normal_sq < NUMERIC_EPSILON * NUMERIC_EPSILON:
                result.distance = 0.0
                result.point_a = wp.quat_rotate(shape_a.rot, point_a) + shape_a.pos
                result.point_b = wp.quat_rotate(shape_a.rot, point_b) + shape_a.pos
                return result

            if not hit:
                d = wp.dot(normal, v1_btoa)
                hit = d >= 0.0

            v4_b, v4_btoa = minkowski_support(sa, sb, normal)

            temp3 = v4_btoa - v3_btoa
            delta = wp.dot(temp3, normal)
            penetration = wp.dot(v4_btoa, normal)

            if delta * delta <= MPR_COLLIDE_EPSILON * MPR_COLLIDE_EPSILON * normal_sq or penetration <= 0.0:
                if hit:
                    inv_normal = 1.0 / wp.sqrt(normal_sq)
                    penetration = penetration * inv_normal
                    normal = normal * inv_normal

                    # Barycentric interpolation for witness points
                    temp3 = wp.cross(v1_btoa, temp1)
                    gamma = wp.dot(temp3, normal) * inv_normal
                    temp3 = wp.cross(temp2, v1_btoa)
                    beta = wp.dot(temp3, normal) * inv_normal
                    alpha = 1.0 - gamma - beta

                    point_a = alpha * (v1_b + v1_btoa) + beta * (v2_b + v2_btoa) + gamma * (v3_b + v3_btoa)
                    point_b = alpha * v1_b + beta * v2_b + gamma * v3_b

                    result.distance = penetration
                    result.normal = wp.quat_rotate(shape_a.rot, normal)
                    result.point_a = wp.quat_rotate(shape_a.rot, point_a) + shape_a.pos
                    result.point_b = wp.quat_rotate(shape_a.rot, point_b) + shape_a.pos
                    return result
                else:
                    result.distance = 0.0
                    result.point_a = wp.quat_rotate(shape_a.rot, point_a) + shape_a.pos
                    result.point_b = wp.quat_rotate(shape_a.rot, point_b) + shape_a.pos
                    return result

            # Determine which region of the wedge the origin is in
            temp1 = wp.cross(v4_btoa, v0_btoa)
            dot_val = wp.dot(temp1, v1_btoa)
            if dot_val >= 0.0:
                dot_val = wp.dot(temp1, v2_btoa)
                if dot_val >= 0.0:
                    v1_b = v4_b
                    v1_btoa = v4_btoa
                else:
                    v3_b = v4_b
                    v3_btoa = v4_btoa
            else:
                dot_val = wp.dot(temp1, v3_btoa)
                if dot_val >= 0.0:
                    v2_b = v4_b
                    v2_btoa = v4_btoa
                else:
                    v1_b = v4_b
                    v1_btoa = v4_btoa

        # Max iterations reached
        result.distance = 0.0
        result.point_a = wp.quat_rotate(shape_a.rot, point_a) + shape_a.pos
        result.point_b = wp.quat_rotate(shape_a.rot, point_b) + shape_a.pos
        return result

    # ===================================================================
    # Combined GJK + MPR
    # ===================================================================

    @wp.func
    def gjk_mpr(shape_a: ShapeData, shape_b: ShapeData) -> GJKResult:
        """Compute signed distance between two shapes.

        MPR first (handles penetration, exits early for separated).
        GJK fallback for accurate separation distance and normal.
        Returns positive distance (separated) or negative (penetrating).
        Witness points are on shape surfaces (after margin adjustment).
        """
        total_margin = shape_a.margin + shape_b.margin

        # MPR first — handles penetration directly
        mpr_result = mpr_depth(shape_a, shape_b)
        if mpr_result.distance > 0.0:
            # MPR found collision
            result = GJKResult()
            result.distance = -(mpr_result.distance + total_margin)
            result.normal = mpr_result.normal
            result.point_a = mpr_result.point_a + mpr_result.normal * shape_a.margin
            result.point_b = mpr_result.point_b - mpr_result.normal * shape_b.margin
            return result

        # MPR says no collision — GJK for accurate separation distance
        result = gjk_distance(shape_a, shape_b)
        core_dist = result.distance
        result.distance = core_dist - total_margin
        if core_dist > 0.0:
            result.point_a = result.point_a + result.normal * shape_a.margin
            result.point_b = result.point_b - result.normal * shape_b.margin

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

            pi_xyz = wp.vec3(poly[i][0], poly[i][1], poly[i][2])
            pj_xyz = wp.vec3(poly[j][0], poly[j][1], poly[j][2])
            di = wp.dot(plane_n, pi_xyz) - plane_d
            dj = wp.dot(plane_n, pj_xyz) - plane_d

            if di <= 0.0:
                if out_count < CLIP_MAX_POINTS:
                    out[out_count] = poly[i]
                    out_count = out_count + 1
                if dj > 0.0:
                    denom = di - dj
                    if wp.abs(denom) < 1.0e-10:
                        denom = 1.0e-10
                    t = di / denom
                    ix = pi_xyz[0] + t * (pj_xyz[0] - pi_xyz[0])
                    iy = pi_xyz[1] + t * (pj_xyz[1] - pi_xyz[1])
                    iz = pi_xyz[2] + t * (pj_xyz[2] - pi_xyz[2])
                    iw = poly[i][3] + t * (poly[j][3] - poly[i][3])
                    if out_count < CLIP_MAX_POINTS:
                        out[out_count] = wp.vec4(ix, iy, iz, iw)
                        out_count = out_count + 1
            elif dj <= 0.0:
                denom = di - dj
                if wp.abs(denom) < 1.0e-10:
                    denom = 1.0e-10
                t = di / denom
                ix = pi_xyz[0] + t * (pj_xyz[0] - pi_xyz[0])
                iy = pi_xyz[1] + t * (pj_xyz[1] - pi_xyz[1])
                iz = pi_xyz[2] + t * (pj_xyz[2] - pi_xyz[2])
                iw = poly[i][3] + t * (poly[j][3] - poly[i][3])
                if out_count < CLIP_MAX_POINTS:
                    out[out_count] = wp.vec4(ix, iy, iz, iw)
                    out_count = out_count + 1

        return out, out_count

    @wp.func
    def get_face_point(face: ContactFaceResult, idx: int) -> wp.vec3:
        if idx == 0:
            return face.p0
        elif idx == 1:
            return face.p1
        elif idx == 2:
            return face.p2
        return face.p3

    @wp.func
    def make_face_clip_planes(
        face: ContactFaceResult,
        axis: wp.vec3,
        planes: ClipPlanes,
        num_planes: int,
    ) -> tuple[ClipPlanes, int]:
        """Build clip planes from a face's edges extending along the axis."""
        fc = face.count
        eps = 1.0e-6
        for ei in range(4):
            if ei >= fc:
                continue
            if num_planes >= CLIP_MAX_POINTS:
                continue
            ej = ei + 1
            if ej >= fc:
                ej = int(0)
            s = get_face_point(face, ei)
            e = get_face_point(face, ej)
            n = wp.cross(e - s, axis)
            n_len = wp.length(n)
            if n_len < eps:
                continue
            n = n / n_len
            pos_count = int(0)
            neg_count = int(0)
            for ki in range(4):
                if ki >= fc:
                    continue
                dd = wp.dot(get_face_point(face, ki) - s, n)
                if dd > eps:
                    pos_count = pos_count + 1
                elif dd < -eps:
                    neg_count = neg_count + 1
            if pos_count > 0 and neg_count > 0:
                continue
            if pos_count > 0:
                n = -n
            planes[num_planes] = wp.vec4(n[0], n[1], n[2], wp.dot(n, s))
            num_planes = num_planes + 1
        return planes, num_planes

    @wp.func
    def generate_contact_patch(
        shape_a: ShapeData,
        shape_b: ShapeData,
        gjk_result: GJKResult,
    ) -> ContactResult:
        """Generate contact patch — single function, all clip paths.

        PhysX FaceClipper pattern: clipNone / clip2x2 / clipNxN.
        All paths produce (midpoint, depth) via face plane projection.
        ClipPoly stores (x, y, z, depth) per point.
        """
        result = ContactResult()
        result.p0 = (gjk_result.point_a + gjk_result.point_b) * 0.5
        result.d0 = gjk_result.distance
        result.normal = gjk_result.normal
        result.count = 1
        return result

        # TODO: restore contact generation after GJK/MPR is fixed
        axis = gjk_result.normal
        result.normal = axis

        face_a = contact_face_world(shape_a, axis, gjk_result.point_a)
        face_b = contact_face_world(shape_b, -axis, gjk_result.point_b)

        na = face_a.normal
        nb = face_b.normal
        da_val = wp.dot(na, face_a.p0)
        db_val = wp.dot(nb, face_b.p0)
        denom_a = wp.dot(na, axis)
        denom_b = wp.dot(nb, axis)
        if wp.abs(denom_a) < 1.0e-10:
            denom_a = 1.0e-10
        if wp.abs(denom_b) < 1.0e-10:
            denom_b = 1.0e-10

        poly = ClipPoly()
        num_pts = int(0)

        # === clipNone ===
        if face_a.count < 2 or face_b.count < 2:
            mid = (gjk_result.point_a + gjk_result.point_b) * 0.5
            result.p0 = mid
            result.d0 = gjk_result.distance
            result.count = 1
            return result

        # === clip2x2 (edge-edge) ===
        if face_a.count == 2 and face_b.count == 2:
            a0 = face_a.p0
            a1 = face_a.p1
            b0 = face_b.p0
            b1 = face_b.p1
            seg_ab = a1 - a0
            seg_cd = b1 - b0
            len_ab_sq = wp.dot(seg_ab, seg_ab)
            len_cd_sq = wp.dot(seg_cd, seg_cd)
            eps = 1.0e-5
            t_val = wp.dot(b0 - a0, seg_ab)
            if t_val > -eps * len_ab_sq and t_val < len_ab_sq * (1.0 + eps) and len_ab_sq > eps:
                proj = a0 + seg_ab * (t_val / len_ab_sq)
                mid = (b0 + proj) * 0.5
                d = wp.dot(axis, proj - b0)
                poly[num_pts] = wp.vec4(mid[0], mid[1], mid[2], d)
                num_pts = num_pts + 1
            t_val = wp.dot(b1 - a0, seg_ab)
            if t_val > -eps * len_ab_sq and t_val < len_ab_sq * (1.0 + eps) and len_ab_sq > eps:
                proj = a0 + seg_ab * (t_val / len_ab_sq)
                mid = (b1 + proj) * 0.5
                d = wp.dot(axis, proj - b1)
                poly[num_pts] = wp.vec4(mid[0], mid[1], mid[2], d)
                num_pts = num_pts + 1
            t_val = wp.dot(a0 - b0, seg_cd)
            if t_val > -eps * len_cd_sq and t_val < len_cd_sq * (1.0 + eps) and len_cd_sq > eps:
                proj = b0 + seg_cd * (t_val / len_cd_sq)
                mid = (a0 + proj) * 0.5
                d = wp.dot(axis, proj - a0)
                is_dup = int(0)
                for di_idx in range(CLIP_MAX_POINTS):
                    if di_idx >= num_pts:
                        continue
                    pv = wp.vec3(poly[di_idx][0], poly[di_idx][1], poly[di_idx][2])
                    if wp.length_sq(mid - pv) < eps:
                        is_dup = int(1)
                if is_dup == 0:
                    poly[num_pts] = wp.vec4(mid[0], mid[1], mid[2], d)
                    num_pts = num_pts + 1
            t_val = wp.dot(a1 - b0, seg_cd)
            if t_val > -eps * len_cd_sq and t_val < len_cd_sq * (1.0 + eps) and len_cd_sq > eps:
                proj = b0 + seg_cd * (t_val / len_cd_sq)
                mid = (a1 + proj) * 0.5
                d = wp.dot(axis, proj - a1)
                is_dup = int(0)
                for di_idx in range(CLIP_MAX_POINTS):
                    if di_idx >= num_pts:
                        continue
                    pv = wp.vec3(poly[di_idx][0], poly[di_idx][1], poly[di_idx][2])
                    if wp.length_sq(mid - pv) < eps:
                        is_dup = int(1)
                if is_dup == 0:
                    poly[num_pts] = wp.vec4(mid[0], mid[1], mid[2], d)
                    num_pts = num_pts + 1
        else:
            # === clipNxN (2xN, Nx2, NxN) ===
            planes = ClipPlanes()
            num_planes = int(0)
            planes, num_planes = make_face_clip_planes(face_a, axis, planes, num_planes)
            planes, num_planes = make_face_clip_planes(face_b, axis, planes, num_planes)

            if face_a.count == 2:
                poly[0] = wp.vec4(face_a.p0[0], face_a.p0[1], face_a.p0[2], 0.0)
                poly[1] = wp.vec4(face_a.p1[0], face_a.p1[1], face_a.p1[2], 0.0)
                num_pts = int(2)
            elif face_b.count == 2:
                poly[0] = wp.vec4(face_b.p0[0], face_b.p0[1], face_b.p0[2], 0.0)
                poly[1] = wp.vec4(face_b.p1[0], face_b.p1[1], face_b.p1[2], 0.0)
                num_pts = int(2)
            else:
                # Bounding quad (PhysX makePolygon)
                u = wp.cross(axis, wp.vec3(0.0, 0.0, 1.0))
                if wp.length(u) < 1.0e-6:
                    u = wp.cross(axis, wp.vec3(0.0, 1.0, 0.0))
                u = wp.normalize(u)
                bv = wp.cross(axis, u)
                min_u = float(1.0e30)
                max_u = float(-1.0e30)
                min_v = float(1.0e30)
                max_v = float(-1.0e30)
                for fi in range(4):
                    if fi < face_a.count:
                        pa = get_face_point(face_a, fi)
                        pu = wp.dot(pa, u)
                        pv_val = wp.dot(pa, bv)
                        min_u = wp.min(min_u, pu)
                        max_u = wp.max(max_u, pu)
                        min_v = wp.min(min_v, pv_val)
                        max_v = wp.max(max_v, pv_val)
                    if fi < face_b.count:
                        pb = get_face_point(face_b, fi)
                        pu = wp.dot(pb, u)
                        pv_val = wp.dot(pb, bv)
                        min_u = wp.min(min_u, pu)
                        max_u = wp.max(max_u, pu)
                        min_v = wp.min(min_v, pv_val)
                        max_v = wp.max(max_v, pv_val)
                ref = (gjk_result.point_a + gjk_result.point_b) * 0.5
                ref_u = wp.dot(ref, u)
                ref_v = wp.dot(ref, bv)
                c0 = ref + u * (min_u - ref_u) + bv * (min_v - ref_v)
                c1 = ref + u * (max_u - ref_u) + bv * (min_v - ref_v)
                c2 = ref + u * (max_u - ref_u) + bv * (max_v - ref_v)
                c3 = ref + u * (min_u - ref_u) + bv * (max_v - ref_v)
                poly[0] = wp.vec4(c0[0], c0[1], c0[2], 0.0)
                poly[1] = wp.vec4(c1[0], c1[1], c1[2], 0.0)
                poly[2] = wp.vec4(c2[0], c2[1], c2[2], 0.0)
                poly[3] = wp.vec4(c3[0], c3[1], c3[2], 0.0)
                num_pts = int(4)

            # Clip
            for pl_i in range(CLIP_MAX_POINTS):
                if pl_i >= num_planes:
                    continue
                pn = wp.vec3(planes[pl_i][0], planes[pl_i][1], planes[pl_i][2])
                pd = planes[pl_i][3]
                poly, num_pts = clip_poly_against_plane(poly, num_pts, pn, pd)

            # Project onto face planes -> midpoint + depth
            for ci in range(CLIP_MAX_POINTS):
                if ci >= num_pts:
                    continue
                pt = wp.vec3(poly[ci][0], poly[ci][1], poly[ci][2])
                t_a = (da_val - wp.dot(na, pt)) / denom_a
                t_b = (db_val - wp.dot(nb, pt)) / denom_b
                p_on_a = pt + axis * t_a
                p_on_b = pt + axis * t_b
                mid = (p_on_a + p_on_b) * 0.5
                depth = wp.dot(axis, p_on_b - p_on_a)
                poly[ci] = wp.vec4(mid[0], mid[1], mid[2], depth)

        # Fallback
        if num_pts == 0:
            mid = (gjk_result.point_a + gjk_result.point_b) * 0.5
            result.p0 = mid
            result.d0 = gjk_result.distance
            result.count = 1
            return result

        # Reduce to 4 (deepest first, then largest quad)
        if num_pts > 4:
            best_w = float(1.0e30)
            idx0 = int(0)
            for i in range(CLIP_MAX_POINTS):
                if i >= num_pts:
                    continue
                if poly[i][3] < best_w:
                    best_w = poly[i][3]
                    idx0 = i
            p0 = wp.vec3(poly[idx0][0], poly[idx0][1], poly[idx0][2])
            best_dist = float(-1.0)
            idx1 = int(0)
            for i in range(CLIP_MAX_POINTS):
                if i >= num_pts:
                    continue
                if i == idx0:
                    continue
                pi = wp.vec3(poly[i][0], poly[i][1], poly[i][2])
                dd = wp.length_sq(pi - p0)
                if dd > best_dist:
                    best_dist = dd
                    idx1 = i
            p1 = wp.vec3(poly[idx1][0], poly[idx1][1], poly[idx1][2])
            perp = wp.cross(axis, p1 - p0)
            best_pos = float(-1.0e30)
            best_neg = float(1.0e30)
            idx2 = int(-1)
            idx3 = int(-1)
            for k in range(CLIP_MAX_POINTS):
                if k >= num_pts:
                    continue
                if k == idx0 or k == idx1:
                    continue
                pk = wp.vec3(poly[k][0], poly[k][1], poly[k][2])
                dd = wp.dot(pk - p0, perp)
                if dd > best_pos:
                    best_pos = dd
                    idx2 = k
                if dd < best_neg:
                    best_neg = dd
                    idx3 = k
            out = ClipPoly()
            out[0] = poly[idx0]
            out[1] = poly[idx1]
            out_count = int(2)
            if idx2 >= 0:
                out[out_count] = poly[idx2]
                out_count = out_count + 1
            if idx3 >= 0 and idx3 != idx2:
                out[out_count] = poly[idx3]
                out_count = out_count + 1
            poly = out
            num_pts = out_count

        # Store in ContactResult
        count = int(0)
        for ci in range(CLIP_MAX_POINTS):
            if ci >= num_pts:
                continue
            if count >= 4:
                continue
            if count == 0:
                result.p0 = wp.vec3(poly[ci][0], poly[ci][1], poly[ci][2])
                result.d0 = poly[ci][3]
            elif count == 1:
                result.p1 = wp.vec3(poly[ci][0], poly[ci][1], poly[ci][2])
                result.d1 = poly[ci][3]
            elif count == 2:
                result.p2 = wp.vec3(poly[ci][0], poly[ci][1], poly[ci][2])
                result.d2 = poly[ci][3]
            elif count == 3:
                result.p3 = wp.vec3(poly[ci][0], poly[ci][1], poly[ci][2])
                result.d3 = poly[ci][3]
            count = count + 1
        result.count = count
        return result

        result.count = 0
        axis = gjk_result.normal
        result.normal = axis

        face_a = contact_face_world(shape_a, axis, gjk_result.point_a)
        face_b = contact_face_world(shape_b, -axis, gjk_result.point_b)

        na = face_a.normal
        nb = face_b.normal
        da_val = wp.dot(na, face_a.p0)
        db_val = wp.dot(nb, face_b.p0)
        denom_a = wp.dot(na, axis)
        denom_b = wp.dot(nb, axis)
        if wp.abs(denom_a) < 1.0e-10:
            denom_a = 1.0e-10
        if wp.abs(denom_b) < 1.0e-10:
            denom_b = 1.0e-10

        poly = ClipPoly()
        num_pts = int(0)

        # === clipNone ===
        if face_a.count < 2 or face_b.count < 2:
            mid = (gjk_result.point_a + gjk_result.point_b) * 0.5
            result.p0 = mid
            result.d0 = gjk_result.distance
            result.count = 1
            return result

        # === clip2x2 (edge-edge) ===
        if face_a.count == 2 and face_b.count == 2:
            a0 = face_a.p0
            a1 = face_a.p1
            b0 = face_b.p0
            b1 = face_b.p1
            seg_ab = a1 - a0
            seg_cd = b1 - b0
            len_ab_sq = wp.dot(seg_ab, seg_ab)
            len_cd_sq = wp.dot(seg_cd, seg_cd)
            eps = 1.0e-5
            # Project each endpoint onto the other segment
            t_val = wp.dot(b0 - a0, seg_ab)
            if t_val > -eps * len_ab_sq and t_val < len_ab_sq * (1.0 + eps) and len_ab_sq > eps:
                proj = a0 + seg_ab * (t_val / len_ab_sq)
                mid = (b0 + proj) * 0.5
                d = wp.dot(axis, proj - b0)
                poly[num_pts] = wp.vec4(mid[0], mid[1], mid[2], d)
                num_pts = num_pts + 1
            t_val = wp.dot(b1 - a0, seg_ab)
            if t_val > -eps * len_ab_sq and t_val < len_ab_sq * (1.0 + eps) and len_ab_sq > eps:
                proj = a0 + seg_ab * (t_val / len_ab_sq)
                mid = (b1 + proj) * 0.5
                d = wp.dot(axis, proj - b1)
                poly[num_pts] = wp.vec4(mid[0], mid[1], mid[2], d)
                num_pts = num_pts + 1
            t_val = wp.dot(a0 - b0, seg_cd)
            if t_val > -eps * len_cd_sq and t_val < len_cd_sq * (1.0 + eps) and len_cd_sq > eps:
                proj = b0 + seg_cd * (t_val / len_cd_sq)
                mid = (a0 + proj) * 0.5
                d = wp.dot(axis, proj - a0)
                is_dup = int(0)
                for di_idx in range(CLIP_MAX_POINTS):
                    if di_idx >= num_pts:
                        continue
                    pv = wp.vec3(poly[di_idx][0], poly[di_idx][1], poly[di_idx][2])
                    if wp.length_sq(mid - pv) < eps:
                        is_dup = int(1)
                if is_dup == 0:
                    poly[num_pts] = wp.vec4(mid[0], mid[1], mid[2], d)
                    num_pts = num_pts + 1
            t_val = wp.dot(a1 - b0, seg_cd)
            if t_val > -eps * len_cd_sq and t_val < len_cd_sq * (1.0 + eps) and len_cd_sq > eps:
                proj = b0 + seg_cd * (t_val / len_cd_sq)
                mid = (a1 + proj) * 0.5
                d = wp.dot(axis, proj - a1)
                is_dup = int(0)
                for di_idx in range(CLIP_MAX_POINTS):
                    if di_idx >= num_pts:
                        continue
                    pv = wp.vec3(poly[di_idx][0], poly[di_idx][1], poly[di_idx][2])
                    if wp.length_sq(mid - pv) < eps:
                        is_dup = int(1)
                if is_dup == 0:
                    poly[num_pts] = wp.vec4(mid[0], mid[1], mid[2], d)
                    num_pts = num_pts + 1
        else:
            # === clipNxN (2xN, Nx2, NxN) ===
            planes = ClipPlanes()
            num_planes = int(0)
            planes, num_planes = make_face_clip_planes(face_a, axis, planes, num_planes)
            planes, num_planes = make_face_clip_planes(face_b, axis, planes, num_planes)

            # Initial polygon
            if face_a.count == 2:
                poly[0] = wp.vec4(face_a.p0[0], face_a.p0[1], face_a.p0[2], 0.0)
                poly[1] = wp.vec4(face_a.p1[0], face_a.p1[1], face_a.p1[2], 0.0)
                num_pts = int(2)
            elif face_b.count == 2:
                poly[0] = wp.vec4(face_b.p0[0], face_b.p0[1], face_b.p0[2], 0.0)
                poly[1] = wp.vec4(face_b.p1[0], face_b.p1[1], face_b.p1[2], 0.0)
                num_pts = int(2)
            else:
                # Bounding quad (PhysX makePolygon)
                u = wp.cross(axis, wp.vec3(0.0, 0.0, 1.0))
                if wp.length(u) < 1.0e-6:
                    u = wp.cross(axis, wp.vec3(0.0, 1.0, 0.0))
                u = wp.normalize(u)
                v = wp.cross(axis, u)
                min_u = float(1.0e30)
                max_u = float(-1.0e30)
                min_v = float(1.0e30)
                max_v = float(-1.0e30)
                for fi in range(4):
                    if fi < face_a.count:
                        pa = get_face_point(face_a, fi)
                        pu = wp.dot(pa, u)
                        pv_val = wp.dot(pa, v)
                        min_u = wp.min(min_u, pu)
                        max_u = wp.max(max_u, pu)
                        min_v = wp.min(min_v, pv_val)
                        max_v = wp.max(max_v, pv_val)
                    if fi < face_b.count:
                        pb = get_face_point(face_b, fi)
                        pu = wp.dot(pb, u)
                        pv_val = wp.dot(pb, v)
                        min_u = wp.min(min_u, pu)
                        max_u = wp.max(max_u, pu)
                        min_v = wp.min(min_v, pv_val)
                        max_v = wp.max(max_v, pv_val)
                ref = (gjk_result.point_a + gjk_result.point_b) * 0.5
                ref_u = wp.dot(ref, u)
                ref_v = wp.dot(ref, v)
                c0 = ref + u * (min_u - ref_u) + v * (min_v - ref_v)
                c1 = ref + u * (max_u - ref_u) + v * (min_v - ref_v)
                c2 = ref + u * (max_u - ref_u) + v * (max_v - ref_v)
                c3 = ref + u * (min_u - ref_u) + v * (max_v - ref_v)
                poly[0] = wp.vec4(c0[0], c0[1], c0[2], 0.0)
                poly[1] = wp.vec4(c1[0], c1[1], c1[2], 0.0)
                poly[2] = wp.vec4(c2[0], c2[1], c2[2], 0.0)
                poly[3] = wp.vec4(c3[0], c3[1], c3[2], 0.0)
                num_pts = int(4)

            # Clip
            for pl_i in range(CLIP_MAX_POINTS):
                if pl_i >= num_planes:
                    continue
                pn = wp.vec3(planes[pl_i][0], planes[pl_i][1], planes[pl_i][2])
                pd = planes[pl_i][3]
                poly, num_pts = clip_poly_against_plane(poly, num_pts, pn, pd)

            # Project onto face planes → midpoint + depth
            for ci in range(CLIP_MAX_POINTS):
                if ci >= num_pts:
                    continue
                pt = wp.vec3(poly[ci][0], poly[ci][1], poly[ci][2])
                t_a = (da_val - wp.dot(na, pt)) / denom_a
                t_b = (db_val - wp.dot(nb, pt)) / denom_b
                p_on_a = pt + axis * t_a
                p_on_b = pt + axis * t_b
                mid = (p_on_a + p_on_b) * 0.5
                depth = wp.dot(axis, p_on_b - p_on_a)
                poly[ci] = wp.vec4(mid[0], mid[1], mid[2], depth)

        # Fallback
        if num_pts == 0:
            mid = (gjk_result.point_a + gjk_result.point_b) * 0.5
            result.p0 = mid
            result.d0 = gjk_result.distance
            result.count = 1
            return result

        # Reduce to 4 (deepest first, then largest quad)
        if num_pts > 4:
            best_w = float(1.0e30)
            idx0 = int(0)
            for i in range(CLIP_MAX_POINTS):
                if i >= num_pts:
                    continue
                if poly[i][3] < best_w:
                    best_w = poly[i][3]
                    idx0 = i
            p0 = wp.vec3(poly[idx0][0], poly[idx0][1], poly[idx0][2])
            best_dist = float(-1.0)
            idx1 = int(0)
            for i in range(CLIP_MAX_POINTS):
                if i >= num_pts:
                    continue
                if i == idx0:
                    continue
                pi = wp.vec3(poly[i][0], poly[i][1], poly[i][2])
                dd = wp.length_sq(pi - p0)
                if dd > best_dist:
                    best_dist = dd
                    idx1 = i
            p1 = wp.vec3(poly[idx1][0], poly[idx1][1], poly[idx1][2])
            perp = wp.cross(axis, p1 - p0)
            best_pos = float(-1.0e30)
            best_neg = float(1.0e30)
            idx2 = int(-1)
            idx3 = int(-1)
            for k in range(CLIP_MAX_POINTS):
                if k >= num_pts:
                    continue
                if k == idx0 or k == idx1:
                    continue
                pk = wp.vec3(poly[k][0], poly[k][1], poly[k][2])
                dd = wp.dot(pk - p0, perp)
                if dd > best_pos:
                    best_pos = dd
                    idx2 = k
                if dd < best_neg:
                    best_neg = dd
                    idx3 = k
            out = ClipPoly()
            out[0] = poly[idx0]
            out[1] = poly[idx1]
            out_count = int(2)
            if idx2 >= 0:
                out[out_count] = poly[idx2]
                out_count = out_count + 1
            if idx3 >= 0 and idx3 != idx2:
                out[out_count] = poly[idx3]
                out_count = out_count + 1
            poly = out
            num_pts = out_count

        # Store in ContactResult
        count = int(0)
        for ci in range(CLIP_MAX_POINTS):
            if ci >= num_pts:
                break
            if count >= 4:
                break
            if count == 0:
                result.p0 = wp.vec3(poly[ci][0], poly[ci][1], poly[ci][2])
                result.d0 = poly[ci][3]
            elif count == 1:
                result.p1 = wp.vec3(poly[ci][0], poly[ci][1], poly[ci][2])
                result.d1 = poly[ci][3]
            elif count == 2:
                result.p2 = wp.vec3(poly[ci][0], poly[ci][1], poly[ci][2])
                result.d2 = poly[ci][3]
            elif count == 3:
                result.p3 = wp.vec3(poly[ci][0], poly[ci][1], poly[ci][2])
                result.d3 = poly[ci][3]
            count = count + 1
        result.count = count
        return result

    @wp.func
    def generate_contacts(shape_a: ShapeData, shape_b: ShapeData) -> ContactResult:
        """Generate contact patch (runs GJK/MPR internally)."""
        gjk_result = gjk_mpr(shape_a, shape_b)
        if gjk_result.distance > 0.0:
            result = ContactResult()
            result.count = 0
            return result
        return generate_contact_patch(shape_a, shape_b, gjk_result)

    @wp.func
    def generate_contacts_from_gjk(shape_a: ShapeData, shape_b: ShapeData, gjk_result: GJKResult) -> ContactResult:
        """Backward-compat wrapper."""
        return generate_contact_patch(shape_a, shape_b, gjk_result)

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
    def gjk_mpr_kernel(
        shapes_a: wp.array(dtype=ShapeData),
        shapes_b: wp.array(dtype=ShapeData),
        results_dist: wp.array(dtype=float),
        results_point_a: wp.array(dtype=wp.vec3),
        results_point_b: wp.array(dtype=wp.vec3),
        results_normal: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        r = gjk_mpr(shapes_a[tid], shapes_b[tid])
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
        gjk_result = gjk_mpr(sa, sb)

        # Generate full contact patch if within contact_distance
        if gjk_result.distance <= contact_distance:
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
        gjk_mpr=gjk_mpr,
        mpr_depth=mpr_depth,
        generate_contacts=generate_contacts,
        support_kernel=support_kernel,
        contact_face_kernel=contact_face_kernel,
        aabb_kernel=aabb_kernel,
        gjk_kernel=gjk_kernel,
        gjk_mpr_kernel=gjk_mpr_kernel,
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
