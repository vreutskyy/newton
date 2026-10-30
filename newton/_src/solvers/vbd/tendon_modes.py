# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Circular tendon condensation adapted from 6bcd973e; no profile routing."""

import warp as wp

from ..tendon_damping import linear_damped_tension, tendon_damping_tension
from ..tendon_kernels import tendon_material_tangent, tendon_material_tension

_MIN_TENDON_COMPLIANCE = wp.constant(1.0e-8)


@wp.struct
class TendonMaterialMode:
    """A condensed material block with unilateral forces and per-guide secants."""

    first: int
    last: int
    next: int
    tension: float
    stiffness: float
    transfer_weight: float
    """Outgoing guide's friction stiffness divided by total transfer stiffness."""


@wp.kernel
def prepare_tendon_material_modes(
    tendon_start: wp.array[int],
    cone_l: wp.array[int],
    cone_r: wp.array[int],
    cap_ratio: wp.array[float],
    rest: wp.array[float],
    length: wp.array[float],
    stretch: wp.array[float],
    compliance: wp.array[float],
    damping: wp.array[float],
    damping_tension: wp.array[float],
    active: wp.array[int],
    dt: float,
    ea_low: float,
    ea_ratio: float,
    transition: float,
    width: float,
    material_transfer: wp.array[float],
    sliding_curvature: int,
    modes: wp.array[TendonMaterialMode],
):
    """Condense material modes, optionally using per-guide friction secants.

    The sparse path uses friction secants as a positive solver metric, not the
    exact derivative of the capstan projection. Projected tensions are unchanged.
    """
    tendon = wp.tid()
    first = tendon_start[tendon] - tendon
    end = tendon_start[tendon + 1] - tendon - 1
    for seg in range(first, end):
        mode = TendonMaterialMode()
        mode.first = seg
        mode.last = seg
        mode.next = -1
        if active[seg] != 0:
            comp = wp.max(compliance[seg], _MIN_TENDON_COMPLIANCE)
            material = stretch[seg] / comp
            tangent = 1.0 / comp
            if ea_low > 0.0:
                material = tendon_material_tension(length[seg], rest[seg], comp, ea_low, ea_ratio, transition, width)
                tangent = tendon_material_tangent(length[seg], rest[seg], comp, ea_low, ea_ratio, transition, width)
            damped = tendon_damping_tension(stretch[seg], damping_tension[seg], damping[seg], dt)
            mode.tension = wp.max(material + damped, 0.0)
            if ea_low <= 0.0:
                mode.tension = linear_damped_tension(stretch[seg], comp, damping_tension[seg], damping[seg] / dt)
            if mode.tension > 0.0:
                mode.stiffness = tangent + damping[seg] / dt
        modes[seg] = mode

    for link in range(tendon_start[tendon] + 1, tendon_start[tendon + 1] - 1):
        left = cone_l[link]
        right = cone_r[link]
        if left < first or right >= end or left < 0 or right <= left:
            continue
        if active[left] == 0 or active[right] == 0 or rest[left] <= 1.001e-6 or rest[right] <= 1.001e-6:
            continue
        if modes[left].stiffness <= 0.0 or modes[right].stiffness <= 0.0:
            continue
        # A finite-friction cone boundary is nonsmooth: reverse motion can
        # stick. Keep that conservative tangent instead of extending a sliding
        # branch in both directions. A unit ratio has no sticking interval.
        join = cap_ratio[link] == 1.0
        weight = float(0.0)
        if sliding_curvature != 0 and cap_ratio[link] > 1.0:
            slip = wp.abs(material_transfer[link])
            if slip > 0.0:
                # Positive friction secant: resistance / slip. It tends to the
                # sticking curvature as slip vanishes, rather than abruptly
                # releasing a material coordinate at the friction boundary.
                rho = cap_ratio[link]
                resistance = (rho - 1.0) / (rho + 1.0) * (modes[left].tension + modes[right].tension)
                # Transfer stretches one span while relaxing the other: its
                # elastic stiffness is k_left+k_right, not their series stiffness.
                transfer_stiffness = modes[left].stiffness + modes[right].stiffness
                weight = resistance / (resistance + slip * transfer_stiffness)
                join = True
        if join:
            l = modes[left]
            l.next = right
            l.transfer_weight = weight
            modes[left] = l

    head = first
    while head < end:
        last = head
        while modes[last].next >= 0:
            last = modes[last].next
        for seg in range(head, last + 1):
            mode = modes[seg]
            mode.first = head
            mode.last = last
            modes[seg] = mode
        head = last + 1


@wp.func
def tendon_segment_body_gradient(
    body: int,
    world_com: wp.vec3,
    seg: int,
    tendon_link_body: wp.array[int],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
):
    """Return the negative straight-span length gradient on one body."""
    linear = wp.vec3(0.0)
    angular = wp.vec3(0.0)
    body_l = tendon_link_body[seg_active_link_l[seg]]
    body_r = tendon_link_body[seg_active_link_r[seg]]
    left = seg_attachment_l[seg]
    right = seg_attachment_r[seg]
    length = wp.length(right - left)
    # A span internal to one body has exactly zero length derivative.
    if body_l != body_r and length > 1.0e-8:
        direction = (right - left) / length
        if body_l == body:
            linear = direction
            angular = wp.cross(left - world_com, direction)
        elif body_r == body:
            linear = -direction
            angular = -wp.cross(right - world_com, direction)
    return linear, angular


@wp.func
def tendon_mode_body_pair_hessians(
    row: int,
    col: int,
    com_row: wp.vec3,
    com_col: wp.vec3,
    first: int,
    modes: wp.array[TendonMaterialMode],
    tendon_link_body: wp.array[int],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
):
    """Eliminate individual transfer secants in linear time and constant storage.

    Each elimination contributes a positive weighted gradient outer product.
    This is the Schur complement of a quadratic friction-secant metric, not
    the exact derivative of the nonsymmetric capstan projection.
    """
    ll = wp.mat33(0.0)
    la = wp.mat33(0.0)
    al = wp.mat33(0.0)
    aa = wp.mat33(0.0)
    seg = first
    stiffness = modes[seg].stiffness
    if stiffness > 0.0:
        lin_row, ang_row = tendon_segment_body_gradient(
            row,
            com_row,
            seg,
            tendon_link_body,
            seg_attachment_l,
            seg_attachment_r,
            seg_active_link_l,
            seg_active_link_r,
        )
        lin_col, ang_col = tendon_segment_body_gradient(
            col,
            com_col,
            seg,
            tendon_link_body,
            seg_attachment_l,
            seg_attachment_r,
            seg_active_link_l,
            seg_active_link_r,
        )
        while modes[seg].next >= 0:
            next_seg = modes[seg].next
            next_k = modes[next_seg].stiffness
            weight = modes[seg].transfer_weight
            # weight = f/(f + k_seg + k_next). Recover K/(K+f)
            # without dividing by a possibly tiny material displacement.
            free = stiffness * (1.0 - weight)
            resistance = weight * (modes[seg].stiffness + next_k)
            carry = free / (free + resistance)
            kept_k = stiffness * (1.0 - carry)
            ll += kept_k * wp.outer(lin_row, lin_col)
            la += kept_k * wp.outer(lin_row, ang_col)
            al += kept_k * wp.outer(ang_row, lin_col)
            aa += kept_k * wp.outer(ang_row, ang_col)
            next_lin_row, next_ang_row = tendon_segment_body_gradient(
                row,
                com_row,
                next_seg,
                tendon_link_body,
                seg_attachment_l,
                seg_attachment_r,
                seg_active_link_l,
                seg_active_link_r,
            )
            next_lin_col, next_ang_col = tendon_segment_body_gradient(
                col,
                com_col,
                next_seg,
                tendon_link_body,
                seg_attachment_l,
                seg_attachment_r,
                seg_active_link_l,
                seg_active_link_r,
            )
            lin_row = carry * lin_row + next_lin_row
            ang_row = carry * ang_row + next_ang_row
            lin_col = carry * lin_col + next_lin_col
            ang_col = carry * ang_col + next_ang_col
            stiffness = next_k / (1.0 + carry * next_k / stiffness)
            seg = next_seg
        ll += stiffness * wp.outer(lin_row, lin_col)
        la += stiffness * wp.outer(lin_row, ang_col)
        al += stiffness * wp.outer(ang_row, lin_col)
        aa += stiffness * wp.outer(ang_row, ang_col)
    return ll, la, al, aa


@wp.func
def evaluate_condensed_tendon_force_hessians(
    body: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_adj_segments: wp.array[int],
    body_adj_segments_offsets: wp.array[int],
    modes: wp.array[TendonMaterialMode],
    tendon_link_body: wp.array[int],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
):
    """Evaluate body loads from material and geometry projected before this body color."""
    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    h_ll = wp.mat33(0.0)
    h_al = wp.mat33(0.0)
    h_aa = wp.mat33(0.0)

    adjacent_start = body_adj_segments_offsets[body]
    adjacent_end = body_adj_segments_offsets[body + 1]
    for adjacent_index in range(adjacent_start, adjacent_end):
        seg = body_adj_segments[adjacent_index]
        if seg_active[seg] == 0:
            continue

        link_l = seg_active_link_l[seg]
        link_r = seg_active_link_r[seg]
        body_l = tendon_link_body[link_l]
        body_r = tendon_link_body[link_r]
        if body != body_l and body != body_r:
            continue

        attachment_l = seg_attachment_l[seg]
        attachment_r = seg_attachment_r[seg]
        direction = attachment_r - attachment_l
        length = wp.length(direction)
        if length <= 1.0e-8:
            continue
        direction = direction / length

        tension = modes[seg].tension

        if tension <= 0.0:
            continue

        if body_l == body_r:
            # Internal straight-span forces and moments cancel on one body.
            continue

        world_com = wp.transform_point(body_q[body], body_com[body])
        if body == body_l:
            attachment = attachment_l
            body_force = tension * direction
        else:
            attachment = attachment_r
            body_force = -tension * direction

        moment_arm = attachment - world_com
        body_torque = wp.cross(moment_arm, body_force)

        force = force + body_force
        torque = torque + body_torque

    # Use the same per-interface elimination for diagonal and cross-body terms.
    previous = int(-1)
    world_com = wp.transform_point(body_q[body], body_com[body])
    for adjacent_index in range(adjacent_start, adjacent_end):
        adjacent_seg = body_adj_segments[adjacent_index]
        if seg_active[adjacent_seg] == 0:
            continue
        mode = modes[adjacent_seg]
        if mode.first == previous:
            continue
        previous = mode.first
        ll, _la, al, aa = tendon_mode_body_pair_hessians(
            body,
            body,
            world_com,
            world_com,
            mode.first,
            modes,
            tendon_link_body,
            seg_attachment_l,
            seg_attachment_r,
            seg_active_link_l,
            seg_active_link_r,
        )
        h_ll += ll
        h_al += al
        h_aa += aa

    return force, torque, h_ll, h_al, h_aa
