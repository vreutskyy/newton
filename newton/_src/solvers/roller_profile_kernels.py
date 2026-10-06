# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Planar convex-profile routing with optional dynamic circular rollers."""

import warp as wp

from ..geometry.roller_profile import (
    ProfileData,
    _ProfileKind,
    profile_arc_length,
    profile_parameter_delta,
    profile_span_tangent,
)
from ..sim.tendon import TendonLinkFlags, TendonLinkType


@wp.func
def _link_tangent(
    gl: int,
    gr: int,
    body_q: wp.array[wp.transform],
    bodies: wp.array[int],
    orientations: wp.array[int],
    offsets: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    axes_x: wp.array[wp.vec3],
    profiles: wp.array[ProfileData],
):
    ql = body_q[bodies[gl]]
    qr = body_q[bodies[gr]]
    cl = wp.transform_point(ql, offsets[gl])
    cr = wp.transform_point(qr, offsets[gr])
    al, ar = cl, cr
    sl, sr = float(0.0), float(0.0)
    valid = True
    left_roller = profiles[gl].kind != int(_ProfileKind.POINT)
    right_roller = profiles[gr].kind != int(_ProfileKind.POINT)
    if left_roller or right_roller:
        nl = wp.transform_vector(ql, axes[gl])
        nr = wp.transform_vector(qr, axes[gr])
        xl = wp.transform_vector(ql, axes_x[gl])
        xr = wp.transform_vector(qr, axes_x[gr])
        al, ar, sl, sr, _support_normal, valid = profile_span_tangent(
            profiles[gl],
            profiles[gr],
            cl,
            cr,
            xl,
            wp.cross(nl, xl),
            xr,
            wp.cross(nr, xr),
            nl if left_roller else nr,
            orientations[gl],
            orientations[gr],
        )
        if left_roller and right_roller and wp.dot(nl, nr) < 1.0 - 1.0e-5:
            valid = False
    return al, ar, sl, sr, valid


@wp.kernel
def update_profile_link_active(
    body_q: wp.array[wp.transform],
    bodies: wp.array[int],
    flags: wp.array[int],
    orientations: wp.array[int],
    offsets: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    axes_x: wp.array[wp.vec3],
    profiles: wp.array[ProfileData],
    activation_tol: float,
    initialize: bool,
    active: wp.array[bool],
    route_rest: wp.array[float],
):
    """Test circles against the exact bypass tangent of their fixed neighbors."""
    link = wp.tid()
    if (flags[link] & int(TendonLinkFlags.DYNAMIC)) == 0:
        return
    # The builder rejects terminal and consecutive dynamic rollers, so the
    # two neighbors always exist and remain active throughout this decision.
    al, ar, _sl, _sr, valid = _link_tangent(
        link - 1, link + 1, body_q, bodies, orientations, offsets, axes, axes_x, profiles
    )
    if initialize:
        route_rest[link] = wp.length(ar - al) if valid else -1.0
    if not valid:
        wp.printf("ERROR: Dynamic profile link %d has no valid coplanar bypass tangent.\n", link)
        return
    pose = body_q[bodies[link]]
    normal = wp.normalize(wp.transform_vector(pose, axes[link]))
    span = ar - al
    offset = wp.transform_point(pose, offsets[link]) - al
    span -= wp.dot(span, normal) * normal
    offset -= wp.dot(offset, normal) * normal
    span_sq = wp.length_sq(span)
    next_active = False
    if span_sq > 1.0e-12:
        alpha = wp.dot(offset, span) / span_sq
        distance = float(orientations[link]) * wp.dot(offset, wp.cross(normal, span)) / wp.sqrt(span_sq)
        radius = profiles[link].radii[0]
        if not active[link]:
            radius *= 1.0 - activation_tol
        next_active = alpha > 0.0 and alpha < 1.0 and distance <= radius
    active[link] = next_active


@wp.kernel
def update_profile_attachments(
    body_q: wp.array[wp.transform],
    bodies: wp.array[int],
    orientations: wp.array[int],
    offsets: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    axes_x: wp.array[wp.vec3],
    profiles: wp.array[ProfileData],
    seg_active: wp.array[int],
    link_active: wp.array[bool],
    link_active_step: wp.array[bool],
    seg_link_l: wp.array[int],
    seg_link_r: wp.array[int],
    parameter_l_step: wp.array[float],
    parameter_r_step: wp.array[float],
    rolling: bool,
    has_dynamic: bool,
    attachment_l: wp.array[wp.vec3],
    attachment_r: wp.array[wp.vec3],
    local_l: wp.array[wp.vec3],
    local_r: wp.array[wp.vec3],
    delta_l: wp.array[float],
    delta_r: wp.array[float],
    lengths: wp.array[float],
    status: wp.array[int],
    parameter_l: wp.array[float],
    parameter_r: wp.array[float],
):
    """Update exact common tangents and full material transport since step start."""
    seg = wp.tid()
    delta_l[seg] = 0.0
    delta_r[seg] = 0.0
    status[seg] = 0
    if has_dynamic and seg_active[seg] == 0:
        attachment_l[seg] = wp.vec3(0.0)
        attachment_r[seg] = wp.vec3(0.0)
        local_l[seg] = wp.vec3(0.0)
        local_r[seg] = wp.vec3(0.0)
        lengths[seg] = 0.0
        parameter_l[seg] = 0.0
        parameter_r[seg] = 0.0
        return
    gl = seg_link_l[seg]
    gr = seg_link_r[seg]
    ql = body_q[bodies[gl]]
    qr = body_q[bodies[gr]]
    left_roller = profiles[gl].kind != int(_ProfileKind.POINT)
    right_roller = profiles[gr].kind != int(_ProfileKind.POINT)
    al, ar, sl, sr, valid = _link_tangent(gl, gr, body_q, bodies, orientations, offsets, axes, axes_x, profiles)
    if not valid:
        status[seg] = 1
        wp.printf("ERROR: Profile tendon segment %d has no valid coplanar supporting tangent.\n", seg)
        return
    if rolling:
        if left_roller and (not has_dynamic or link_active[gl] == link_active_step[gl]):
            old = parameter_l_step[seg]
            delta = profile_parameter_delta(profiles[gl], old, sl)
            delta_l[seg] = -float(orientations[gl]) * profile_arc_length(profiles[gl], old, delta)
        if right_roller and (not has_dynamic or link_active[gr] == link_active_step[gr]):
            # The persistent right endpoint moves between slots on a switch;
            # its old boundary coordinate belongs to the same neighboring link.
            old_seg = seg
            if has_dynamic:
                if link_active[gl] and not link_active_step[gl]:
                    old_seg = seg - 1
                elif gr == gl + 2 and not link_active[gl + 1] and link_active_step[gl + 1]:
                    old_seg = seg + 1
            old = parameter_r_step[old_seg]
            delta = profile_parameter_delta(profiles[gr], old, sr)
            delta_r[seg] = float(orientations[gr]) * profile_arc_length(profiles[gr], old, delta)
    attachment_l[seg] = al
    attachment_r[seg] = ar
    local_l[seg] = wp.transform_point(wp.transform_inverse(ql), al)
    local_r[seg] = wp.transform_point(wp.transform_inverse(qr), ar)
    lengths[seg] = wp.length(ar - al)
    parameter_l[seg] = sl
    parameter_r[seg] = sr


@wp.kernel
def update_profile_cones(
    body_q: wp.array[wp.transform],
    tendon_start: wp.array[int],
    link_tendon: wp.array[int],
    bodies: wp.array[int],
    types: wp.array[int],
    orientations: wp.array[int],
    friction: wp.array[float],
    axes: wp.array[wp.vec3],
    profiles: wp.array[ProfileData],
    link_active: wp.array[bool],
    seg_active: wp.array[int],
    seg_link_l: wp.array[int],
    seg_link_r: wp.array[int],
    lengths: wp.array[float],
    attachment_l: wp.array[wp.vec3],
    attachment_r: wp.array[wp.vec3],
    parameter_l: wp.array[float],
    parameter_r: wp.array[float],
    report: bool,
    has_dynamic: bool,
    cone_l: wp.array[int],
    cone_r: wp.array[int],
    cap_ratio: wp.array[float],
    wrap_length: wp.array[float],
    status: wp.array[int],
):
    """Use tangent turning, not polar angle, for general-profile capstan friction."""
    link = wp.tid()
    tendon = link_tendon[link]
    start = tendon_start[tendon]
    end = tendon_start[tendon + 1]
    cone_l[link] = -1
    cone_r[link] = -1
    cap_ratio[link] = 1.0
    if report:
        wrap_length[link] = 0.0
    status[link] = 0
    if link == start or link == end - 1 or types[link] == int(TendonLinkType.ATTACHMENT):
        return
    if has_dynamic and not link_active[link]:
        return
    left = link - tendon - 1
    right = left + 1
    if types[link] == int(TendonLinkType.ROLLING):
        if has_dynamic:
            if seg_active[left] == 0:
                left -= 1
            if left < start - tendon or seg_active[left] == 0 or seg_link_r[left] != link:
                return
            if seg_active[right] == 0 or seg_link_l[right] != link:
                return
    else:
        # Match pinhole semantics: skip inactive or degenerate adjacent spans.
        while left >= start - tendon:
            if seg_active[left] != 0 and lengths[left] > 1.0e-5:
                break
            left -= 1
        while right < end - tendon - 1:
            if seg_active[right] != 0 and lengths[right] > 1.0e-5:
                break
            right += 1
        if left < start - tendon or right >= end - tendon - 1:
            return
    incoming = wp.normalize(attachment_r[left] - attachment_l[left])
    outgoing = wp.normalize(attachment_r[right] - attachment_l[right])
    theta = wp.atan2(wp.length(wp.cross(incoming, outgoing)), wp.dot(incoming, outgoing))
    if types[link] == int(TendonLinkType.ROLLING):
        normal = wp.transform_vector(body_q[bodies[link]], axes[link])
        signed_turn = float(orientations[link]) * wp.atan2(
            wp.dot(wp.cross(incoming, outgoing), normal), wp.dot(incoming, outgoing)
        )
        # Keep the prototype's supported wrap range explicit. Multi-turn routing
        # is not inferred from a reversed tangent turn.
        if signed_turn < -1.0e-5 and signed_turn > -wp.pi + 1.0e-5:
            status[link] = 1
            if report:
                wp.printf("ERROR: Profile tendon link %d exceeds the supported tangent turn [0, pi].\n", link)
        # The material solve uses incremental boundary travel and the tangent
        # turn, not this diagnostic. Integrate the full wrap only at the accepted
        # pose (and initialization), rather than during every VBD iteration.
        if report:
            entry = parameter_r[left]
            exit = parameter_l[right]
            delta = float(orientations[link]) * (exit - entry)
            if delta < 0.0:
                delta += profiles[link].period
            # Independent tangent queries can reverse coincident contacts by a
            # few ULPs. Do not interpret a zero-wrap boundary as a full circuit.
            # Retain finite boundary travel along a sector's straight edge.
            if wp.abs(signed_turn) <= 1.0e-5 and delta >= (1.0 - 2.0e-6) * profiles[link].period:
                delta = 0.0
            wrap_length[link] = wp.abs(profile_arc_length(profiles[link], entry, float(orientations[link]) * delta))
    cone_l[link] = left
    cone_r[link] = right
    cap_ratio[link] = wp.exp(wp.min(friction[link] * theta, 20.0))
