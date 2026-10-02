# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prescribed convex-profile routing; the circular/dynamic path stays unchanged."""

import warp as wp

from ..geometry.roller_profile import (
    ProfileData,
    _ProfileKind,
    profile_arc_length,
    profile_parameter_delta,
    profile_span_tangent,
)
from ..sim.tendon import TendonLinkType


@wp.kernel
def update_profile_attachments(
    body_q: wp.array[wp.transform],
    bodies: wp.array[int],
    orientations: wp.array[int],
    offsets: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    axes_x: wp.array[wp.vec3],
    profiles: wp.array[ProfileData],
    seg_link_l: wp.array[int],
    seg_link_r: wp.array[int],
    parameter_l_step: wp.array[float],
    parameter_r_step: wp.array[float],
    rolling: bool,
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
    gl = seg_link_l[seg]
    gr = seg_link_r[seg]
    ql = body_q[bodies[gl]]
    qr = body_q[bodies[gr]]
    cl = wp.transform_point(ql, offsets[gl])
    cr = wp.transform_point(qr, offsets[gr])
    nl = wp.transform_vector(ql, axes[gl])
    nr = wp.transform_vector(qr, axes[gr])
    xl = wp.transform_vector(ql, axes_x[gl])
    xr = wp.transform_vector(qr, axes_x[gr])
    yl = wp.cross(nl, xl)
    yr = wp.cross(nr, xr)
    left_roller = profiles[gl].kind != int(_ProfileKind.POINT)
    right_roller = profiles[gr].kind != int(_ProfileKind.POINT)
    normal = nl if left_roller else nr
    al = cl
    ar = cr
    sl = float(0.0)
    sr = float(0.0)
    valid = True
    if left_roller or right_roller:
        al, ar, sl, sr, _support_normal, valid = profile_span_tangent(
            profiles[gl],
            profiles[gr],
            cl,
            cr,
            xl,
            yl,
            xr,
            yr,
            normal,
            orientations[gl],
            orientations[gr],
        )
        if left_roller and right_roller and wp.dot(nl, nr) < 1.0 - 1.0e-5:
            valid = False
    delta_l[seg] = 0.0
    delta_r[seg] = 0.0
    status[seg] = 0
    if not valid:
        status[seg] = 1
        wp.printf("ERROR: Profile tendon segment %d has no valid coplanar supporting tangent.\n", seg)
        return
    if rolling:
        if left_roller:
            old = parameter_l_step[seg]
            delta = profile_parameter_delta(profiles[gl], old, sl)
            delta_l[seg] = -float(orientations[gl]) * profile_arc_length(profiles[gl], old, delta)
        if right_roller:
            old = parameter_r_step[seg]
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
    attachment_l: wp.array[wp.vec3],
    attachment_r: wp.array[wp.vec3],
    parameter_l: wp.array[float],
    parameter_r: wp.array[float],
    report: bool,
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
    left = link - tendon - 1
    right = left + 1
    incoming = wp.normalize(attachment_r[left] - attachment_l[left])
    outgoing = wp.normalize(attachment_r[right] - attachment_l[right])
    theta = wp.atan2(wp.length(wp.cross(incoming, outgoing)), wp.dot(incoming, outgoing))
    if types[link] == int(TendonLinkType.ROLLING):
        normal = wp.transform_vector(body_q[bodies[link]], axes[link])
        signed_turn = float(orientations[link]) * wp.atan2(
            wp.dot(wp.cross(incoming, outgoing), normal), wp.dot(incoming, outgoing)
        )
        # Keep the prototype's supported wrap range explicit. Multi-turn routing
        # and activation are separate features, not silently inferred here.
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
            wrap_length[link] = wp.abs(profile_arc_length(profiles[link], entry, float(orientations[link]) * delta))
    cone_l[link] = left
    cone_r[link] = right
    cap_ratio[link] = wp.exp(wp.min(friction[link] * theta, 20.0))
