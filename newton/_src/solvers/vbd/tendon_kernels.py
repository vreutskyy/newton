# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...sim.tendon import TendonLinkType
from ..tendon_kernels import (
    tendon_material_tangent,
    tendon_material_tension,
    tendon_segment_length_rate_from_poses,
)

wp.set_module_options({"enable_backward": False})

_MIN_TENDON_COMPLIANCE = wp.constant(1.0e-8)


@wp.struct
class TendonForceElementAdjacencyInfo:
    """CSR adjacency between VBD rigid bodies and tendon segments."""

    body_adj_segments: wp.array[wp.int32]
    body_adj_segments_offsets: wp.array[wp.int32]

    def to(self, device):
        """Copy the adjacency to a device."""
        if device == self.body_adj_segments.device:
            return self

        adjacency = TendonForceElementAdjacencyInfo()
        adjacency.body_adj_segments = self.body_adj_segments.to(device)
        adjacency.body_adj_segments_offsets = self.body_adj_segments_offsets.to(device)
        return adjacency


@wp.kernel
def snapshot_tendon_segment_length_reference(
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_length_prev: wp.array[float],
):
    """Snapshot previous-pose segment lengths for final VBD diagnostics."""
    seg = wp.tid()
    seg_length_prev[seg] = 0.0
    if seg_active[seg] == 0:
        return

    link_l = seg_active_link_l[seg]
    link_r = seg_active_link_r[seg]
    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]
    attachment_l = wp.transform_point(body_q[body_l], seg_attachment_l_local[seg])
    attachment_r = wp.transform_point(body_q[body_r], seg_attachment_r_local[seg])
    length_rate = tendon_segment_length_rate_from_poses(
        dt,
        body_q,
        body_q_prev,
        body_com,
        tendon_link_body,
        tendon_link_type,
        tendon_link_offset,
        tendon_link_axis,
        link_l,
        link_r,
        seg_attachment_l_local[seg],
        seg_attachment_r_local[seg],
        attachment_l,
        attachment_r,
    )
    seg_length_prev[seg] = wp.length(attachment_r - attachment_l) - dt * length_rate


@wp.kernel
def update_tendon_segment_diagnostics(
    dt: float,
    body_q: wp.array[wp.transform],
    tendon_link_body: wp.array[int],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_length_prev: wp.array[float],
    sigmoid_ea_low: float,
    sigmoid_ea_ratio: float,
    sigmoid_transition_strain: float,
    sigmoid_transition_width: float,
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_material_tension: wp.array[float],
    seg_damping_tension: wp.array[float],
    seg_lambda: wp.array[float],
):
    """Update tendon geometry and damping from the accepted VBD pose."""
    seg = wp.tid()
    seg_attachment_l[seg] = wp.vec3(0.0)
    seg_attachment_r[seg] = wp.vec3(0.0)
    seg_material_tension[seg] = 0.0
    seg_damping_tension[seg] = 0.0
    # Native VBD does not solve XPBD constraint multipliers.
    seg_lambda[seg] = 0.0
    if seg_active[seg] == 0:
        return

    link_l = seg_active_link_l[seg]
    link_r = seg_active_link_r[seg]
    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]

    attachment_l = wp.transform_point(body_q[body_l], seg_attachment_l_local[seg])
    attachment_r = wp.transform_point(body_q[body_r], seg_attachment_r_local[seg])
    seg_attachment_l[seg] = attachment_l
    seg_attachment_r[seg] = attachment_r
    length = wp.length(attachment_r - attachment_l)
    length_rate = (length - seg_length_prev[seg]) / dt
    compliance = wp.max(seg_active_compliance[seg], _MIN_TENDON_COMPLIANCE)
    seg_material_tension[seg] = tendon_material_tension(
        length,
        seg_rest_length[seg],
        compliance,
        sigmoid_ea_low,
        sigmoid_ea_ratio,
        sigmoid_transition_strain,
        sigmoid_transition_width,
    )
    seg_damping_tension[seg] = seg_active_damping[seg] * length_rate


@wp.func
def _tendon_span_tension(
    seg: int,
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    sigmoid_ea_low: float,
    sigmoid_ea_ratio: float,
    sigmoid_transition_strain: float,
    sigmoid_transition_width: float,
) -> float:
    """Evaluate the selected VBD material law at the current trial poses."""
    link_l = seg_active_link_l[seg]
    link_r = seg_active_link_r[seg]
    attachment_l = wp.transform_point(body_q[tendon_link_body[link_l]], seg_attachment_l_local[seg])
    attachment_r = wp.transform_point(body_q[tendon_link_body[link_r]], seg_attachment_r_local[seg])
    length = wp.length(attachment_r - attachment_l)
    rest_length = seg_rest_length[seg]
    # Preserve the VBD slack/damping gate used by the body force assembly.
    if length <= 1.0e-8 or length <= rest_length:
        return 0.0
    compliance = wp.max(seg_active_compliance[seg], _MIN_TENDON_COMPLIANCE)
    length_rate = tendon_segment_length_rate_from_poses(
        dt,
        body_q,
        body_q_prev,
        body_com,
        tendon_link_body,
        tendon_link_type,
        tendon_link_offset,
        tendon_link_axis,
        link_l,
        link_r,
        seg_attachment_l_local[seg],
        seg_attachment_r_local[seg],
        attachment_l,
        attachment_r,
    )
    stiffness = 1.0 / compliance
    material_tension = stiffness * (length - rest_length)
    if sigmoid_ea_low > 0.0:
        material_tension = tendon_material_tension(
            length,
            rest_length,
            compliance,
            sigmoid_ea_low,
            sigmoid_ea_ratio,
            sigmoid_transition_strain,
            sigmoid_transition_width,
        )
    return wp.max(material_tension + seg_active_damping[seg] * length_rate, 0.0)


@wp.func
def _rolling_spin_axis_component(
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    tendon_link_cone_seg_l: wp.array[int],
    tendon_link_cone_seg_r: wp.array[int],
    tendon_link_cap_ratio: wp.array[float],
    seg_rest_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    link: int,
    seg: int,
    tension: float,
    attachment: wp.vec3,
    direction: wp.vec3,
    sigmoid_ea_low: float,
    sigmoid_ea_ratio: float,
    sigmoid_transition_strain: float,
    sigmoid_transition_width: float,
) -> wp.vec3:
    """Remove only the rim moment forbidden by the material cone.

    Sticking transmits the full tension difference. A body trial outside the
    cone is limited by its current adjacent span forces; cached diagnostic
    tensions would be stale after a preceding VBD body update.
    """
    if tendon_link_type[link] != int(TendonLinkType.ROLLING):
        return wp.vec3(0.0)
    seg_left = tendon_link_cone_seg_l[link]
    seg_right = tendon_link_cone_seg_r[link]
    if seg_left < 0 or seg_right < 0 or seg_left >= seg_active.shape[0] or seg_right >= seg_active.shape[0]:
        return wp.vec3(0.0)
    if seg_active[seg_left] == 0 or seg_active[seg_right] == 0:
        return wp.vec3(0.0)
    if seg_active_link_r[seg_left] != link or seg_active_link_l[seg_right] != link:
        return wp.vec3(0.0)

    spin_scale = float(0.0)
    # The two local contact radii rotate together during a body update, so the
    # cached cone angle stays valid until the route is retangented.
    cap_ratio = tendon_link_cap_ratio[link]
    if cap_ratio > 1.0:
        # The caller already evaluated this span at the same trial poses.
        neighbor = seg_right if seg == seg_left else seg_left
        neighbor_tension = _tendon_span_tension(
            neighbor,
            dt,
            body_q,
            body_q_prev,
            body_com,
            tendon_link_body,
            tendon_link_type,
            tendon_link_offset,
            tendon_link_axis,
            seg_rest_length,
            seg_attachment_l_local,
            seg_attachment_r_local,
            seg_active_compliance,
            seg_active_damping,
            seg_active_link_l,
            seg_active_link_r,
            sigmoid_ea_low,
            sigmoid_ea_ratio,
            sigmoid_transition_strain,
            sigmoid_transition_width,
        )
        beta = (cap_ratio - 1.0) / (cap_ratio + 1.0)
        allowed_difference = beta * (tension + neighbor_tension)
        spin_scale = wp.min(1.0, allowed_difference / wp.max(wp.abs(tension - neighbor_tension), 1.0e-8))

    pose = body_q[tendon_link_body[link]]
    center = wp.transform_point(pose, tendon_link_offset[link])
    normal = wp.normalize(wp.transform_vector(pose, tendon_link_axis[link]))
    radial = attachment - center
    return (1.0 - spin_scale) * wp.dot(wp.cross(radial, direction), normal) * normal


@wp.func
def evaluate_tendon_force_hessians(
    body: int,
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    adjacency: TendonForceElementAdjacencyInfo,
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    tendon_link_cone_seg_l: wp.array[int],
    tendon_link_cone_seg_r: wp.array[int],
    tendon_link_cap_ratio: wp.array[float],
    seg_rest_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    sigmoid_ea_low: float,
    sigmoid_ea_ratio: float,
    sigmoid_transition_strain: float,
    sigmoid_transition_width: float,
):
    """Evaluate unilateral tendon spring-damper forces for one VBD body.

    Freeze the capstan limiter when forming the positive-semidefinite local
    Hessian approximation; its derivative is not included.
    """
    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    h_ll = wp.mat33(0.0)
    h_al = wp.mat33(0.0)
    h_aa = wp.mat33(0.0)

    adjacent_start = adjacency.body_adj_segments_offsets[body]
    adjacent_end = adjacency.body_adj_segments_offsets[body + 1]
    for adjacent_index in range(adjacent_start, adjacent_end):
        seg = adjacency.body_adj_segments[adjacent_index]
        if seg_active[seg] == 0:
            continue

        link_l = seg_active_link_l[seg]
        link_r = seg_active_link_r[seg]
        body_l = tendon_link_body[link_l]
        body_r = tendon_link_body[link_r]
        if body != body_l and body != body_r:
            continue

        attachment_l = wp.transform_point(body_q[body_l], seg_attachment_l_local[seg])
        attachment_r = wp.transform_point(body_q[body_r], seg_attachment_r_local[seg])
        direction = attachment_r - attachment_l
        length = wp.length(direction)
        if length <= 1.0e-8:
            continue
        direction = direction / length

        compliance = wp.max(seg_active_compliance[seg], _MIN_TENDON_COMPLIANCE)

        rest_length = seg_rest_length[seg]
        if length <= rest_length:
            continue

        length_rate = tendon_segment_length_rate_from_poses(
            dt,
            body_q,
            body_q_prev,
            body_com,
            tendon_link_body,
            tendon_link_type,
            tendon_link_offset,
            tendon_link_axis,
            link_l,
            link_r,
            seg_attachment_l_local[seg],
            seg_attachment_r_local[seg],
            attachment_l,
            attachment_r,
        )

        stiffness = 1.0 / compliance
        damping = seg_active_damping[seg]
        tension = stiffness * (length - rest_length) + damping * length_rate
        if sigmoid_ea_low > 0.0:
            stiffness = tendon_material_tangent(
                length,
                rest_length,
                compliance,
                sigmoid_ea_low,
                sigmoid_ea_ratio,
                sigmoid_transition_strain,
                sigmoid_transition_width,
            )
            tension = (
                tendon_material_tension(
                    length,
                    rest_length,
                    compliance,
                    sigmoid_ea_low,
                    sigmoid_ea_ratio,
                    sigmoid_transition_strain,
                    sigmoid_transition_width,
                )
                + damping * length_rate
            )
        tension = wp.max(tension, 0.0)

        if tension <= 0.0:
            continue

        if body_l == body_r:
            # Both endpoints ride this body: the endpoint forces and their base
            # torques cancel exactly, but the rolling spin corrections are
            # asymmetric, leaving a net roller-axis torque — the same net row
            # XPBD's combined same-body Jacobian applies. Without it, a cable
            # that wraps a roller and terminates on the same body transmits no
            # torque at all (toy3 cable B: R3 -> tip on link1).
            fix_l = _rolling_spin_axis_component(
                dt,
                body_q,
                body_q_prev,
                body_com,
                tendon_link_body,
                tendon_link_type,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_cone_seg_l,
                tendon_link_cone_seg_r,
                tendon_link_cap_ratio,
                seg_rest_length,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active_compliance,
                seg_active_damping,
                seg_active,
                seg_active_link_l,
                seg_active_link_r,
                link_l,
                seg,
                tension,
                attachment_l,
                direction,
                sigmoid_ea_low,
                sigmoid_ea_ratio,
                sigmoid_transition_strain,
                sigmoid_transition_width,
            )
            fix_r = _rolling_spin_axis_component(
                dt,
                body_q,
                body_q_prev,
                body_com,
                tendon_link_body,
                tendon_link_type,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_cone_seg_l,
                tendon_link_cone_seg_r,
                tendon_link_cap_ratio,
                seg_rest_length,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active_compliance,
                seg_active_damping,
                seg_active,
                seg_active_link_l,
                seg_active_link_r,
                link_r,
                seg,
                tension,
                attachment_r,
                direction,
                sigmoid_ea_low,
                sigmoid_ea_ratio,
                sigmoid_transition_strain,
                sigmoid_transition_width,
            )
            net_moment_axis = fix_l - fix_r
            torque = torque - tension * net_moment_axis
            same_body_stiffness = stiffness + damping / dt
            h_aa = h_aa + same_body_stiffness * wp.outer(net_moment_axis, net_moment_axis)
            continue

        world_com = wp.transform_point(body_q[body], body_com[body])
        if body == body_l:
            attachment = attachment_l
            link = link_l
            body_force = tension * direction
        else:
            attachment = attachment_r
            link = link_r
            body_force = -tension * direction

        moment_arm = attachment - world_com
        moment_axis = wp.cross(moment_arm, direction)
        body_torque = wp.cross(moment_arm, body_force)

        spin_fix = _rolling_spin_axis_component(
            dt,
            body_q,
            body_q_prev,
            body_com,
            tendon_link_body,
            tendon_link_type,
            tendon_link_offset,
            tendon_link_axis,
            tendon_link_cone_seg_l,
            tendon_link_cone_seg_r,
            tendon_link_cap_ratio,
            seg_rest_length,
            seg_attachment_l_local,
            seg_attachment_r_local,
            seg_active_compliance,
            seg_active_damping,
            seg_active,
            seg_active_link_l,
            seg_active_link_r,
            link,
            seg,
            tension,
            attachment,
            direction,
            sigmoid_ea_low,
            sigmoid_ea_ratio,
            sigmoid_transition_strain,
            sigmoid_transition_width,
        )
        moment_axis = moment_axis - spin_fix
        endpoint_sign = 1.0 if body == body_l else -1.0
        body_torque = body_torque - endpoint_sign * tension * spin_fix

        effective_stiffness = stiffness + damping / dt

        force = force + body_force
        torque = torque + body_torque
        # The axial Gauss-Newton approximation remains positive semidefinite.
        h_ll = h_ll + effective_stiffness * wp.outer(direction, direction)
        h_al = h_al + effective_stiffness * wp.outer(moment_axis, direction)
        h_aa = h_aa + effective_stiffness * wp.outer(moment_axis, moment_axis)

    return force, torque, h_ll, h_al, h_aa
