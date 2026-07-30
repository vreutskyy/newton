# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...sim.tendon import TendonLinkType

wp.set_module_options({"enable_backward": False})


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
    body_q_prev: wp.array[wp.transform],
    tendon_link_body: wp.array[int],
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
    attachment_l_prev = wp.transform_point(body_q_prev[body_l], seg_attachment_l_local[seg])
    attachment_r_prev = wp.transform_point(body_q_prev[body_r], seg_attachment_r_local[seg])
    seg_length_prev[seg] = wp.length(attachment_r_prev - attachment_l_prev)


@wp.kernel
def update_tendon_segment_diagnostics(
    dt: float,
    body_q: wp.array[wp.transform],
    tendon_link_body: wp.array[int],
    seg_rest_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_length_prev: wp.array[float],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    seg_lambda: wp.array[float],
):
    """Update tendon geometry and multipliers from the accepted VBD pose."""
    seg = wp.tid()
    seg_attachment_l[seg] = wp.vec3(0.0)
    seg_attachment_r[seg] = wp.vec3(0.0)
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
    rest_length = seg_rest_length[seg]
    if length <= rest_length:
        return

    compliance = seg_active_compliance[seg]
    if compliance <= 0.0:
        return

    length_rate = (length - seg_length_prev[seg]) / dt

    tension = (length - rest_length) / compliance + seg_active_damping[seg] * length_rate
    seg_lambda[seg] = -wp.max(tension, 0.0) * dt


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
    tendon_link_radius: wp.array[float],
    tendon_link_mu: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    tendon_link_seg_left: wp.array[int],
    seg_rest_length: wp.array[float],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
):
    """Evaluate unilateral tendon spring-damper forces for one VBD body."""
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

        compliance = seg_active_compliance[seg]
        if compliance <= 0.0:
            continue

        rest_length = seg_rest_length[seg]
        if length <= rest_length:
            continue

        attachment_l_prev = wp.transform_point(body_q_prev[body_l], seg_attachment_l_local[seg])
        attachment_r_prev = wp.transform_point(body_q_prev[body_r], seg_attachment_r_local[seg])
        length_prev = wp.length(attachment_r_prev - attachment_l_prev)
        length_rate = (length - length_prev) / dt

        stiffness = 1.0 / compliance
        damping = seg_active_damping[seg]
        tension = stiffness * (length - rest_length) + damping * length_rate
        tension = wp.max(tension, 0.0)

        if tension <= 0.0 or body_l == body_r:
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

        if tendon_link_type[link] == int(TendonLinkType.ROLLING):
            # Free-span tension still loads the body, but only capstan friction
            # transmits the rolling-axis part of its moment.
            seg_left = tendon_link_seg_left[link]
            if seg_left >= 0:
                seg_right = seg_left + 1

                if seg_right < seg_active.shape[0] and seg_active[seg_left] != 0 and seg_active[seg_right] != 0:
                    pose = body_q[body]
                    center = wp.transform_point(pose, tendon_link_offset[link])
                    normal = wp.normalize(wp.transform_vector(pose, tendon_link_axis[link]))
                    point_left = wp.transform_point(pose, seg_attachment_r_local[seg_left])
                    point_right = wp.transform_point(pose, seg_attachment_l_local[seg_right])
                    radial_left = point_left - center
                    radial_right = point_right - center
                    radial_left = radial_left - wp.dot(radial_left, normal) * normal
                    radial_right = radial_right - wp.dot(radial_right, normal) * normal
                    radial_left_length = wp.length(radial_left)
                    radial_right_length = wp.length(radial_right)

                    theta = float(0.0)
                    if tendon_link_radius[link] > 0.0 and radial_left_length > 1.0e-8 and radial_right_length > 1.0e-8:
                        unit_left = radial_left / radial_left_length
                        unit_right = radial_right / radial_right_length
                        theta = wp.abs(
                            wp.atan2(
                                wp.dot(wp.cross(unit_left, unit_right), normal),
                                wp.dot(unit_left, unit_right),
                            )
                        )

                    cap_ratio = wp.exp(wp.min(wp.max(tendon_link_mu[link], 0.0) * theta, 20.0))
                    spin_scale = (cap_ratio - 1.0) / (cap_ratio + 1.0)
                    moment_axis = moment_axis - (1.0 - spin_scale) * wp.dot(moment_axis, normal) * normal
                    body_torque = body_torque - (1.0 - spin_scale) * wp.dot(body_torque, normal) * normal

        effective_stiffness = stiffness + damping / dt

        force = force + body_force
        torque = torque + body_torque
        # The axial Gauss-Newton approximation remains positive semidefinite.
        h_ll = h_ll + effective_stiffness * wp.outer(direction, direction)
        h_al = h_al + effective_stiffness * wp.outer(moment_axis, direction)
        h_aa = h_aa + effective_stiffness * wp.outer(moment_axis, moment_axis)

    return force, torque, h_ll, h_al, h_aa
