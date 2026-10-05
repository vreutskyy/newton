# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ..tendon_kernels import (
    tendon_material_tangent,
    tendon_material_tension,
    tendon_segment_length_rate_from_poses,
)

wp.set_module_options({"enable_backward": False})

_MIN_TENDON_COMPLIANCE = wp.constant(1.0e-8)


@wp.struct
class TendonMaterialMode:
    """A freely sliding material block and its current unilateral force."""

    first: int
    last: int
    next: int
    tension: float
    stiffness: float


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
    modes: wp.array[TendonMaterialMode],
):
    """Condense free material modes, retaining the sticking tangent at friction bounds."""
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
            # Match the material projection's unilateral Kelvin-Voigt force.
            # Damping can support positive tension at nonpositive elastic
            # stretch; testing stretch alone would discard that force.
            mode.tension = wp.max(material + damping_tension[seg], 0.0)
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
        if cap_ratio[link] == 1.0:
            l = modes[left]
            l.next = right
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
def evaluate_tendon_force_hessians(
    body: int,
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    adjacency: TendonForceElementAdjacencyInfo,
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

    # Eliminate freely sliding material coordinates before assembling the
    # Gauss-Newton curvature: series compliance is sum(1/k_i), and the path
    # gradient is sum(J_i). This preserves cancellation between adjacent
    # spans (including a circle's spin nullspace) without inspecting profiles.
    # Finite-friction interfaces retain their sticking-side tangent; the
    # capstan projection still determines all forces above.
    previous = int(-1)
    world_com = wp.transform_point(body_q[body], body_com[body])
    for adjacent_index in range(adjacent_start, adjacent_end):
        adjacent_seg = adjacency.body_adj_segments[adjacent_index]
        if seg_active[adjacent_seg] == 0:
            continue
        mode = modes[adjacent_seg]
        if mode.first == previous:
            continue
        previous = mode.first
        linear = wp.vec3(0.0)
        angular = wp.vec3(0.0)
        mode_compliance = float(0.0)
        for seg in range(mode.first, mode.last + 1):
            if seg_active[seg] == 0 or modes[seg].stiffness <= 0.0:
                continue
            left = seg_attachment_l[seg]
            right = seg_attachment_r[seg]
            length = wp.length(right - left)
            if length <= 1.0e-8:
                continue
            stiffness = modes[seg].stiffness
            mode_compliance += 1.0 / stiffness
            direction = (right - left) / length
            if tendon_link_body[seg_active_link_l[seg]] == body:
                linear += direction
                angular += wp.cross(left - world_com, direction)
            if tendon_link_body[seg_active_link_r[seg]] == body:
                linear -= direction
                angular -= wp.cross(right - world_com, direction)
        if mode_compliance > 0.0:
            h_ll += wp.outer(linear, linear) / mode_compliance
            h_al += wp.outer(angular, linear) / mode_compliance
            h_aa += wp.outer(angular, angular) / mode_compliance

    return force, torque, h_ll, h_al, h_aa
