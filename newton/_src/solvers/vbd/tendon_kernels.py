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
    seg_alm_lambda: wp.array[float],
    seg_alm_k: wp.array[float],
    alm_enabled: int,
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
    # Native VBD does not solve XPBD constraint multipliers; with the compliant-ALM
    # stretch row enabled the reported multiplier is the row's tension multiplier [N].
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
    if alm_enabled != 0:
        # Report the ALM row's tension at the accepted pose (its own length rate and the multiplier after the last
        # ascent), split so that ``material + damping`` is that tension: on a taut span material = T - damping,
        # on a slack span the row applies no damping, so damping is reported as 0 and material = T >= 0.
        stretch = length - seg_rest_length[seg]
        lam = seg_alm_lambda[seg]
        seg_lambda[seg] = lam
        seg_material_tension[seg] = 0.0
        if stretch <= 0.0:
            seg_damping_tension[seg] = 0.0
        if stretch > 0.0 or lam > 0.0:
            k_sec = _tendon_secant_stiffness(
                length,
                seg_rest_length[seg],
                compliance,
                sigmoid_ea_low,
                sigmoid_ea_ratio,
                sigmoid_transition_strain,
                sigmoid_transition_width,
            )
            alm_tension, _alm_k_eff = _tendon_alm_row(
                k_sec, seg_active_damping[seg], dt, stretch, length_rate, seg_alm_k[seg], lam
            )
            seg_material_tension[seg] = wp.max(alm_tension, 0.0) - seg_damping_tension[seg]


@wp.func
def _rolling_spin_axis_component(
    body_q: wp.array[wp.transform],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_mu: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    tendon_link_seg_left: wp.array[int],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active: wp.array[int],
    link: int,
    attachment: wp.vec3,
    direction: wp.vec3,
) -> wp.vec3:
    """Roller-axis moment row of a unit-tension span at a ROLLING link.

    Returns ``(1 - spin_scale) * dot(cross(radial, direction), normal) * normal``
    — the roller-axis part of the span's moment that an ideal (or partially
    slipping) pulley cannot transmit through its rim. Zero for non-ROLLING
    links or when the wrap geometry is unavailable.
    """
    if tendon_link_type[link] != int(TendonLinkType.ROLLING):
        return wp.vec3(0.0)
    seg_left = tendon_link_seg_left[link]
    if seg_left < 0:
        return wp.vec3(0.0)
    seg_right = seg_left + 1
    if seg_right >= seg_active.shape[0] or seg_active[seg_left] == 0 or seg_active[seg_right] == 0:
        return wp.vec3(0.0)

    body = tendon_link_body[link]
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
    radial = attachment - center
    return (1.0 - spin_scale) * wp.dot(wp.cross(radial, direction), normal) * normal


# --- Compliant-ALM stretch row (``SolverVBD(tendon_alm=True)``) -------------------------------------------------
#
# The legacy VBD tendon is a Gauss-Newton penalty spring: T = T_law(e) + D*e_dot, H = K + D/dt. Its weight in the
# per-body 6x6 solve relative to the inertia term is K/rho with rho = 1/(dt^2 * w) the row's inertial support, so a
# stiff cable (K/rho >> 1) converges slowly under the coloured Gauss-Seidel sweep and needs more substeps. The ALM
# row keeps the material law exact at convergence but bounds the row's Hessian by rho. Per segment s of a tendon:
#
#     row_k = K_sec + D/dt,  e_row = (K_sec*e + D*e_dot)/row_k                    K_sec = T_law(e)/e (secant)
#     s_ = row_k/(row_k + rho),  k_eff = row_k*rho/(row_k + rho)
#     force   T_s = k_eff*e_row + s_*lambda,   Hessian H = k_eff
#
# lambda and rho are shared along the tendon (a frictionless routed cable carries one tension; the material solve's
# rest-length transfer equalizes the segments' stretch, so a uniform row keeps the segment forces equal across the
# rollers). The ascent, once per VBD iteration after all colour sweeps, uses the series-cable row: total effective
# stretch E = sum_s C_s*(K_s*e_s + D*e_dot_s), series stiffness K_tot = 1/sum_s C_s,
#
#     lambda <- max(s_c*(lambda + rho*E), 0),  s_c = K_tot/(K_tot + rho)          (cables only pull)
#
# whose stationary point lambda = K_tot*E is the compliance-weighted mean of the segments' force residuals, i.e. the
# cable tension; with equal segment tensions every T_s then equals lambda. rho = min(scale*support, K_sec) from the
# step-start pose; a tendon whose K_sec is below min_stiffness_ratio*support keeps the legacy penalty row (rho = 0).
# The ascent divides the force residual by K_s (a stretch) while the force row divides by row_k = K_s + D/dt: the
# stationary point is unaffected; with heavy damping the ascent gain is slightly under-scaled (deliberate, simpler).
# Limitations: the shared multiplier is exact for frictionless routes (mu = 0); a route change (dynamic routing)
# resets the tendon's multiplier, so a roller that activates every step keeps the row soft until the ascent rebuilds it.

_TENDON_ALM_STRETCH_EPS = wp.constant(1.0e-9)


@wp.func
def _tendon_secant_stiffness(
    length: float,
    rest_length: float,
    compliance: float,
    sigmoid_ea_low: float,
    sigmoid_ea_ratio: float,
    sigmoid_transition_strain: float,
    sigmoid_transition_width: float,
) -> float:
    """Secant stiffness ``T_law(e)/e`` of the material law [N/m]; the low-strain tangent when ``e`` is tiny."""
    stretch = length - rest_length
    if stretch > _TENDON_ALM_STRETCH_EPS * wp.max(rest_length, 1.0e-3):
        tension = tendon_material_tension(
            length,
            rest_length,
            compliance,
            sigmoid_ea_low,
            sigmoid_ea_ratio,
            sigmoid_transition_strain,
            sigmoid_transition_width,
        )
        return wp.max(tension / stretch, 1.0e-30)
    return tendon_material_tangent(
        rest_length,
        rest_length,
        compliance,
        sigmoid_ea_low,
        sigmoid_ea_ratio,
        sigmoid_transition_strain,
        sigmoid_transition_width,
    )


@wp.func
def _tendon_alm_row(
    k_sec: float,
    damping: float,
    dt: float,
    stretch: float,
    stretch_rate: float,
    rho: float,
    lam: float,
):
    """Compliant-ALM stretch row: returns ``(tension, effective_stiffness)`` for the body solve."""
    row_k = k_sec + damping / dt
    if stretch <= 0.0:
        # Slack span: damping must not make it transmit tension; only a stored multiplier can still pull.
        stretch_rate = 0.0
    e_row = (k_sec * stretch + damping * stretch_rate) / row_k
    if rho <= 0.0:
        # rho = 0: the tendon is in legacy mode (soft-cable guard) or nothing in it can move. With lam = 0 this is
        # the legacy penalty tension (secant stiffness times stretch: identical for the linear law up to rounding,
        # the secant instead of the tangent for the nonlinear laws) and its Hessian.
        return row_k * e_row + lam, row_k
    s = row_k / (row_k + rho)
    k_eff = row_k * rho / (row_k + rho)
    return k_eff * e_row + s * lam, k_eff


@wp.func
def _tendon_row_inverse_mass(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_mu: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    tendon_link_seg_left: wp.array[int],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_active: wp.array[int],
    body: int,
    link: int,
    attachment: wp.vec3,
    direction: wp.vec3,
) -> float:
    """Generalized inverse mass of one endpoint of a unit-tension stretch row (same moment row as the force)."""
    world_com = wp.transform_point(body_q[body], body_com[body])
    moment_axis = wp.cross(attachment - world_com, direction) - _rolling_spin_axis_component(
        body_q,
        tendon_link_body,
        tendon_link_type,
        tendon_link_radius,
        tendon_link_mu,
        tendon_link_offset,
        tendon_link_axis,
        tendon_link_seg_left,
        seg_attachment_l_local,
        seg_attachment_r_local,
        seg_active,
        link,
        attachment,
        direction,
    )
    moment_local = wp.quat_rotate_inv(wp.transform_get_rotation(body_q[body]), moment_axis)
    return body_inv_mass[body] + wp.dot(moment_local, body_inv_inertia[body] * moment_local)


@wp.kernel
def step_tendon_alm_state(
    dt: float,
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    tendon_start: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_radius: wp.array[float],
    tendon_link_mu: wp.array[float],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    tendon_link_seg_left: wp.array[int],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_active_compliance: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    sigmoid_ea_low: float,
    sigmoid_ea_ratio: float,
    sigmoid_transition_strain: float,
    sigmoid_transition_width: float,
    penalty_scale: float,
    min_stiffness_ratio: float,
    lambda_decay: float,
    tendon_alm_mode: wp.array[wp.int32],
    tendon_alm_lambda: wp.array[float],
    tendon_alm_k: wp.array[float],
    seg_alm_lambda: wp.array[float],
    seg_alm_k: wp.array[float],
    seg_alm_link_l: wp.array[wp.int32],
    seg_alm_link_r: wp.array[wp.int32],
):
    """Per-step compliant-ALM tendon maintenance (mirror of ``step_joint_C0_lambda``), one thread per tendon.

    The multiplier and the penalty metric are shared along the tendon: a frictionless routed cable carries one
    tension, and the rest-length transfer in the material solve equalizes the segments' stretch, so a uniform row
    (same ``rho`` for every segment, one ``lambda``) keeps the segment forces equal across the rollers. Over the
    tendon's movable segments (rows with a positive generalized inverse mass ``w``), from the step-start pose:
    ``rho = min_s(scale * support_s)`` with ``support_s = 1/(dt^2 * w_s)``, capped by ``min_s K_sec,s`` so the row is
    never stiffer than the material; multiplier retention (``lambda_decay``); a reset when any segment's active
    route changed. Guard: the first time the tendon has a movable row, it is assigned the legacy penalty row
    (``rho = 0``, no multiplier) if ``min_s K_sec,s < min_stiffness_ratio * min_s support_s`` — two minima over the
    movable segments, a conservative proxy for the row's conditioning number — and the choice is kept for the life
    of the solver (no row switching on borderline cables; it is not re-evaluated if ``dt`` changes).
    """
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    num_segs = tendon_start[tendon_id + 1] - link_start - 1
    seg_offset = link_start - tendon_id

    rho = float(1.0e30)
    rho_cap = float(1.0e30)
    support_min = float(1.0e30)
    k_sec_min = float(1.0e30)
    n_rows = int(0)
    route_changed = int(0)
    for s in range(num_segs):
        seg = seg_offset + s
        if seg_active[seg] == 0:
            if seg_alm_link_l[seg] != -1 or seg_alm_link_r[seg] != -1:
                route_changed = 1
            seg_alm_link_l[seg] = -1
            seg_alm_link_r[seg] = -1
            continue

        link_l = seg_active_link_l[seg]
        link_r = seg_active_link_r[seg]
        if link_l != seg_alm_link_l[seg] or link_r != seg_alm_link_r[seg]:
            route_changed = 1
            seg_alm_link_l[seg] = link_l
            seg_alm_link_r[seg] = link_r

        body_l = tendon_link_body[link_l]
        body_r = tendon_link_body[link_r]
        attachment_l = wp.transform_point(body_q_prev[body_l], seg_attachment_l_local[seg])
        attachment_r = wp.transform_point(body_q_prev[body_r], seg_attachment_r_local[seg])
        direction = attachment_r - attachment_l
        length = wp.length(direction)
        if length <= 1.0e-8:
            continue
        direction = direction / length

        w = float(0.0)
        if body_l == body_r:
            # Same-body span: endpoint forces cancel, only the net roller-axis moment row remains.
            fix_l = _rolling_spin_axis_component(
                body_q_prev,
                tendon_link_body,
                tendon_link_type,
                tendon_link_radius,
                tendon_link_mu,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_seg_left,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active,
                link_l,
                attachment_l,
                direction,
            )
            fix_r = _rolling_spin_axis_component(
                body_q_prev,
                tendon_link_body,
                tendon_link_type,
                tendon_link_radius,
                tendon_link_mu,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_seg_left,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active,
                link_r,
                attachment_r,
                direction,
            )
            net_local = wp.quat_rotate_inv(wp.transform_get_rotation(body_q_prev[body_l]), fix_l - fix_r)
            w = wp.dot(net_local, body_inv_inertia[body_l] * net_local)
        else:
            w = _tendon_row_inverse_mass(
                body_q_prev,
                body_com,
                body_inv_mass,
                body_inv_inertia,
                tendon_link_body,
                tendon_link_type,
                tendon_link_radius,
                tendon_link_mu,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_seg_left,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active,
                body_l,
                link_l,
                attachment_l,
                direction,
            ) + _tendon_row_inverse_mass(
                body_q_prev,
                body_com,
                body_inv_mass,
                body_inv_inertia,
                tendon_link_body,
                tendon_link_type,
                tendon_link_radius,
                tendon_link_mu,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_seg_left,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active,
                body_r,
                link_r,
                attachment_r,
                direction,
            )

        if w > 1.0e-12:
            compliance = wp.max(seg_active_compliance[seg], _MIN_TENDON_COMPLIANCE)
            k_sec = _tendon_secant_stiffness(
                length,
                seg_rest_length[seg],
                compliance,
                sigmoid_ea_low,
                sigmoid_ea_ratio,
                sigmoid_transition_strain,
                sigmoid_transition_width,
            )
            support = 1.0 / (dt * dt * w)
            rho_cap = wp.min(rho_cap, k_sec)
            k_sec_min = wp.min(k_sec_min, k_sec)
            support_min = wp.min(support_min, support)
            rho = wp.min(rho, penalty_scale * support)
            n_rows += 1

    lam = wp.max(lambda_decay * tendon_alm_lambda[tendon_id], 0.0)
    if route_changed == 1 or n_rows == 0:
        # The multiplier belongs to a route that no longer exists, or nothing can move: start over.
        lam = 0.0
    mode = tendon_alm_mode[tendon_id]
    if mode < 0 and n_rows > 0:
        # Decide once per tendon from its stiffness-to-support ratio (the penalty row's conditioning number):
        # a soft cable keeps the well-conditioned penalty row; the decision is sticky so borderline cables
        # do not switch rows from step to step.
        if k_sec_min >= min_stiffness_ratio * support_min:
            mode = 1
        else:
            mode = 0
        tendon_alm_mode[tendon_id] = mode
    if n_rows == 0 or mode == 0:
        # Nothing can move, or the legacy penalty row was chosen (rho = 0 gives the legacy tension and Hessian).
        lam = 0.0
        rho = 0.0
        rho_cap = 0.0
    rho = wp.min(rho, rho_cap)
    tendon_alm_lambda[tendon_id] = lam
    tendon_alm_k[tendon_id] = rho
    for s in range(num_segs):
        seg = seg_offset + s
        if seg_active[seg] == 0:
            seg_alm_lambda[seg] = 0.0
            seg_alm_k[seg] = 0.0
        else:
            seg_alm_lambda[seg] = lam
            seg_alm_k[seg] = rho


@wp.kernel
def update_duals_tendon(
    dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    tendon_start: wp.array[int],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_attachment_l_local: wp.array[wp.vec3],
    seg_attachment_r_local: wp.array[wp.vec3],
    seg_rest_length: wp.array[float],
    seg_active_compliance: wp.array[float],
    seg_active_damping: wp.array[float],
    seg_active: wp.array[int],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    sigmoid_ea_low: float,
    sigmoid_ea_ratio: float,
    sigmoid_transition_strain: float,
    sigmoid_transition_width: float,
    tendon_alm_lambda: wp.array[float],
    tendon_alm_k: wp.array[float],
    seg_alm_lambda: wp.array[float],
    seg_alm_k: wp.array[float],
):
    """Per-iteration dual ascent of the shared stretch multiplier of each tendon (after all colour sweeps).

    Series-cable row: total effective stretch and series stiffness over the tendon's loaded segments (see the module
    comment), evaluated on the pose the colour sweeps just produced with the rest lengths this iteration's material
    solve gave the force kernel.
    """
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    num_segs = tendon_start[tendon_id + 1] - link_start - 1
    seg_offset = link_start - tendon_id

    lam = tendon_alm_lambda[tendon_id]
    rho = tendon_alm_k[tendon_id]
    if rho <= 0.0:
        # Legacy-mode tendon (or nothing movable): no multiplier, nothing to update.
        tendon_alm_lambda[tendon_id] = 0.0
        for s in range(num_segs):
            seg_alm_lambda[seg_offset + s] = 0.0
        return
    # The tendon is a series of springs carrying one tension: its row has the total effective stretch
    # E = sum_s C_s * (K_s*e_stab,s + D*e_dot,s) and the series stiffness K_tot = 1 / sum_s C_s, so the ascent
    # gain is rho * E (the whole cable's stretch, not one stiff wrap span's) and the stationary point
    # lambda = K_tot * E is the compliance-weighted mean of the segments' force residuals — the cable tension.
    stretch_eff_sum = float(0.0)
    compliance_sum = float(0.0)
    n_rows = int(0)
    for s in range(num_segs):
        seg = seg_offset + s
        if seg_active[seg] == 0:
            continue
        link_l = seg_active_link_l[seg]
        link_r = seg_active_link_r[seg]
        body_l = tendon_link_body[link_l]
        body_r = tendon_link_body[link_r]
        attachment_l = wp.transform_point(body_q[body_l], seg_attachment_l_local[seg])
        attachment_r = wp.transform_point(body_q[body_r], seg_attachment_r_local[seg])
        length = wp.length(attachment_r - attachment_l)
        if length <= 1.0e-8:
            continue
        rest_length = seg_rest_length[seg]
        stretch = length - rest_length
        if stretch <= 0.0 and lam <= 0.0:
            continue
        length_rate = float(0.0)
        if stretch > 0.0:
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
        compliance = wp.max(seg_active_compliance[seg], _MIN_TENDON_COMPLIANCE)
        k_sec = _tendon_secant_stiffness(
            length,
            rest_length,
            compliance,
            sigmoid_ea_low,
            sigmoid_ea_ratio,
            sigmoid_transition_strain,
            sigmoid_transition_width,
        )
        damping = seg_active_damping[seg]
        stretch_eff_sum += (k_sec * stretch + damping * length_rate) / k_sec
        compliance_sum += 1.0 / k_sec
        n_rows += 1

    if n_rows == 0:
        lam = 0.0
    else:
        k_tot = 1.0 / compliance_sum
        s_ = k_tot / (k_tot + rho)
        lam = wp.max(s_ * (lam + rho * stretch_eff_sum), 0.0)
    tendon_alm_lambda[tendon_id] = lam
    for s in range(num_segs):
        seg = seg_offset + s
        if seg_active[seg] == 0:
            seg_alm_lambda[seg] = 0.0
        else:
            seg_alm_lambda[seg] = lam


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
    sigmoid_ea_low: float,
    sigmoid_ea_ratio: float,
    sigmoid_transition_strain: float,
    sigmoid_transition_width: float,
    seg_alm_lambda: wp.array[float],
    seg_alm_k: wp.array[float],
    alm_enabled: int,
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

        compliance = wp.max(seg_active_compliance[seg], _MIN_TENDON_COMPLIANCE)

        rest_length = seg_rest_length[seg]
        tension = float(0.0)
        effective_stiffness = float(0.0)
        if alm_enabled == 0:
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
            effective_stiffness = stiffness + damping / dt
        else:
            # Compliant-ALM stretch row: a segment whose multiplier still carries tension is not skipped at
            # the slack boundary; the multiplier releases through the clamped ascent instead.
            stretch = length - rest_length
            lam = seg_alm_lambda[seg]
            if stretch <= 0.0 and lam <= 0.0:
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
            k_sec = _tendon_secant_stiffness(
                length,
                rest_length,
                compliance,
                sigmoid_ea_low,
                sigmoid_ea_ratio,
                sigmoid_transition_strain,
                sigmoid_transition_width,
            )
            alm_tension, alm_k_eff = _tendon_alm_row(
                k_sec, seg_active_damping[seg], dt, stretch, length_rate, seg_alm_k[seg], lam
            )
            tension = wp.max(alm_tension, 0.0)
            effective_stiffness = alm_k_eff

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
                body_q,
                tendon_link_body,
                tendon_link_type,
                tendon_link_radius,
                tendon_link_mu,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_seg_left,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active,
                link_l,
                attachment_l,
                direction,
            )
            fix_r = _rolling_spin_axis_component(
                body_q,
                tendon_link_body,
                tendon_link_type,
                tendon_link_radius,
                tendon_link_mu,
                tendon_link_offset,
                tendon_link_axis,
                tendon_link_seg_left,
                seg_attachment_l_local,
                seg_attachment_r_local,
                seg_active,
                link_r,
                attachment_r,
                direction,
            )
            net_moment_axis = fix_l - fix_r
            torque = torque - tension * net_moment_axis
            h_aa = h_aa + effective_stiffness * wp.outer(net_moment_axis, net_moment_axis)
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
                    radial = attachment - center
                    spin_moment_axis = wp.cross(radial, direction)
                    spin_torque = wp.cross(radial, body_force)
                    moment_axis = moment_axis - (1.0 - spin_scale) * wp.dot(spin_moment_axis, normal) * normal
                    body_torque = body_torque - (1.0 - spin_scale) * wp.dot(spin_torque, normal) * normal

        force = force + body_force
        torque = torque + body_torque
        # The axial Gauss-Newton approximation remains positive semidefinite.
        h_ll = h_ll + effective_stiffness * wp.outer(direction, direction)
        h_al = h_al + effective_stiffness * wp.outer(moment_axis, direction)
        h_aa = h_aa + effective_stiffness * wp.outer(moment_axis, moment_axis)

    return force, torque, h_ll, h_al, h_aa
