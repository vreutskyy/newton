# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared routed-tendon solver state helpers.

The routed tendon geometry is solver-independent: XPBD and VBD both need the
same tangent attachments, mutable free-span rest lengths, and segment-to-link
mapping before applying their own numerical solve.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..sim import Model
from ..sim.tendon import TendonLinkFlags, TendonLinkType
from .tendon_kernels import (
    prepare_tendon_route,
    snapshot_tendon_link_active,
    solve_tendon_material,
    solve_tendon_material_direct,
    update_tendon_attachments,
    update_tendon_cone_rows,
    update_tendon_link_active,
)
from .tendon_material import TendonMaterialStatus
from .tendon_material_cooperative_kernels import solve_tendon_material_cooperative
from .tendon_material_nonlinear import TendonMaterialNonlinearStatus, allocate_tendon_material_nonlinear_state
from .tendon_material_state import TendonMaterialState


def _transform_point_np(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a Newton transform (px,py,pz,qx,qy,qz,qw) to a 3D point using numpy."""
    p = pose[:3]
    q = pose[3:]
    t = 2.0 * np.cross(q[:3], point)
    return point + q[3] * t + np.cross(q[:3], t) + p


def _transform_vector_np(pose: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by the quaternion in a Newton transform."""
    q = pose[3:]
    t = 2.0 * np.cross(q[:3], vec)
    return vec + q[3] * t + np.cross(q[:3], t)


def _tangent_point_circle_np(
    point: np.ndarray,
    center: np.ndarray,
    radius: float,
    plane_normal: np.ndarray,
    orientation: int,
) -> np.ndarray:
    """Compute the tangent point on a circle from an external point."""
    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1.0e-12)

    d = center - point
    d_proj = d - np.dot(d, normal) * normal
    dist = float(np.linalg.norm(d_proj))
    if dist <= radius:
        if dist < 1.0e-8:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            fallback -= np.dot(fallback, normal) * normal
            return center + radius * fallback / max(float(np.linalg.norm(fallback)), 1.0e-12)
        return center - radius * d_proj / dist

    u = d_proj / dist
    v = np.cross(normal, u)
    phi = np.arcsin(min(radius / dist, 1.0))
    angle = -0.5 * np.pi - phi if orientation > 0 else 0.5 * np.pi + phi
    return center + radius * (np.cos(angle) * u + np.sin(angle) * v)


def _segment_attachment_points_np(
    center_l: np.ndarray,
    center_r: np.ndarray,
    type_l: int,
    type_r: int,
    radius_l: float,
    radius_r: float,
    orient_l: int,
    orient_r: int,
    normal_l: np.ndarray,
    normal_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute free-span endpoints with the same tangent cases as the Warp kernel."""
    new_l = np.asarray(center_l, dtype=np.float64)
    new_r = np.asarray(center_r, dtype=np.float64)
    rolling = int(TendonLinkType.ROLLING)

    if type_l == rolling and type_r == rolling and radius_l > 0.0 and radius_r > 0.0:
        for _iter in range(10):
            new_r = _tangent_point_circle_np(new_l, center_r, radius_r, normal_r, orient_r)
            new_l = _tangent_point_circle_np(new_r, center_l, radius_l, normal_l, -orient_l)
    elif type_l == rolling and radius_l > 0.0:
        new_l = _tangent_point_circle_np(center_r, center_l, radius_l, normal_l, -orient_l)
        new_r = np.asarray(center_r, dtype=np.float64)
    elif type_r == rolling and radius_r > 0.0:
        new_l = np.asarray(center_l, dtype=np.float64)
        new_r = _tangent_point_circle_np(center_l, center_r, radius_r, normal_r, orient_r)

    return new_l, new_r


class TendonStateMixin:
    """Mixin that allocates routed-tendon mutable state on a solver instance."""

    def _init_direct_tendon_material(self, model: Model) -> None:
        """Validate and allocate the opt-in direct solver's packed workspace."""
        if not getattr(self, "tendon_material_direct", False):
            return
        if model.requires_grad:
            raise ValueError("tendon_material_direct does not support differentiable simulation")
        if model.tendon_segment_count:
            compliance = model.tendon_seg_compliance.numpy()
            if not np.all(np.isfinite(compliance) & (compliance >= 1.0e-25)):
                raise ValueError("tendon_material_direct requires finite segment compliance >= 1e-25")
            starts = model.tendon_start.numpy()
            kinds = model.tendon_link_type.numpy()
            for tendon in range(model.tendon_count):
                count = 0
                for kind in kinds[starts[tendon] + 1 : starts[tendon + 1]]:
                    count += 1
                    if count > 32:
                        raise ValueError(
                            "tendon_material_direct supports at most 32 authored spans between attachments "
                            f"(tendon {tendon})"
                        )
                    if kind == int(TendonLinkType.ATTACHMENT):
                        count = 0

        state = self._tendon_material_state
        state.enabled = True
        state.raw_compliance = model.tendon_seg_compliance
        self._tendon_material_kernel = solve_tendon_material_direct
        for name in ("initial", "compliance", "cap", "upper", "output"):
            setattr(state, name, wp.zeros(model.tendon_segment_count, dtype=float, device=model.device))
        for name in ("status", "faces", "valid", "pieces", "count", "next_link", "incoming"):
            setattr(state, name, wp.zeros(model.tendon_segment_count, dtype=int, device=model.device))
        for name in ("ids", "edge_ids"):
            setattr(state, name, wp.full(model.tendon_segment_count, -1, dtype=int, device=model.device))
        state.failure = wp.zeros(model.tendon_count, dtype=int, device=model.device)
        state.failure_component = wp.full(model.tendon_count, -1, dtype=int, device=model.device)
        state.nonlinear_enabled = self.tendon_sigmoid_ea_low > 0.0
        if state.nonlinear_enabled:
            state.ea_low = self.tendon_sigmoid_ea_low
            state.ea_ratio = self.tendon_sigmoid_ea_ratio
            state.transition_strain = self.tendon_sigmoid_transition_strain
            state.transition_width = self.tendon_sigmoid_transition_width
            state.nonlinear = allocate_tendon_material_nonlinear_state(
                model.tendon_segment_count, model.tendon_segment_count, model.tendon_segment_count, device=model.device
            )
            for name in ("cap", "output", "status", "faces", "valid"):
                setattr(state.nonlinear, name, getattr(state, name))
            if model.device.is_cuda:
                self._tendon_material_kernel = solve_tendon_material_cooperative
                self._tendon_material_lanes = 32
                self._tendon_material_block_dim = 32

    def _validate_direct_tendon_law(self) -> None:
        """Keep the experimental packed law synchronized with constructor settings."""
        state = self._tendon_material_state
        if not state.enabled:
            return
        expected = (
            self.tendon_sigmoid_ea_low,
            self.tendon_sigmoid_ea_ratio,
            self.tendon_sigmoid_transition_strain,
            self.tendon_sigmoid_transition_width,
        )
        if state.nonlinear_enabled != (self.tendon_sigmoid_ea_low > 0.0):
            raise ValueError("Reconstruct the solver to change the direct tendon material law")
        if state.nonlinear_enabled and expected != (
            state.ea_low,
            state.ea_ratio,
            state.transition_strain,
            state.transition_width,
        ):
            raise ValueError("Reconstruct the solver to change the direct tendon material law")

    def check_tendon_material(self) -> None:
        """Check the experimental direct material solve for latched failures.

        Call outside CUDA graph capture, after a step or graph replay and before
        consuming its results. This synchronizes the failure status to the host.
        GPU execution is not rolled back on failure: discard the entire affected
        step/frame/batch and reconstruct the solver after correcting the input.
        No material sweeps are used as a fallback. Does nothing in sweep mode.

        Raises:
            RuntimeError: A direct solve received unsupported input, could not
                satisfy the rest-length bound, or failed numerical validation.
        """
        state = self._tendon_material_state
        if not state.enabled:
            return
        failures = state.failure.numpy()
        failed = np.flatnonzero(failures)
        if failed.size:
            tendon = int(failed[0])
            code = int(failures[tendon])
            names = {-102: "INVALID_ROUTE", -103: "COMPONENT_TOO_LARGE"}
            try:
                reason = (
                    TendonMaterialNonlinearStatus(code + 200).name if code < -200 else TendonMaterialStatus(code).name
                )
            except ValueError:
                reason = names.get(code, "UNKNOWN_FAILURE")
            component = int(state.failure_component.numpy()[tendon])
            raise RuntimeError(
                f"Direct tendon material solve failed: {reason} ({code}), tendon {tendon}, "
                f"component starting at segment {component}. Discard this step/frame/batch; "
                "correct the input and reconstruct the solver. No sweep fallback was applied."
            )

    def _init_tendon_state(self, model: Model, allocate_xpbd_lambdas: bool = True) -> None:
        """Allocate mutable tendon state arrays and build segment/link mappings."""
        self._tendon_material_state = TendonMaterialState()
        self._tendon_material_state.enabled = False
        self._tendon_material_kernel = solve_tendon_material
        self._tendon_material_lanes = 1
        self._tendon_material_block_dim = 256
        self._has_dynamic_tendon_links = False
        # Solver-level cable cone parameters (a solver may override before calling this).
        if not hasattr(self, "tendon_max_sweeps"):
            self.tendon_max_sweeps = 256
        if not hasattr(self, "tendon_settle_tol"):
            self.tendon_settle_tol = 1.0e-3
        if not hasattr(self, "tendon_activation_tol"):
            self.tendon_activation_tol = 2.0e-3
        if not hasattr(self, "tendon_sigmoid_ea_low"):
            self.tendon_sigmoid_ea_low = 0.0
        if not hasattr(self, "tendon_sigmoid_ea_ratio"):
            self.tendon_sigmoid_ea_ratio = 1.0
        if not hasattr(self, "tendon_sigmoid_transition_strain"):
            self.tendon_sigmoid_transition_strain = 0.0
        if not hasattr(self, "tendon_sigmoid_transition_width"):
            self.tendon_sigmoid_transition_width = 1.0
        if not 1 <= self.tendon_max_sweeps <= 256:
            raise ValueError(f"tendon_max_sweeps must be between 1 and 256, got {self.tendon_max_sweeps}")
        if self.tendon_settle_tol < 0.0:
            raise ValueError(f"tendon_settle_tol must be non-negative, got {self.tendon_settle_tol}")
        if not 0.0 <= self.tendon_activation_tol < 1.0:
            raise ValueError(
                f"tendon_activation_tol must be between 0 (inclusive) and 1 (exclusive), "
                f"got {self.tendon_activation_tol}"
            )
        if self.tendon_sigmoid_ea_low < 0.0:
            raise ValueError(f"tendon_sigmoid_ea_low must be non-negative, got {self.tendon_sigmoid_ea_low}")
        if self.tendon_sigmoid_ea_low > 0.0:
            if self.tendon_sigmoid_ea_ratio < 1.0:
                raise ValueError(f"tendon_sigmoid_ea_ratio must be at least 1, got {self.tendon_sigmoid_ea_ratio}")
            if self.tendon_sigmoid_transition_strain < 0.0:
                raise ValueError(
                    "tendon_sigmoid_transition_strain must be non-negative, "
                    f"got {self.tendon_sigmoid_transition_strain}"
                )
            if self.tendon_sigmoid_transition_width <= 0.0:
                raise ValueError(
                    f"tendon_sigmoid_transition_width must be positive, got {self.tendon_sigmoid_transition_width}"
                )
        self._init_direct_tendon_material(model)
        if model.tendon_segment_count == 0:
            self.tendon_seg_rest_length = None
            self.tendon_seg_rest_length_step = None
            self.tendon_seg_route_rest_length = None
            self.tendon_seg_stretch = None
            self.tendon_seg_material_tension = None
            self.tendon_seg_damping_tension = None
            self.tendon_seg_attachment_l = None
            self.tendon_seg_attachment_r = None
            self.tendon_seg_length = None
            self.tendon_seg_attachment_l_local = None
            self.tendon_seg_attachment_r_local = None
            self.tendon_seg_attachment_l_local_step = None
            self.tendon_seg_attachment_r_local_step = None
            self.tendon_seg_lambda = None
            self.tendon_seg_delta_lambda = None
            self.tendon_seg_rolling_delta_l = None
            self.tendon_seg_rolling_delta_r = None
            self.tendon_cone_sweep_count = None
            self.tendon_seg_link_l = None
            self.tendon_seg_active = None
            self.tendon_seg_active_link_l = None
            self.tendon_seg_active_link_r = None
            self.tendon_seg_active_compliance = None
            self.tendon_seg_active_damping = None
            self.tendon_link_active = None
            self.tendon_link_active_step = None
            self.tendon_link_route_rest_length = None
            self.tendon_link_seg_left = None
            self.tendon_link_tendon = None
            self.tendon_link_cone_seg_l = None
            self.tendon_link_cone_seg_r = None
            self.tendon_link_cap_ratio = None
            self.tendon_total_cable = None
            return

        with wp.ScopedDevice(model.device):
            self.tendon_seg_attachment_l = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_length = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_seg_attachment_l_local = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r_local = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_l_local_step = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r_local_step = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            # Unilateral constitutive tension; the signed damping term is reported separately.
            self.tendon_seg_material_tension = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_seg_lambda = (
                wp.zeros(model.tendon_segment_count, dtype=float) if allocate_xpbd_lambdas else None
            )
            self.tendon_seg_delta_lambda = (
                wp.zeros(model.tendon_segment_count, dtype=float) if allocate_xpbd_lambdas else None
            )
            self.tendon_seg_rolling_delta_l = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_seg_rolling_delta_r = wp.zeros(model.tendon_segment_count, dtype=float)
            # Cached instantaneous damping term used by routing and slip projections.
            self.tendon_seg_damping_tension = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_cone_sweep_count = wp.zeros(model.tendon_count, dtype=wp.int32)
            self.tendon_seg_active = wp.ones(model.tendon_segment_count, dtype=wp.int32)
            self.tendon_seg_active_link_l = wp.zeros(model.tendon_segment_count, dtype=wp.int32)
            self.tendon_seg_active_link_r = wp.zeros(model.tendon_segment_count, dtype=wp.int32)
            self.tendon_seg_active_compliance = wp.array(
                model.tendon_seg_compliance.numpy().copy(), dtype=float, device=model.device
            )
            self.tendon_seg_active_damping = wp.array(
                model.tendon_seg_damping.numpy().copy(), dtype=float, device=model.device
            )
            self.tendon_link_active = wp.ones(model.tendon_link_count, dtype=bool)
            self.tendon_link_active_step = wp.ones(model.tendon_link_count, dtype=bool)
            self.tendon_link_route_rest_length = wp.zeros(model.tendon_link_count, dtype=float)
            self.tendon_link_cone_seg_l = wp.full(model.tendon_link_count, -1, dtype=wp.int32)
            self.tendon_link_cone_seg_r = wp.full(model.tendon_link_count, -1, dtype=wp.int32)
            self.tendon_link_cap_ratio = wp.ones(model.tendon_link_count, dtype=float)
            self.tendon_total_cable = wp.zeros(model.tendon_count, dtype=float)

            tendon_start_np = model.tendon_start.numpy()
            seg_link_l = []
            link_seg_left = np.full(model.tendon_link_count, -1, dtype=np.int32)
            link_tendon = np.empty(model.tendon_link_count, dtype=np.int32)
            seg = 0
            for t in range(model.tendon_count):
                start = tendon_start_np[t]
                end = tendon_start_np[t + 1]
                link_tendon[start:end] = t
                for link_idx in range(start, end - 1):
                    seg_link_l.append(link_idx)
                    if link_idx + 1 < end - 1:
                        link_seg_left[link_idx + 1] = seg
                    seg += 1

            self.tendon_seg_link_l = wp.array(seg_link_l, dtype=wp.int32, device=model.device)
            self.tendon_seg_active_link_l = wp.array(seg_link_l, dtype=wp.int32, device=model.device)
            self.tendon_seg_active_link_r = wp.array(
                np.asarray(seg_link_l, dtype=np.int32) + 1, dtype=wp.int32, device=model.device
            )
            self.tendon_link_seg_left = wp.array(link_seg_left, dtype=wp.int32, device=model.device)
            self.tendon_link_tendon = wp.array(link_tendon, dtype=wp.int32, device=model.device)

            rest_np = model.tendon_seg_rest_length.numpy().copy()
            auto_mask = rest_np < 0.0
            rest_np[auto_mask] = 0.0
            self.tendon_seg_rest_length = wp.array(rest_np, dtype=float, device=model.device)
            self.tendon_seg_rest_length_step = wp.array(rest_np.copy(), dtype=float, device=model.device)
            self.tendon_seg_route_rest_length = wp.array(rest_np.copy(), dtype=float, device=model.device)
            # scratch: per-segment stretch d = len - rest, snapshot+telescoped inside the capstan
            # transport (kept at its own scale so stiff-cable friction transfers survive float32)
            self.tendon_seg_stretch = wp.zeros_like(self.tendon_seg_rest_length)

            link_type_np = model.tendon_link_type.numpy()
            link_flags_np = model.tendon_link_flags.numpy()
            self._has_dynamic_tendon_links = bool(
                np.any(
                    (link_type_np == int(TendonLinkType.ROLLING))
                    & ((link_flags_np & int(TendonLinkFlags.DYNAMIC)) != 0)
                )
            )
            if self._has_dynamic_tendon_links and model.body_q is not None:
                # Resolve the initial topology before measuring its free-span rest lengths.
                self._update_tendon_link_active(model, model.body_q)
                wp.copy(self.tendon_link_active_step, self.tendon_link_active)

            route_rest_np, route_seg_mask = self._compute_active_route_rest_lengths(model)
            self.tendon_link_route_rest_length = wp.array(route_rest_np, dtype=float, device=model.device)

            self._init_tendon_attachment_points(model, auto_mask, route_seg_mask)

    def _snapshot_tendon_step_state(self) -> None:
        """Snapshot mutable tendon material state at the start of a time step."""
        if self.tendon_seg_rest_length is None:
            return

        wp.copy(self.tendon_seg_rest_length_step, self.tendon_seg_rest_length)
        wp.copy(self.tendon_seg_attachment_l_local_step, self.tendon_seg_attachment_l_local)
        wp.copy(self.tendon_seg_attachment_r_local_step, self.tendon_seg_attachment_r_local)
        if self._has_dynamic_tendon_links:
            wp.launch(
                kernel=snapshot_tendon_link_active,
                dim=self.tendon_link_active.shape[0],
                inputs=[self.tendon_link_active, self.tendon_link_active_step, self.model.tendon_link_flags],
                device=self.tendon_link_active.device,
            )

    def _update_tendon_link_active(self, model: Model, body_q: wp.array[wp.transform]) -> None:
        """Update solver-owned dynamic routing flags from the current body poses."""
        if not self._has_dynamic_tendon_links:
            return

        wp.launch(
            kernel=update_tendon_link_active,
            dim=model.tendon_count,
            inputs=[
                body_q,
                model.tendon_start,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_flags,
                model.tendon_link_radius,
                model.tendon_link_orientation,
                model.tendon_link_offset,
                model.tendon_link_axis,
                self.tendon_activation_tol,
                self.tendon_link_active,
            ],
            device=model.device,
        )

    def _prepare_tendon_route(
        self,
        model: Model,
        body_q: wp.array[wp.transform],
        compliance_floor: float = 0.0,
    ) -> None:
        """Build the active route and merged segment properties for one solver step."""
        if model.tendon_segment_count == 0:
            return

        if self._tendon_material_state.enabled:
            # Also track array replacement outside capture, not just in-place updates.
            self._tendon_material_state.raw_compliance = model.tendon_seg_compliance

        wp.launch(
            kernel=prepare_tendon_route,
            dim=model.tendon_count,
            inputs=[
                body_q,
                model.tendon_start,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_flags,
                model.tendon_link_radius,
                model.tendon_link_offset,
                model.tendon_link_axis,
                self.tendon_seg_rest_length_step,
                model.tendon_seg_compliance,
                model.tendon_seg_damping,
                self.tendon_link_active,
                self.tendon_link_active_step,
                self.tendon_link_route_rest_length,
                self.tendon_seg_attachment_l_local_step,
                self.tendon_seg_attachment_r_local_step,
                compliance_floor,
            ],
            outputs=[
                self.tendon_seg_route_rest_length,
                self.tendon_seg_active,
                self.tendon_seg_active_link_l,
                self.tendon_seg_active_link_r,
                self.tendon_seg_active_compliance,
                self.tendon_seg_active_damping,
            ],
            device=model.device,
        )

    def _update_tendon_cone_rows(
        self,
        model: Model,
        body_q: wp.array[wp.transform],
        report_unsupported_wrap: bool,
    ) -> None:
        """Cache geometry-dependent segment pairs and capstan ratios for material rows."""
        wp.launch(
            kernel=update_tendon_cone_rows,
            dim=model.tendon_link_count,
            inputs=[
                body_q,
                model.tendon_start,
                self.tendon_link_tendon,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_radius,
                model.tendon_link_orientation,
                model.tendon_link_mu,
                model.tendon_link_offset,
                model.tendon_link_axis,
                self.tendon_link_active,
                self.tendon_seg_active,
                self.tendon_seg_active_link_l,
                self.tendon_seg_active_link_r,
                self.tendon_seg_attachment_l,
                self.tendon_seg_attachment_r,
                self.tendon_seg_length,
                int(report_unsupported_wrap),
                self._tendon_material_state.enabled,
            ],
            outputs=[
                self.tendon_link_cone_seg_l,
                self.tendon_link_cone_seg_r,
                self.tendon_link_cap_ratio,
            ],
            device=model.device,
        )

    def _compute_active_route_rest_lengths(self, model: Model) -> tuple[np.ndarray, np.ndarray]:
        """Compute bypass material lengths for dynamically routed rolling links."""
        route_rest = np.zeros(model.tendon_link_count, dtype=np.float32)
        route_seg_mask = np.zeros(model.tendon_segment_count, dtype=bool)
        body_q = model.body_q
        if body_q is None:
            return route_rest, route_seg_mask

        tendon_start = model.tendon_start.numpy()
        link_body = model.tendon_link_body.numpy()
        link_type = model.tendon_link_type.numpy()
        link_radius = model.tendon_link_radius.numpy()
        link_orientation = model.tendon_link_orientation.numpy()
        link_flags = model.tendon_link_flags.numpy()
        link_active = self.tendon_link_active.numpy()
        link_offset = model.tendon_link_offset.numpy()
        link_axis = model.tendon_link_axis.numpy()
        body_q_np = body_q.numpy()

        seg_base = 0
        for t in range(model.tendon_count):
            start = tendon_start[t]
            end = tendon_start[t + 1]
            for i in range(start + 1, end - 1):
                if link_type[i] != int(TendonLinkType.ROLLING) or (link_flags[i] & int(TendonLinkFlags.DYNAMIC)) == 0:
                    continue

                left_seg = seg_base + (i - start) - 1
                right_seg = left_seg + 1
                if not link_active[i]:
                    route_seg_mask[left_seg] = True
                    route_seg_mask[right_seg] = True

                link_l = i - 1
                link_r = i + 1
                pose_l = body_q_np[link_body[link_l]]
                pose_r = body_q_np[link_body[link_r]]
                center_l = _transform_point_np(pose_l, link_offset[link_l]).astype(np.float64)
                center_r = _transform_point_np(pose_r, link_offset[link_r]).astype(np.float64)
                normal_l = _transform_vector_np(pose_l, link_axis[link_l])
                normal_r = _transform_vector_np(pose_r, link_axis[link_r])
                p0, p1 = _segment_attachment_points_np(
                    center_l,
                    center_r,
                    int(link_type[link_l]),
                    int(link_type[link_r]),
                    float(link_radius[link_l]),
                    float(link_radius[link_r]),
                    int(link_orientation[link_l]),
                    int(link_orientation[link_r]),
                    normal_l,
                    normal_r,
                )
                route_rest[i] = float(np.linalg.norm(p1 - p0))

            seg_base += end - start - 1

        return route_rest, route_seg_mask

    def _init_tendon_attachment_points(self, model: Model, auto_mask: np.ndarray, route_seg_mask: np.ndarray) -> None:
        """Compute initial tendon tangent attachments and rest lengths."""
        body_q = model.body_q
        if body_q is None:
            return

        tendon_start_np = model.tendon_start.numpy()
        link_body_np = model.tendon_link_body.numpy()
        link_offset_np = model.tendon_link_offset.numpy()
        body_q_np = body_q.numpy()

        att_l = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)
        att_r = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)
        att_l_local = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)
        att_r_local = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)

        seg = 0
        for t in range(model.tendon_count):
            start = tendon_start_np[t]
            end = tendon_start_np[t + 1]
            for i in range(start, end - 1):
                body_l = link_body_np[i]
                body_r = link_body_np[i + 1]
                off_l = link_offset_np[i]
                off_r = link_offset_np[i + 1]
                att_l[seg] = _transform_point_np(body_q_np[body_l], off_l)
                att_r[seg] = _transform_point_np(body_q_np[body_r], off_r)
                att_l_local[seg] = off_l
                att_r_local[seg] = off_r
                seg += 1

        with wp.ScopedDevice(model.device):
            self.tendon_seg_attachment_l = wp.array(att_l, dtype=wp.vec3, device=model.device)
            self.tendon_seg_attachment_r = wp.array(att_r, dtype=wp.vec3, device=model.device)
            self.tendon_seg_attachment_l_local = wp.array(att_l_local, dtype=wp.vec3, device=model.device)
            self.tendon_seg_attachment_r_local = wp.array(att_r_local, dtype=wp.vec3, device=model.device)

        self._prepare_tendon_route(model, body_q)

        wp.launch(
            kernel=update_tendon_attachments,
            dim=model.tendon_segment_count,
            inputs=[
                body_q,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_flags,
                model.tendon_link_radius,
                model.tendon_link_orientation,
                model.tendon_link_offset,
                model.tendon_link_axis,
                self.tendon_seg_active,
                self.tendon_seg_active_link_l,
                self.tendon_seg_active_link_r,
                self.tendon_link_active,
                self.tendon_link_active_step,
                self.tendon_seg_attachment_l_local_step,
                self.tendon_seg_attachment_r_local_step,
                0,
            ],
            outputs=[
                self.tendon_seg_attachment_l,
                self.tendon_seg_attachment_r,
                self.tendon_seg_attachment_l_local,
                self.tendon_seg_attachment_r_local,
                self.tendon_seg_rolling_delta_l,
                self.tendon_seg_rolling_delta_r,
                self.tendon_seg_length,
            ],
            device=model.device,
        )

        wp.launch(
            kernel=self._tendon_material_kernel,
            dim=model.tendon_count * self._tendon_material_lanes,
            block_dim=self._tendon_material_block_dim,
            inputs=[
                body_q,
                model.body_qd,
                body_q,
                model.body_com,
                model.tendon_start,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_radius,
                model.tendon_link_offset,
                model.tendon_link_axis,
                self.tendon_seg_rest_length,
                self.tendon_seg_rest_length_step,
                self.tendon_seg_route_rest_length,
                self.tendon_seg_stretch,
                self.tendon_seg_damping_tension,
                self.tendon_seg_active,
                self.tendon_seg_active_link_l,
                self.tendon_seg_active_link_r,
                self.tendon_seg_active_compliance,
                self.tendon_seg_active_damping,
                self.tendon_link_active,
                self.tendon_link_active_step,
                self.tendon_link_route_rest_length,
                self.tendon_seg_attachment_l,
                self.tendon_seg_attachment_r,
                self.tendon_seg_length,
                self.tendon_seg_attachment_l_local,
                self.tendon_seg_attachment_r_local,
                self.tendon_seg_rolling_delta_l,
                self.tendon_seg_rolling_delta_r,
                self.tendon_link_cone_seg_l,
                self.tendon_link_cone_seg_r,
                self.tendon_link_cap_ratio,
                self.tendon_cone_sweep_count,
                0,
                0.0,
                0,
                0,
                0,
                self.tendon_max_sweeps,
                self.tendon_settle_tol,
                self.tendon_sigmoid_ea_low,
                self.tendon_sigmoid_ea_ratio,
                self.tendon_sigmoid_transition_strain,
                self.tendon_sigmoid_transition_width,
                self._tendon_material_state,
            ],
            device=model.device,
        )

        att_l_np = self.tendon_seg_attachment_l.numpy()
        att_r_np = self.tendon_seg_attachment_r.numpy()
        rest_np = self.tendon_seg_rest_length.numpy()
        for i in range(model.tendon_segment_count):
            if auto_mask[i] and not route_seg_mask[i]:
                rest_np[i] = np.linalg.norm(att_r_np[i] - att_l_np[i])
        self.tendon_seg_rest_length = wp.array(rest_np, dtype=float, device=model.device)
        self._snapshot_tendon_step_state()

        link_type_np = model.tendon_link_type.numpy()
        link_radius_np = model.tendon_link_radius.numpy()
        link_offset_np = model.tendon_link_offset.numpy()
        link_axis_np = model.tendon_link_axis.numpy()
        link_active_np = self.tendon_link_active.numpy()
        seg_active_np = self.tendon_seg_active.numpy()
        seg_active_link_l_np = self.tendon_seg_active_link_l.numpy()
        seg_active_link_r_np = self.tendon_seg_active_link_r.numpy()

        total_cable = np.zeros(model.tendon_count, dtype=np.float32)
        seg = 0
        for t in range(model.tendon_count):
            start = tendon_start_np[t]
            end = tendon_start_np[t + 1]
            num_links = end - start
            seg_base = seg
            cable_len = 0.0
            for s in range(num_links - 1):
                if seg_active_np[seg_base + s] != 0:
                    cable_len += rest_np[seg_base + s]
            for i in range(start + 1, end - 1):
                if link_type_np[i] == int(TendonLinkType.ROLLING):
                    if not link_active_np[i]:
                        continue
                    body_idx = link_body_np[i]
                    q = body_q_np[body_idx]
                    center = _transform_point_np(q, link_offset_np[i])
                    normal = _transform_vector_np(q, link_axis_np[i])
                    radius = link_radius_np[i]
                    pt_left = None
                    pt_right = None
                    for s in range(num_links - 1):
                        seg_idx = seg_base + s
                        if seg_active_np[seg_idx] == 0:
                            continue
                        if seg_active_link_r_np[seg_idx] == i:
                            pt_left = att_r_np[seg_idx]
                        if seg_active_link_l_np[seg_idx] == i:
                            pt_right = att_l_np[seg_idx]
                    if pt_left is None or pt_right is None:
                        continue

                    r_l = pt_left - center
                    r_r = pt_right - center
                    cross_val = np.dot(np.cross(r_l, r_r), normal)
                    dot_val = np.dot(r_l, r_r)
                    theta = abs(np.arctan2(cross_val, dot_val))
                    cable_len += theta * radius
            total_cable[t] = cable_len
            seg += num_links - 1

        self.tendon_total_cable = wp.array(total_cable, dtype=float, device=model.device)
