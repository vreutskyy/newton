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
from .tendon_kernels import snapshot_tendon_link_active, update_tendon_attachments, update_tendon_link_active


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


class TendonStateMixin:
    """Mixin that allocates routed-tendon mutable state on a solver instance."""

    def _init_tendon_state(self, model: Model, allocate_xpbd_lambdas: bool = True) -> None:
        """Allocate mutable tendon state arrays and build segment/link mappings."""
        self._has_dynamic_tendon_links = False
        # Solver-level cable cone parameters (a solver may override before calling this).
        if not hasattr(self, "tendon_max_sweeps"):
            self.tendon_max_sweeps = 256
        if not hasattr(self, "tendon_settle_tol"):
            self.tendon_settle_tol = 1.0e-3
        if not 1 <= self.tendon_max_sweeps <= 256:
            raise ValueError(f"tendon_max_sweeps must be between 1 and 256, got {self.tendon_max_sweeps}")
        if self.tendon_settle_tol < 0.0:
            raise ValueError(f"tendon_settle_tol must be non-negative, got {self.tendon_settle_tol}")
        if model.tendon_segment_count == 0:
            self.tendon_seg_rest_length = None
            self.tendon_seg_rest_length_step = None
            self.tendon_seg_stretch = None
            self.tendon_seg_attachment_l = None
            self.tendon_seg_attachment_r = None
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
            self.tendon_link_wrap_angle = None
            self.tendon_link_wrap_angle_step = None
            self.tendon_link_seg_left = None
            self.tendon_total_cable = None
            return

        with wp.ScopedDevice(model.device):
            self.tendon_seg_attachment_l = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_l_local = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r_local = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_l_local_step = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_attachment_r_local_step = wp.zeros(model.tendon_segment_count, dtype=wp.vec3)
            self.tendon_seg_lambda = (
                wp.zeros(model.tendon_segment_count, dtype=float) if allocate_xpbd_lambdas else None
            )
            self.tendon_seg_delta_lambda = (
                wp.zeros(model.tendon_segment_count, dtype=float) if allocate_xpbd_lambdas else None
            )
            self.tendon_seg_rolling_delta_l = wp.zeros(model.tendon_segment_count, dtype=float)
            self.tendon_seg_rolling_delta_r = wp.zeros(model.tendon_segment_count, dtype=float)
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
            link_active_np = model.tendon_link_active.numpy().copy()
            self.tendon_link_active = wp.array(link_active_np, dtype=bool, device=model.device)
            self.tendon_link_active_step = wp.array(link_active_np, dtype=bool, device=model.device)
            self.tendon_link_wrap_angle = wp.zeros(model.tendon_link_count, dtype=float)
            self.tendon_link_wrap_angle_step = wp.zeros(model.tendon_link_count, dtype=float)
            self.tendon_total_cable = wp.zeros(model.tendon_count, dtype=float)

            seg_link_l = model.tendon_seg_link_l.numpy().copy()
            seg_link_r = model.tendon_seg_link_r.numpy().copy()
            link_seg_left = np.full(model.tendon_link_count, -1, dtype=np.int32)
            for seg, link_r in enumerate(seg_link_r):
                link_seg_left[link_r] = seg

            self.tendon_seg_link_l = wp.array(seg_link_l, dtype=wp.int32, device=model.device)
            self.tendon_seg_active_link_l = wp.array(seg_link_l, dtype=wp.int32, device=model.device)
            self.tendon_seg_active_link_r = wp.array(seg_link_r, dtype=wp.int32, device=model.device)
            self.tendon_link_seg_left = wp.array(link_seg_left, dtype=wp.int32, device=model.device)

            rest_np = model.tendon_seg_rest_length.numpy().copy()
            auto_mask = rest_np < 0.0
            rest_np[auto_mask] = 0.0
            self.tendon_seg_rest_length = wp.array(rest_np, dtype=float, device=model.device)
            self.tendon_seg_rest_length_step = wp.array(rest_np.copy(), dtype=float, device=model.device)
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

            self._init_tendon_attachment_points(model, auto_mask)

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
        wp.copy(self.tendon_link_wrap_angle_step, self.tendon_link_wrap_angle)

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
                model.tendon_closed,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_flags,
                model.tendon_link_wrap_turns,
                self.tendon_link_wrap_angle,
                model.tendon_link_radius,
                model.tendon_link_orientation,
                model.tendon_link_offset,
                model.tendon_link_axis,
                self.tendon_link_active,
            ],
            device=model.device,
        )

    def _init_tendon_attachment_points(self, model: Model, auto_mask: np.ndarray) -> None:
        """Compute initial tendon tangent attachments and rest lengths."""
        body_q = model.body_q
        if body_q is None:
            return

        tendon_start_np = model.tendon_start.numpy()
        tendon_seg_start_np = model.tendon_seg_start.numpy()
        link_body_np = model.tendon_link_body.numpy()
        link_offset_np = model.tendon_link_offset.numpy()
        seg_link_l_np = model.tendon_seg_link_l.numpy()
        seg_link_r_np = model.tendon_seg_link_r.numpy()
        body_q_np = body_q.numpy()

        att_l = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)
        att_r = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)
        att_l_local = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)
        att_r_local = np.zeros((model.tendon_segment_count, 3), dtype=np.float32)

        for seg in range(model.tendon_segment_count):
            link_l = seg_link_l_np[seg]
            link_r = seg_link_r_np[seg]
            body_l = link_body_np[link_l]
            body_r = link_body_np[link_r]
            off_l = link_offset_np[link_l]
            off_r = link_offset_np[link_r]
            att_l[seg] = _transform_point_np(body_q_np[body_l], off_l)
            att_r[seg] = _transform_point_np(body_q_np[body_r], off_r)
            att_l_local[seg] = off_l
            att_r_local[seg] = off_r

        with wp.ScopedDevice(model.device):
            self.tendon_seg_attachment_l = wp.array(att_l, dtype=wp.vec3, device=model.device)
            self.tendon_seg_attachment_r = wp.array(att_r, dtype=wp.vec3, device=model.device)
            self.tendon_seg_attachment_l_local = wp.array(att_l_local, dtype=wp.vec3, device=model.device)
            self.tendon_seg_attachment_r_local = wp.array(att_r_local, dtype=wp.vec3, device=model.device)

        wp.launch(
            kernel=update_tendon_attachments,
            dim=model.tendon_count,
            inputs=[
                body_q,
                model.tendon_start,
                model.tendon_seg_start,
                model.tendon_closed,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_flags,
                model.tendon_link_wrap_turns,
                model.tendon_link_radius,
                model.tendon_link_orientation,
                model.tendon_link_mu,
                model.tendon_link_offset,
                model.tendon_link_axis,
                self.tendon_seg_rest_length,
                self.tendon_seg_rest_length_step,
                self.tendon_seg_stretch,
                model.tendon_seg_compliance,
                model.tendon_seg_damping,
                model.tendon_seg_link_l,
                model.tendon_seg_link_r,
                self.tendon_seg_active,
                self.tendon_seg_active_link_l,
                self.tendon_seg_active_link_r,
                self.tendon_seg_active_compliance,
                self.tendon_seg_active_damping,
                self.tendon_link_active,
                self.tendon_link_active_step,
                self.tendon_link_wrap_angle,
                self.tendon_link_wrap_angle_step,
                self.tendon_seg_attachment_l,
                self.tendon_seg_attachment_r,
                self.tendon_seg_attachment_l_local,
                self.tendon_seg_attachment_r_local,
                self.tendon_seg_attachment_l_local_step,
                self.tendon_seg_attachment_r_local_step,
                self.tendon_seg_rolling_delta_l,
                self.tendon_seg_rolling_delta_r,
                self.tendon_cone_sweep_count,
                1,
                0,
                0,
                0,
                self.tendon_max_sweeps,
                self.tendon_settle_tol,
            ],
            device=model.device,
        )

        att_l_np = self.tendon_seg_attachment_l.numpy()
        att_r_np = self.tendon_seg_attachment_r.numpy()
        rest_np = self.tendon_seg_rest_length.numpy()
        model_rest_np = model.tendon_seg_rest_length.numpy()
        tendon_closed_np = model.tendon_closed.numpy()
        seg_active_np = self.tendon_seg_active.numpy()
        seg_active_link_l_np = self.tendon_seg_active_link_l.numpy()
        seg_active_link_r_np = self.tendon_seg_active_link_r.numpy()
        for t in range(model.tendon_count):
            link_start = tendon_start_np[t]
            link_end = tendon_start_np[t + 1]
            seg_start = tendon_seg_start_np[t]
            seg_end = tendon_seg_start_np[t + 1]
            for seg_idx in range(seg_start, seg_end):
                if seg_active_np[seg_idx] == 0:
                    continue

                link_l = seg_active_link_l_np[seg_idx]
                link_r = seg_active_link_r_np[seg_idx]
                source_segments = []
                source_link = link_l
                for _ in range(link_end - link_start):
                    if source_link == link_r:
                        break
                    source_segments.append(seg_start + source_link - link_start)
                    source_link += 1
                    if tendon_closed_np[t] and source_link == link_end:
                        source_link = link_start

                if np.any(auto_mask[source_segments]):
                    rest_np[seg_idx] = np.linalg.norm(att_r_np[seg_idx] - att_l_np[seg_idx])
                else:
                    rest_np[seg_idx] = np.sum(model_rest_np[source_segments])
        self.tendon_seg_rest_length = wp.array(rest_np, dtype=float, device=model.device)
        self._snapshot_tendon_step_state()

        link_type_np = model.tendon_link_type.numpy()
        link_flags_np = model.tendon_link_flags.numpy()
        link_radius_np = model.tendon_link_radius.numpy()
        link_offset_np = model.tendon_link_offset.numpy()
        link_axis_np = model.tendon_link_axis.numpy()
        link_active_np = self.tendon_link_active.numpy()
        link_wrap_angle_np = self.tendon_link_wrap_angle.numpy()
        seg_active_np = self.tendon_seg_active.numpy()

        total_cable = np.zeros(model.tendon_count, dtype=np.float32)
        for t in range(model.tendon_count):
            start = tendon_start_np[t]
            end = tendon_start_np[t + 1]
            seg_start = tendon_seg_start_np[t]
            seg_end = tendon_seg_start_np[t + 1]
            cable_len = 0.0
            for seg_idx in range(seg_start, seg_end):
                if seg_active_np[seg_idx] != 0:
                    cable_len += rest_np[seg_idx]
            for i in range(start, end):
                if link_type_np[i] == int(TendonLinkType.ROLLING):
                    if not link_active_np[i]:
                        continue
                    if (link_flags_np[i] & int(TendonLinkFlags.CONTINUOUS_WRAP)) != 0:
                        cable_len += abs(link_wrap_angle_np[i]) * link_radius_np[i]
                        continue
                    body_idx = link_body_np[i]
                    q = body_q_np[body_idx]
                    center = _transform_point_np(q, link_offset_np[i])
                    normal = _transform_vector_np(q, link_axis_np[i])
                    radius = link_radius_np[i]
                    pt_left = None
                    pt_right = None
                    for seg_idx in range(seg_start, seg_end):
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

        self.tendon_total_cable = wp.array(total_cable, dtype=float, device=model.device)
