# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Equilibrium
#
# Two equal-mass, vertically guided weights connected over a dynamic pulley.
# After the small elastic settling, neither side should move — the system stays in
# static equilibrium, verifying that the XPBD tendon constraint
# correctly balances symmetric loads.
#
# Command: python -m newton.examples tendon_equilibrium
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Axis, TendonLinkType
from newton.examples.cable.cable import assert_tendon_total_length, get_tendon_cable_lines


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, -9.81))

        self.pulley_radius = 0.15
        pulley_mass = 0.5
        weight_mass = 2.0

        pulley = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.5), q=wp.quat_identity()),
            mass=0.0,
        )
        q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
        pulley_cfg = newton.ModelBuilder.ShapeConfig(density=pulley_mass / (np.pi * self.pulley_radius**2 * 0.08))
        builder.add_shape_cylinder(
            pulley, xform=wp.transform(q=q_cyl), radius=self.pulley_radius, half_height=0.04, cfg=pulley_cfg
        )

        j_pulley = builder.add_joint_revolute(
            parent=-1,
            child=pulley,
            axis=Axis.Y,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.5), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )

        weight_cfg = newton.ModelBuilder.ShapeConfig(density=weight_mass / (8.0 * 0.08**3))

        self.left_idx = left = builder.add_link(
            xform=wp.transform(p=wp.vec3(-0.5, 0.0, 2.0), q=wp.quat_identity()),
            mass=0.0,
        )
        builder.add_shape_box(left, hx=0.08, hy=0.08, hz=0.08, cfg=weight_cfg)
        j1 = builder.add_joint_prismatic(
            parent=-1,
            child=left,
            axis=Axis.Z,
            parent_xform=wp.transform(p=wp.vec3(-0.5, 0.0, 2.0), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )

        self.right_idx = right = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.5, 0.0, 2.0), q=wp.quat_identity()),
            mass=0.0,
        )
        builder.add_shape_box(right, hx=0.08, hy=0.08, hz=0.08, cfg=weight_cfg)
        j2 = builder.add_joint_prismatic(
            parent=-1,
            child=right,
            axis=Axis.Z,
            parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, 2.0), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )

        builder.add_articulation([j_pulley])
        builder.add_articulation([j1])
        builder.add_articulation([j2])

        axis = (0.0, 1.0, 0.0)
        builder.add_tendon()
        builder.add_tendon_link(
            body=left,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.08),
            axis=axis,
        )
        builder.add_tendon_link(
            body=pulley,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.pulley_radius,
            orientation=1,
            mu=10.0,
            offset=(0.0, 0.0, 0.0),
            axis=axis,
            compliance=1.0e-6,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=right,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.08),
            axis=axis,
            compliance=1.0e-6,
            damping=0.1,
            rest_length=-1.0,
        )

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            # Resolve coupled sticking rows so material tension balances gravity.
            iterations=32,
            joint_linear_relaxation=0.8,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        bq = self.state_0.body_q.numpy()
        self.y_left_0 = float(bq[self.left_idx][2])
        self.y_right_0 = float(bq[self.right_idx][2])

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -4.0, 2.0), pitch=5.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        assert_tendon_total_length(self)
        arrival = self.solver.tendon_seg_attachment_r_local.numpy()[0]
        departure = self.solver.tendon_seg_attachment_l_local.numpy()[1]
        angle = np.arctan2(np.cross(arrival, departure)[1], np.dot(arrival, departure))
        assert np.isfinite(angle) and angle >= -1.0e-5, (
            f"Equilibrium pulley left the supported half-wrap range: {np.degrees(angle):.4f} degrees"
        )
        if self.sim_time < self.frame_dt * 1.5:
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            body_q = self.state_0.body_q.numpy()
            pulley_z = body_q[0][2]
            assert att_r[0][2] > pulley_z, (
                f"Cable should wrap over pulley: left tangent z={att_r[0][2]:.3f} <= center z={pulley_z:.3f}"
            )
            assert att_l[1][2] > pulley_z, (
                f"Cable should wrap over pulley: right tangent z={att_l[1][2]:.3f} <= center z={pulley_z:.3f}"
            )

    def test_final(self):
        assert_tendon_total_length(self)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"

        y_left = float(body_q[self.left_idx][2])
        y_right = float(body_q[self.right_idx][2])
        drift_left = abs(y_left - self.y_left_0)
        drift_right = abs(y_right - self.y_right_0)
        drift_diff = abs((y_left - self.y_left_0) - (y_right - self.y_right_0))

        assert drift_left < 0.05, f"Left weight drifted {drift_left:.4f} m"
        assert drift_right < 0.05, f"Right weight drifted {drift_right:.4f} m"
        assert drift_diff < 0.02, f"Asymmetric drift: {drift_diff:.4f} m"

        if self.sim_time >= 1.0:
            # A stationary pose alone can hide unconverged material tension.
            spans = self.solver.tendon_seg_attachment_r.numpy() - self.solver.tendon_seg_attachment_l.numpy()
            vertical_fraction = np.abs(spans[:, 2]) / np.linalg.norm(spans, axis=1)
            masses = self.model.body_mass.numpy()[[self.left_idx, self.right_idx]]
            expected = masses * 9.81 / vertical_fraction
            reported = np.maximum(
                self.solver.tendon_seg_material_tension.numpy() + self.solver.tendon_seg_damping_tension.numpy(), 0.0
            )
            np.testing.assert_allclose(
                reported, expected, rtol=0.05, atol=0.0, err_msg="Equilibrium tendon tension should balance gravity"
            )

            # Independently recover straight tangent lengths from the accepted
            # body pose, not the solver's pre-correction attachment cache.
            offsets = self.model.tendon_link_offset.numpy()[[0, 2]]
            anchors = np.array(
                [
                    wp.transform_point(wp.transform(*body_q[body]), wp.vec3(offset))
                    for body, offset in zip((self.left_idx, self.right_idx), offsets, strict=True)
                ],
                dtype=np.float64,
            )
            distance = np.linalg.norm(anchors - body_q[0, :3], axis=1)
            lengths = np.sqrt(distance**2 - self.pulley_radius**2)
            extension = lengths - self.solver.tendon_seg_rest_length.numpy().astype(np.float64)
            material = np.maximum(extension, 0.0) / self.solver.tendon_seg_active_compliance.numpy()
            np.testing.assert_allclose(
                material,
                expected,
                rtol=0.05,
                atol=0.0,
                err_msg="Accepted cable extension should agree with the equilibrium force",
            )

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(0.9, 0.2, 0.2), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
