# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drive circular, elliptical, and pizza-slice rollers against a loaded slider.

Left cable ends move vertically. Right ends are dynamic sliders pulled downward
by a 3 N load. Rollers rotate freely, with explicit viscous bearing drag and
finite cable friction. No roller angle or slider position is prescribed.

Command: python -m newton.examples roller_profiles
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import RollerProfileCircle, RollerProfileEllipse, RollerProfileSector
from newton.solvers import SolverVBD, SolverXPBD


@wp.kernel
def drive_and_load(
    initial: wp.array[wp.transform],
    anchors: wp.array[int],
    sliders: wp.array[int],
    rollers: wp.array[int],
    frame: wp.array[int],
    frame_dt: float,
    offset: float,
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    forces: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    time = float(frame[0]) * frame_dt + offset
    phase = 2.0 * wp.pi * time / 4.0
    anchor = anchors[i]
    position = wp.transform_get_translation(initial[anchor])
    position[2] -= 0.04 * (1.0 - wp.cos(phase))
    q[anchor] = wp.transform(position, wp.quat_identity())
    qd[anchor] = wp.spatial_vector(0.0, 0.0, -0.02 * wp.pi * wp.sin(phase), 0.0, 0.0, 0.0)
    slider = sliders[i]
    forces[slider] = wp.spatial_vector(
        0.0, 0.0, -3.0 * (1.0 - wp.exp(-8.0 * time)) - 0.5 * qd[slider][2], 0.0, 0.0, 0.0
    )
    roller = rollers[i]
    forces[roller] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, -0.005 * qd[roller][4], 0.0)


@wp.kernel
def advance_frame(frame: wp.array[int]):
    frame[0] += 1


def _boundary(profile, coordinate):
    coordinate = np.asarray(coordinate)
    if isinstance(profile, RollerProfileCircle):
        return profile.radius * np.stack((np.cos(coordinate), np.sin(coordinate)), axis=-1)
    if isinstance(profile, RollerProfileEllipse):
        return np.stack((profile.a * np.cos(coordinate), profile.b * np.sin(coordinate)), axis=-1)
    radius, angle = profile.radius, profile.angle
    coordinate = coordinate % (radius * (2 + angle))
    theta = (coordinate - radius) / radius - angle / 2
    points = radius * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    lower = coordinate[..., None] * np.array((math.cos(angle / 2), -math.sin(angle / 2)))
    upper = (radius * (2 + angle) - coordinate)[..., None] * np.array((math.cos(angle / 2), math.sin(angle / 2)))
    return np.where(
        (coordinate < radius)[..., None], lower, np.where((coordinate > radius * (1 + angle))[..., None], upper, points)
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60
        self.sim_substeps = getattr(args, "substeps", 10)
        solver_name = getattr(args, "solver", "vbd")
        if solver_name not in ("vbd", "xpbd"):
            raise ValueError("solver must be 'vbd' or 'xpbd'")
        iterations = getattr(args, "iterations", None)
        if iterations is None:
            iterations = 32
        friction = getattr(args, "friction", 0.3)
        if self.sim_substeps < 1 or iterations < 1:
            raise ValueError("substeps and iterations must be positive")
        if not math.isfinite(friction) or friction < 0:
            raise ValueError("friction must be finite and nonnegative")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.profiles = [RollerProfileCircle(0.06), RollerProfileEllipse(0.08, 0.045), RollerProfileSector(0.08, 2.0)]
        builder = newton.ModelBuilder(gravity=0.0)
        marker = builder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        anchors, sliders, rollers = [], [], []
        for i, profile in enumerate(self.profiles):
            offset = 0.42 * (i - 1)
            anchor = builder.add_body(xform=wp.transform(p=(offset - 0.12, 0, 0.70)), is_kinematic=True)
            slider_pose = wp.transform(p=(offset + 0.12, 0, 0.70))
            slider = builder.add_link(xform=slider_pose, mass=0.1, inertia=wp.mat33(np.eye(3) * 1e-5))
            slider_joint = builder.add_joint_prismatic(-1, slider, parent_xform=slider_pose, axis=newton.Axis.Z)
            builder.add_articulation([slider_joint])
            angle = 0.0 if i == 0 else -0.4 if i == 1 else -0.8
            pose = wp.transform(p=(offset, 0, 1.0), q=wp.quat_from_axis_angle(wp.vec3(0, 1, 0), angle))
            roller = builder.add_link(xform=pose, mass=0.1, inertia=wp.mat33(np.eye(3) * 2e-4))
            hinge = builder.add_joint_revolute(
                -1, roller, parent_xform=pose, axis=newton.Axis.Y, target_ke=0.0, target_kd=0.0
            )
            builder.add_articulation([hinge])
            builder.add_shape_sphere(anchor, radius=0.009, cfg=marker, color=(0.3, 0.8, 0.3))
            builder.add_shape_box(slider, hx=0.012, hy=0.008, hz=0.016, cfg=marker, color=(0.9, 0.3, 0.2))
            builder.add_shape_sphere(roller, radius=0.004, cfg=marker, color=(1.0, 1.0, 1.0))
            builder.add_tendon()
            builder.add_tendon_link(anchor, newton.TendonLinkType.ATTACHMENT)
            builder.add_tendon_link(
                roller,
                newton.TendonLinkType.ROLLING,
                profile=profile,
                axis=(0, 1, 0),
                mu=friction,
                compliance=1e-5,
                damping=1.0,
            )
            builder.add_tendon_link(slider, newton.TendonLinkType.ATTACHMENT, compliance=1e-5, damping=1.0)
            anchors.append(anchor)
            sliders.append(slider)
            rollers.append(roller)
        builder.color()
        self.model = builder.finalize()
        solver_type = SolverVBD if solver_name == "vbd" else SolverXPBD
        self.solver = solver_type(self.model, iterations=iterations, tendon_settle_tol=1e-5)
        if solver_name == "xpbd":
            # Coupled cable impulses interact with the separately solved hinges.
            # The default 0.7 chatters at the sector's stick/slip transition here.
            self.solver.joint_linear_relaxation = 0.5
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.anchors = wp.array(anchors, dtype=int, device=self.model.device)
        self.sliders = wp.array(sliders, dtype=int, device=self.model.device)
        self.rollers = wp.array(rollers, dtype=int, device=self.model.device)
        self.roller_indices = rollers
        self.material_total = np.array(self.solver.tendon_total_cable.numpy(), copy=True)
        self.frame = wp.zeros(1, dtype=int, device=self.model.device)
        self.frame_count = 0
        self.graphs = []
        if self.model.device.is_cuda and not getattr(args, "disable_cuda_graph", False):
            # Odd step counts alternate the input/output buffers. Capture both
            # directions so each replay starts from the preceding output.
            for _ in range(2 if self.sim_substeps % 2 else 1):
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self.simulate()
                self.graphs.append(capture.graph)
        if viewer is not None:
            viewer.set_model(self.model)
            viewer.set_camera(pos=wp.vec3(0.0, -1.55, 0.88), pitch=0.0, yaw=90.0)

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                drive_and_load,
                dim=3,
                inputs=[
                    self.model.body_q,
                    self.anchors,
                    self.sliders,
                    self.rollers,
                    self.frame,
                    self.frame_dt,
                    substep * self.sim_dt,
                ],
                outputs=[self.state_0.body_q, self.state_0.body_qd, self.state_0.body_f],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(advance_frame, dim=1, inputs=[self.frame], device=self.model.device)

    def step(self):
        if self.graphs:
            wp.capture_launch(self.graphs[self.frame_count % len(self.graphs)])
            if self.sim_substeps % 2:
                self.state_0, self.state_1 = self.state_1, self.state_0
        else:
            self.simulate()
        self.frame_count += 1
        self.sim_time = self.frame_count * self.frame_dt

    def test_post_step(self):
        """Check each route's validity and conserved free-plus-wrapped material."""
        assert np.isfinite(self.state_0.body_q.numpy()).all()
        assert np.isfinite(self.state_0.body_qd.numpy()).all()
        # Material conservation alone does not detect an under-converged body
        # solve: also check the fixed hinges and frictionless circular symmetry.
        np.testing.assert_allclose(
            self.state_0.body_q.numpy()[self.roller_indices, :3],
            self.model.body_q.numpy()[self.roller_indices, :3],
            atol=1e-5,
            rtol=0,
        )
        if self.model.tendon_link_mu.numpy()[1] == 0.0:
            # The ellipse and sector may rotate rapidly under normal forces;
            # a centered frictionless circle cannot receive that spin torque.
            circle = self.roller_indices[0]
            np.testing.assert_allclose(self.state_0.body_q.numpy()[circle, 3:], [0, 0, 0, 1], atol=0.0025, rtol=0)
            assert abs(float(self.state_0.body_qd.numpy()[circle, 4])) < 0.03
        assert not np.any(self.solver.tendon_profile_tangent_status.numpy())
        assert not np.any(self.solver.tendon_profile_wrap_status.numpy())
        total = self.solver.tendon_seg_rest_length.numpy().reshape(3, 2).sum(axis=1)
        total += self.solver.tendon_profile_wrap_length.numpy().reshape(3, 3).sum(axis=1)
        np.testing.assert_allclose(total, self.material_total, atol=2e-5, rtol=0)

    def test_final(self):
        """Retain supported, finite, material-conserving routes."""
        self.test_post_step()

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        poses = self.state_0.body_q.numpy()
        parameter_l = self.solver.tendon_profile_parameter_l.numpy()
        parameter_r = self.solver.tendon_profile_parameter_r.numpy()
        wrap_length = self.solver.tendon_profile_wrap_length.numpy()
        attachment_l = self.solver.tendon_seg_attachment_l.numpy()
        attachment_r = self.solver.tendon_seg_attachment_r.numpy()
        starts, ends, colors = [], [], []
        tension = np.maximum(
            self.solver.tendon_seg_material_tension.numpy() + self.solver.tendon_seg_damping_tension.numpy(), 0
        )
        for i, profile in enumerate(self.profiles):
            body_pose = poses[self.roller_indices[i]]
            rotation = np.array(wp.quat_to_matrix(wp.quat(*body_pose[3:]))).reshape(3, 3)
            basis = np.stack((rotation[:, 0], -rotation[:, 2]))

            def world(p, basis=basis, origin=body_pose[:3]):
                return np.asarray(p) @ basis + origin

            period = (
                2 * math.pi if not isinstance(profile, RollerProfileSector) else profile.radius * (2 + profile.angle)
            )
            corners = (
                []
                if not isinstance(profile, RollerProfileSector)
                else [profile.radius, profile.radius * (1 + profile.angle)]
            )

            def curve(a, b, color, profile=profile, corners=corners, period=period, world=world):
                samples = sorted(
                    [
                        *np.linspace(a, b, 81),
                        *[s + k * period for s in corners for k in (0, 1) if a < s + k * period < b],
                    ]
                )
                points = world(_boundary(profile, samples))
                starts.extend(points[:-1])
                ends.extend(points[1:])
                colors.extend([color] * (len(points) - 1))

            curve(0, period, (0.25, 0.55, 1.0))
            entry = float(parameter_r[2 * i])
            exit = float(parameter_l[2 * i + 1])
            # Use the solver's zero-wrap classification: modulo of independently
            # rounded coincident parameters can otherwise draw a full circuit.
            if wrap_length[3 * i + 1] > 0.0:
                curve(entry, entry + (exit - entry) % period, (1.0, 0.75, 0.2))
            starts.append(world((0, 0)))
            ends.append(world(_boundary(profile, 0)))
            colors.append((0.5, 0.8, 1.0))
            for seg in (2 * i, 2 * i + 1):
                value = min(float(tension[seg]) / 6.0, 1.0)
                starts.append(attachment_l[seg])
                ends.append(attachment_r[seg])
                colors.append((value, 0.3, 1.0 - value))
        self.viewer.log_lines(
            "profiles_and_cables",
            wp.array(starts, dtype=wp.vec3, device=self.model.device),
            wp.array(ends, dtype=wp.vec3, device=self.model.device),
            colors=wp.array(colors, dtype=wp.vec3, device=self.model.device),
            width=0.0015,
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=None, help="Default: 32 for either solver")
    parser.add_argument("--friction", type=float, default=0.3)
    parser.add_argument("--solver", choices=("vbd", "xpbd"), default="vbd")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
