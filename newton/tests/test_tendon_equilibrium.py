# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.example_tendon_equilibrium import Example
from newton.tests.unittest_utils import add_function_test, sanitize_identifier


def build_atwood_equal_weights(mass=2.0, pulley_mass=0.5, pulley_radius=0.15):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, -9.81))

    pulley = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.5), q=wp.quat_identity()),
        mass=0.0,
    )
    q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
    pulley_cfg = newton.ModelBuilder.ShapeConfig(density=pulley_mass / (np.pi * pulley_radius**2 * 0.08))
    builder.add_shape_cylinder(
        pulley, xform=wp.transform(q=q_cyl), radius=pulley_radius, half_height=0.04, cfg=pulley_cfg
    )

    j_pulley = builder.add_joint_revolute(
        parent=-1,
        child=pulley,
        axis=Axis.Y,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.5), q=wp.quat_identity()),
        child_xform=wp.transform(),
    )
    builder.add_articulation([j_pulley])

    weight_cfg = newton.ModelBuilder.ShapeConfig(density=mass / (8.0 * 0.08**3))

    left = builder.add_link(
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
    builder.add_articulation([j1])

    right = builder.add_link(
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
        radius=pulley_radius,
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

    return builder.finalize(), left, right, pulley


class TestTendonEquilibrium(unittest.TestCase):
    pass


def assert_supported_atwood_wrap(solver):
    """Require a two-span Atwood test to stay within valid fixed-roller geometry."""
    arrival = solver.tendon_seg_attachment_r_local.numpy()[0]
    departure = solver.tendon_seg_attachment_l_local.numpy()[1]
    angle = np.arctan2(np.cross(arrival, departure)[1], np.dot(arrival, departure))
    # Opposite radius vectors have the same half-wrap at +pi and -pi; allow
    # roundoff at that branch cut, not a genuinely unsupported turn.
    if angle < -np.pi + 1.0e-5:
        angle += 2.0 * np.pi
    if not np.isfinite(angle) or angle < -1.0e-5:
        raise AssertionError(f"Invalid Atwood wrap: {np.degrees(angle):.4f} degrees")


def add_test(cls, name, devices):
    def run(test, device):
        """Keep guided equal weights balanced without leaving the supported wrap range."""
        with wp.ScopedDevice(device):
            model, left_idx, right_idx, pulley_idx = build_atwood_equal_weights()
            dt = 1.0 / 60.0 / 16

            solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=0.8)
            s0 = model.state()
            s1 = model.state()
            control = model.control()
            contacts = model.contacts()

            bq0 = s0.body_q.numpy()
            y_left_0 = float(bq0[left_idx][2])
            y_right_0 = float(bq0[right_idx][2])

            s0.clear_forces()
            model.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0

            att_r = solver.tendon_seg_attachment_r.numpy()
            att_l = solver.tendon_seg_attachment_l.numpy()
            pulley_z = float(s0.body_q.numpy()[pulley_idx][2])
            test.assertGreater(
                att_r[0][2],
                pulley_z,
                f"Cable should wrap over pulley: left tangent z={att_r[0][2]:.3f} <= center z={pulley_z:.3f}",
            )
            test.assertGreater(
                att_l[1][2],
                pulley_z,
                f"Cable should wrap over pulley: right tangent z={att_l[1][2]:.3f} <= center z={pulley_z:.3f}",
            )

            num_frames = 120
            for _ in range(num_frames):
                for _ in range(16):
                    s0.clear_forces()
                    model.collide(s0, contacts)
                    solver.step(s0, s1, control, contacts, dt)
                    s0, s1 = s1, s0
                assert_supported_atwood_wrap(solver)

            bq = s0.body_q.numpy()
            y_left = float(bq[left_idx][2])
            y_right = float(bq[right_idx][2])

            drift_left = abs(y_left - y_left_0)
            drift_right = abs(y_right - y_right_0)
            drift_diff = abs((y_left - y_left_0) - (y_right - y_right_0))

            test.assertLess(drift_left, 0.05, f"Left weight drifted {drift_left:.4f} m")
            test.assertLess(drift_right, 0.05, f"Right weight drifted {drift_right:.4f} m")
            test.assertLess(drift_diff, 0.02, f"Asymmetric drift: {drift_diff:.4f} m")
            test.assertTrue(np.isfinite(bq).all(), "Non-finite body positions")

    for device in devices:
        test_name = f"test_{sanitize_identifier(name)}_{sanitize_identifier(device)}"
        setattr(cls, test_name, lambda self, d=device: run(self, d))


devices = ["cpu"]
if wp.is_cuda_available():
    devices.append("cuda:0")

add_test(TestTendonEquilibrium, "equal_weight_atwood", devices)


def test_example_force_balance(test, device):
    """Require both reported tension and accepted cable extension to support the weights."""
    with wp.ScopedDevice(device):
        example = Example(None, None)
        graph = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture() as capture:
                example.simulate()
            graph = capture.graph
        for _ in range(120):
            if graph is None:
                example.simulate()
            else:
                wp.capture_launch(graph)
            example.sim_time += example.frame_dt
            example.test_post_step()
        example.test_final()


add_function_test(TestTendonEquilibrium, "test_example_force_balance", test_example_force_balance, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
