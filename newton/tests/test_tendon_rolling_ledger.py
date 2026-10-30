# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check conservative rolling transport independently of material friction."""

import sys
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.tendon_kernels import update_tendon_attachments
from newton.tests.unittest_utils import (
    StdOutCapture,
    add_function_test,
    capture_native_stdout_in_subprocess,
    get_test_devices,
)


def _fixture(offset=(0.0, 0.0, 0.0), *, dynamic=False, half_turn=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_tendon()
    ends = ((0.1, -0.4, 0.0), (-0.1, -0.4, 0.0)) if half_turn else ((-0.4, 0.0, 0.0), (0.4, 0.0, 0.0))
    for i, point in enumerate((ends[0], -np.asarray(offset), ends[1])):
        body = builder.add_body(xform=wp.transform(p=wp.vec3(point)), mass=0.0, is_kinematic=True)
        builder.add_tendon_link(
            body=body,
            link_type=newton.TendonLinkType.ROLLING if i == 1 else newton.TendonLinkType.ATTACHMENT,
            offset=offset if i == 1 else (0.0, 0.0, 0.0),
            axis=(0.0, 0.0, 1.0),
            radius=0.1 if i == 1 else 0.0,
            mu=0.1,
            compliance=1.0e-5,
            dynamic=dynamic and i == 1,
        )
    model = builder.finalize()
    return model, newton.solvers.SolverXPBD(model, iterations=1)


def _geometry(model, solver):
    wp.launch(
        update_tendon_attachments,
        dim=model.tendon_segment_count,
        inputs=[
            model.body_q,
            model.tendon_link_body,
            model.tendon_link_type,
            model.tendon_link_flags,
            model.tendon_link_radius,
            model.tendon_link_orientation,
            model.tendon_link_offset,
            model.tendon_link_axis,
            solver.tendon_seg_active,
            solver.tendon_seg_active_link_l,
            solver.tendon_seg_active_link_r,
            solver.tendon_link_active,
            solver.tendon_link_active_step,
            solver.tendon_seg_attachment_l_local_step,
            solver.tendon_seg_attachment_r_local_step,
            1,
        ],
        outputs=[
            solver.tendon_seg_attachment_l,
            solver.tendon_seg_attachment_r,
            solver.tendon_seg_attachment_l_local,
            solver.tendon_seg_attachment_r_local,
            solver.tendon_seg_rolling_delta_l,
            solver.tendon_seg_rolling_delta_r,
            solver.tendon_seg_length,
        ],
    )
    solver._update_tendon_cone_rows(model, model.body_q, False)


def _transport(solver):
    return np.array(
        [solver.tendon_seg_rolling_delta_r.numpy()[0], solver.tendon_seg_rolling_delta_l.numpy()[1]],
        dtype=np.float64,
    )


def _arc(model, solver):
    values = model.body_q.numpy()[1]
    pose = wp.transform(p=wp.vec3(values[:3]), q=wp.quat(values[3:]))
    center = np.array(wp.transform_point(pose, wp.vec3(model.tendon_link_offset.numpy()[1])), dtype=np.float64)
    normal = np.array(wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0)), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    a = solver.tendon_seg_attachment_r.numpy()[0].astype(np.float64) - center
    b = solver.tendon_seg_attachment_l.numpy()[1].astype(np.float64) - center
    a -= normal * np.dot(a, normal)
    b -= normal * np.dot(b, normal)
    return float(np.arctan2(np.dot(np.cross(a, b), normal), np.dot(a, b))) * 0.1


def test_stationary_circle_spin_preserves_arc(test, device):
    """Keep paired transport exactly conservative when a circle only spins."""
    with wp.ScopedDevice(device):
        model, solver = _fixture()
        initial_length = solver.tendon_seg_length.numpy().copy()
        for angle in np.linspace(0.0, 5.0, 151):
            solver._snapshot_tendon_step_state()
            poses = model.body_q.numpy()
            poses[1, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(angle)))
            model.body_q.assign(poses)
            _geometry(model, solver)
            delta = _transport(solver)
            test.assertLessEqual(abs(float(np.sum(delta))), 1.0e-12)
            np.testing.assert_array_equal(solver.tendon_seg_length.numpy(), initial_length)


def test_moving_circle_books_world_arc(test, device):
    """Book changing wraps in arbitrary planes and replay the ledger in a CUDA graph."""
    with wp.ScopedDevice(device):
        offset = (0.03, -0.02, 0.01)
        model, solver = _fixture(offset)
        for angle in np.linspace(0.0, 1.7, 45):
            before = _arc(model, solver)
            solver._snapshot_tendon_step_state()
            frame = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), float(angle))
            spin = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(2.0 * angle))
            rotation = frame * spin
            origin = wp.vec3(float(0.1 * angle), 0.2, -0.3)
            poses = model.body_q.numpy()
            poses[0, :3] = np.asarray(origin + wp.quat_rotate(frame, wp.vec3(-0.4, 0.0, 0.0)))
            poses[2, :3] = np.asarray(origin + wp.quat_rotate(frame, wp.vec3(0.4, float(0.05 * np.sin(angle)), 0.0)))
            poses[1, :3] = np.asarray(origin - wp.quat_rotate(rotation, wp.vec3(offset)))
            poses[1, 3:] = np.asarray(rotation)
            model.body_q.assign(poses)
            _geometry(model, solver)
            delta = _transport(solver)
            after = _arc(model, solver)
            test.assertAlmostEqual(float(np.sum(delta)), before - after, delta=4.0e-8)
            _geometry(model, solver)
            np.testing.assert_array_equal(_transport(solver), delta)
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                solver._snapshot_tendon_step_state()
                _geometry(model, solver)
            for _ in range(3):
                wp.capture_launch(capture.graph)
                test.assertLessEqual(abs(float(np.sum(_transport(solver)))), 1.0e-12)


def _emit_half_turn_report(device, x):
    """Report the same half-turn geometry in an isolated native stdout process."""
    with wp.ScopedDevice(device):
        model, solver = _fixture(half_turn=True)
        poses = model.body_q.numpy()
        poses[2, 0] = x
        model.body_q.assign(poses)
        _geometry(model, solver)
        solver._update_tendon_cone_rows(model, model.body_q, True)
        wp.synchronize()


def test_half_turn_rolling_has_no_circumference_jump(test, device):
    """Keep rolling continuous across atan2's cut without declaring the route supported."""
    with wp.ScopedDevice(device):
        model, solver = _fixture(half_turn=True)
        previous = float(solver._tendon_link_arc.numpy()[1])
        test.assertAlmostEqual(previous, np.pi * 0.1, delta=4.0e-8)
        for x in (-0.1001, -0.0999, -0.1, -0.1001):
            solver._snapshot_tendon_step_state()
            poses = model.body_q.numpy()
            poses[2, 0] = x
            model.body_q.assign(poses)
            _geometry(model, solver)
            principal = _arc(model, solver)
            # Independent nearest-angle continuation of the measured world arc.
            continued = principal + 2.0 * np.pi * 0.1 * round((previous - principal) / (2.0 * np.pi * 0.1))
            transport = float(np.sum(_transport(solver)))
            test.assertLess(abs(transport), 1.0e-4)
            test.assertAlmostEqual(transport, previous - continued, delta=4.0e-8)
            previous = continued
            if principal < -3.0 * 0.1:
                if sys.platform == "win32" and device.is_cpu:
                    output = capture_native_stdout_in_subprocess(
                        "newton.tests.test_tendon_rolling_ledger", "_emit_half_turn_report", device=device.alias, x=x
                    )
                else:
                    capture = StdOutCapture()
                    capture.begin()
                    try:
                        solver._update_tendon_cone_rows(model, model.body_q, True)
                        wp.synchronize()
                    finally:
                        output = capture.end()
                test.assertIn("crossed the supported wrap range", output)


def test_dynamic_merge_consumes_snapshot_arc(test, device):
    """Merge precisely the arc booked by the accepted geometry, without reconstructing it."""
    with wp.ScopedDevice(device):
        for angle in np.linspace(0.0, 2.0, 21):
            model, solver = _fixture((0.03, -0.02, 0.01), dynamic=True)
            test.assertTrue(solver.tendon_link_active.numpy()[1])
            solver.tendon_seg_rest_length.fill_(0.05)
            solver._snapshot_tendon_step_state()
            rest = solver.tendon_seg_rest_length_step.numpy()
            expected = np.float32(rest[0] + rest[1] + solver._tendon_link_arc_step.numpy()[1])

            # Deactivation occurs at a new pose, but it consumes the previous
            # material state. Its arc must not change through a pose round trip.
            poses = model.body_q.numpy()
            poses[1, :3] = (0.2, -0.3, 0.1)
            poses[1, 3:] = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), float(angle)))
            model.body_q.assign(poses)
            active = solver.tendon_link_active.numpy()
            active[1] = False
            solver.tendon_link_active.assign(active)
            solver._prepare_tendon_route(model)
            test.assertEqual(solver.tendon_seg_route_rest_length.numpy()[0], expected)


class TestTendonRollingLedger(unittest.TestCase):
    """Verify the circular transport ledger on CPU and CUDA."""


for _test in (
    test_stationary_circle_spin_preserves_arc,
    test_moving_circle_books_world_arc,
    test_half_turn_rolling_has_no_circumference_jump,
    test_dynamic_merge_consumes_snapshot_arc,
):
    add_function_test(
        TestTendonRollingLedger,
        _test.__name__,
        _test,
        devices=get_test_devices(),
        # This case captures and checks the deliberately emitted diagnostic.
        check_output=_test is not test_half_turn_rolling_has_no_circumference_jump,
    )


if __name__ == "__main__":
    unittest.main()
