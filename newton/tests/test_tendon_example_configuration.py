# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check authored masses, joints, and route geometry in tendon examples."""

import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.cable.example_tendon_cable_machine import Example as CableMachineExample
from newton.examples.cable.example_tendon_capstan_friction import Example as DynamicCapstanExample
from newton.examples.cable.example_tendon_capstan_kinematic import Example
from newton.examples.cable.example_tendon_compound_pulley import Example as CompoundPulleyExample
from newton.examples.cable.example_tendon_equilibrium import Example as EquilibriumExample
from newton.examples.cable.example_tendon_gear_pulley import Example as GearPulleyExample
from newton.examples.cable.example_tendon_pulley import Example as PulleyExample
from newton.examples.cable.example_tendon_rolling_pulley import Example as RollingPulleyExample
from newton.tests.test_tendon_capstan import build_dynamic_pulley_atwood, build_kinematic_pulley_atwood
from newton.tests.test_tendon_equilibrium import assert_supported_atwood_wrap, build_atwood_equal_weights


class TestTendonExampleConfiguration(unittest.TestCase):
    def setUp(self):
        """Construct the example without advancing or rendering it."""
        with wp.ScopedDevice("cpu"):
            self.example = Example(None, SimpleNamespace())

    def test_kinematic_capstan_weight_masses(self):
        """Do not add visual-shape mass on top of the specified weight mass."""
        masses = self.example.model.body_mass.numpy()
        np.testing.assert_allclose(masses[self.example.left_indices], 1.0, atol=1e-6)
        np.testing.assert_allclose(masses[self.example.right_indices], 1.2, atol=1e-6)

    def test_kinematic_capstan_vertical_guides(self):
        """Prevent pendulum motion from driving fixed rollers past a half wrap."""
        model = self.example.model
        weight_joints = np.isin(model.joint_child.numpy(), self.example.left_indices + self.example.right_indices)
        np.testing.assert_array_equal(model.joint_type.numpy()[weight_joints], int(newton.JointType.PRISMATIC))
        dofs = model.joint_qd_start.numpy()[:-1][weight_joints]
        np.testing.assert_allclose(model.joint_axis.numpy()[dofs], np.tile([0.0, 0.0, 1.0], (6, 1)), atol=1e-7)

    def test_kinematic_capstan_travel_stays_below_pulley(self):
        """Bound long-running demonstrations before a weight reaches the pulley."""
        model = self.example.model
        weight_joints = np.isin(model.joint_child.numpy(), self.example.left_indices + self.example.right_indices)
        dofs = model.joint_qd_start.numpy()[:-1][weight_joints]
        np.testing.assert_allclose(model.joint_limit_lower.numpy()[dofs], -1.2, atol=1e-7)
        np.testing.assert_allclose(model.joint_limit_upper.numpy()[dofs], 1.2, atol=1e-7)


class TestCompoundPulleyConfiguration(unittest.TestCase):
    def test_unique_pulley_hinges(self):
        """Give each pulley one revolute joint, without a parallel free joint."""
        with wp.ScopedDevice("cpu"):
            example = CompoundPulleyExample(None, None)
        children = example.model.joint_child.numpy()
        joint_types = example.model.joint_type.numpy()
        for body in (example.p1_idx, example.p2_idx):
            with self.subTest(pulley=body):
                np.testing.assert_array_equal(joint_types[children == body], [int(newton.JointType.REVOLUTE)])


class TestRollingPulleyConfiguration(unittest.TestCase):
    def setUp(self):
        """Construct the rolling pulley without advancing it."""
        with wp.ScopedDevice("cpu"):
            self.example = RollingPulleyExample(None, SimpleNamespace())

    def test_weight_masses(self):
        """Give the weights their specified 1 kg and 2 kg masses."""
        example = self.example
        np.testing.assert_allclose(example.model.body_mass.numpy()[[example.left_idx, example.right_idx]], [1, 2])

    def test_vertical_guides_and_travel(self):
        """Prevent weight rotation and keep both attachments below the pulley."""
        example = self.example
        model = example.model
        selected = np.isin(model.joint_child.numpy(), [example.left_idx, example.right_idx])
        np.testing.assert_array_equal(model.joint_type.numpy()[selected], int(newton.JointType.PRISMATIC))
        dofs = model.joint_qd_start.numpy()[:-1][selected]
        np.testing.assert_allclose(model.joint_axis.numpy()[dofs], [[0, 0, 1], [0, 0, 1]], atol=1e-7)
        initial = model.body_q.numpy()[[example.left_idx, example.right_idx], 2]
        upper = model.joint_limit_upper.numpy()[dofs]
        self.assertTrue(np.all(initial + upper + [0.08, 0.10] < example.pulley_height - example.pulley_radius))
        np.testing.assert_allclose(
            model.body_q.numpy()[[example.left_idx, example.right_idx], 0],
            [-example.pulley_radius, example.pulley_radius],
            atol=1e-7,
        )


class TestEquilibriumConfiguration(unittest.TestCase):
    def test_masses_and_guides(self):
        """Give the example and unit fixture their stated masses and vertical guides."""
        with wp.ScopedDevice("cpu"):
            example_model = EquilibriumExample(None, None).model
            fixture_model, left, right, pulley = build_atwood_equal_weights()
        for model in (example_model, fixture_model):
            with self.subTest(model=model):
                np.testing.assert_allclose(model.body_mass.numpy()[[pulley, left, right]], [0.5, 2.0, 2.0], atol=1e-6)
                self.assertEqual(model.joint_count, 3)
                selected = np.isin(model.joint_child.numpy(), [left, right])
                np.testing.assert_array_equal(model.joint_type.numpy()[selected], int(newton.JointType.PRISMATIC))
                dofs = model.joint_qd_start.numpy()[:-1][selected]
                np.testing.assert_allclose(model.joint_axis.numpy()[dofs], [[0, 0, 1], [0, 0, 1]], atol=1e-7)


class TestPulleyConfiguration(unittest.TestCase):
    def test_weight_mass(self):
        """Give the freely swinging weight its intended 2 kg mass."""
        with wp.ScopedDevice("cpu"):
            model = PulleyExample(None, None).model
        self.assertAlmostEqual(float(model.body_mass.numpy()[1]), 2.0, places=6)

    def test_weight_has_one_unrestricted_joint(self):
        """Retain all six free DOFs without defining the body twice in the joint graph."""
        with wp.ScopedDevice("cpu"):
            model = PulleyExample(None, None).model
        selected = model.joint_child.numpy() == 1
        self.assertEqual(int(np.count_nonzero(selected)), 1)
        np.testing.assert_array_equal(model.joint_dof_dim.numpy()[selected], [[3, 3]])


class TestAtwoodFixtureConfiguration(unittest.TestCase):
    def test_flywheel_inertia_is_preserved(self):
        """Author a valid flywheel tensor without changing the pulley's gravitational mass."""
        with wp.ScopedDevice("cpu"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, _, _, pulley = build_dynamic_pulley_atwood(pulley_mass=5.0, pulley_inertia_y=225.0)
        self.assertFalse(any("Inertia validation corrected" in str(w.message) for w in caught))
        self.assertAlmostEqual(float(model.body_mass.numpy()[pulley]), 5.0)
        self.assertAlmostEqual(float(model.body_inertia.numpy()[pulley, 1, 1]), 225.0)

    def test_wrap_guard_handles_half_turn_branch_cut(self):
        """Accept a rounded half wrap, but reject genuinely negative turns and nonfinite contacts."""
        with wp.ScopedDevice("cpu"):
            model, _, _, _ = build_dynamic_pulley_atwood()
            solver = newton.solvers.SolverXPBD(model)
        arrival = solver.tendon_seg_attachment_r_local.numpy()
        departure = solver.tendon_seg_attachment_l_local.numpy()
        radius = float(model.tendon_link_radius.numpy()[1])
        arrival[0] = [radius, 0.0, 0.0]
        solver.tendon_seg_attachment_r_local.assign(arrival)
        for angle in (0.0, np.pi - 1e-7, -np.pi + 1e-7, -0.1, -np.pi + 1e-3, np.nan):
            with self.subTest(angle=angle):
                departure[1] = [radius * np.cos(angle), 0.0, -radius * np.sin(angle)]
                solver.tendon_seg_attachment_l_local.assign(departure)
                if angle in (-0.1, -np.pi + 1e-3) or not np.isfinite(angle):
                    with self.assertRaisesRegex(AssertionError, "Invalid Atwood wrap"):
                        assert_supported_atwood_wrap(solver)
                else:
                    assert_supported_atwood_wrap(solver)

    def test_authored_masses(self):
        """Use the specified masses in both finite-friction Atwood fixtures."""
        for build in (build_dynamic_pulley_atwood, build_kinematic_pulley_atwood):
            with self.subTest(fixture=build.__name__), wp.ScopedDevice("cpu"):
                model, left, right, _ = build(mass_left=1.0, mass_right=3.0)
                np.testing.assert_allclose(model.body_mass.numpy()[[left, right]], [1.0, 3.0], atol=1e-6)

    def test_guided_weights_and_single_hinge(self):
        """Keep friction fixtures below the pulley with no redundant pulley joint."""
        for build in (build_dynamic_pulley_atwood, build_kinematic_pulley_atwood):
            with self.subTest(fixture=build.__name__), wp.ScopedDevice("cpu"):
                model, left, right, pulley = build()
                selected = np.isin(model.joint_child.numpy(), [left, right])
                self.assertEqual(np.count_nonzero(selected), 2)
                np.testing.assert_array_equal(model.joint_type.numpy()[selected], int(newton.JointType.PRISMATIC))
                dofs = model.joint_qd_start.numpy()[:-1][selected]
                np.testing.assert_allclose(model.joint_axis.numpy()[dofs], [[0, 0, 1], [0, 0, 1]])
                np.testing.assert_allclose(model.joint_limit_lower.numpy()[dofs], -1.0)
                np.testing.assert_allclose(model.joint_limit_upper.numpy()[dofs], 1.0)
                if build is build_dynamic_pulley_atwood:
                    pulley_joints = model.joint_child.numpy() == pulley
                    self.assertEqual(np.count_nonzero(pulley_joints), 1)
                    np.testing.assert_array_equal(
                        model.joint_type.numpy()[pulley_joints], int(newton.JointType.REVOLUTE)
                    )


class TestCableMachineConfiguration(unittest.TestCase):
    def setUp(self):
        """Construct the cable machine without advancing it."""
        with wp.ScopedDevice("cpu"):
            self.example = CableMachineExample(None, None)

    def test_weight_masses(self):
        """Keep the capsule lighter than the box, using the specified 1 kg and 2 kg masses."""
        example = self.example
        np.testing.assert_allclose(example.model.body_mass.numpy()[[example.left_idx, example.right_idx]], [1.0, 2.0])

    def test_guides_and_travel_clearance(self):
        """Bound both weights' travel below their adjacent pulleys."""
        example = self.example
        model = example.model
        bodies = [example.left_idx, example.right_idx]
        selected = np.isin(model.joint_child.numpy(), bodies)
        np.testing.assert_array_equal(model.joint_type.numpy()[selected], int(newton.JointType.PRISMATIC))
        dofs = model.joint_qd_start.numpy()[:-1][selected]
        np.testing.assert_allclose(model.joint_axis.numpy()[dofs], [[0, 0, 1], [0, 0, 1]])
        heights = model.body_q.numpy()[bodies, 2]
        offsets = model.tendon_link_offset.numpy()[[0, -1], 2]
        crown = model.body_q.numpy()[[example.p1_idx, example.p3_idx], 2] - [example.r1, example.r3]
        self.assertTrue(np.all(heights + offsets + model.joint_limit_upper.numpy()[dofs] < crown))
        self.assertTrue(np.isfinite(model.joint_limit_lower.numpy()[dofs]).all())

    def test_capsule_is_vertical(self):
        """Align the capsule's long axis with its top-mounted cable attachment."""
        example = self.example
        shape = np.flatnonzero(example.model.shape_body.numpy() == example.left_idx)[0]
        rotation = example.model.shape_transform.numpy()[shape, 3:]
        axis = wp.quat_rotate(wp.quat(*rotation), wp.vec3(0.0, 0.0, 1.0))
        np.testing.assert_allclose(np.array(axis), [0, 0, 1], atol=1e-7)

    def test_unique_pulley_hinges(self):
        """Give each pulley only its revolute joint."""
        example = self.example
        children = example.model.joint_child.numpy()
        types = example.model.joint_type.numpy()
        for body in (example.p1_idx, example.p2_idx, example.p3_idx):
            with self.subTest(pulley=body):
                np.testing.assert_array_equal(types[children == body], [int(newton.JointType.REVOLUTE)])


class TestGearPulleyConfiguration(unittest.TestCase):
    def setUp(self):
        """Construct the block-and-tackle without advancing it."""
        with wp.ScopedDevice("cpu"):
            self.example = GearPulleyExample(None, None)

    def test_initial_wraps(self):
        """Start every pulley with a supported, nonzero oriented wrap."""
        example = self.example
        centers = example.model.tendon_link_offset.numpy()[1:-1]
        axes = example.model.tendon_link_axis.numpy()[1:-1]
        arrival = example.solver.tendon_seg_attachment_r_local.numpy()[:-1] - centers
        departure = example.solver.tendon_seg_attachment_l_local.numpy()[1:] - centers
        angles = np.arctan2(np.sum(np.cross(arrival, departure) * axes, axis=1), np.sum(arrival * departure, axis=1))
        self.assertTrue(np.isfinite(angles).all())
        self.assertTrue(np.all((angles > 0.0) & (angles < np.pi)), f"Invalid initial wraps: {np.degrees(angles)}")

    def test_hinges_and_free_counterweight(self):
        """Give each pulley one hinge and retain the unconstrained counterweight."""
        example = self.example
        model = example.model
        children = model.joint_child.numpy()
        types = model.joint_type.numpy()
        for body in example._pulley_indices:
            with self.subTest(pulley=body):
                np.testing.assert_array_equal(types[children == body], [int(newton.JointType.REVOLUTE)])
        np.testing.assert_array_equal(types[children == example.right_idx], [int(newton.JointType.FREE)])
        self.assertGreater(float(model.body_inv_mass.numpy()[example.right_idx]), 0.0)
        self.assertGreater(float(np.linalg.norm(model.body_inv_inertia.numpy()[example.right_idx])), 0.0)

    def test_counterweight_collides_with_floor(self):
        """Do not filter out the floor that bounds the counterweight's travel."""
        model = self.example.model
        bodies = model.shape_body.numpy()
        groups = model.shape_collision_group.numpy()
        floor = groups[bodies == -1]
        self.assertEqual(len(floor), 1)
        self.assertGreater(int(floor[0]), 0)
        np.testing.assert_array_equal(groups[bodies == self.example.right_idx], floor)

    def test_motion_check_accepts_startup_reversal(self):
        """Detect a moving pulley even when a startup reversal cancels its net angle."""
        example = self.example
        frames = example._direction_validation_frames
        progress = np.linspace(0.0, 1.0, frames)
        lower = np.tile(np.array(example.lower_block_pos), (frames, 1))
        right = np.tile(np.array(example.right_initial_pos), (frames, 1))
        lower[:, 2] += 0.15 * progress
        right[:, 2] -= 0.60 * progress
        rotations = np.tile(progress[:, None], (1, len(example._pulley_indices)))
        rotations[:, 1] = np.interp(np.arange(frames), [0, 20, 45, frames - 1], [0.0, -0.02, 0.0, 1.0])
        example._lower_history = lower.tolist()
        example._right_history = right.tolist()
        example._pulley_rotation_history = rotations.tolist()
        example._axis_error_history = np.zeros_like(rotations).tolist()
        example.test_final()
        rotations[:, 1] = 0.0
        example._pulley_rotation_history = rotations.tolist()
        with self.assertRaisesRegex(AssertionError, "inactive=.*upper_large"):
            example.test_final()


class TestDynamicCapstanConfiguration(unittest.TestCase):
    def setUp(self):
        """Construct the dynamic capstan without advancing it."""
        with wp.ScopedDevice("cpu"):
            self.example = DynamicCapstanExample(None, SimpleNamespace())

    def test_weight_masses(self):
        """Use the specified masses instead of adding default-density shape mass."""
        example = self.example
        masses = example.model.body_mass.numpy()
        np.testing.assert_allclose(masses[example.left_indices], 1.0, atol=1e-6)
        np.testing.assert_allclose(masses[example.right_indices], 1.2, atol=1e-6)

    def test_vertical_guides_and_travel(self):
        """Keep the weights below the pulley without allowing sideways swing."""
        example = self.example
        model = example.model
        weights = example.left_indices + example.right_indices
        selected = np.isin(model.joint_child.numpy(), weights)
        self.assertEqual(np.count_nonzero(selected), len(weights))
        np.testing.assert_array_equal(model.joint_type.numpy()[selected], int(newton.JointType.PRISMATIC))
        dofs = model.joint_qd_start.numpy()[:-1][selected]
        np.testing.assert_allclose(model.joint_axis.numpy()[dofs], np.tile([0.0, 0.0, 1.0], (6, 1)), atol=1e-7)
        np.testing.assert_allclose(model.joint_limit_lower.numpy()[dofs], -example.weight_travel)
        np.testing.assert_allclose(model.joint_limit_upper.numpy()[dofs], example.weight_travel)
        self.assertLess(2.0 + example.weight_travel + 0.09, example.pulley_z - example.pulley_radius)

    def test_pulley_has_only_its_hinge(self):
        """Do not create a free joint in parallel with the revolute joint."""
        model = self.example.model
        for body in self.example.pulley_indices:
            selected = model.joint_child.numpy() == body
            self.assertEqual(np.count_nonzero(selected), 1)
            np.testing.assert_array_equal(model.joint_type.numpy()[selected], int(newton.JointType.REVOLUTE))

    def test_reject_unsupported_wrap(self):
        """Reject an invalid route even when the bodies remain finite and bounded."""
        example = self.example
        example._assert_supported_wraps()
        arrival = example.solver.tendon_seg_attachment_r_local.numpy()
        departure = example.solver.tendon_seg_attachment_l_local.numpy()
        center = example.model.tendon_link_offset.numpy()[1]
        arrival[0] = center + np.array([example.pulley_radius, 0.0, 0.0])
        departure[1] = center + np.array([0.0, 0.0, example.pulley_radius])
        example.solver.tendon_seg_attachment_r_local.assign(arrival)
        example.solver.tendon_seg_attachment_l_local.assign(departure)
        with self.assertRaisesRegex(AssertionError, "supported half-wrap"):
            example._assert_supported_wraps()


if __name__ == "__main__":
    unittest.main()
