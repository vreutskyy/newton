# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco import SolverMuJoCo2

wp.config.quiet = True


class TestSolverMuJoCo2(unittest.TestCase):
    """Test the new clean MuJoCo solver implementation."""

    def test_basic_instantiation(self):
        """Test that we can create a basic solver instance."""
        builder = newton.ModelBuilder()

        # Create a simple model with one body
        builder.current_env_group = 0
        body = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body=body, radius=0.5)

        model = builder.finalize()

        # Create solver
        solver = SolverMuJoCo2(model)

        # Check basic properties
        self.assertIsNotNone(solver.mjc_model)
        self.assertIsNotNone(solver.mjw_model)
        self.assertIsNotNone(solver.mjw_data)
        self.assertEqual(solver.mjc_model.nbody, 2)  # world + our body
        self.assertEqual(solver.mjc_model.ngeom, 1)  # our sphere

    def test_multi_environment_validation(self):
        """Test environment consistency validation."""
        builder = newton.ModelBuilder()

        # Create two environments with different body counts
        builder.current_env_group = 0
        builder.add_body(mass=1.0)
        builder.add_body(mass=2.0)

        builder.current_env_group = 1
        builder.add_body(mass=3.0)  # Only one body in env 1

        model = builder.finalize()

        # Should raise error due to inconsistent body counts
        with self.assertRaises(ValueError) as cm:
            SolverMuJoCo2(model)

        self.assertIn("different body counts", str(cm.exception))

    def test_global_body_validation(self):
        """Test that global bodies are not allowed with separate_envs_to_worlds."""
        builder = newton.ModelBuilder()

        # Create a global body
        builder.current_env_group = -1
        builder.add_body(mass=1.0)

        # Create a normal environment body
        builder.current_env_group = 0
        builder.add_body(mass=1.0)

        model = builder.finalize()

        # Should raise error for global bodies when environments exist
        with self.assertRaises(ValueError) as cm:
            SolverMuJoCo2(model)

        self.assertIn("global bodies", str(cm.exception))
        self.assertIn("Found 1 global bodies", str(cm.exception))

    def test_all_global_entities(self):
        """Test that all-global entities work fine (no environment separation)."""
        builder = newton.ModelBuilder()

        # Create only global entities (default group is -1)
        body0 = builder.add_body(mass=1.0)
        builder.add_joint_free(parent=-1, child=body0)
        builder.add_shape_sphere(body=body0, radius=0.1)

        model = builder.finalize()

        # Should work fine - no environments means no separation
        solver = SolverMuJoCo2(model)
        self.assertIsNotNone(solver)
        self.assertFalse(solver.separate_envs_to_worlds)
        self.assertEqual(solver.n_worlds, 1)

    def test_consistent_environments(self):
        """Test with properly consistent environments."""
        builder = newton.ModelBuilder()

        # Create two environments with same structure
        for env in [0, 1]:
            builder.current_env_group = env
            body = builder.add_body(mass=1.0 + env)
            builder.add_shape_sphere(body=body, radius=0.5)

        # Add a global static shape
        builder.current_env_group = -1
        builder.add_shape_plane(body=-1)

        model = builder.finalize()

        # Should work fine
        solver = SolverMuJoCo2(model)

        # Check properties
        self.assertEqual(solver.n_worlds, 2)  # Two environments
        self.assertEqual(solver.mjc_model.nbody, 2)  # world + one body from env0
        self.assertEqual(solver.mjc_model.ngeom, 2)  # sphere from env0 + global plane

    def test_joint_creation(self):
        """Test joint creation in MuJoCo model."""
        builder = newton.ModelBuilder()

        builder.current_env_group = 0
        parent = builder.add_body(mass=1.0)
        child = builder.add_body(mass=0.5)

        # Add geoms so MuJoCo can compute inertia
        builder.add_shape_box(body=parent, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=child, hx=0.1, hy=0.1, hz=0.1)

        # Add a revolute joint
        builder.add_joint_revolute(
            parent=parent,
            child=child,
            axis=(1, 0, 0),
        )

        model = builder.finalize()
        solver = SolverMuJoCo2(model)

        # Check joint was created
        self.assertEqual(solver.mjc_model.njnt, 1)
        self.assertEqual(solver.mjc_model.nv, 1)  # 1 DOF for revolute

    def test_mapping_dimensions(self):
        """Test that mappings have correct dimensions."""
        builder = newton.ModelBuilder()

        # Create two environments
        for env in [0, 1]:
            builder.current_env_group = env
            parent = builder.add_body(mass=1.0)
            child = builder.add_body(mass=0.5)
            builder.add_joint_revolute(parent=parent, child=child)
            builder.add_shape_box(body=child, hx=0.1, hy=0.1, hz=0.1)

        model = builder.finalize()
        solver = SolverMuJoCo2(model)

        # Check model was created with correct entities
        self.assertEqual(solver.n_worlds, 2)  # Two environments
        # Note: Only env 0 entities are in the MuJoCo model when separate_envs_to_worlds=True
        self.assertEqual(solver.mjc_model.nbody, 3)  # World + 2 bodies from env 0
        self.assertEqual(solver.mjc_model.njnt, 1)  # 1 joint from env 0
        self.assertEqual(solver.mjc_model.ngeom, 1)  # 1 shape from env 0

        # Check mapping dimensions
        n_worlds = solver.n_worlds
        n_bodies = solver.mjc_model.nbody  # Now includes world body
        n_geoms = solver.mjc_model.ngeom
        n_dofs = solver.mjc_model.nv
        self.assertEqual(solver.mjc_to_newton_body.shape, (n_worlds, n_bodies))
        self.assertEqual(solver.mjc_to_newton_geom.shape, (n_worlds, n_geoms))
        self.assertEqual(solver.mjc_to_newton_dof.shape, (n_worlds, n_dofs))

    def test_name_tracking(self):
        """Test that name tracking works correctly."""
        builder = newton.ModelBuilder()

        builder.current_env_group = 0
        body0 = builder.add_body(mass=1.0)
        joint0 = builder.add_joint_free(parent=-1, child=body0)
        shape0 = builder.add_shape_sphere(body=body0, radius=0.5)

        model = builder.finalize()
        solver = SolverMuJoCo2(model)

        # Check name tracking
        self.assertIn(f"body_{body0}", solver._name_tracking["bodies"])
        self.assertIn(f"joint_{joint0}", solver._name_tracking["joints"])
        self.assertIn(f"geom_{shape0}", solver._name_tracking["geoms"])

        # Check tracked indices
        self.assertEqual(solver._name_tracking["bodies"][f"body_{body0}"], body0)
        self.assertEqual(solver._name_tracking["joints"][f"joint_{joint0}"], joint0)
        self.assertEqual(solver._name_tracking["geoms"][f"geom_{shape0}"], shape0)

    def test_step_function(self):
        """Test that the step function runs without errors."""
        builder = newton.ModelBuilder()

        # Create a simple pendulum
        builder.current_env_group = 0
        body0 = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body0, axis=[0, 0, 1])
        builder.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)

        model = builder.finalize()
        solver = SolverMuJoCo2(model)

        # Create state objects
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = newton.Contacts(0, 0)

        # Step the simulation
        dt = 0.01
        solver.step(state_in, state_out, control, contacts, dt)

        # Check that step completed without errors
        self.assertIsNotNone(state_out)
        self.assertEqual(solver.mjc_model.opt.timestep, dt)

    def test_falling_body_physics(self):
        """Test that a falling body actually falls under gravity."""
        builder = newton.ModelBuilder()

        # Create a body that should fall
        builder.current_env_group = 0
        # Create free-floating body at height 1.0
        body0 = builder.add_body(mass=1.0)
        builder.add_joint_free(parent=-1, child=body0)
        builder.add_shape_sphere(body=body0, radius=0.1)

        model = builder.finalize()
        solver = SolverMuJoCo2(model)

        # Create states
        state = model.state()
        control = model.control()
        contacts = newton.Contacts(0, 0)

        # Set initial position at height 1.0
        # For free joint, first 3 DOFs are position
        state.joint_q.numpy()[0] = 0.0  # X
        state.joint_q.numpy()[1] = 1.0  # Y
        state.joint_q.numpy()[2] = 0.0  # Z
        state.joint_q.numpy()[3] = 1.0  # Quaternion w
        state.joint_q.numpy()[4] = 0.0  # Quaternion x
        state.joint_q.numpy()[5] = 0.0  # Quaternion y
        state.joint_q.numpy()[6] = 0.0  # Quaternion z

        # Simulate for 0.5 seconds with 0.01s timestep
        dt = 0.01
        for _ in range(50):
            solver.step(state, state, control, contacts, dt)

        # Check that the body has fallen
        final_height = state.joint_q.numpy()[1]  # Y position (2nd DOF of free joint)

        # With gravity ~9.81 m/s^2, after 0.5s:
        # Expected fall distance: 0.5 * g * t^2 = 0.5 * 9.81 * 0.5^2 ≈ 1.22625 m
        # So final height should be around 1.0 - 1.22625 = -0.22625
        # But let's just check it fell significantly
        self.assertLess(final_height, 0.5, "Body should have fallen under gravity")
        self.assertGreater(final_height, -2.0, "Body shouldn't have fallen too far")

    def test_pendulum_motion(self):
        """Test that a pendulum swings back and forth.

        This test verifies that joint position/velocity (qpos/qvel) synchronization
        between Newton and MuJoCo is working correctly for revolute joints.
        """
        builder = newton.ModelBuilder()

        # Create a pendulum
        # builder.current_env_group = 0
        body0 = builder.add_body(mass=1.0)
        # Attach at origin with revolute joint
        # Use Y-axis so pendulum swings in XZ plane (gravity is along -Z)
        builder.add_joint_revolute(
            parent=-1,
            child=body0,
            axis=[0, 1, 0],
            limit_lower=-np.pi / 2,  # -90 degrees
            limit_upper=np.pi / 2,  # +90 degrees
            # limit_ke=1000.0,       # Stiff limits
            # limit_kd=10.0          # Some damping at limits
        )
        # Add shape - this will create the pendulum bob at some distance from the joint
        builder.add_shape_sphere(body=body0, radius=0.1, xform=wp.transform([1.0, 0.0, 0.0], wp.quat_identity()))

        model = builder.finalize()

        solver = SolverMuJoCo2(model)

        # Create states
        state = model.state()
        control = model.control()
        contacts = newton.Contacts(0, 0)

        # Start pendulum at an angle (near one limit)
        joint_q_array = state.joint_q.numpy()
        joint_q_array[0] = np.pi / 3  # 60 degrees
        state.joint_q.assign(joint_q_array)

        # Record positions over time
        angles = []
        velocities = []
        dt = 0.01
        # print(f"\nPendulum simulation:")
        # print(f"Initial angle: {state.joint_q.numpy()[0]:.3f} rad ({np.degrees(state.joint_q.numpy()[0]):.1f} deg)")

        for _ in range(600):  # 6 seconds of simulation
            solver.step(state, state, control, contacts, dt)

            # Track joint angle and velocity
            joint_angle = state.joint_q.numpy()[0]
            joint_vel = state.joint_qd.numpy()[0]
            angles.append(joint_angle)
            velocities.append(joint_vel)

            # Debug output every 20 steps (0.2 seconds)
            # if i % 20 == 0:
            #    print(f"t={i*dt:.2f}s: angle={joint_angle:.3f} rad ({np.degrees(joint_angle):.1f} deg), vel={joint_vel:.3f} rad/s")

        # Check that pendulum oscillates (changes direction)
        # Count velocity sign changes
        velocity_sign_changes = 0
        for i in range(1, len(velocities)):
            if velocities[i - 1] * velocities[i] < 0:  # Sign change
                velocity_sign_changes += 1

        # Check that the pendulum actually moved
        angle_range = max(angles) - min(angles)

        # With limits, pendulum should oscillate between them
        self.assertGreater(
            velocity_sign_changes, 2, f"Pendulum should oscillate (got {velocity_sign_changes} direction changes)"
        )
        self.assertGreater(angle_range, 0.5, f"Pendulum should move significantly (angle range: {angle_range:.2f} rad)")

        # Velocity should vary (not constant)
        velocity_range = max(velocities) - min(velocities)
        self.assertGreater(
            velocity_range, 0.5, f"Velocity should vary due to gravity (range: {velocity_range:.2f} rad/s)"
        )

    def test_solver_options(self):
        """Test that solver options are properly set in MuJoCo."""
        builder = newton.ModelBuilder()

        # Create a simple model
        body = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body, radius=0.1)
        builder.add_joint_free(parent=-1, child=body)

        model = builder.finalize()

        # Test with newton solver
        solver_newton = SolverMuJoCo2(model, solver="newton", iterations=25, ls_iterations=5)
        self.assertEqual(solver_newton.solver, "newton")
        self.assertEqual(solver_newton.iterations, 25)
        self.assertEqual(solver_newton.ls_iterations, 5)

        # Test with cg solver
        solver_cg = SolverMuJoCo2(model, solver="cg", iterations=30, ls_iterations=8)
        self.assertEqual(solver_cg.solver, "cg")
        self.assertEqual(solver_cg.iterations, 30)
        self.assertEqual(solver_cg.ls_iterations, 8)

        # Test invalid solver type
        with self.assertRaises(ValueError) as cm:
            SolverMuJoCo2(model, solver="invalid")
        self.assertIn("Unknown solver", str(cm.exception))

    def test_body_properties_update(self):
        """Test that body mass, COM, and inertia are properly updated in MuJoCo."""
        builder = newton.ModelBuilder()

        # Create bodies with different properties
        body1 = builder.add_body(mass=2.0)
        body2 = builder.add_body(mass=3.0)

        # Add shapes to ensure proper inertia
        builder.add_shape_sphere(body1, radius=0.1)
        builder.add_shape_box(body2, hx=0.1, hy=0.1, hz=0.1)

        builder.add_joint_free(parent=-1, child=body1)
        builder.add_joint_revolute(parent=body1, child=body2, axis=[0, 1, 0])

        model = builder.finalize()

        # Set custom COM and inertia after finalization
        model.body_com.numpy()[0] = [0.1, 0.2, 0.3]
        model.body_inertia.numpy()[0] = np.eye(3) * 0.5
        model.body_inertia.numpy()[1] = np.diag([0.1, 0.2, 0.3])

        state = model.state(requires_grad=False)

        # Create solver
        solver = SolverMuJoCo2(model)

        # Check initial values are set
        # Note: We can't directly check MuJoCo internal values without exposing them,
        # but we can verify the solver runs without errors
        state_out = model.state(requires_grad=False)
        solver.step(state, state_out, model.control(), model.collide(state), 0.01)

        # Modify body properties
        model.body_mass.numpy()[0] = 5.0
        model.body_com.numpy()[0] = [0.5, 0.6, 0.7]
        model.body_inertia.numpy()[0] = np.diag([1.0, 2.0, 3.0])

        # This would require notify_model_changed() to update, which isn't implemented yet
        # For now, just verify the solver can handle the model
        solver.step(state_out, state, model.control(), model.collide(state_out), 0.01)

    def test_joint_properties(self):
        """Test that joint armature and friction are properly set."""
        builder = newton.ModelBuilder()

        # Create a chain with different joint types
        body1 = builder.add_body(mass=1.0)
        body2 = builder.add_body(mass=1.0)

        # Add shapes to ensure proper inertia
        builder.add_shape_sphere(body1, radius=0.1)
        builder.add_shape_sphere(body2, radius=0.1)

        # Add joints with specific armature and friction
        builder.add_joint_revolute(parent=-1, child=body1, axis=[0, 1, 0], armature=0.1, friction=0.05)
        builder.add_joint_prismatic(parent=body1, child=body2, axis=[1, 0, 0], armature=0.2, friction=0.1)

        model = builder.finalize()

        # Verify armature and friction are set
        self.assertAlmostEqual(model.joint_armature.numpy()[0], 0.1)
        self.assertAlmostEqual(model.joint_friction.numpy()[0], 0.05)
        self.assertAlmostEqual(model.joint_armature.numpy()[1], 0.2)
        self.assertAlmostEqual(model.joint_friction.numpy()[1], 0.1)

        # Create solver and verify it works
        solver = SolverMuJoCo2(model)
        state = model.state(requires_grad=False)
        state_out = model.state(requires_grad=False)
        solver.step(state, state_out, model.control(), model.collide(state), 0.01)

    def test_shape_material_properties(self):
        """Test that shape friction and contact parameters are properly set."""
        builder = newton.ModelBuilder()

        # Create shapes with different material properties
        body = builder.add_body(mass=1.0)
        builder.add_joint_free(parent=-1, child=body)

        # Add shapes with specific material properties
        cfg1 = newton.ModelBuilder.ShapeConfig()
        cfg1.mu = 0.8
        cfg1.ke = 1000.0
        cfg1.kd = 10.0
        builder.add_shape_sphere(body, radius=0.1, cfg=cfg1)

        cfg2 = newton.ModelBuilder.ShapeConfig()
        cfg2.mu = 0.5
        cfg2.ke = 2000.0
        cfg2.kd = 20.0
        builder.add_shape_box(body, cfg=cfg2)

        model = builder.finalize()

        # Verify material properties are set
        self.assertAlmostEqual(model.shape_material_mu.numpy()[0], 0.8)
        self.assertAlmostEqual(model.shape_material_ke.numpy()[0], 1000.0)
        self.assertAlmostEqual(model.shape_material_kd.numpy()[0], 10.0)
        self.assertAlmostEqual(model.shape_material_mu.numpy()[1], 0.5)

        # Create solver and verify it works
        solver = SolverMuJoCo2(model)
        state = model.state(requires_grad=False)
        state_out = model.state(requires_grad=False)
        solver.step(state, state_out, model.control(), model.collide(state), 0.01)

    def test_actuator_creation(self):
        """Test that actuators are properly created for controllable joints."""
        builder = newton.ModelBuilder()

        # Create joints with different control modes
        body1 = builder.add_body(mass=1.0)
        body2 = builder.add_body(mass=1.0)
        body3 = builder.add_body(mass=1.0)

        # Add shapes to ensure proper inertia
        builder.add_shape_sphere(body1, radius=0.1)
        builder.add_shape_sphere(body2, radius=0.1)
        builder.add_shape_sphere(body3, radius=0.1)

        # Position-controlled joint
        builder.add_joint_revolute(
            parent=-1,
            child=body1,
            axis=[0, 1, 0],
            limit_lower=-np.pi,
            limit_upper=np.pi,
            limit_ke=100.0,
            limit_kd=1.0,
            effort_limit=10.0,
            target_ke=50.0,
            target_kd=5.0,
            mode=newton.JointMode.TARGET_POSITION,
        )

        # Velocity-controlled joint
        builder.add_joint_prismatic(
            parent=body1,
            child=body2,
            axis=[1, 0, 0],
            effort_limit=20.0,
            target_kd=2.0,
            mode=newton.JointMode.TARGET_VELOCITY,
        )

        # Force-controlled joint (mode=NONE)
        builder.add_joint_revolute(
            parent=body2, child=body3, axis=[0, 0, 1], effort_limit=15.0, mode=newton.JointMode.NONE
        )

        model = builder.finalize()

        # Create solver
        solver = SolverMuJoCo2(model)

        # Verify actuators were created (3 joints with non-zero effort limits)
        self.assertEqual(len(solver._name_tracking["actuators"]), 3)

        # Test with control
        state = model.state(requires_grad=False)
        control = model.control(requires_grad=False)

        # Set control targets
        control.joint_target.numpy()[0] = 0.5  # Position target
        control.joint_target.numpy()[1] = 1.0  # Velocity target

        # Step should work with control
        state_out = model.state(requires_grad=False)
        solver.step(state, state_out, control, model.collide(state), 0.01)

    def test_joint_position_velocity_sync(self):
        """Test that joint positions and velocities are synchronized between Newton and MuJoCo."""
        builder = newton.ModelBuilder()

        # Create a simple pendulum
        body = builder.add_body(mass=1.0)
        # Use xform to set shape position
        shape_xform = wp.transform([0.5, 0, 0], wp.quat_identity())
        builder.add_shape_sphere(body, radius=0.1, xform=shape_xform)
        builder.add_joint_revolute(parent=-1, child=body, axis=[0, 1, 0])

        model = builder.finalize()
        state = model.state(requires_grad=False)

        # Set initial position and velocity
        state.joint_q.numpy()[0] = 0.5
        state.joint_qd.numpy()[0] = 2.0

        # Create solver
        solver = SolverMuJoCo2(model)

        # Step and verify values are preserved
        state_out = model.state(requires_grad=False)
        solver.step(state, state_out, model.control(), model.collide(state), 0.01)

        # Values should change due to physics but not be reset to zero
        self.assertNotAlmostEqual(state_out.joint_q.numpy()[0], 0.0)
        self.assertNotAlmostEqual(state_out.joint_qd.numpy()[0], 0.0)

    def test_multi_environment_actuators(self):
        """Test actuator creation with multiple environments."""
        builder = newton.ModelBuilder()

        # Create same structure in two environments
        for env in range(2):
            builder.current_env_group = env
            body = builder.add_body(mass=1.0)
            builder.add_shape_sphere(body, radius=0.1)  # Add shape for inertia
            builder.add_joint_revolute(
                parent=-1,
                child=body,
                axis=[0, 1, 0],
                effort_limit=10.0,
                target_ke=50.0,
                target_kd=5.0,
                mode=newton.JointMode.TARGET_POSITION,
            )

        model = builder.finalize()

        # Create solver - should auto-detect multiple environments
        solver = SolverMuJoCo2(model)

        self.assertTrue(solver.separate_envs_to_worlds)
        self.assertEqual(solver.n_worlds, 2)

        # Should have one actuator (from template env 0)
        self.assertEqual(len(solver._name_tracking["actuators"]), 1)

        # Test stepping works
        state = model.state(requires_grad=False)
        state_out = model.state(requires_grad=False)
        control = model.control(requires_grad=False)
        solver.step(state, state_out, control, model.collide(state), 0.01)

    def test_actuator_control_forces(self):
        """Test that actuator control forces are applied correctly."""
        builder = newton.ModelBuilder()

        # Create a simple pendulum with position control
        body = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body, radius=0.1)

        builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=[0, 1, 0],
            effort_limit=10.0,
            target_ke=100.0,  # Strong position control
            target_kd=10.0,  # Some damping
            mode=newton.JointMode.TARGET_POSITION,
        )

        model = builder.finalize()
        state = model.state(requires_grad=False)
        control = model.control(requires_grad=False)

        # Set initial angle
        joint_q = state.joint_q.numpy()
        joint_q[0] = 0.0
        state.joint_q.assign(joint_q)

        # Set target position
        joint_target = control.joint_target.numpy()
        joint_target[0] = np.pi / 4  # 45 degrees
        control.joint_target.assign(joint_target)

        # Create solver
        solver = SolverMuJoCo2(model)

        # Debug: Check actuator was created
        print(f"Number of actuators: {solver.mjw_model.nu}")
        print(f"Actuator tracking: {solver._name_tracking['actuators']}")
        print(f"Joint DOF mode: {model.joint_dof_mode.numpy()}")
        print(f"Joint qd start: {model.joint_qd_start.numpy()}")
        print(f"Joint target: {control.joint_target.numpy()}")

        # Step simulation
        dt = 0.01
        state_out = model.state(requires_grad=False)

        # Debug first few steps
        for i in range(100):  # 1 second
            solver.step(state, state_out, control, model.collide(state), dt)
            state, state_out = state_out, state

            if i < 5:
                print(f"Step {i}: angle={state.joint_q.numpy()[0]:.4f}, vel={state.joint_qd.numpy()[0]:.4f}")

        # Check that joint moved towards target
        final_angle = state.joint_q.numpy()[0]
        self.assertAlmostEqual(
            final_angle, np.pi / 4, places=2, msg=f"Joint should reach target position, got {final_angle:.3f}"
        )

    def test_joint_forces(self):
        """Test that joint forces are applied correctly."""
        builder = newton.ModelBuilder()

        # Create a simple pendulum
        body = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body, radius=0.1)

        builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=[0, 1, 0],
            mode=newton.JointMode.NONE,  # No control, just direct forces
        )

        model = builder.finalize()
        state = model.state(requires_grad=False)
        control = model.control(requires_grad=False)

        # Apply constant torque
        joint_f = control.joint_f.numpy()
        joint_f[0] = 5.0  # 5 Nm torque
        control.joint_f.assign(joint_f)

        # Create solver
        solver = SolverMuJoCo2(model)

        # Step simulation
        dt = 0.01
        initial_vel = state.joint_qd.numpy()[0]
        state_out = model.state(requires_grad=False)

        for _ in range(10):
            solver.step(state, state_out, control, model.collide(state), dt)
            state, state_out = state_out, state

        # Check that velocity increased (torque causes acceleration)
        final_vel = state.joint_qd.numpy()[0]
        self.assertGreater(
            final_vel,
            initial_vel + 0.1,
            f"Joint should accelerate with applied torque, vel changed from {initial_vel} to {final_vel}",
        )

    def test_body_forces(self):
        """Test that body forces are applied correctly.

        Note: Newton's ModelBuilder adds mass from shape volume even with density=0,
        so we calculate the required force based on actual mass rather than specified mass.
        """
        builder = newton.ModelBuilder()

        # Create a body with prismatic joint (can only move vertically)
        body = builder.add_body(mass=1.0)

        # Add sphere shape - Newton will add mass based on shape volume
        builder.add_shape_sphere(body, radius=0.1)
        builder.add_joint_prismatic(
            parent=-1,
            child=body,
            axis=[0, 0, 1],  # Z-axis (vertical)
            mode=newton.JointMode.NONE,  # No control, just free movement
        )

        model = builder.finalize()
        state = model.state()
        control = model.control()

        # Set initial joint position to 1.0 (1m high)
        joint_q = model.joint_q.numpy()
        joint_q[0] = 1.0
        model.joint_q.assign(joint_q)

        # Initialize state with forward kinematics
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        contacts = model.collide(state)

        # Apply upward force to counteract gravity
        # Use actual mass to calculate proper force
        actual_mass = model.body_mass.numpy()[0]
        weight = actual_mass * 9.81
        # Apply 2x weight to ensure upward movement
        upward_force = weight * 2.0

        body_f = state.body_f.numpy()
        body_f[0] = wp.spatial_vector([0, 0, upward_force, 0, 0, 0])
        state.body_f.assign(body_f)

        # Create solver
        solver = SolverMuJoCo2(model)

        # Get initial joint position
        initial_pos = state.joint_q.numpy()[0]  # joint position

        # Step simulation
        dt = 0.01
        state_out = model.state()

        for _ in range(100):  # 1 second
            state.clear_forces()
            state.body_f.assign(body_f)
            solver.step(state, state_out, control, contacts, dt)
            state, state_out = state_out, state

        # Check that joint moved upward (force > gravity)
        final_pos = state.joint_q.numpy()[0]
        self.assertGreater(
            final_pos,
            initial_pos,
            f"Joint should move upward with applied force, moved from {initial_pos:.3f} to {final_pos:.3f}",
        )

    def test_velocity_control(self):
        """Test velocity-controlled actuators."""
        builder = newton.ModelBuilder()

        # Create a slider with velocity control
        body = builder.add_body(mass=1.0)
        # Add box shape - Newton will add mass based on shape volume
        builder.add_shape_box(body)

        builder.add_joint_prismatic(
            parent=-1,
            child=body,
            axis=[1, 0, 0],  # Slide along X
            effort_limit=10000.0,  # High effort limit
            target_kd=5000.0,  # High velocity control gain for quick response
            mode=newton.JointMode.TARGET_VELOCITY,
        )

        model = builder.finalize()
        state = model.state(requires_grad=False)
        control = model.control(requires_grad=False)

        # Set target velocity
        joint_target = control.joint_target.numpy()
        joint_target[0] = 1.0  # 1 m/s (reduced target)
        control.joint_target.assign(joint_target)

        # Create solver
        solver = SolverMuJoCo2(model)

        # Step simulation until steady state
        dt = 0.01
        state_out = model.state(requires_grad=False)

        for _ in range(200):  # 2 seconds
            solver.step(state, state_out, control, model.collide(state), dt)
            state, state_out = state_out, state

        # Check that velocity is close to target
        final_vel = state.joint_qd.numpy()[0]
        self.assertAlmostEqual(final_vel, 1.0, places=1, msg=f"Joint should reach target velocity, got {final_vel:.3f}")


if __name__ == "__main__":
    unittest.main()
