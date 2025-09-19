# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
import warp as wp
import newton
from newton._src.solvers.mujoco import SolverMuJoCo2


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
            solver = SolverMuJoCo2(model)
        
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
        joint0 = builder.add_joint_free(parent=-1, child=body0)
        shape0 = builder.add_shape_sphere(body=body0, radius=0.1)
        
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
        self.assertIn(f"body_{body0}", solver._name_tracking['bodies'])
        self.assertIn(f"joint_{joint0}", solver._name_tracking['joints'])
        self.assertIn(f"geom_{shape0}", solver._name_tracking['geoms'])
        
        # Check tracked indices
        self.assertEqual(solver._name_tracking['bodies'][f"body_{body0}"], body0)
        self.assertEqual(solver._name_tracking['joints'][f"joint_{joint0}"], joint0)
        self.assertEqual(solver._name_tracking['geoms'][f"geom_{shape0}"], shape0)
    
    def test_step_function(self):
        """Test that the step function runs without errors."""
        builder = newton.ModelBuilder()
        
        # Create a simple pendulum
        builder.current_env_group = 0
        body0 = builder.add_body(mass=1.0)
        joint0 = builder.add_joint_revolute(parent=-1, child=body0, axis=[0, 0, 1])
        shape0 = builder.add_shape_box(body=body0, hx=0.5, hy=0.5, hz=0.5)
        
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
        joint0 = builder.add_joint_free(parent=-1, child=body0)
        shape0 = builder.add_shape_sphere(body=body0, radius=0.1)
        
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
        #builder.current_env_group = 0
        body0 = builder.add_body(mass=1.0)
        # Attach at origin with revolute joint
        # Use Y-axis so pendulum swings in XZ plane (gravity is along -Z)
        joint0 = builder.add_joint_revolute(
            parent=-1, 
            child=body0, 
            axis=[0, 1, 0],
            limit_lower=-np.pi/2,  # -90 degrees
            limit_upper=np.pi/2,   # +90 degrees
            #limit_ke=1000.0,       # Stiff limits
            #limit_kd=10.0          # Some damping at limits
        )
        # Add shape - this will create the pendulum bob at some distance from the joint
        shape0 = builder.add_shape_sphere(
            body=body0, 
            radius=0.1, 
            xform=wp.transform([1.0, 0.0, 0.0], wp.quat_identity())
        )
        
        model = builder.finalize()
        
        solver = SolverMuJoCo2(model)
        
        # Create states
        state = model.state()
        control = model.control()
        contacts = newton.Contacts(0, 0)
        
        # Start pendulum at an angle (near one limit)
        joint_q_array = state.joint_q.numpy()
        joint_q_array[0] = np.pi/3  # 60 degrees
        state.joint_q.assign(joint_q_array)
        
        # Record positions over time
        angles = []
        velocities = []
        dt = 0.01
        print(f"\nPendulum simulation:")
        print(f"Initial angle: {state.joint_q.numpy()[0]:.3f} rad ({np.degrees(state.joint_q.numpy()[0]):.1f} deg)")
        
        for i in range(600):  # 6 seconds of simulation
            solver.step(state, state, control, contacts, dt)
            
            # Track joint angle and velocity
            joint_angle = state.joint_q.numpy()[0]
            joint_vel = state.joint_qd.numpy()[0]
            angles.append(joint_angle)
            velocities.append(joint_vel)
            
            # Debug output every 20 steps (0.2 seconds)
            if i % 20 == 0:
                print(f"t={i*dt:.2f}s: angle={joint_angle:.3f} rad ({np.degrees(joint_angle):.1f} deg), vel={joint_vel:.3f} rad/s")
        
        # Check that pendulum oscillates (changes direction)
        # Count velocity sign changes
        velocity_sign_changes = 0
        for i in range(1, len(velocities)):
            if velocities[i-1] * velocities[i] < 0:  # Sign change
                velocity_sign_changes += 1
        
        # Check that the pendulum actually moved
        angle_range = max(angles) - min(angles)
        
        # With limits, pendulum should oscillate between them
        self.assertGreater(velocity_sign_changes, 2, 
                          f"Pendulum should oscillate (got {velocity_sign_changes} direction changes)")
        self.assertGreater(angle_range, 0.5, 
                          f"Pendulum should move significantly (angle range: {angle_range:.2f} rad)")
        
        # Velocity should vary (not constant)
        velocity_range = max(velocities) - min(velocities)
        self.assertGreater(velocity_range, 0.5, 
                          f"Velocity should vary due to gravity (range: {velocity_range:.2f} rad/s)")


if __name__ == "__main__":
    unittest.main()
