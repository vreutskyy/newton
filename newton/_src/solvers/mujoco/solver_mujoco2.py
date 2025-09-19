# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import warp as wp

from ...core.types import override
from ...geometry import GeoType
from ...sim import (
    Contacts,
    Control,
    JointType,
    Model,
    State,
)
from ..solver import SolverBase

try:
    import mujoco_warp
    from mujoco_warp import Model as MjWarpModel, Data as MjWarpData
except ImportError:
    raise ImportError("mujoco_warp is required for SolverMuJoCo2. Please install it.")


class SolverMuJoCo2(SolverBase):
    """Clean MuJoCo solver.
    
    Automatically detects whether to separate environments to worlds based on
    entity groups. If all entities are in the global environment (group -1),
    uses a single world. Otherwise, creates separate worlds for each environment.
    All non-global environments are expected to have the same number of bodies,
    shapes, and joints.
    """
    
    def __init__(self, model: Model):
        super().__init__(model)
        
        # Automatically determine if we should separate environments to worlds
        # by checking if any entities are in non-global environments
        all_groups = np.concatenate([
            model.body_group.numpy(),
            model.shape_group.numpy(),
            model.joint_group.numpy()
        ])
        # Check if any entities are in non-global environments (group >= 0)
        has_environments = np.any(all_groups >= 0)
        self.separate_envs_to_worlds = has_environments
         
        # Validate environment consistency if separating
        if self.separate_envs_to_worlds:
            self._validate_environment_consistency(model)
            
            # Compute actual number of environments from entity groups
            env_groups = all_groups[all_groups >= 0]
            if len(env_groups) > 0:
                self.n_worlds = int(np.max(env_groups) + 1)
            else:
                # This shouldn't happen after the has_environments check
                self.n_worlds = 1
        else:
            # Everything is global, single world
            self.n_worlds = 1
        
        # Build MuJoCo model
        self.mjc_model, self.mjw_model, self.mjw_data = self._build_mjc_model(model)
        
        # Create mappings (MuJoCo → Newton only!)
        # These will be initialized in _create_mappings
        self.mjc_to_newton_body: wp.array2d = None  # [nworld, n_mjc_bodies] -> Newton body idx
        self.mjc_to_newton_joint: wp.array2d = None  # [nworld, n_mjc_joints] -> Newton joint idx  
        self.mjc_to_newton_geom: wp.array2d = None  # [nworld, n_mjc_geoms] -> Newton shape idx
        self.mjc_to_newton_dof: wp.array2d = None  # [nworld, n_mjc_dofs] -> encoded (joint<<3)|dof
        
        # Create mappings
        self._create_mappings(model)
        
        # Create MuJoCo joint info arrays once (to avoid creating them in update methods)
        if self.mjc_model.njnt > 0:
            self.mjc_jnt_qposadr = wp.array(self.mjc_model.jnt_qposadr, dtype=wp.int32, device=model.device)
            self.mjc_jnt_dofadr = wp.array(self.mjc_model.jnt_dofadr, dtype=wp.int32, device=model.device)
            self.mjc_jnt_type = wp.array(self.mjc_model.jnt_type, dtype=wp.int32, device=model.device)
        else:
            self.mjc_jnt_qposadr = None
            self.mjc_jnt_dofadr = None
            self.mjc_jnt_type = None
    
    def _validate_environment_consistency(self, model: Model):
        """Validate that all non-global environments have the same structure."""
        # Group entities by environment
        body_groups = model.body_group.numpy()
        shape_groups = model.shape_group.numpy()
        joint_groups = model.joint_group.numpy()
        
        # Check for global bodies (not allowed with separate_envs_to_worlds)
        global_bodies = np.where(body_groups < 0)[0]
        if len(global_bodies) > 0:
            raise ValueError(
                f"Cannot use separate_envs_to_worlds=True with bodies in the global environment (group -1). "
                f"Found {len(global_bodies)} global bodies. Global dynamic bodies cannot be properly "
                f"synchronized across multiple MuJoCo worlds. Please assign all bodies to specific "
                f"environments (group >= 0). Only static shapes (body=-1) should be global."
            )
        
        # Count entities per environment
        env_body_counts = {}
        env_shape_counts = {}
        env_joint_counts = {}
        
        # Find unique non-global environments
        all_envs = set()
        all_envs.update(body_groups[body_groups >= 0])
        all_envs.update(shape_groups[shape_groups >= 0])
        all_envs.update(joint_groups[joint_groups >= 0])
        
        for env in sorted(all_envs):
            env_body_counts[env] = np.sum(body_groups == env)
            env_shape_counts[env] = np.sum(shape_groups == env)
            env_joint_counts[env] = np.sum(joint_groups == env)
        
        # Check consistency
        body_counts = list(env_body_counts.values())
        shape_counts = list(env_shape_counts.values())
        joint_counts = list(env_joint_counts.values())
        
        if len(body_counts) > 0 and len(set(body_counts)) > 1:
            raise ValueError(f"Environments have different body counts: {env_body_counts}")
        if len(shape_counts) > 0 and len(set(shape_counts)) > 1:
            raise ValueError(f"Environments have different shape counts: {env_shape_counts}")
        if len(joint_counts) > 0 and len(set(joint_counts)) > 1:
            raise ValueError(f"Environments have different joint counts: {env_joint_counts}")
    
    def _build_mjc_model(self, model: Model):
        """Build MuJoCo model from Newton model."""
        import mujoco
        
        spec = mujoco.MjSpec()
        
        # Configure options
        spec.option.gravity = model.gravity  # Set gravity from Newton model
        spec.option.timestep = 0.01  # Will be set during step()
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        
        # Allow MuJoCo to compute inertia from geoms when needed
        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO
        
        # Track names for later mapping
        self._name_tracking = {
            'bodies': {},   # mjc_name -> newton_idx
            'joints': {},   # mjc_name -> newton_idx
            'geoms': {},    # mjc_name -> newton_idx
        }
        
        if self.separate_envs_to_worlds:
            # Filter to first environment + global
            selected_bodies, selected_joints, selected_shapes = self._select_first_env(model)
            self._build_from_selection(model, spec, selected_bodies, selected_joints, selected_shapes)
        else:
            # Use all entities
            self._build_all(model, spec)
        
        # Compile
        mjc_model = spec.compile()
        
        # Create MuJoCo Warp model
        mjw_model = mujoco_warp.put_model(mjc_model)
        
        # Expand fields for multi-world if needed
        if self.separate_envs_to_worlds:
            self._expand_model_fields(mjw_model, self.n_worlds)
        
        # Create regular MuJoCo data first
        mjc_data = mujoco.MjData(mjc_model)
        
        # Determine nconmax and njmax  
        nconmax = model.rigid_contact_max if hasattr(model, 'rigid_contact_max') else 100
        njmax = 100  # Default from original solver
        
        # Create MuJoCo Warp data from regular data
        mjw_data = mujoco_warp.put_data(
            mjc_model,
            mjc_data,
            nworld=self.n_worlds,
            nconmax=nconmax,
            njmax=njmax,
        )
        
        return mjc_model, mjw_model, mjw_data
    
    def _expand_model_fields(self, mjw_model: MjWarpModel, n_worlds: int):
        """Expand MuJoCo model fields for multi-world support."""
        # MuJoCo Warp automatically expands fields when creating Data with multiple worlds
        # But we may need to manually expand some fields like timestep
        if hasattr(mjw_model.opt, 'timestep') and not hasattr(mjw_model.opt.timestep, 'fill_'):
            # Convert scalar timestep to array
            timestep = mjw_model.opt.timestep
            mjw_model.opt.timestep = wp.full(n_worlds, timestep, dtype=float, device=self.model.device)
    
    def _select_first_env(self, model: Model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select entities from first environment + global."""
        # Get entity groups
        body_groups = model.body_group.numpy()
        shape_groups = model.shape_group.numpy()
        joint_groups = model.joint_group.numpy()
        
        # Use environment 0 as the template
        first_env = 0
        
        # Select entities from first env + global
        selected_bodies = np.where((body_groups == first_env) | (body_groups < 0))[0]
        selected_joints = np.where((joint_groups == first_env) | (joint_groups < 0))[0]
        selected_shapes = np.where((shape_groups == first_env) | (shape_groups < 0))[0]
        
        return selected_bodies, selected_joints, selected_shapes
    
    def _build_all(self, model: Model, spec):
        """Build MuJoCo model using all Newton entities."""
        # For now, just build from all entities as if they were selected
        n_bodies = model.body_count
        n_joints = model.joint_count
        n_shapes = model.shape_count
        
        selected_bodies = np.arange(n_bodies)
        selected_joints = np.arange(n_joints)
        selected_shapes = np.arange(n_shapes)
        
        self._build_from_selection(model, spec, selected_bodies, selected_joints, selected_shapes)
    
    def _build_from_selection(self, model: Model, spec, 
                             selected_bodies: np.ndarray, 
                             selected_joints: np.ndarray, 
                             selected_shapes: np.ndarray):
        """Build MuJoCo entities from selected Newton entities."""
        import mujoco
        
        # Get Newton data as numpy arrays
        body_mass = model.body_mass.numpy()
        body_com = model.body_com.numpy()
        body_inertia = model.body_inertia.numpy()
        
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_type = model.joint_type.numpy()
        joint_axis = model.joint_axis.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_limit_lower = model.joint_limit_lower.numpy()
        joint_limit_upper = model.joint_limit_upper.numpy()
        
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_collision_radius = model.shape_collision_radius.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_transforms = model.shape_transform.numpy()
        
        # 1. Find root bodies (not children of any joint)
        children = set()
        for ji in selected_joints:
            child = joint_child[ji]
            if child >= 0:
                children.add(child)
        
        root_bodies = [bi for bi in selected_bodies if bi not in children]
        
        # 2. Add root bodies
        mjc_bodies = {}  # newton_idx -> mjc_body
        for bi in root_bodies:
            name = f"body_{bi}"
            mass = body_mass[bi]
            
            # Add body to worldbody
            if mass > 0 and np.any(body_inertia[bi] != 0):
                # Has valid mass and inertia
                body = spec.worldbody.add_body(
                    name=name,
                    mass=mass,
                    pos=[0, 0, 0],  # Initial position
                    ipos=body_com[bi].tolist(),
                    fullinertia=self._get_fullinertia(body_inertia[bi]),
                    explicitinertial=True,
                )
            else:
                # Let MuJoCo compute inertia from geoms
                body = spec.worldbody.add_body(
                    name=name,
                    pos=[0, 0, 0],  # Initial position
                )
            mjc_bodies[bi] = body
            self._name_tracking['bodies'][name] = bi
        
        # 3. Add joints and their child bodies in topological order
        # TODO: Implement proper topological sort if needed
        for ji in selected_joints:
            parent_idx = joint_parent[ji]
            child_idx = joint_child[ji]
            
            # Skip if child is invalid
            if child_idx < 0:
                continue
                
            # Add child body if not exists
            if child_idx not in mjc_bodies:
                name = f"body_{child_idx}"
                mass = body_mass[child_idx]
                
                # Get parent body
                if parent_idx >= 0 and parent_idx in mjc_bodies:
                    parent_body = mjc_bodies[parent_idx]
                else:
                    parent_body = spec.worldbody
                
                if mass > 0 and np.any(body_inertia[child_idx] != 0):
                    # Has valid mass and inertia
                    body = parent_body.add_body(
                        name=name,
                        mass=mass,
                        pos=[0, 0, 0],  # Will be set by joint
                        ipos=body_com[child_idx].tolist(),
                        fullinertia=self._get_fullinertia(body_inertia[child_idx]),
                        explicitinertial=True,
                    )
                else:
                    # Let MuJoCo compute inertia from geoms
                    body = parent_body.add_body(
                        name=name,
                        pos=[0, 0, 0],  # Will be set by joint
                    )
                mjc_bodies[child_idx] = body
                self._name_tracking['bodies'][name] = child_idx
            
            # Add joint
            child_body = mjc_bodies[child_idx]
            joint_name = f"joint_{ji}"
            
            # Add appropriate joint type
            jtype = joint_type[ji]
            if jtype == JointType.FREE:
                child_body.add_freejoint(name=joint_name)
            elif jtype == JointType.BALL:
                child_body.add_joint(
                    name=joint_name,
                    type=mujoco.mjtJoint.mjJNT_BALL,
                )
            elif jtype == JointType.REVOLUTE:
                # Get axis using DOF index
                qd_start = joint_qd_start[ji]
                axis = joint_axis[qd_start].tolist()
                
                # Get joint limits
                lower = joint_limit_lower[qd_start]
                upper = joint_limit_upper[qd_start]
                
                # Check if joint has limits (Newton uses large values for unlimited)
                has_limits = abs(lower) < 1e5 and abs(upper) < 1e5
                
                joint_params = {
                    "name": joint_name,
                    "type": mujoco.mjtJoint.mjJNT_HINGE,
                    "axis": axis,
                }
                
                if has_limits:
                    joint_params["limited"] = True
                    joint_params["range"] = (np.rad2deg(lower), np.rad2deg(upper))
                
                child_body.add_joint(**joint_params)
            elif jtype == JointType.PRISMATIC:
                # Get axis using DOF index
                qd_start = joint_qd_start[ji]
                axis = joint_axis[qd_start].tolist()
                
                # Get joint limits
                lower = joint_limit_lower[qd_start]
                upper = joint_limit_upper[qd_start]
                
                # Check if joint has limits (Newton uses large values for unlimited)
                has_limits = abs(lower) < 1e5 and abs(upper) < 1e5
                
                joint_params = {
                    "name": joint_name,
                    "type": mujoco.mjtJoint.mjJNT_SLIDE,
                    "axis": axis,
                }
                
                if has_limits:
                    joint_params["limited"] = True
                    joint_params["range"] = (lower, upper)
                
                child_body.add_joint(**joint_params)
            # TODO: Add other joint types
            
            self._name_tracking['joints'][joint_name] = ji
        
        # 4. Add shapes/geoms
        for si in selected_shapes:
            body_idx = shape_body[si]
            
            # Get MuJoCo body (or worldbody for static shapes)
            if body_idx >= 0 and body_idx in mjc_bodies:
                mjc_body = mjc_bodies[body_idx]
            else:
                mjc_body = spec.worldbody
            
            geom_name = f"geom_{si}"
            stype = shape_type[si]
            
            # Get the size vector for this shape
            size = shape_scale[si]
            
            # Get shape transform relative to body
            shape_tf = shape_transforms[si]
            shape_pos = [shape_tf[0], shape_tf[1], shape_tf[2]]
            shape_quat = [shape_tf[3], shape_tf[4], shape_tf[5], shape_tf[6]]
            
            # Add geom based on type
            if stype == GeoType.SPHERE:
                # For spheres, MuJoCo wants a list with radius
                mjc_body.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=size.tolist(),
                    pos=shape_pos,
                    quat=shape_quat,
                )
            elif stype == GeoType.BOX:
                # For boxes, shape_scale is half-extents
                mjc_body.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=size.tolist(),
                    pos=shape_pos,
                    quat=shape_quat,
                )
            elif stype == GeoType.CAPSULE:
                # For capsules, size is [radius, half-height, ...]
                mjc_body.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=size.tolist(),
                    pos=shape_pos,
                    quat=shape_quat,
                )
            elif stype == GeoType.CYLINDER:
                # For cylinders, size is [radius, half-height, ...]
                mjc_body.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=size.tolist(),
                    pos=shape_pos,
                    quat=shape_quat,
                )
            elif stype == GeoType.PLANE:
                # MuJoCo plane
                mjc_body.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_PLANE,
                    size=[1000, 1000, 0.1],  # Large plane
                    pos=shape_pos,
                    quat=shape_quat,
                )
            # TODO: Add mesh support
            
            self._name_tracking['geoms'][geom_name] = si
    
    def _get_fullinertia(self, inertia_matrix: np.ndarray) -> list:
        """Convert 3x3 inertia matrix to MuJoCo's fullinertia format."""
        # MuJoCo expects: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        return [
            inertia_matrix[0, 0],
            inertia_matrix[1, 1],
            inertia_matrix[2, 2],
            inertia_matrix[0, 1],
            inertia_matrix[0, 2],
            inertia_matrix[1, 2],
        ]
    
    def _get_joint_dof_count(self, mjc_joint_idx: int) -> int:
        """Get number of DOFs for a MuJoCo joint."""
        import mujoco
        
        joint_type = self.mjc_model.jnt_type[mjc_joint_idx]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 3
        elif joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
            return 1
        else:
            return 0
    
    def _create_mappings(self, model: Model):
        """Create MuJoCo→Newton mappings using name lookups."""
        import mujoco
        
        n_worlds = self.n_worlds
            
        n_bodies = self.mjw_model.nbody
        n_joints = self.mjw_model.njnt
        n_geoms = self.mjw_model.ngeom
        n_dofs = self.mjw_model.nv
        
        # Initialize 2D mappings
        body_mapping = np.full((n_worlds, n_bodies), -1, dtype=np.int32)
        joint_mapping = np.full((n_worlds, n_joints), -1, dtype=np.int32)
        geom_mapping = np.full((n_worlds, n_geoms), -1, dtype=np.int32)
        dof_mapping = np.full((n_worlds, n_dofs), -1, dtype=np.int32)
        
        # Get entity lists per environment
        body_groups = model.body_group.numpy()
        shape_groups = model.shape_group.numpy()
        joint_groups = model.joint_group.numpy()
        
        # Build per-environment entity lists
        env_bodies = {}
        env_shapes = {}
        env_joints = {}
        
        # Find all environments
        all_envs = set()
        all_envs.update(body_groups[body_groups >= 0])
        all_envs.update(shape_groups[shape_groups >= 0])
        all_envs.update(joint_groups[joint_groups >= 0])
        
        for env in sorted(all_envs):
            env_bodies[env] = np.where(body_groups == env)[0]
            env_shapes[env] = np.where(shape_groups == env)[0]
            env_joints[env] = np.where(joint_groups == env)[0]
        
        # Global entities (env -1)
        global_bodies = np.where(body_groups < 0)[0]
        global_shapes = np.where(shape_groups < 0)[0]
        global_joints = np.where(joint_groups < 0)[0]
        
        # Get first environment index
        env_list = sorted(env_bodies.keys())
        if not env_list:
            # No environments found
            if self.separate_envs_to_worlds:
                raise ValueError("No environments found but separate_envs_to_worlds=True")
            # For non-separated case, treat all entities as belonging to world 0
            first_env = None
        else:
            first_env = env_list[0]
        
        # Special handling when there are no environments (all entities are global)
        if first_env is None:
            # Map all entities directly to world 0
            for mjc_name, newton_idx in self._name_tracking['bodies'].items():
                mjc_idx = mujoco.mj_name2id(self.mjc_model, mujoco.mjtObj.mjOBJ_BODY, mjc_name)
                if mjc_idx >= 0:
                    body_mapping[0, mjc_idx] = newton_idx
        else:
            # Body mapping with environment support
            for mjc_name, newton_idx_in_env0 in self._name_tracking['bodies'].items():
                mjc_idx = mujoco.mj_name2id(self.mjc_model, mujoco.mjtObj.mjOBJ_BODY, mjc_name)
                if mjc_idx < 0:
                    continue
                    
                # Find position of this body within env0's body list
                if newton_idx_in_env0 in global_bodies:
                    # Global body - same index for all worlds
                    for world in range(n_worlds):
                        body_mapping[world, mjc_idx] = newton_idx_in_env0
                else:
                    # Environment-specific body
                    env0_bodies = env_bodies[first_env]
                    pos_in_env = np.where(env0_bodies == newton_idx_in_env0)[0]
                    if len(pos_in_env) == 0:
                        continue
                    pos_in_env = pos_in_env[0]
                    
                    for world_idx, env in enumerate(env_list[:n_worlds]):
                        if pos_in_env < len(env_bodies[env]):
                            body_mapping[world_idx, mjc_idx] = env_bodies[env][pos_in_env]
        
        # Geom/Shape mapping (similar logic)
        if first_env is None:
            # Map all entities directly to world 0
            for mjc_name, newton_idx in self._name_tracking['geoms'].items():
                mjc_idx = mujoco.mj_name2id(self.mjc_model, mujoco.mjtObj.mjOBJ_GEOM, mjc_name)
                if mjc_idx >= 0:
                    geom_mapping[0, mjc_idx] = newton_idx
        else:
            for mjc_name, newton_idx_in_env0 in self._name_tracking['geoms'].items():
                mjc_idx = mujoco.mj_name2id(self.mjc_model, mujoco.mjtObj.mjOBJ_GEOM, mjc_name)
                if mjc_idx < 0:
                    continue
                    
                if newton_idx_in_env0 in global_shapes:
                    # Global shape - same index for all worlds
                    for world in range(n_worlds):
                        geom_mapping[world, mjc_idx] = newton_idx_in_env0
                else:
                    # Environment-specific shape
                    env0_shapes = env_shapes[first_env]
                    pos_in_env = np.where(env0_shapes == newton_idx_in_env0)[0]
                    if len(pos_in_env) == 0:
                        continue
                    pos_in_env = pos_in_env[0]
                    
                    for world_idx, env in enumerate(env_list[:n_worlds]):
                        if pos_in_env < len(env_shapes[env]):
                            geom_mapping[world_idx, mjc_idx] = env_shapes[env][pos_in_env]
        
        # DOF mapping with encoded joint+dof indices
        joint_qd_start = model.joint_qd_start.numpy()
        
        if first_env is None:
            # Map all joints directly to world 0
            for mjc_name, newton_idx in self._name_tracking['joints'].items():
                mjc_joint_idx = mujoco.mj_name2id(self.mjc_model, mujoco.mjtObj.mjOBJ_JOINT, mjc_name)
                if mjc_joint_idx < 0:
                    continue
                    
                # Joint mapping
                joint_mapping[0, mjc_joint_idx] = newton_idx
                
                # Get DOF range for this joint
                dof_start = self.mjc_model.jnt_dofadr[mjc_joint_idx]
                n_joint_dofs = self._get_joint_dof_count(mjc_joint_idx)
                
                # Map DOFs
                for dof_offset in range(n_joint_dofs):
                    mjc_dof_idx = dof_start + dof_offset
                    dof_mapping[0, mjc_dof_idx] = (newton_idx << 3) | dof_offset
        else:
            for mjc_name, newton_idx_in_env0 in self._name_tracking['joints'].items():
                mjc_joint_idx = mujoco.mj_name2id(self.mjc_model, mujoco.mjtObj.mjOBJ_JOINT, mjc_name)
                if mjc_joint_idx < 0:
                    continue
                    
                # Get DOF range for this joint
                dof_start = self.mjc_model.jnt_dofadr[mjc_joint_idx]
                n_joint_dofs = self._get_joint_dof_count(mjc_joint_idx)
                
                # Map DOFs
                for dof_offset in range(n_joint_dofs):
                    mjc_dof_idx = dof_start + dof_offset
                    
                    if newton_idx_in_env0 in global_joints:
                        # Global joint
                        for world in range(n_worlds):
                            joint_mapping[world, mjc_joint_idx] = newton_idx_in_env0
                            # Encode: (joint_idx << 3) | dof_offset (3 bits for offset is enough)
                            dof_mapping[world, mjc_dof_idx] = (newton_idx_in_env0 << 3) | dof_offset
                    else:
                        # Environment-specific joint
                        env0_joints = env_joints[first_env]
                        pos_in_env = np.where(env0_joints == newton_idx_in_env0)[0]
                        if len(pos_in_env) == 0:
                            continue
                        pos_in_env = pos_in_env[0]
                        
                        for world_idx, env in enumerate(env_list[:n_worlds]):
                            if pos_in_env < len(env_joints[env]):
                                newton_joint_idx = env_joints[env][pos_in_env]
                                joint_mapping[world_idx, mjc_joint_idx] = newton_joint_idx
                                dof_mapping[world_idx, mjc_dof_idx] = (newton_joint_idx << 3) | dof_offset
        
        # Create warp arrays
        self.mjc_to_newton_body = wp.array(body_mapping, dtype=wp.int32, shape=(n_worlds, n_bodies))
        self.mjc_to_newton_joint = wp.array(joint_mapping, dtype=wp.int32, shape=(n_worlds, n_joints))
        self.mjc_to_newton_geom = wp.array(geom_mapping, dtype=wp.int32, shape=(n_worlds, n_geoms))
        self.mjc_to_newton_dof = wp.array(dof_mapping, dtype=wp.int32, shape=(n_worlds, n_dofs))
    
    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        """Step the simulation."""
        # 1. Update Newton → MuJoCo
        self._update_newton_to_mjc(state_in, control)
        
        # 2. Step MuJoCo
        self.mjw_model.opt.timestep.fill_(dt)
        with wp.ScopedDevice(self.model.device):
            mujoco_warp.step(self.mjw_model, self.mjw_data)
        
        # 3. Update MuJoCo → Newton  
        self._update_mjc_to_newton(state_out)
    
    def _update_newton_to_mjc(self, state: State, control: Control):
        """Update MuJoCo from Newton state."""
        n_worlds = self.n_worlds
        
        # Update joint positions and velocities
        if self.mjw_model.njnt > 0:
            wp.launch(
                kernel=update_joint_positions_newton_to_mjc,
                dim=(n_worlds, self.mjw_model.njnt),
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    self.mjc_to_newton_joint,
                    self.mjw_data.qpos,
                    self.mjw_data.qvel,
                    self.mjc_jnt_qposadr,
                    self.mjc_jnt_dofadr,
                    self.mjc_jnt_type,
                ],
                device=self.model.device
            )
        
        # Update joint properties (armature, damping, etc.)
        if self.mjw_model.njnt > 0 and self.mjw_model.nv > 0:
            # Launch kernel to update from Newton
            # MuJoCo Warp already has 2D arrays, so we can use them directly
            wp.launch(
                kernel=update_joint_properties_newton_to_mjc,
                dim=(n_worlds, self.mjw_model.nv),
                inputs=[
                    self.model.joint_armature,
                    self.model.joint_qd_start,
                    self.mjc_to_newton_dof,
                    self.mjw_model.dof_armature,  # MuJoCo Warp array is already 2D
                ],
                device=self.model.device
            )
        
        # Update static shape positions if they move
        # NOTE: In the initial implementation, we assume static shapes don't move during simulation
        # This kernel would be used if we support moving static shapes in the future
    
    def _update_mjc_to_newton(self, state: State):
        """Update Newton state from MuJoCo."""
        n_worlds = self.n_worlds
        
        # Update joint positions and velocities
        if self.mjw_model.njnt > 0:
            wp.launch(
                kernel=update_joint_positions_mjc_to_newton,
                dim=(n_worlds, self.mjw_model.njnt),
                inputs=[
                    self.mjw_data.qpos,
                    self.mjw_data.qvel,
                    self.mjc_to_newton_joint,
                    state.joint_q,
                    state.joint_qd,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    self.mjc_jnt_qposadr,
                    self.mjc_jnt_dofadr,
                    self.mjc_jnt_type,
                ],
                device=self.model.device
            )
        
        # Update body transforms
        if self.mjw_model.nbody > 0:
            # MuJoCo Warp already has 2D arrays for xpos and xquat
            # Launch kernel directly (includes world body in mapping)
            wp.launch(
                kernel=update_body_properties_mjc_to_newton,
                dim=(n_worlds, self.mjw_model.nbody),
                inputs=[
                    self.mjw_data.xpos,    # Already 2D in MuJoCo Warp
                    self.mjw_data.xquat,   # Already 2D in MuJoCo Warp
                    self.mjc_to_newton_body,
                    state.body_q,
                ],
                device=self.model.device
            )


# Kernel definitions

@wp.kernel
def update_body_properties_mjc_to_newton(
    # MuJoCo data (2D: [world, body])
    mjc_body_pos: wp.array2d(dtype=wp.vec3),
    mjc_body_quat: wp.array2d(dtype=wp.quatf),
    # Mapping (2D!)
    mjc_to_newton_body: wp.array2d(dtype=wp.int32),
    # Newton data (1D)
    newton_body_q: wp.array(dtype=wp.transform),
):
    world_id, mjc_body_id = wp.tid()
    
    # Get Newton body index for this specific MuJoCo body in this world
    newton_body_idx = mjc_to_newton_body[world_id, mjc_body_id]
    if newton_body_idx < 0:
        return  # No mapping
    
    # Read from MuJoCo
    pos = mjc_body_pos[world_id, mjc_body_id]
    quat = mjc_body_quat[world_id, mjc_body_id]
    
    # Write to Newton
    newton_body_q[newton_body_idx] = wp.transform(pos, quat)


@wp.kernel  
def update_joint_properties_newton_to_mjc(
    # Newton data
    newton_joint_armature: wp.array(dtype=float),
    newton_joint_qd_start: wp.array(dtype=wp.int32),
    # Mapping (2D!)
    mjc_to_newton_dof: wp.array2d(dtype=wp.int32),
    # MuJoCo data (2D: [world, dof])
    mjc_dof_armature: wp.array2d(dtype=float),
):
    world_id, mjc_dof_id = wp.tid()
    
    # Get encoded Newton joint+DOF index
    encoded = mjc_to_newton_dof[world_id, mjc_dof_id]
    if encoded < 0:
        return
    
    # Decode: joint_idx = encoded >> 3, dof_offset = encoded & 0x7 (3 bits)
    newton_joint_idx = encoded >> 3
    dof_offset = encoded & 0x7
    
    # Get actual Newton DOF index
    newton_dof_idx = newton_joint_qd_start[newton_joint_idx] + dof_offset
    
    # Read from Newton and write to MuJoCo
    mjc_dof_armature[world_id, mjc_dof_id] = newton_joint_armature[newton_dof_idx]


@wp.kernel
def update_static_geom_positions_newton_to_mjc(
    # Newton data
    newton_shape_transforms: wp.array(dtype=wp.transform),
    # Mapping (2D!)
    mjc_to_newton_geom: wp.array2d(dtype=wp.int32),
    # MuJoCo data (2D: [world, geom])
    mjc_geom_pos: wp.array2d(dtype=wp.vec3),
    mjc_geom_quat: wp.array2d(dtype=wp.quatf),
):
    world_id, mjc_geom_id = wp.tid()
    
    # Get Newton shape index
    newton_shape_idx = mjc_to_newton_geom[world_id, mjc_geom_id]
    if newton_shape_idx < 0:
        return
    
    # Read from Newton (same shape for all worlds if it's global/static)
    tf = newton_shape_transforms[newton_shape_idx]
    
    # Write to MuJoCo
    mjc_geom_pos[world_id, mjc_geom_id] = tf.p
    mjc_geom_quat[world_id, mjc_geom_id] = tf.q


@wp.kernel
def update_joint_positions_newton_to_mjc(
    # Newton data
    newton_joint_q: wp.array(dtype=float),
    newton_joint_qd: wp.array(dtype=float),
    newton_joint_q_start: wp.array(dtype=wp.int32),
    newton_joint_qd_start: wp.array(dtype=wp.int32),
    # Mapping (2D!)
    mjc_to_newton_joint: wp.array2d(dtype=wp.int32),
    # MuJoCo data (2D: [world, coord/dof])
    mjc_qpos: wp.array2d(dtype=float),
    mjc_qvel: wp.array2d(dtype=float),
    # Joint info
    mjc_jnt_qposadr: wp.array(dtype=wp.int32),
    mjc_jnt_dofadr: wp.array(dtype=wp.int32),
    mjc_jnt_type: wp.array(dtype=wp.int32),
):
    world_id, mjc_joint_id = wp.tid()
    
    # Get Newton joint index
    newton_joint_idx = mjc_to_newton_joint[world_id, mjc_joint_id]
    if newton_joint_idx < 0:
        return
    
    # Get joint type and DOF info
    joint_type = mjc_jnt_type[mjc_joint_id]
    qpos_start = mjc_jnt_qposadr[mjc_joint_id]
    dof_start = mjc_jnt_dofadr[mjc_joint_id]
    
    # Get Newton's q and qd start indices
    newton_q_start = newton_joint_q_start[newton_joint_idx]
    newton_qd_start = newton_joint_qd_start[newton_joint_idx]
    
    # Copy based on joint type
    # FREE = 0, BALL = 1, SLIDE = 2, HINGE = 3
    if joint_type == 0:  # FREE
        # Position (3 DOFs)
        for i in range(3):
            mjc_qpos[world_id, qpos_start + i] = newton_joint_q[newton_q_start + i]
            mjc_qvel[world_id, dof_start + i] = newton_joint_qd[newton_qd_start + i]
        # Quaternion (4 coords -> 3 DOFs for angular velocity)
        for i in range(4):
            mjc_qpos[world_id, qpos_start + 3 + i] = newton_joint_q[newton_q_start + 3 + i]
        for i in range(3):
            mjc_qvel[world_id, dof_start + 3 + i] = newton_joint_qd[newton_qd_start + 3 + i]
    elif joint_type == 1:  # BALL
        # Quaternion (4 coords, 3 DOFs)
        for i in range(4):
            mjc_qpos[world_id, qpos_start + i] = newton_joint_q[newton_q_start + i]
        for i in range(3):
            mjc_qvel[world_id, dof_start + i] = newton_joint_qd[newton_qd_start + i]
    elif joint_type == 2 or joint_type == 3:  # SLIDE or HINGE
        # Single DOF
        mjc_qpos[world_id, qpos_start] = newton_joint_q[newton_q_start]
        mjc_qvel[world_id, dof_start] = newton_joint_qd[newton_qd_start]


@wp.kernel
def update_joint_positions_mjc_to_newton(
    # MuJoCo data (2D: [world, coord/dof])
    mjc_qpos: wp.array2d(dtype=float),
    mjc_qvel: wp.array2d(dtype=float),
    # Mapping (2D!)
    mjc_to_newton_joint: wp.array2d(dtype=wp.int32),
    # Newton data
    newton_joint_q: wp.array(dtype=float),
    newton_joint_qd: wp.array(dtype=float),
    newton_joint_q_start: wp.array(dtype=wp.int32),
    newton_joint_qd_start: wp.array(dtype=wp.int32),
    # Joint info
    mjc_jnt_qposadr: wp.array(dtype=wp.int32),
    mjc_jnt_dofadr: wp.array(dtype=wp.int32),
    mjc_jnt_type: wp.array(dtype=wp.int32),
):
    world_id, mjc_joint_id = wp.tid()
    
    # Get Newton joint index
    newton_joint_idx = mjc_to_newton_joint[world_id, mjc_joint_id]
    if newton_joint_idx < 0:
        return
    
    # Get joint type and DOF info
    joint_type = mjc_jnt_type[mjc_joint_id]
    qpos_start = mjc_jnt_qposadr[mjc_joint_id]
    dof_start = mjc_jnt_dofadr[mjc_joint_id]
    
    # Get Newton's q and qd start indices
    newton_q_start = newton_joint_q_start[newton_joint_idx]
    newton_qd_start = newton_joint_qd_start[newton_joint_idx]
    
    # Copy based on joint type
    if joint_type == 0:  # FREE
        # Position (3 DOFs)
        for i in range(3):
            newton_joint_q[newton_q_start + i] = mjc_qpos[world_id, qpos_start + i]
            newton_joint_qd[newton_qd_start + i] = mjc_qvel[world_id, dof_start + i]
        # Quaternion (4 coords -> 3 DOFs for angular velocity)
        for i in range(4):
            newton_joint_q[newton_q_start + 3 + i] = mjc_qpos[world_id, qpos_start + 3 + i]
        for i in range(3):
            newton_joint_qd[newton_qd_start + 3 + i] = mjc_qvel[world_id, dof_start + 3 + i]
    elif joint_type == 1:  # BALL
        # Quaternion (4 coords, 3 DOFs)
        for i in range(4):
            newton_joint_q[newton_q_start + i] = mjc_qpos[world_id, qpos_start + i]
        for i in range(3):
            newton_joint_qd[newton_qd_start + i] = mjc_qvel[world_id, dof_start + i]
    elif joint_type == 2 or joint_type == 3:  # SLIDE or HINGE
        # Single DOF
        newton_joint_q[newton_q_start] = mjc_qpos[world_id, qpos_start]
        newton_joint_qd[newton_qd_start] = mjc_qvel[world_id, dof_start]
