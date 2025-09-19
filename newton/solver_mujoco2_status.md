# SolverMuJoCo2 Implementation Status

## Initialization (Model Creation)

### ✅ Already Implemented

#### Solver Options
- ✅ `gravity` - from `model.gravity`
- ✅ `timestep` - default 0.01
- ✅ `integrator` - set to IMPLICITFAST

#### Basic Structure
- ✅ Bodies created with proper hierarchy
- ✅ Joints created with correct types (FREE, BALL, REVOLUTE, PRISMATIC)
- ✅ Shapes/geoms created with correct types and sizes
- ✅ Joint limits passed (after user fix)
- ✅ Joint axes computed correctly
- ✅ Shape transforms (position and orientation)
- ✅ Multi-world support with environment separation

### ❌ Missing in Initialization

#### Solver Options
- ❌ `solver` - constraint solver type (CG/Newton)
- ❌ `iterations` - solver iterations (default 20)
- ❌ `ls_iterations` - line search iterations (default 10)
- ❌ `cone` - friction cone type (pyramidal/elliptic)
- ❌ `impratio` - impedance ratio
- ❌ `tolerance` - solver tolerance
- ❌ `disableflags` - feature disable flags

#### Default Geom Properties
- ❌ `geom.condim` - contact dimension (default 3)
- ❌ `geom.solref` - default contact stiffness/damping
- ❌ `geom.solimp` - default contact impedance
- ❌ `geom.friction` - default friction coefficients

#### Body Properties
- ❌ `mass` - from `body_mass`
- ❌ `ipos` - center of mass from `body_com`
- ❌ `fullinertia` - from `body_inertia`
- ❌ `explicitinertial` - flag for explicit inertia

#### Joint Properties
- ❌ `pos` - joint position for non-revolute joints
- ❌ `armature` - from `joint_armature`
- ❌ `frictionloss` - from `joint_friction`
- ❌ `damping` - explicit damping values
- ❌ `solref_limit`, `solimp_limit` - customizable limit parameters

#### Geom/Shape Properties
- ❌ `contype`, `conaffinity` - collision filtering from shape colors
- ❌ `friction` - from `shape_material_mu` with torsional/rolling
- ❌ `solref` - contact stiffness/damping from `shape_material_ke/kd`
- ❌ `solimp` - contact impedance parameters
- ❌ `rgba` - visualization colors

#### Actuator System
- ❌ Actuators for each controllable DOF
- ❌ `gear` - actuator gear ratios
- ❌ `forcerange` - from `joint_effort_limit`
- ❌ `gainprm`, `biasprm` - PD control parameters
- ❌ Mapping from DOFs to actuators

#### Other
- ❌ Mesh support for complex geometries
- ❌ Equality constraints (weld, joint constraints)
- ❌ Proper up-axis handling (Y-up vs Z-up)

## Runtime Updates

### ✅ Already Implemented

#### Newton → MuJoCo
- ✅ Joint positions (`joint_q` → `qpos`)
- ✅ Joint velocities (`joint_qd` → `qvel`)

#### MuJoCo → Newton
- ✅ Joint positions (`qpos` → `joint_q`)
- ✅ Joint velocities (`qvel` → `joint_qd`)
- ✅ Body transforms (`xpos`, `xquat` → `body_q`)

### ❌ Missing Runtime Updates

#### Newton → MuJoCo Updates

##### Per-Step Updates
- ❌ Control forces:
  - ❌ `control.joint_target` → `ctrl` (via actuators)
  - ❌ `control.joint_f` → `qfrc_applied`
  - ❌ `state.body_f` → `xfrc_applied`

##### Dynamic Property Updates (via notify_model_changed)
- ❌ Body properties:
  - ❌ `body_mass` → `body_mass`
  - ❌ `body_com` → `body_ipos`
  - ❌ `body_inertia` → `body_inertia`, `body_iquat`

- ❌ Joint properties:
  - ❌ `joint_armature` → `dof_armature`
  - ❌ `joint_friction` → `dof_frictionloss`
  - ❌ `joint_X_p`, `joint_X_c` → `jnt_pos`, `jnt_axis`
  - ❌ Body transforms from joint transforms

- ❌ Shape properties:
  - ❌ `shape_transform` → `geom_pos`, `geom_quat` (dynamic)
  - ❌ `shape_collision_radius` → `geom_rbound`
  - ❌ `shape_material_mu` → `geom_friction`
  - ❌ `shape_material_ke/kd` → `geom_solref`
  - ❌ `shape_scale` → `geom_size`

- ❌ Actuator properties:
  - ❌ `joint_target_ke/kd` → `actuator_gainprm`, `actuator_biasprm`
  - ❌ `joint_effort_limit` → `actuator_forcerange`

#### MuJoCo → Newton Updates

##### Contact Information
- ❌ MuJoCo contacts → Newton contact format (if using MuJoCo collision detection)

## Key Missing Systems

### 1. Actuator System
- Need to create actuators during initialization
- Map DOFs to actuator indices
- Support different control modes (FORCE, TARGET_POSITION, TARGET_VELOCITY)
- Update actuator parameters dynamically

### 2. Dynamic Updates
- Implement `notify_model_changed()` method
- Track which properties have changed
- Update only changed properties for efficiency

### 3. Control Application
- Implement proper control force application
- Handle joint forces in different joint types
- Apply body forces with proper coordinate transforms

### 4. Contact Handling
- Option to use MuJoCo or Newton collision detection
- Convert between contact formats if needed

### 5. Multi-DOF Joints
- Handle complex joints (UNIVERSAL, D6)
- Proper DOF indexing and mapping

## Implementation Priority

### High Priority (Required for basic physics)
1. Body mass, COM, and inertia initialization
2. Joint armature and friction
3. Control force application
4. Basic actuator system

### Medium Priority (Required for accurate simulation)
1. Shape contact properties (friction, stiffness)
2. Dynamic property updates
3. Collision filtering
4. Proper solver parameters

### Low Priority (Nice to have)
1. Mesh support
2. Equality constraints
3. Advanced actuator features
4. Visualization improvements
