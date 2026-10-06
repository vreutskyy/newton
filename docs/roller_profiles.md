# Experimental roller profiles

This prototype is based on the combined experimental branch, not the tendon PR.
It retains `TendonLinkType` and `ModelBuilder.add_tendon_link`; it does not introduce
the PR's guide naming or complete-route builder API.

Circular rollers remain the default: existing `radius=...` calls are unchanged.
Explicit immutable profiles are available from `newton.geometry`:

- `RollerProfileCircle(radius)` uses analytical circle support, tangent, and arc queries.
- `RollerProfileEllipse(a, b)` takes the two semiaxes.
- `RollerProfileSector(radius, angle)` is an exact convex circular sector, with
  two radial edges and a circular arc. The angle is in radians, in `(0, pi]`.

Each shape has only its own parameters. `RollerProfile` is the common base;
the packed Warp representation is private. The sector apex is at the profile
origin, and its symmetry direction is local +X. `profile_axis` specifies this
direction in body coordinates; `axis` specifies the plane normal.

```python
builder.add_tendon_link(
    body=roller,
    link_type=newton.TendonLinkType.ROLLING,
    profile=newton.geometry.RollerProfileSector(0.08, 2.0),
    axis=(0.0, 1.0, 0.0),
    profile_axis=(1.0, 0.0, 0.0),
    compliance=1.0e-5,
    mu=0.3,
)
```

Explicit profiles select planar profile routing for the whole model in XPBD
and VBD. Circles (either `radius=...` or `RollerProfileCircle`) may be dynamic,
including between fixed ellipse/sector neighbors. Ellipses and sectors retain
prescribed routing: `dynamic=True` is rejected for these shapes. The existing
restriction against consecutive dynamic rollers remains. Activation uses the
neighbors' common supporting tangent and the same radius-relative
`tendon_activation_tol` hysteresis as ordinary circles. Segment split/merge
conserves free-plus-wrapped material and carries the fixed neighbors' boundary
contact history across changing segment slots.

Terminal rollers and autodiff are rejected in profile mode.
Builder merging/replication and fixed-joint collapse with explicit
profiles are also rejected until their profile bookkeeping is implemented.
The supported tangent turn is 0–180 degrees; multiple wraps and 3D routing are
not implemented. Wrapped boundary length remains inextensible, as in the
existing reduced-order tendon model. No polygon approximation is used.
This is an experimental builder/runtime API; USD import/export support is not
included.

The solver retains the boundary coordinates returned by the tangent query
instead of recovering them repeatedly from transformed contact points.
Common tangents use safeguarded Newton iterations with bisection inside the
bracket; boundary coordinates are evaluated only at the final tangent. The
full wrapped length is diagnostic and is integrated only at the accepted pose,
while incremental boundary travel and friction are evaluated every iteration.
## Common mechanics

All rollers, including radius-authored circles, now use physical contact forces
and their full moments about the body COM. There is no circle-specific spin
removal, friction-scaled torque restoration, or `remove_roller_spin` switch.
A future profile supplies tangent/contact geometry and boundary travel; it does
not supply a new force/torque formulation. This applies within the planar routing
and wrap restrictions above, not to arbitrary unsupported topology.

The complete no-slip material trial is assembled before finite-friction capstan
projection. XPBD then groups spans whose current tension is on a sliding capstan
bound. Their tension ratios are fixed while summing their material equations
eliminates internal rest-length transfer. One scalar impulse amplitude is solved
for each group, including cross terms from shared bodies and opposing moments.
Sticking spans use the same kernel as single-row groups. The actual accumulated
impulses and body corrections are updated, not just reported tension values.

The previous independent segment solves could give different impulses on the
two sides of a frictionless circle even after the material solve equalized their
tensions. Their moments canceled only after many outer iterations. The coupled
solve accounts for this cancellation inside its effective mass instead. Ellipse
and sector moments do not generally cancel, even at zero friction, and are kept.

The reduction uses one row per segment and no dense matrix, but currently costs
O(m²) for a sliding group of m spans. Active capstan bounds are recognized within
the material-settling tolerance; depleted spans are not condensed together. A
singular or non-monotone scalar block emits an error and is not solved. General
long-route scalability and these degenerate cases still need validation.

This revises the split stretch/slip numerical pipeline recorded in
`cable_joints_slip_plan.md`: the capstan law is retained, but its active bounds
are used for condensation instead of adding a separate roller-spin impulse.
It remains an experimental numerical change, not an upstream compatibility claim.

XPBD updates contact history, material, and constitutive tension once more at
the accepted pose after its final correction. Geometry and material balance are
refreshed before each VBD body color for every roller shape. Otherwise a preceding
body update can invalidate capstan balance and create artificial spin torque.
Shape-specific dispatch remains in geometry queries, not in the mechanics.

VBD also eliminates freely sliding material coordinates from its axial
Gauss-Newton curvature. Summing independent span Hessians incorrectly gave a
frictionless circle spin stiffness, even when its physical torque was zero;
translation/rotation coupling in the body update then produced angular drift.
Series compliance and the combined path gradient preserve that nullspace for
any profile. Finite-friction boundaries retain the conservative sticking-side
tangent, because reverse motion can stick rather than continue sliding. This
affects the curvature approximation, not the projected forces or capstan law.
Body coloring separates all bodies in each material-connected route section;
internal attachments separate sections. Custom coloring is validated against
that coupling. Long routes may consequently require more body colors.

If the rolling trial exhausts a free span, missing material is transferred from
other free spans in the same section before capstan projection. Independently
clipping each rest length would create cable. This bound correction does not
renormalize the initial total or borrow across attachments. If the section has
insufficient free material altogether, the route remains unsupported and emits
a warning about the length added by the unavoidable minimum-rest clamp.

## Controlled example

Run `uv run -m newton.examples roller_profiles`. Use `--friction 0` to compare
frictionless motion, or `--friction 0.3` for the default. The cases are circle, ellipse,
and sector, from left to right. A green anchor moves vertically, while a red
dynamic slider is pulled downward by 3 N (ramped on at startup). The roller
rotation is not prescribed. Each roller has inertia `2e-4 kg·m²`, and each span
has compliance `1e-5 m/N`. Explicit viscous bearing drag is `0.005 N·m·s/rad`;
slider drag is `0.5 N·s/m`. This does not rely on unsupported passive joint
damping. The display includes an orientation spoke and span colors from blue
(0 N) to red (6 N or greater).

CUDA graph capture is enabled by default. The drive clock lives on the device,
so replay advances the moving anchor and load ramp. Odd substep counts use two
graphs to alternate state buffers correctly. CPU execution and
`--disable-cuda-graph` use the same uncaptured simulation loop.

The interactive default is 10 steps per frame and 32 iterations for either solver.
Select `--solver xpbd` for XPBD; the default solver is VBD.
Override with `--substeps` and `--iterations`.
These are settings for this example, not general-purpose solver recommendations.
The XPBD example explicitly uses `joint_linear_relaxation=0.5`: the stronger
coupled cable corrections interact with the separately solved hinges. At the
global default of 0.7, the frictional sector briefly displaced its hinge by
19.9 micrometers at 32 iterations and 18.3 micrometers at 256, exceeding the
unchanged 10-micrometer check. More iterations alone did not remove that
stick/slip transient. The solver's global relaxation default remains unchanged.

Before the coupled solve, XPBD at 10 × 32 gave the frictionless circle an
artificial angular velocity of 1.33 rad/s after 0.2 s. Even 128 iterations
exceeded the example's full-cycle angular-drift check; 256 passed. That was a
failure of independent contact-impulse convergence, not a requirement of the
roller geometry itself. The coupled solve passes at 32 iterations; the precise
validation settings and results are recorded below.

In the preceding VBD-only validation, on an RTX 3080 Laptop GPU, an eight-second,
two-cycle comparison measured about 44 ms/frame at 10 × 32, excluding initialization and
rendering. Against 10 × 128 at the same timestep:

| Cable friction | Peak ellipse / sector angle difference | Segment-tension RMS difference |
| --- | ---: | ---: |
| 0 | 1.23° / 0.61° | 0.015 N |
| 0.3 | 0.0015° / 0.0008° | 0.012 N |

The frictionless circle's maximum angle drift was 0.13°, instead of continuous
rotation. Free-plus-wrapped material drift stayed below 1.7 micrometers. These
comparisons measure iteration convergence, not timestep or physical accuracy.
At 10 × 8, the frictionless sector's initial flip differed by as much as 36°
from the tightly iterated run despite a similar final pose, so those earlier
fast settings are no longer the default. Historical comparisons with the old
once-per-iteration material update do not validate the corrected physics.

The geometry tests compare circle tangents with analytical references, ellipse
arc lengths with numerical reference integration, and profile torques with
finite differences of total path length. VBD tests check force/torque response,
finite-friction material transfer, and free-plus-wrapped material conservation.
Additional tests compare non-circular VBD forces and torques with independent
float64 routed-length derivatives. A moving-endpoint regression verifies that
a frictionless circle remains stationary, with and without tendon damping,
while finite cable friction still produces rotation. This regression fails
with the old once-per-iteration material update.
Graph tests compare captured and uncaptured motion, material state, drive time,
and odd/even state-buffer handling. Tangent tests include 96 randomized cases
against independent double-precision bisection on both CPU and CUDA.

The follow-up validation includes neighboring ellipse/sector rollers, shared-body
routes with offset COMs, both winding directions, and material conservation over
full profile rotations. XPBD tests check physical endpoint impulses (including
damping and friction), small-step accelerations, accepted-pose material history,
and graph replay. Two zero-wrap regressions cover roundoff turning coincident
contacts into a full perimeter, and inconsistent sector-face support ties on
CUDA. Flat radial edges retain their boundary length; they are not treated as
curved arcs or indiscriminately discarded at zero tangent turn.

## Validation status

The dated entries below retain earlier development checkpoints, including failures
that were subsequently fixed. The latest results and remaining limitations are in
[Closure follow-up](#closure-follow-up-october-5).

Run the focused tests from the repository root:

```sh
uv run --extra dev -m unittest newton.tests.test_roller_profile newton.tests.test_roller_profile_vbd
uv run --extra dev -m unittest newton.tests.test_roller_profile_routing newton.tests.test_roller_profile_xpbd
uv run --extra dev -m unittest newton.tests.test_tendon_stretch_blocks
```

On October 5, 2026, the final five-module run completed 57 tests: 55 passed and
two CPU graph tests were skipped. This includes nonlinear/damped loads and the
frictional-sector transition. Independent dense float64 algebra checks cover
sliding and sticking groups, unequal initial impulses, damping, and repeated
bodies. Explicit-circle and radius-authored routes are compared in both solvers.
The zero-wrap tests failed before their respective fixes, including the
sector-edge failure reproduced specifically on CUDA.

The old isolated same-body-span test expected torque from the removed spin
correction. It now verifies cancellation of the two internal contact forces,
and separately verifies nonzero torque from the complete taut route. The old
beta-scaled-slip unit assertion is replaced by physical capstan-ratio and impulse
checks. These are corrected physical expectations and removed implementation-
detail assumptions, not tolerance relaxations.

Final four-second CUDA cycles use 10 substeps, relaxation 0.5, and otherwise
identical settings at 32 and 256 XPBD iterations:

| Friction | Peak circle / ellipse / sector angle difference | Maximum hinge displacement at 32 iterations |
| --- | --- | --- |
| 0 | <0.000001° / 0.0449° / 0.0211° | 2.81 micrometers |
| 0.3 | 0.00445° / 0.00624° / 0.01136° | 4.24 micrometers |

All per-frame checks passed. The frictionless circle's angular velocity remained
zero at stored precision, and material drift stayed below 1.2 micrometers in all
four runs. Disabling only cross-span coupling, with the same 32 iterations and
0.5 relaxation, reproduced artificial circle spin of -2.594 rad/s after 0.2 s.
Thus lowering relaxation alone does not explain the cancellation improvement.
These are convergence/consistency checks, not a wall-clock performance claim.

Six additional CUDA legacy tests passed: XPBD and VBD friction-controlled dynamic
and kinematic capstans, the XPBD dynamic-route neighbor matrix, and VBD depleted-
span rolling transfer. Some of those historical fixtures emit unsupported-wrap
warnings; their passing assertions do not certify those invalid configurations.

On October 2, 2026, the first two modules ran 33 tests: 32 passed and one CPU graph test
was skipped because graph capture requires CUDA. The moving-endpoint regression
also failed with the previous material-update ordering restored, confirming
that it detects the artificial frictionless spin.

A separate 24-test legacy tendon subset had 20 passes, two CPU graph skips,
and two failures: `test_vbd_rolling_chain_tension_matches_accepted_pose` on CPU
and CUDA. Its tension spread was 0.02384 N against a 0.005 N tolerance, with the
same result when the pre-fix update ordering was restored. This remains an
open validation issue; the full legacy suite has not been certified by this
prototype. The branch is for technical review, not merge-ready integration.

On October 5, a 32-test legacy subset covering same-body/off-center mechanics,
zero compliance, diagnostics, sigmoid projection, angular Jacobians, accepted-pose
activation, gravity, and rolling chains had 30 passes and the same two failures
above, with the identical 0.023841858 N spread on CPU and CUDA. No other failures
were found in that subset; this is not a full-suite result.

The subsequent combined `test_tendon_capstan` / `test_tendon_vbd` full run was
stopped after approximately 30 minutes without a completion summary. It is an
incomplete validation run, not a passing full-suite result. The follow-up below
supersedes the earlier open rolling-chain and long-block checks, but does not
claim full-suite clearance.

### Regression follow-up, October 5

The expanded profile/block tests, generic XPBD tests, and two new rolling
regressions completed **137 tests: 135 passed, two CPU graph tests skipped**.
The dense reference now includes 1, 8, 32, and 128 spans, repeated bodies,
sticking boundaries, and depleted spans. A separate finite-friction Atwood
reference checks both acceleration and impulse-derived tension analytically.
Singular/non-monotone blocks remain explicitly unsupported; these tests do not
validate every possible conditioning or route configuration.

Two production bugs were reproduced with failing regressions before fixing them:

- The linear material projection discarded the receiving span's negative
  stretch when solving a capstan bound. It now retains that slack in the root,
  instead of leaving a false tension that decays only over repeated sweeps.
  An unloaded spinning roller cannot maintain tension against a slack tail
  under a finite capstan ratio.
- VBD reconstructed tension from rounded `length - rest` after material
  projection. Different span-length scales then reintroduced unequal forces
  and false circle torque, increasing with stiffness. VBD now uses the projected
  stretch and damping when assembling a body update. The stationary-circle
  regression covers compliance from `1e-4` through `1e-8`.

The VBD balanced-compound-pulley CPU regression and frictionless motor CUDA
regression pass after these corrections. Targeted nonlinear projection,
material-tangent, and accepted-pose checks also pass. The rolling-chain fixture
was corrected to positive wraps (its negative wraps occurred at the committed
baseline too), compliance `1e-4`, and settle tolerance `1e-5`. Its original
0.005 N spread tolerance is unchanged. At the old compliance `1e-5`, one
rest-length ULP alone represented about 0.012 N.

Other fixture changes are explicit: the dynamic capstan demonstration uses
middle friction 0.05 instead of 0.4, which already locked in the revised
mechanics; the pinhole friction comparison stops before the rising weight
reaches the guide; the switch penetration check includes the configured
activation gap; the sigmoid diagnostic checks the accepted pose; and prescribed
no-slip transfer tests are pre-tensioned so both spans remain taut.

**The complete CUDA capstan module is not green: 50 of 55 tests passed.**
The remaining tests are unchanged and visible:

- `motorized_pulley_drives_slider`
- `motorized_pulley_couples_without_delay`
- `motorized_pulley_updates_rest_in_first_step`
- `rolling_transfer_saturates_at_zero_span`
- `moving_rolling_route_conserves_material` (its required clamp-coverage assertion,
  not the conservation assertion)

The first four expect an initially unloaded motor-driven roller to transport
material without maintaining a taut return span. The last deliberately creates
one short stretched span and one slack span and expects it to stay near the
minimum rest length; projection now redistributes that inconsistent initial
state. These fixtures need a physically admissible loaded-drive/depleted-span
replacement, not looser numerical tolerances. The corresponding VBD motor-drive
check also fails. A diagnostic loaded-return carriage moves under friction and
stays stationary without friction, but it has not replaced those tests.

Full eight-second CUDA showcase cycles at 10 x 32 pass every frame in XPBD for
both friction 0 and 0.3. Maximum material drift is respectively 1.02 and 0.84
micrometers; the frictionless circle's angular speed is zero at stored precision.
The VBD frictional cycle also passes, but **the frictionless VBD cycle fails**
its unchanged angle tolerance at frame 401 (6.683 s). Peak circle angle is
0.00572 rad (0.328 degrees), versus a roughly 0.005 rad limit.

An ablation that reconstructs stretch from rounded lengths again lowers that
VBD drift to 0.00269 rad, while reintroducing much larger spurious instantaneous
torques. That does not justify restoring the rounding bug: force accuracy and
long-run integration drift are separate checks. The cause of the residual drift
is not yet established. No iteration, damping, or assertion threshold was tuned
to hide it. The earlier full-cycle VBD results do not clear the revised code.

The superseded combined CPU/CUDA legacy run was stopped after the new fixes,
because it was still executing its old imported code. Its partial results are
not a completed suite result. Formatting, lint, and `git diff --check` pass.
At that checkpoint the prototype was uncommitted and not fully regression-cleared.

### Closure follow-up, October 5

The earlier open items above were investigated further:

- The VBD drift came from false spin curvature, not a missing physical torque.
  A diagnostic that removed only that curvature reduced the eight-second circle
  angle drift from 0.328 degrees to about 0.00003 degrees. The production fix
  uses shape-independent material condensation, not a circle-axis special case.
  Independent dense Schur-complement tests cover 1, 2, 8, and 32 spans, damping,
  repeated bodies, finite-friction boundaries, and depleted spans. The centered
  circle regression failed before the curvature fix.
- Body coloring now includes material coupling between non-neighboring links.
  A four-body pinhole route reproduced insufficient coloring before the fix;
  automatic coloring now separates it and VBD rejects insufficient custom colors.
- **Physical expectation corrected:** unloaded motor tests now assert no
  traction against a slack tail. They were renamed to state that expectation.
  **New invariant added:** a separate loaded-drive fixture uses two equal hanging
  masses and checks every step's signed material transfer against `R * theta`,
  and body travel against the same independent reference, in both solvers.
  The VBD unloaded check allows 0.5 mm residual drift over the complete motor run,
  accounting for finite-iteration hinge motion and float32 stretch; it no longer
  expects the physically unjustified 0.2 m drive motion.
- **New invariant added:** depletion fixtures keep both spans taut and explicitly
  check conservation, including radius-authored and explicit-profile circles.
  This exposed a real production error: independently clamping a negative trial
  rest length created 1.001 mm of cable in the first failing step. The conservative
  bound projection fixes it. A separate regression verifies that it cannot borrow
  through an internal attachment. The existing moving-wrap test now exercises
  depletion while conserving free-plus-wrapped material.
- **Physical expectation corrected:** the legacy spin/damping check assumed
  that roller-axis motion had been removed from the contact Jacobian. With full
  physical contact moments, its damping rate must use the same material-contact
  motion. A separate float64 finite-pose reference now checks that rate while
  verifying that the circle's geometric tangent lengths remain unchanged.
  The same-body test checks both slack relief without motion and torque from
  the complete taut route; it no longer requires an unfinished material solve
  to retain tension in a slack cable.

The final focused run completed **138 tests: 136 passed and two CPU graph tests
skipped**. The generic VBD module completed **18 tests, all passed**. Additional
CPU/CUDA explicit-profile depletion, internal-attachment, loaded-drive, coloring,
and inactive-route-gap curvature tests passed.

All four complete eight-second CUDA showcase runs pass every frame at the
unchanged **10 substeps and 32 iterations**:

| Solver | Friction | Maximum material drift | Peak circle angular speed |
| --- | --- | --- | --- |
| XPBD | 0 | 1.01 micrometers | 0 rad/s |
| XPBD | 0.3 | 0.83 micrometers | 1.032 rad/s |
| VBD | 0 | 1.55 micrometers | 5.43e-6 rad/s |
| VBD | 0.3 | 1.73 micrometers | 1.032 rad/s |

The nonzero circle speed in the frictional runs is intended physical motion,
not drift. The earlier frictionless VBD failure is resolved without changing
its angle tolerance, iteration count, or damping.

The complete legacy run finished: **207 tests, 11 failures, 13 skips, and two
unexpected successes** in the raw report. Five failures were the obsolete
spin/damping, slack-tension, and first-step rest-transfer expectations discussed
above; their corrected CPU/CUDA checks pass separately. The two unexpected
successes were stale gear-pulley expected-failure markers, now removed.
This is not a fresh all-green 207-test result.

The other six failures were long-running VBD examples. All six independently
fail at the committed baseline `06ec530` too. Further investigation distinguished
the following cases:

- The cable machine enters a negative wrap at its third roller. In the updated
  run, free-plus-wrapped material stays within about 10 micrometers of its initial
  4.169868 m through frame 50, while wraps are valid. The third wrap is negative
  by frame 60; the existing length assertion fails at frame 94. The baseline
  also fails the same length assertion. Its geometry and tolerance are unchanged.
- The dynamic-capstan example lets a light weight pass its pulley. The baseline
  fails the final crest-height assertion; the updated run fails the earlier
  near-side assertion. The frictionless case is unaffected by the change to the
  middle case's friction coefficient. The current run first exceeds a half wrap
  at frame 40, before the side assertion fails at frame 84.
- The kinematic-capstan example enters unsupported wraps and fails its 6% length
  check in both versions: 6.24% at baseline and 6.35% in the updated run. The first
  unsupported wrap is at frame 30; the length assertion fails at frame 62.
- The rolling-pulley example exceeds a half wrap at frame 25, long before its
  attachment crosses the pulley crest at frame 143. The baseline also fails its
  crest assertion. Collision does not guarantee that an unconstrained weight
  stays on one side of a roller or that its cable wrap stays below pi.
- The original equilibrium example failed its 5 cm vertical-drift assertion:
  7.20 cm at baseline and 12.94 cm in the updated run. Reversing body-color order
  reversed which weight fell. At 80 and 320 iterations the left drift was about
  7.28 and 7.03 cm, but those runs entered unsupported wraps and are not valid
  accuracy references. Alternating the order restored symmetry, yet both weights
  swung inward and crossed a half wrap by frame 40. That diagnostic remains
  outside the production solver.

  **Fixture corrected, not a solver-convergence fix:** both the example and its
  dedicated Atwood fixture now constrain the weights to vertical motion. Their
  inclined cables require horizontal guide reactions and attachment-torque
  reactions for the claimed static equilibrium. With the guides, 20-iteration
  VBD settles by about 0.15 mm and stops drifting, in either color order. The
  vertical-drift limits were tightened from 5 cm to 2 mm, and asymmetry from
  2 cm to 1 mm. All five XPBD CPU/CUDA, VBD CPU/CUDA, and full-duration VBD
  example checks passed. This does not claim to fix the original free-swinging
  system's finite-iteration order sensitivity.
  The corrected example also passes all 120 frames with its unchanged XPBD
  configuration (16 substeps, 8 iterations) and the tightened assertions.
- The XY-table prefix has a separate rotation-check issue: the lower-guide
  mirror error is 0.10525 rad at 30 iterations and 0.09647 rad at 120, against a
  0.035 rad limit. The baseline fails an earlier all-guides-must-rotate assertion.
  Increasing iterations alone does not clear it. Captured and uncaptured
  30-iteration runs reproduce the same positions and error with the same
  time-dependent drive targets.

  **Measurement corrected:** its recorder previously discarded the first
  frame's pulley rotations. Initializing the reference from the actual initial
  pose fixes that; CPU/CUDA regressions fail before and pass after. Rerunning
  the same 30-frame trajectory gives a 0.07940 rad mirror error, still above
  0.035 rad. No force or pose changed. Early samples have slack spans despite
  the no-slip kinematic assertions. The recorded per-frame table path remains
  inside the existing positional bounds.

  **Phase assertion corrected:** the original final check simultaneously
  required all guides to rotate and the lower guides to stay nearly stationary
  during pure X travel. It now requires rotation of the X-path guides in that
  prefix and retains the separate lower-guide rotation check once the Y phase
  has run. CPU/CUDA tests with synthetic traces fail before this correction and
  pass after it, including checking that stationary lower guides still fail in
  the Y phase. The 0.035 rad mirror limit and all trajectory limits are unchanged.

  **Loading investigated, not changed in the example:** a diagnostic with
  uniform 20 N pretension keeps every span taut. An independent float64 model
  reduces the mechanism to nine coordinates (table X/Y and seven pulley angles)
  with exact joints, full material-contact gradients, cable stiffness/damping,
  and the same time-dependent velocity drives. This is a sticking-branch
  reference, not a reference for slack or slipping motion. At 120 VBD iterations,
  maximum table-position error is 7.91 micrometers and maximum pulley-angle error
  is 0.00060 rad over the 30-frame prefix. Monitoring every simulation step
  confirms at least 9.77 N total tension and at most 7.49 nm material transfer
  beyond the sticking rolling trial. The reference itself has a nonzero transient
  lower-guide mirror error (0.01713 rad); VBD gives 0.01740 rad at 120 iterations
  and 0.03152 rad at 30, both below the existing 0.035 rad limit.

  With the original weak pretension and unchanged 30-iteration configuration,
  the same monitor instead records zero tension, up to 0.953 mm slack extension,
  and 64.8 micrometers of material transfer beyond the sticking trial in one
  step. Thus the original setup does not maintain the assumed no-slip branch.
  This explains why a taut-cable kinematic check is not an adequate accuracy
  reference for that run; it does not independently validate all of its slack
  dynamics. The original loading and its failing mirror assertion remain
  unchanged pending a decision about the example's intended operating regime.

**Diagnostic corrected:** the wrap warning rejected atan2's negative-pi
representation of an exact half wrap, and tiny roundoff below zero. It now
allows 1e-6 radians of endpoint roundoff without changing geometry, forces, or
material transfer. CPU/CUDA boundary-classification tests fail with the old
predicate and pass with the correction, including genuinely invalid angles near
both ends. This warning-only correction does not fix XY-table motion.

A temporary cable-machine ablation with vertical guides passes its original
100-frame assertions. It has not been applied to that intentionally free-swinging
example; changing the other examples' physical setup needs an explicit decision.

The remaining legacy failures stay visible rather than being hidden with new
skips, expected-failure markers, or relaxed tolerances.
Both previously expected CUDA gear-pulley failures now pass their original
assertions; their stale expected-failure markers were removed. The direction
test also passed a separate rerun on the final code.

### Guided legacy examples and damped-force consistency

The follow-up now includes the approved changes to the legacy examples' physical
setups. The cable machine, rolling pulley, dynamic capstan, and kinematic capstan
have vertical weight guides and finite travel stops. This keeps their circular
wraps inside the supported range instead of relying on collision to prevent
weights swinging over a roller. The rolling-pulley weights also start at
horizontal offsets of +/-0.30 m, so the guided motion still reaches the contact
neighborhood checked by the example. Body masses are unchanged.

The examples now assert valid oriented wraps and conservation of free-plus-wrapped
material separately from geometric path length. A stop may legitimately leave a
slack cable: the kinematic-capstan geometric check permits shortening, while its
material check still has a 1 mm absolute tolerance and no relative tolerance.
Friction comparisons and smooth-slip checks use the first half second, before
stop impacts; their numerical thresholds are unchanged. The cable-machine motion
check uses peak excursion rather than final displacement because a weight can
return toward its starting point after hitting its stop.

This investigation also exposed a solver inconsistency, not just invalid example
geometry. The linear material projection uses total unilateral Kelvin-Voigt
tension, `max(stretch/compliance + damping_tension, 0)`. Positive damping can
therefore support positive total tension while elastic stretch is nonpositive.
The VBD force preparation incorrectly discarded such spans. It now retains the
total force and tangent, and includes those spans in frictionless material-mode
condensation. XPBD's sliding-block classification now uses the same signed
elastic residual instead of adding damping to an independently clamped spring
force. This does not change the exposed nonnegative elastic-tension diagnostic.

CPU and CUDA force-preparation regressions failed before these corrections and
pass afterward. They cover positive/zero/negative stretch, damping of both signs,
slack total force, and joining equal-tension spans when one has negative elastic
stretch. A separate implicit Atwood reference checks actual mass accelerations
with damping and friction coefficients 0, 0.1, and 0.2 in both solvers. This is an
independent dynamics check, not a comparison between reconstructed diagnostics.
The combined targeted check completed 14 tests, all passed.

The XY-table example now initializes a uniform 20 N pretension by shortening each
free span by its compliance times the requested tension; wrapped length is not
scaled. CPU/CUDA tests check the resulting tension, material budget, and
idempotence. Its no-slip reference needs a taut cable. Iteration-count and
full-trajectory verification are recorded separately below; prefix success is
not evidence that all later motion phases pass.

The former slack-damping test assumed that geometric slack always disables the
damper. That is not the force law used by XPBD or the material projection: for
its original fast-extension input, the implicit signed spring-plus-damper
equation predicts positive force even though the accepted geometric length is
below rest length. The revised test checks that independent equation in both
solvers at four velocities, including inactive and damping-dominated cases. All
CPU/CUDA cases pass. This explicitly preserves the existing Kelvin-Voigt model;
it does not claim that model is equivalent to a damper disabled whenever a cable
is geometrically slack. Such a law would require a separate, consistent change
to material transfer and both body solvers.

The longer XY diagnostic also isolated a **joint-coordinate issue without any
tendon**: XPBD's velocity drive about a swing coordinate loses effectiveness
near a half turn. In a one-step, initially stationary drive test, the original
world-Z/local-Z representation gives zero angular response near pi, while the
implicit reference gives 2.97265 rad/s. Expressing that same world-Z hinge axis
as local-X twist gives the reference response on either side of pi. The example
now uses this equivalent joint-frame representation for its pulleys. CPU/CUDA
regressions copy its actual drive into a fresh cable-free model and fail with
the old frames and pass with the new frames. The general XPBD swing-coordinate
drive implementation is not changed by this branch.

The final profile/geometry and generic XPBD/VBD regression selection completed
**162 tests: 160 passed, two CPU graph tests skipped**. All four complete
eight-second profile cycles pass again after the damped-force correction, at
the same 10 substeps and 32 iterations. Material drift and circle speeds agree
with the table above. These are correctness runs, not performance measurements.

All eight full guided-demo runs pass their per-frame and final assertions:
cable machine (100 frames), rolling pulley (180), dynamic capstan (100), and
kinematic capstan (100), each in XPBD and VBD. Both separate CPU dynamic-capstan
friction-ordering and rotation-transient tests pass. The affected legacy
nonlinear, damping, diagnostic, and graph-capture selection completed another
26 tests: 24 passed and two CPU graph tests skipped.

The XY example uses 32 substeps, 64 XPBD iterations, and linear relaxation 0.5.
Its full 600-frame/10-second trajectory now passes every original numerical
trajectory bound: RMS position error is 2.804 mm and maximum error is 5.240 mm.
The original joint frames failed the same run even at 64 iterations, with
24.80 mm RMS and 98.15 mm maximum error; the late failure was not resolved by
increasing iterations alone. The equivalent twist frames are therefore part
of the example correction, not a general solver-convergence claim. Earlier
20-iteration and stronger-drive diagnostics are retained as failed comparisons.

The final VBD XY prefix passes all assertions for 40 frames at 32 substeps and
64 iterations, including the unchanged 0.01 rad lower-guide and 0.035 rad mirror
limits. Its position error is 2.316 mm RMS and 3.501 mm maximum.

The subsequent full 600-frame/ten-second VBD run uses the same final frames,
32 substeps, and 64 iterations. It completes every frame and passes the
trajectory, symmetry, rotation, and velocity-sign checks: position error is
2.624 mm RMS and 5.040 mm maximum. It originally failed the raw acceleration
assertion: 285.505 rad/s² versus the existing 250 rad/s² bound.

Saved-trace analysis locates both bound exceedances at commanded velocity
changes, at 6.75 and 7.0 seconds. Perfect tracking of the piecewise-linear angle
reference itself produces up to 294.852 rad/s² under the test's frame-based
second difference, so it would also fail this assertion. In the same validation
window, VBD's maximum acceleration error relative to that reference is
9.408 rad/s². The saved XPBD trace has 241.400 rad/s² raw acceleration and
63.001 rad/s² acceleration error: its smoother response passes the raw bound.
These measurements identify a flaw in the vibration criterion, not evidence
that this VBD spike is uncommanded oscillation. The assertion now checks
acceleration error relative to the commanded trajectory, using identical
finite differences and retaining the 250 rad/s² bound. The reference uses the
same target scale and hold time as the drive commands.

The new CPU/CUDA regression first failed with the original assertion. It now
accepts perfect tracking at both 60 and 120 Hz and at target scales 1.0 and
0.5, while rejecting injected positive and negative 400 rad/s² spikes. The
actual corrected vibration-check block also passes when replayed on both saved
600-frame traces, and rejects an injected spike in each. This follow-up reuses
the full simulation traces; it does not rerun those long simulations or
reconstruct unrecorded state.

The follow-up XY-table and guided-validation selection passes **14 tests:
13 passed and one expected CPU skip**, including the existing VBD simulation
prefix. All pre-commit hooks pass after the assertion correction.

Small departures from nominal half wraps caused by finite joint error can
still trigger the existing circular-wrap diagnostic. Legacy tests also still
print unsupported-wrap diagnostics; passing their assertions does not validate
those unsupported states. The coordinate-frame change does not extend the
legacy circular wrap range.

No newly skipped tests, expected-failure markers, or loosened numerical bounds
were used to hide a failure. Physical setup changes, measurement corrections,
force-law expectations, and numerical configuration changes are listed above
explicitly. The complete earlier 207-test legacy selection was not rerun as one
suite at the time of the targeted results above. The final follow-up now runs
all three massless-tendon regression modules (`test_tendon_capstan`,
`test_tendon_equilibrium`, and `test_tendon_vbd`) together: **223 tests, 210
passed and 13 expected CPU skips, with no failures or unexpected successes**.
All nine registered full VBD example-assertion cases pass. That suite run
predates the acceleration-check correction and the two new regression cases;
the registered XY simulation test covers its prefix only. Full-duration
vibration-check verification is the saved-trace replay described above.

`uvx pre-commit run -a` passes, as does an explicit hook run over the three new
untracked regression files. Ruff checks, Ruff format checks, and
`git diff --check` pass. The acceleration-check follow-up changes example
validation and adds a regression, not solver behavior. These checks cover the
XPBD/VBD follow-up on `2976-roller-profiles`.

## Dynamic circular profiles — 2026-10-06

Circular activation now works in mixed profile models in both solvers. The
switching circle uses the actual bypass tangent between its fixed neighbors,
not circle approximations to an ellipse or sector. Existing isolated-circle
segment split/merge rules are retained. Persistent neighboring profiles carry
their old boundary coordinates to the new segment slot; a newly active circle
gets its initial wrap from the split instead of a stale rolling delta.
Inactive placeholder segments are excluded from the initial total material.

No dynamic ellipses/sectors, consecutive dynamic rollers, multi-turn wrapping,
or nonplanar routing are added. Dynamic means contact activation, not whether
the roller's rigid body can translate or rotate. No extra persistent arrays or
runtime host readbacks are introduced.

`test_roller_profile_dynamic` covers both solvers and devices: reflected routes,
ellipse/sector neighbors in both orders, active/inactive initial states,
radius-only/explicit-circle authoring, moving neighboring profiles, repeated
switches, multiple independently switching tendons, radius-relative hysteresis,
and graph replay with a moving loaded endpoint. Total material is checked
against independent double-precision tangent/arc integration. Loaded circular
routes also match the existing radius-only implementation. Non-circular dynamic
profiles are explicitly rejected. The initial mixed-profile regressions failed
against the previous builder restrictions before implementing this extension.

The final focused selection runs all five profile modules, the equilibrium
module, and relevant legacy route, material, pinhole, and shared-body force
checks: **142 tests, 138 passed, 4 expected CPU graph skips**. The broad legacy
demo rerun was stopped before completion after checking its previous runtime
(over an hour); it is not claimed as a completed current validation.

Performance evidence and reproducible benchmark/test scripts are saved outside
the repository in the local `roller-profiles-20261006` artifact directory.
The reference is commit `6bcd973e`, immediately before this dynamic-circle
extension, not the original VBD-only prototype. Benchmarks use CUDA graphs on
the RTX 3080 Laptop GPU, no rendering, no per-step readbacks, and exclude setup,
compilation, and graph capture. Both variants use the same frictional profile
example at 10 substeps and 32 iterations, with 30 warm-up frames followed by
four batches of 120 frames. Final poses and rest lengths match the reference
exactly for both solvers.

An initial implementation added 4–5% to this fixed-route workload. The final
implementation bypasses unneeded activation/history checks when a model has
no dynamic links. One intermediate run recovered the reference XPBD timing,
but the final back-to-back repeat still measured a small regression:

| Solver | Reference ms/frame | Updated ms/frame | Change |
|---|---:|---:|---:|
| XPBD | 26.97 | 27.88 | +3.4% |
| VBD | 64.29 | 66.57 | +3.5% |

These are timings for this example and laptop, not a general scalability claim.
No further optimization was made to chase the remaining few percent.
A separate taut ellipse–circle–sector microbenchmark
(same geometry, material, and tension; circle fixed versus dynamic but active)
measured 23.57 versus 24.15 ms per 10 steps in XPBD (+2.5%) and 84.97 versus
85.97 ms in VBD (+1.2%). This measures switching bookkeeping on a stable route,
not the relative cost of different switching trajectories or long tendons.

### Known material-sweep artifact at activation

A loaded ellipse–dynamic-circle example exposed a material-sweep artifact at
finite friction. When the circle reactivates, the sweeps abruptly reduce the
anchor–ellipse tension and the ellipse briefly rotates clockwise before
resuming its counterclockwise motion. Increasing XPBD iterations from 32 to
128 does not remove it. Total material is conserved, but its distribution is
incorrect: an earlier junction transfer is not retracted after the next
junction update makes that transfer excessive. Satisfying the friction bounds
alone does not establish a consistent net slip.

An isolated adapter tested the existing direct material solver from commit
`1674c49d` against the same profile geometry, damping, loads, and integration
settings (friction 0.2, XPBD 10 substeps and 32 iterations). No direct-solver
modules were changed. It removes the clockwise activation kick in both cycles
of the eight-second run, with no solver failures or invalid-wrap diagnostics
and less than 1 micrometer of total-material drift. Around the second
activation, it also matches an independent two-junction reference within
0.00014 degrees and 0.00045 N.

The side-by-side comparison uses sweeps on the left and the existing direct
solver on the right; the second activation near 4.45 seconds makes the
difference clearest. Scripts, traces, and videos are retained outside the
repository in the same artifact directory. This is a correctness comparison,
not a performance measurement.

The direct solver is **not integrated into this branch**. The material-sweep
correction is deferred as a separate follow-up; this prototype still uses
sweeps and retains this known limitation.
