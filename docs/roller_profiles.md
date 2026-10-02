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

Explicit profiles currently select prescribed planar VBD routing for the whole
model. Mixing them with dynamic links, terminal rollers, autodiff, or XPBD is
rejected. Builder merging/replication and fixed-joint collapse with explicit
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
The no-slip material trial is assembled before finite-friction projection.
General profiles retain physical endpoint moments about the body COM.
Geometry and material balance are refreshed before each VBD body color. A
preceding body update can otherwise invalidate the capstan tension balance and
apply artificial spin torque even to a frictionless circle. Radius-only models
keep their existing solver path.

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

The interactive default is 10 steps per frame and 32 VBD iterations. Override
with `--substeps` and `--iterations`.
These are settings for this example, not general-purpose solver recommendations.

On an RTX 3080 Laptop GPU, an eight-second, two-cycle comparison of the corrected
solver measured about 44 ms/frame at 10 × 32, excluding initialization and
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

## Validation status

Run the focused tests from the repository root:

```sh
uv run --extra dev -m unittest newton.tests.test_roller_profile newton.tests.test_roller_profile_vbd
```

On October 2, 2026, these modules ran 33 tests: 32 passed and one CPU graph test
was skipped because graph capture requires CUDA. The moving-endpoint regression
also failed with the previous material-update ordering restored, confirming
that it detects the artificial frictionless spin.

A separate 24-test legacy tendon subset had 20 passes, two CPU graph skips,
and two failures: `test_vbd_rolling_chain_tension_matches_accepted_pose` on CPU
and CUDA. Its tension spread was 0.02384 N against a 0.005 N tolerance, with the
same result when the pre-fix update ordering was restored. This remains an
open validation issue; the full legacy suite has not been certified by this
prototype. The branch is for technical review, not merge-ready integration.
