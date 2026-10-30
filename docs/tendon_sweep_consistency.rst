.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Tendon material-sweep consistency: review notes
=============================================

Scope
-----

This experimental correction is based on ``10715f13``. It uses circular
rollers and the existing sweep solver; it does not require the non-circular
profile prototype or the direct material solver. No public solver options or
defaults are added. The solver correction and the independent example setup
corrections are separate commits.

What changes
------------

* Retain each junction's net slip within a material solve. Before revisiting
  that junction, undo its previous trial transfer, retain neighboring
  transfers, and project again. This permits retraction of an excessive
  earlier transfer instead of stopping at an incorrect point inside the
  friction cone.
* Form the complete rolling no-slip trial before friction projection. Remove
  the beta-scaled differential rolling transport formerly applied after the
  sweeps. Friction limits slip and transmitted torque; it does not scale the
  prescribed rolling displacement.
* In XPBD, project the accumulated roller spin impulse and apply its change,
  including retraction on unloading. Independently limiting each iteration's
  impulse increment would not limit their accumulated result.
* In VBD, transmit the full sticking moment and limit out-of-cone trial
  moments using current adjacent tensions. Reuse the already evaluated span
  tension, but evaluate its neighbor at the current body trial, not from a
  stale diagnostic cache. The local positive-semidefinite Hessian freezes the
  limiter; its derivative is not included.

The capstan factor ``(rho - 1) / (rho + 1)`` remains in the physical bound on
the tension difference. It is the unconditional scaling of rolling motion
and reaction that is removed, not that bound.

This is not an early-out fix. On a frozen four-span diagnostic, disabling
early-out changed the old solve from 2 to 256 sweeps with bitwise identical,
incorrect output. The underlying iteration was stuck. The corrected
iteration needs more sweeps to satisfy the net-slip condition.

Validation
----------

The new ``test_tendon_material_sweeps`` module covers CPU and CUDA:

* analytical finite-friction net-slip solutions, both traversal directions,
  with adaptive early-out enabled and disabled;
* linear and sigmoid compliance, damping, slack and sticking;
* full rolling transport, absence of unloaded grip, and the rest-length floor;
* accumulated XPBD reaction, reversal and unloading;
* VBD sticking and sliding reactions, including a slack adjacent span.

Four independent analytical material tests were run against the unfixed
kernel with only its signature adapted; their assertions fail there.
The corrected focused material/reaction suite passes all 16 CPU/CUDA cases.

The one-circle and two-circles-on-one-body finite-friction reproducers run
for eight simulated seconds and include two dynamic-roller activations. In
XPBD, the backward rotation at the second activation falls from about
15.26 degrees to 0.00015 degrees for one circle, and from 9.34 degrees to
0.00124 degrees for two circles. The maximum trajectory-angle differences
from the direct reference are about 0.0019 and 0.0143 degrees, respectively.
The corresponding VBD reproducers also remove the activation glitch.

Existing tests were corrected where they demanded grip without tension,
strictly increasing rotation above the sticking threshold, or material-floor
saturation in an unloaded fixture. Loaded transport now has explicit
pretension and an analytical displacement check. The fixed rolling-chain
fixture uses supported signed wraps and a series-compliance tension
reference with float32-aware tolerances.

Reproduce the focused suite from this checkout::

    uv run --extra dev -m unittest newton.tests.test_tendon_material_sweeps

The broader validation covered these modules on CPU and available CUDA::

    uv run --extra dev -m unittest newton.tests.test_tendon_capstan newton.tests.test_tendon_material_sweeps newton.tests.test_tendon_vbd newton.tests.test_tendon_equilibrium newton.tests.test_cable newton.tests.test_fixed_tendon newton.tests.test_spatial_tendon

That broader run is not all green; see the known limitations below. Subsequent
focused reruns cover the VBD tension-reuse optimization and corrected fixtures.

Performance
-----------

Matched measurements on 2026-10-07 use the same circular pair scene, finite
friction coefficient 0.2, compliance 1e-4 m/N, damping 1 N s/m, 10 substeps,
32 body iterations and material settle tolerance 1e-5. Each run covers eight
simulated seconds (480 frames). Only warmed CUDA graph replay and completion
are timed; compilation, rendering and diagnostic readbacks are excluded.

For each solver, run order was unfixed, corrected, corrected, unfixed. On an
RTX 3080 Laptop GPU with Warp 1.15.0.dev20260626, the results were:

======  ====================  =====================  ================
Solver  Unfixed runs [s]      Corrected runs [s]     Mean-time change
======  ====================  =====================  ================
XPBD    14.585, 14.377        14.145, 13.711         -3.8%
VBD     21.768, 21.775        23.636, 23.719         +8.8%
======  ====================  =====================  ================

During these linear runs, periodic GPU-clock samples were 1800--1815 MHz
with no thermal-throttling flag. The nonlinear full-scene follow-up did hit
thermal throttling and is deliberately excluded from this table. These are
two runs per variant on a small reproducer, not a general speed guarantee.
The trajectories differ because the original response was incorrect; this
is a before/after cost comparison, not an equal-accuracy benchmark.

The material kernel itself can be substantially slower even when the whole
scene overhead is small. Earlier isolated frozen-state tests at default
settle tolerance showed about 1.8x cost for four linear spans, 3.9x for four
sigmoid spans, and up to 17x for sixteen sigmoid spans. In those cases the
old solver stopped after two sweeps with an incorrect net-slip result; the
corrected solver took 7--32 sweeps. These are material-only stress cases,
not whole-simulation slowdowns, and are not general performance guarantees.

Known limitations and separate example corrections
--------------------------------------------------

The correction is ready for review, not a claim that all tendon scenarios or
legacy examples are validated. In particular:

* The original October 7 validation recorded failures in the VBD cable-machine,
  rolling-pulley, dynamic-capstan and kinematic-capstan examples on both
  variants. Later October 27 runs in this experimental checkout passed the
  cable-machine and both capstan examples' assertions. Some still reported
  unsupported fixed wraps, so passing those assertions does not establish a
  valid full trajectory. The rolling-pulley fixture has since been corrected
  to use its stated masses and vertical guides with travel stops. Removing
  XPBD's small-velocity cutoff lets it match analytical free motion at its
  original 32 substeps and 48 iterations; the ten-second run also passes its
  bounded-motion checks after the stops engage. Its VBD expected-failure
  marker is removed.
  The separate high-inertia regression now uses vertical guides and checks
  analytical finite-inertia motion for two pulley inertias. Its former 5 mm
  threshold rejected physically expected motion and passed only because of
  the velocity cutoff. The corrected regression also rejects invalid wraps
  and checks both weights' displacement against the pulley rim travel.
* The original dynamic-capstan fixture exceeds the supported 180-degree wrap
  range at about 0.67 seconds in both local and sparse VBD. Its later motion
  is not an acceptance reference. The corrected fixture constrains each
  hanging weight to vertical motion with 1.2-metre travel stops and sets the
  total weight masses to the stated 1.0 and 1.2 kg, without adding collision
  shape mass on top. It also checks the supported wrap range after each
  frame. This is an explicit change to the model, not evidence that the
  solver now supports the original invalid route. The sparse acceptance
  comparison uses this corrected fixture; the original's valid prefix is
  retained separately as a reproduction check.
* The full VBD XY-table trajectory is not validated by its half-second
  reference-prefix test. That prefix retains per-step checks and now uses
  only prefix-appropriate final checks; its reference tolerances are tighter.
* Total material drift is not eliminated. The October 7 circular-pair runs
  recorded 28--31 micrometres with corrected sweeps versus 18--19 micrometres
  with the old implementation. After the rolling-arc ledger correction, the
  eight-second repro has 0.61/0.82 micrometres with XPBD sweeps/direct and
  0.79/0.43 micrometres with VBD sweeps/direct. These October 27 runs use
  10 substeps, 32 iterations, and independent arc measurements from the same
  local contact geometry as the material state; they have no wrap errors.
  The VBD run maps the old ignored ``k_start`` values to the current finite
  joint stiffness parameters. This is a bounded trajectory check, not proof
  of zero drift or a matched before/after benchmark.
* Unsupported fixed wraps and general stiff-body/joint convergence remain
  limitations. The default XPBD and condensed VBD damping law now excludes
  motion spent taking up slack, as described below. The force-free-slack
  regression passes on CPU and CUDA and is no longer an expected failure.

The separate example correction commit constrains the equal-weight
equilibrium fixture to vertical motion (angled spans otherwise pull the
weights inward), and changes compound-pulley compliance from 1e-6 to 1e-5 m/N
to reduce force sensitivity to position rounding at that scene scale. The
original final assertions pass for both XPBD and VBD. This is a change to
those fixtures, not a solver improvement at the former stiffness.

Experimental-branch integration
------------------------------

This local port applies ``cf643a3c20d12454c463fd4bfa098d4773f9f878``
to experimental base ``1a775c105181b187b09acf3a53b45e2cd0cd91ce``.
It preserves the direct material specialization, cooperative CUDA backend,
explicit route orientation, and accepted-route activation handling. The
scalar and cooperative direct paths also publish net material slip relative
to the full-rolling trial. The experimental VBD sliding-curvature metric
uses this slip; it does not change the projected material solution.
Certified sticking interfaces publish exactly zero slip, so rounded stretch
differences cannot create a false sliding direction or pass it downstream.
This uses the solved friction state, not a displacement threshold: real small
slips and slack redistribution are retained.
Each slipping guide retains its own friction secant. The body stiffness is
assembled by eliminating these transfer coordinates along the route, instead
of assigning the largest sticking weight to every span in a connected block.
This linear-time elimination uses no additional per-step arrays and gives
consistent diagonal and cross-body terms. It is a positive solver metric,
not the exact derivative of the capstan projection. It is enabled for sparse
articulation updates; local VBD retains its existing conservative metric.
During direct solve's opening hold iterations, rest lengths stay frozen but
stretch and damping are refreshed from the current pose. The condensed VBD
body solve must not reuse forces from the previous material projection.

All VBD material modes now merge dynamic rollers that cross zero wrap at the
accepted pose, before checking for unsupported routes. Previously this
cleanup ran only with per-segment ALM, so ordinary direct mode could reject a
normal detachment and sweeps could retain a negative wrap until the next
substep. The merge retains the signed arc, clears obsolete ALM rows when
present, and projects only changed routes again. Fixed negative wraps remain
errors. CPU, CUDA and graph-replay regressions cover material conservation,
damped linear and sigmoid laws, and articulated rollers in the sparse solve.

The corrected sweep and accumulated XPBD reaction are enabled in their
respective paths. VBD's non-direct path uses the current-tension reaction
from the selected correction. Its existing direct/ALM reaction stays intact,
including the per-segment ALM full endpoint-wrench guard from ``f11f0e91``.
Independent clipping of that mode's rim moment would violate internal
force/moment balance.

The generic damping and material/body-coupling changes on the later profile
branch are not included. Elliptical and convex pulley features are not
included. This correction alone does not qualify the complete experimental
branch on current Newton main or certify arbitrary damping/ALM combinations.

The inherited geometry-only roller helper is retained separately for ALM
support-stiffness estimation. Its three callers retain their original
formula; this estimate is not the physical current-tension reaction.

Adaptive stopping and linear-precision follow-up
-----------------------------------------------

The earlier consistency correction above did not change the stopping rule.
This follow-up checks the final capstan error after all neighboring transfers
in a sweep. For a sticking contact, both
tensions must lie inside the capstan band. For nonzero net slip, they must
also lie on the corresponding slipping boundary.

This residual replaces the per-sweep force-change heuristic. Small changes
alone do not establish friction balance, and a large update needs no further
sweep if it already satisfies every contact. Checking the residual directly
also avoids computing intermediate tensions solely for the stopping heuristic.

Each contact uses its current larger tension as the relative scale, rather
than the first sweep's peak tension. Both XPBD and VBD accept the experimental
``tendon_settle_atol`` option (default ``1e-6`` N). The contact threshold is
``tendon_settle_atol + tendon_settle_tol * max(T_left, T_right)``. The absolute
term lets an unloading cable settle as tension approaches zero. Setting
``tendon_settle_tol=0`` still disables early-out and runs the entire sweep
budget. These options do not control the direct material solver.
They are runtime-only numerical solver settings, not tendon material or USD
asset properties.

An unconstrained linear sliding update now computes its correction directly
from the current stretch. Subtracting two much larger accumulated transfers
can otherwise erase the correction on stiff spans. The friction law, material
inventory, and rest-length bounds are unchanged.

The sweep budget remains a cap, not a guarantee of convergence. In particular,
an incompatible rest-length bound or a tolerance below numerical precision
can exhaust it. Material convergence also does not establish convergence of
the coupled body/joint solve.

Exact release of linear slack components
---------------------------------------

Even a very small positive tension residual activates full elastic stiffness
in VBD. Pairwise sweeps can approach a completely slack state asymptotically,
so reducing their tolerance alone does not reliably remove that stiffness.

Before sweeping, linear-law components now check whether their conserved
material inventory permits zero total tension in every span, including the
signed damping contribution and the minimum rest-length bound. If so, a
linear-time projection selects the zero-force allocation with the smallest
sum of squared net guide transfers. Already slack allocations stay unchanged.
Anchors and disabled transfer interfaces separate components; inactive route
slots are skipped. Loaded components retain the finite-friction sweeps.

The force evaluation uses the same represented zero-force endpoint, avoiding
a positive rounding residual when elastic and damping forces cancel. This
does not introduce a force cutoff or change the damping law. The slack
projection currently applies only to linear compliance; sigmoid components
continue through their existing nonlinear material sweeps. It requires
O(segment count) persistent workspace and no extra kernel launch or host
synchronization. Double-precision compensated sums are confined to the slack
allocation; the stored geometry, forces, stretches, and transfers remain floats.

Engagement-aware ALM slack history
---------------------------------

The experimental per-segment ALM material projection uses a damping rule
that excludes motion spent taking up slack. Its zero-force boundary therefore
changes when a span enters or leaves the damping engagement branch. Nonlinear
compliance can change that boundary as material moves, too. A zero-force
solution of a frozen tangent is not necessarily the least-transfer solution
of the actual force law, even if both give zero final tension.

The projection now finds each span's actual zero-force endpoint before
selecting the slack allocation. It includes stored ALM multipliers and the
minimum rest length, then reuses the global minimum-squared-transfer solve.
Already feasible allocations are preserved. Loaded components still require
the full force and friction certificate.

Continuous damping engagement
-----------------------------

Default XPBD and condensed VBD now use the same engagement rule in body
forces, material transfer, and diagnostics. For positive extension rate,
damping is limited to the taut extension divided by the timestep. Motion
spent taking up slack therefore cannot create a pulling force. Negative
damping still reduces the elastic pull, and total tension remains nonnegative.

This is a force-law correction, not a force cutoff. Simply switching damping
off at negative stretch leaves a discontinuity at engagement and can give an
inconsistent implicit step. The limited rule is continuous there and matches
independent backward-Euler references for slack-to-taut motion and taut
loading/unloading, with both linear and sigmoid stiffness.

The material solve retains raw damping until allocation is complete, because
material transfer itself can cross the engagement boundary. Linear sweeps
solve the piecewise-affine pair law. Direct projection retains its fast
constant-damping path when the result stays on that branch; otherwise it
solves and certifies the engagement-dependent force law using the existing
projection machinery with no ALM multipliers or penalty. This does not impose
ALM's compliance floor on the physical material law. Published damping is
then limited using the final stretch, including during material-hold iterations.

Conservative circular rolling transport
--------------------------------------

A circular roller's two contact advances share one wrapped-length balance.
Their antisymmetric part transfers material from one free span to the other;
their sum must equal the decrease in wrapped length. Evaluating both contact
advances independently can violate that balance through float roundoff and
create a small artificial strain even when an unloaded roller only spins.

The geometry update now normalizes the roller's world-space plane normal and
stores its wrapped arc. The existing cone-row kernel reconciles the paired
contact advances with the change from the step-start arc. Dynamic detachment
consumes that same stored arc instead of reconstructing it at a new body pose.
These are float arrays with one additional step-start copy, not an extra
per-iteration kernel launch. Friction and the material solve remain unchanged.

The arc difference follows the continuous angle branch at a half turn, so
crossing ``atan2``'s representation boundary cannot move a circumference of
cable. This is not support for new wrap topologies or multiple full turns;
the existing unsupported-wrap diagnostics still apply.
