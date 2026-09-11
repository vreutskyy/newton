<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Experimental direct tendon material solver

This review branch adds an optional finite-friction material solver to XPBD
and VBD. It does not replace either solver's rigid-body iterations, change their
time step, or change their default material-sweep path.

## Usage and scope

Select `tendon_material_direct=True` when constructing `SolverXPBD` or
`SolverVBD`. Omit the sigmoid parameters for linear per-segment compliance, or
use the existing experimental law:

```python
solver = newton.solvers.SolverXPBD(
    model,
    iterations=64,
    tendon_material_direct=True,
    tendon_sigmoid_ea_low=500.0,
    tendon_sigmoid_ea_ratio=10.0,
    tendon_sigmoid_transition_strain=0.01,
    tendon_sigmoid_transition_width=0.003,
)
```

The same material options work with `SolverVBD`; its body coloring and other
solver requirements still apply. The numbers above describe one test law, not
recommended parameters for arbitrary cables. Reconstruct the solver to change
the selected material mode or sigmoid parameters.

The direct path currently supports at most 32 authored spans per component
between attachment links, capstan ratios in [1, 4], finite input compliance of
at least 1e-25 m/N, and a 1e-6 m minimum segment rest length. It supports finite
friction, sticking/sliding, slack, damping offsets, and dynamic route repacking.
It does not support differentiable simulation or arbitrary constitutive laws.
Existing routing restrictions, including unsupported fixed-roller wraps,
remain unchanged.

Call `solver.check_tendon_material()` outside capture after a step or graph
replay, before consuming its results. A failure is latched; discard the affected
step/frame/batch and correct the input before reconstructing the solver. There
is no fallback to material sweeps. `tendon_max_sweeps` and `tendon_settle_tol`
control sweeps only, not the direct solve.

## Formulation and implementation

Each connected material component is solved at fixed body poses, span lengths,
and damping tensions. The unknowns are signed span extensions; their sum is
fixed by the reference allocation. Adjacent tensions must obey the capstan
bounds. Nonzero net transfer selects the corresponding bound, while sticking
permits tensions inside the cone. Flat force branches choose the allocation
with minimum squared net transfer across links.

The nonlinear path evaluates the actual sigmoid law as rest length changes;
it does not freeze its secant compliance. A log-force active set identifies
slipping blocks. Each block's coupled force equations are solved with Newton
iterations and a line search. There are at most 64 active-set iterations, with
separate bounds on the inner solves.

On CUDA, one complete warp owns a tendon and solves its connected components
in sequence. Span equations run across lanes; the leader handles routing,
active-set decisions, slack allocation, and publication. CPU uses the scalar
exact-block path. Neither path requires host work inside a captured step.

The CUDA Newton predictor and linearized corrections use FP32. Force residuals,
tiny extension accumulation, log-force state, conservation, and publication
checks retain FP64. Removing that remaining precision failed cancellation and
slack regressions. The fully double-precision prototype is retained separately
as an experimental reference, not installed as an additional runtime mode.

A relative physical tolerance of 1e-5 is checked before float32 publication.
`CONVERGED_ROUNDED` means the certified allocation passes but its prescribed
float32 storage exceeds that tolerance; storage error is measured separately.
It does not mean the stored state satisfies exact equilibrium. Failure never
publishes a rejected component allocation.

Direct mode uses the **full rolling displacement** in its reference allocation,
instead of the sweep path's beta-scaled displacement, and enforces friction in
the projection and matching body reaction. The inherited linear-mode dynamics
tests cover driven rollers and analytical reaction checks. VBD's adjacent-span
reaction now evaluates the selected sigmoid law as well. This is an explicit
formulation difference from sweeps, not merely a faster implementation of an
identical iteration. It does not change the friction coefficient.

## Validation and measured performance

The repository tests cover analytical and independently constructed solutions,
both slip directions, zero/positive damping plateaus, least transfer, rounded
storage, 1–32-span components, heterogeneous packed rows, graph replay, failed
row isolation, route changes, and native XPBD/VBD integration:

```text
uv run -m unittest newton.tests.test_tendon_material_direct newton.tests.test_tendon_material_dynamics newton.tests.test_tendon_material_nonlinear newton.tests.test_tendon_material_cooperative newton.tests.test_tendon_material_integration newton.tests.test_tendon_material_nonlinear_integration newton.tests.test_tendon_material_vbd
```

The review-branch run completed 153 tests successfully (5 device-specific
skips) on CPU and CUDA. Its native XPBD integration also completed both phases
of the 366-frame Toy3 recording below: every shared simulation-state array
matched the validated mixed-precision prototype bit for bit. This checks the
integration against the prototype, not against an independent physical ground
truth; analytical and independently constructed cases are covered separately.

Selected-prototype measurements on a GeForce RTX 3080 Laptop GPU (Windows,
Warp 1.15 development build), using the finite-friction sigmoid Toy3 workload:

| Measurement | Sweeps | Direct | Speedup |
|---|---:|---:|---:|
| Identical saved material inputs, cold | 216.22 ms | 62.64 ms | 3.45x |
| Identical saved material inputs, changed-input warm | 216.22 ms | 69.14 ms | 3.13x |
| Full paired XPBD simulation | 387.00 s | 319.57 s | 1.21x |

Material timings replay the last saved material input per component per frame,
not every material invocation in the trajectory. Five interleaved trials used
two graph replays each. The full comparison alternates sweep/direct frames,
with separate evolving histories: 366 frames / 6.1 simulated seconds, 40 steps
per frame, 64 XPBD iterations, mu=0.1, damping=5, EA=500 to 5000 N, transition
strain=0.01, width=0.003, and A:R0 lowered by 5 mm. Compilation, capture, readback,
and rendering are excluded. Sweeps use their native 1e-3 stopping tolerance;
direct uses its distinct physical certificate above.

These are workload-specific measurements, not a general VBD performance claim
or a guarantee of equal trajectories. The full simulation remains dominated
in part by work outside material transfer. The branch deliberately excludes
mass splitting, warm-started rigid constraints, virtual roller radii, bilinear
laws, and the rejected all-FP32 variants.
