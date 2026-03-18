# Experimental Collision Pipeline — Development Plan

## Location: `newton/_src/geometry/collision_pipeline/`

## Sub-element Abstraction

Every shape is a collection of convex sub-elements. Each sub-element provides:
- `support(dir) → point` — farthest point in direction (for GJK/EPA)
- `contact_face(dir, point) → points[1..4], normal` — farthest polygon in direction (for contact patch)

---

## Phase 0 — Scaffold + Sphere Support Function

**Goal:** Minimal abstraction, one shape type, no collision yet. Just prove the interface works.

**Tests:**
- `SphereShape` returns 1 sub-element
- `support(dir)` returns correct point on sphere surface for cardinal directions
- `contact_face(dir, point)` returns 1 point (same as support point)
- Support respects sphere center offset and radius

**Implement:**
- `SubElement` base (dataclass or protocol)
- `ConvexShape` protocol: `num_sub_elements()`, `query_sub_elements(volume)`, `support(sub_id, dir)`, `contact_face(sub_id, dir, point)`
- `SphereShape(center, radius)` implementing `ConvexShape`

---

## Phase 1 — GJK (Distance Query)

**Goal:** Compute distance + closest points between two separated convex sub-elements using support functions only.

**Tests:**
- Two spheres separated → correct distance and closest points
- Two spheres just touching → distance ≈ 0
- Degenerate: identical centers → returns zero distance (or flags overlap)
- Closest points lie on sphere surfaces, not centers

**Implement:**
- `gjk(shape_a, sub_id_a, shape_b, sub_id_b) → distance, point_a, point_b, normal`
- Minkowski difference support: `support_a(dir) - support_b(-dir)`
- Standard GJK simplex evolution (1-simplex → 2-simplex → 3-simplex)
- All in Warp (runs on both CPU and GPU)

---

## Phase 2 — EPA (Penetration Depth)

**Goal:** When GJK detects overlap (simplex encloses origin), find penetration depth + normal.

**Tests:**
- Two overlapping spheres → correct penetration depth and normal
- Barely overlapping → depth ≈ small positive value, normal along center-center axis
- Deeply overlapping → correct depth
- Symmetry: `epa(A, B)` and `epa(B, A)` give consistent (flipped) results

**Implement:**
- `epa(shape_a, sub_id_a, shape_b, sub_id_b, gjk_simplex) → depth, normal, point_a, point_b`
- Polytope expansion from GJK's terminal simplex
- Find closest face to origin, expand until convergence

---

## Phase 3 — Contact Patch Generation

**Goal:** Multi-point contacts via face clipping. This is where `contact_face` pays off.

**Tests:**
- Sphere-sphere: always 1 contact point (trivial, but validates the pipeline end-to-end)
- Box-box face-face (Phase 4, but design tests now): should produce 4 contact points
- Polygon clipping: unit test Sutherland-Hodgman on 2D quads directly
- `reduce_polygon`: 8 coplanar points → 4 forming largest quad
- `reduce_polygon`: 3 points → 3 points (no reduction needed)

**Implement:**
- After GJK/EPA gives `normal`:
  - `face_a = shape_a.contact_face(sub_id_a, normal, point_a)` → up to 4 points
  - `face_b = shape_b.contact_face(sub_id_b, -normal, point_b)` → up to 4 points
- `clip_polygons(face_a, face_b, normal) → clipped_points[]`
- `reduce_polygon(points, normal) → points[max 4]` — largest-quad heuristic
- Full pipeline: `generate_contacts(shape_a, shape_b) → normal, points[1..4], depths[1..4]`

---

## Phase 4 — Box and Capsule Primitives

**Goal:** More shape types to exercise the abstraction properly.

**Tests:**
- `BoxShape.support(dir)` returns correct corner for all octants
- `BoxShape.contact_face(dir)` returns 4 corners of the correct face for axis-aligned dirs, 2 edge points for edge-on dirs
- `CapsuleShape.support(dir)` = hemisphere point
- `CapsuleShape.contact_face(dir)` = 2 axis endpoints (lateral) or 1 point (end-on)
- Box on ground plane: 4 contact points, all at correct depth
- Box edge on box face: 2 contact points
- Box corner on box face: 1 contact point
- Capsule on ground plane: 2 contact points (line contact)
- Capsule-capsule parallel: 2 contact points
- Box-sphere: 1 contact point at correct position

**Implement:**
- `BoxShape(center, half_extents, orientation)`
- `CapsuleShape(center, radius, half_height, orientation)`
- Both implement `ConvexShape` protocol with `support` and `contact_face`
- Brute-force N×N broadphase: all shape pairs tested, AABB overlap filter
  - Shapes provide `get_aabb()`
  - `CollisionWorld`: holds shapes, runs N×N broadphase → narrowphase for each pair
  - This is the first multi-shape scene test (box on plane, stack of boxes, mixed primitives)

---

## Phase 5 — Triangle Sub-Elements (Mesh Collision)

**Goal:** A triangle mesh is a collection of triangle sub-elements. Each triangle has its own support and contact_face.

**Tests:**
- `MeshShape` from a simple mesh (cube = 12 triangles) returns 12 sub-elements
- `query_sub_elements(aabb)` returns only triangles overlapping the AABB
- `support(tri_id, dir)` returns correct vertex of that triangle
- `contact_face(tri_id, dir)` returns triangle vertices (3 points), or edge (2) if edge-on, or vertex (1)
- Sphere vs mesh cube: contacts on correct face
- Box vs mesh cube face-on: 4 contact points

**Implement:**
- `MeshShape(vertices, indices)` implementing `ConvexShape`
- BVH for `query_sub_elements(volume)` — own implementation
- Triangle support = `argmax(dot(vertex, dir))` over 3 vertices
- Triangle contact_face: project `dir` to find face/edge/vertex case

---

## Phase 6 — Convex Surface Patches

**Goal:** Instead of individual triangles, group adjacent coplanar/convex triangles into patches. Fewer sub-elements, better contact quality.

**Tests:**
- Cube mesh (12 tris) → 6 patches (one per face, 4 vertices each)
- Sphere mesh (many tris) → many small patches (each nearly flat)
- `support(patch_id, dir)` = `argmax(dot(v, dir))` over patch vertices — matches brute force over constituent triangles
- `contact_face(patch_id, dir)` returns up to 4 vertices of the patch boundary
- Box vs cube-mesh patch: same contact quality as box-box (4 points on face)
- Convexity invariant: no patch contains a concave vertex angle

**Implement:**
- Greedy region-growing decomposition:
  - Start from seed triangle
  - Try adding adjacent triangles
  - Check convexity (all boundary vertex angles < 180° on surface)
  - Stop when no more can be added or patch hits max size (e.g., 32 vertices)
  - Max size prevents degenerate patches (e.g., a finely tessellated sphere collapsing into one huge patch)
- `ConvexPatch`: vertex list + adjacency
- `MeshShape` option: `use_patches=True` replaces triangle sub-elements with patch sub-elements
- Patch support = `argmax` over patch vertex set

---

## Phase 7 — Optimized Broadphase

**Goal:** Replace brute-force N×N with a scalable broadphase. The N×N from Phase 4 is correct but O(n²).

**Tests:**
- Produces identical pair sets as brute-force N×N on randomized scenes
- Stress: 1000+ shapes, verify correctness against brute force
- Performance: measurably faster than N×N on large scenes
- Incremental update: move one shape, only affected pairs change

**Implement:**
- Sort-and-sweep on one axis:
  - Sort AABB min endpoints
  - Sweep and test overlaps on remaining axes
- Same `Broadphase` interface, swap implementation behind `CollisionWorld`
- Alternative: spatial hash grid for more uniform distributions

---

## Phase 8 — SDF Deep Penetration Handler

**Goal:** For deep penetrations, SDF-vs-SDF gradient descent. Handles concavity correctly.

**Tests:**
- Two overlapping sphere SDFs: gradient descent finds deepest penetration point, depth matches analytical
- Concave mesh SDF (e.g., bowl shape): point inside bowl correctly detected as inside, correct depth
- SDF-vs-SDF on two boxes: penetration depth and normal match analytical
- Symmetry: consistent results regardless of query order
- Edge case: objects just barely overlapping — should still find the penetration

**Implement:**
- `SDFShape(grid, cell_size, origin)` or wrap Newton's existing `SDF` class
- `sdf_penetration(sdf_a, transform_a, sdf_b, transform_b) → depth, normal, point`
- Algorithm: sample points on surface of A, evaluate B's SDF (and vice versa), follow gradient to find deepest penetration
- Returns 1 (or few) deep contact points

---

## Phase 9 — Two-Stage Pipeline

**Goal:** Combine everything. SDF for deep penetrations, GJK/EPA + clipping for shallow/touching.

**Tests:**
- Deep overlap: pipeline uses SDF path, returns valid contacts
- Shallow overlap: pipeline uses GJK/EPA path, returns multi-point contact patch
- Transition: as objects move from deep to shallow penetration, contact points remain stable (no popping)
- Separated objects: only GJK distance query, no contacts
- Mixed: mesh-vs-primitive uses SDF for deep, patches + GJK for shallow
- Full scene: multiple shape pairs processed correctly

**Implement:**
- `CollisionPipeline`:
  1. **Broadphase**: query overlapping shape pairs (own implementation)
  2. **SDF probe**: for each pair, quick SDF sample to estimate penetration depth
  3. **Route**:
     - `depth > threshold` → SDF-vs-SDF gradient descent → 1–2 deep contacts
     - `depth ≤ threshold` → query sub-elements in overlap region → GJK/EPA per pair → contact_face + clip → up to 4 contacts per sub-element pair
  4. **Merge**: collect all contacts, deduplicate nearby points
- Depth threshold is tunable (e.g., 2× cell size of coarsest SDF)

---

## Phase 10 — Integration with Newton

**Goal:** Wire the pipeline into Newton's simulation loop as an alternative collision backend.

**Tests:**
- Existing Newton examples (stacking, pendulum) run with new pipeline, produce stable simulation
- Performance: not a regression on simple scenes
- Contact output format matches what Newton's solver expects

**Implement:**
- Adapter: convert `CollisionPipeline` output to Newton's contact format
- Config flag to switch between old and new pipeline
- Profile and optimize hot paths
