# xcol — Development Plan

## Location: `xcol/` (standalone package at repo root)

## Architecture

**Shape abstraction:** Every shape type registers `support(params, dir)` and
`contact_face(params, dir)` functions via `register_shape()`. Built-in and
custom shapes use the same registration path. `create_collider()` compiles
dispatch functions from the registry at kernel compile time via `wp.static()`.

**Core + margin model:** Each shape has a core geometry and a margin (uniform
inflation). GJK operates on cores; margin is subtracted from the distance
afterward. Point + margin = sphere, segment + margin = capsule, box + margin =
rounded box, etc. This improves GJK convergence for round shapes and enables
composite shapes like disks (zero-height cylinder + margin = wheel).

**API:** `Builder` (Python lists, `add_shape()`, `finalize()` → `Model`) →
`Model` (GPU arrays, preallocated contacts) → `Collider` (compiled kernels,
`collide(model)`).

**Contact output:** Flat SoA Warp arrays — `contact_shape_a`, `contact_shape_b`,
`contact_point`, `contact_normal`, `contact_depth`, `contact_count`.

---

## Done

### Shapes & Registration
- Extensible shape registry: `register_shape()` + `create_collider()`
- Built-in shapes: point (sphere via margin), segment (capsule via margin), box
- Core support, contact face (face/edge/vertex), and AABB functions
- Box `contactFace` with PhysX dual thresholds: dEps1=0.99, dEps2=0.14
- Custom shape test (disk shape registered from user code)

### GJK + MPR
- GJK distance with best-result tracking and simplex restore (adapted from Newton)
- MPR penetration depth (adapted from Newton)
- Combined `gjk_mpr()`: MPR first for penetration, GJK fallback for separation
- Signed distance convention: positive = gap, negative = penetration
- Normal convention: consistent A→B direction for both GJK and MPR paths

### Contact Generation (PhysX FaceClipper pattern)
- `clipNone`: vertex contact → single GJK midpoint
- `clip2x2`: parallel edge-edge → segment endpoint projections, midpoint + depth
- `clip2x2` rejects crossing (non-parallel) edges → falls back to clipNone
- `clipNxN`: face-face/edge-face → bounding quad from both faces, Sutherland-Hodgman
  clipping against edge planes, face plane projection for per-point depth
- `reduce_polygon`: deepest point first, then largest quad
- Depth = `dot(axis, B_side - A_side)` — consistent for both GJK and MPR normals

### Pipeline
- GPU NxN broadphase with world filtering (-1 = global)
- Single `collide_nxn_kernel`: broadphase + narrowphase + contact generation
- `contact_distance` parameter for early detection of approaching shapes
- Nesterov acceleration disabled (causes wrong normals for asymmetric shapes)

### Newton Integration
- `XColPipeline` adapter in example: maps Newton shapes → xcol, bridges contacts
- `example_basic_xcol`: 5 boxes with varied rotations on ground platform
- CUDA graph capture works (same fps as native Newton collision)

### Tests
- 28 tests: shapes, GJK, world filtering, contact rotation sweeps,
  edge-edge depth sign, simulation (box drop, box stack), custom shapes

---

## Next

### Phase 5 — Hierarchical Element Abstraction

**Goal:** Generalize shapes as element trees. Each shape provides a hierarchy
of elements — nodes (bounding volumes) and leaves (convex geometry). Primitives
are trivial single-leaf trees. Meshes are BVH trees with triangle/patch leaves.

**Shape callbacks (per shape type, registered like support/contact_face):**
- `get_aabb(params, element_id) → (min, max)` — bounding box for node or leaf
- `query_subelements(params, element_id, aabb, out_ids) → count` — returns
  child elements overlapping the AABB. Children can be nodes (negative IDs:
  `-actual_id - 1`) or leaves (non-negative IDs). Result fits in a small
  fixed-size type (e.g., `wp.vec4i`) since it's just direct children.
- `support(params, element_id, direction) → point` — only called on leaves
- `contact_face(params, element_id, direction) → ContactFaceResult` — only on leaves

**Element ID encoding:**
- `id >= 0` → leaf (triangle, patch, or primitive)
- `id < 0` → node (decode: `actual_id = -id - 1`)
- Helper: `is_node(id) → bool`

**For primitives:** `query_subelements(0, any_aabb)` returns `[0]` (single leaf).
`get_aabb(0)` returns the shape AABB. One-element tree, midphase is a no-op.

---

### Phase 6 — Midphase (BVH-vs-BVH Traversal)

**Goal:** Parallel BVH-vs-BVH traversal using a work queue. Unifies broadphase
and per-shape BVH traversal into one mechanism.

**Pipeline:**
1. **Broadphase** fills midphase queue with `(shapeA, root_0, shapeB, root_0)`
   for all shape pairs whose root AABBs overlap (+ world filtering).
2. **Midphase** processes the queue:
   - Take pair `(A, elemA, B, elemB)`
   - If both are leaves → push to narrowphase buffer
   - If one/both are nodes → `query_subelements` on the node side with the
     other side's AABB → push `(A, child, B, elemB)` pairs back to queue
   - Expand the larger AABB first to balance traversal
3. **Narrowphase** processes leaf-leaf pairs: GJK/MPR + contact generation
   (same as current `collide_nxn_kernel` but on element pairs, not shape pairs)

**Queue:** Preallocated GPU buffer with atomic read/write heads. Each thread
grabs work from the queue. Queue grows as nodes are expanded.

**Tests:**
- Primitive-vs-primitive: queue has 1 entry, immediately goes to narrowphase
- Mesh-vs-primitive: BVH traversal narrows to relevant triangles
- Mesh-vs-mesh: BVH-vs-BVH traversal
- Results match brute-force (all triangles vs all triangles)

---

### Phase 7 — Triangle Mesh Shape

**Goal:** Register a triangle mesh shape type with BVH hierarchy.

**Tests:**
- Cube mesh (12 triangles): box vs mesh cube face-on → 4 contact points
- Sphere vs mesh cube: contacts on correct face
- Large mesh (1000+ triangles): midphase prunes to small set

**Implement:**
- `register_shape("mesh", ...)` with triangle support/contact_face
- BVH construction (CPU-side, stored in Model arrays)
- `query_subelements` walks BVH, returns children overlapping AABB
- Triangle support = `argmax(dot(vertex, dir))` over 3 vertices
- Triangle contact_face: face (3 pts), edge (2 pts), or vertex (0 pts)

---

### Phase 8 — Convex Surface Patches

**Goal:** Group adjacent convex triangles into patches. Fewer elements, better
contact quality, fewer GJK calls.

**Tests:**
- Cube mesh (12 tris) → 6 patches (one per face, 4 vertices each)
- Patch support matches brute force over constituent triangles
- Box vs cube-mesh patch: same contact quality as box-box

**Implement:**
- Greedy region-growing with convexity check, max patch size (~32 vertices)
- Patch support = `argmax` over patch vertex set
- Patch contact_face = boundary vertices, reduced to 4
- BVH leaves become patches instead of triangles

---

### Phase 9 — SDF Deep Penetration Handler

**Goal:** For deep penetrations, SDF-vs-SDF gradient descent. Handles concavity
correctly — no inside/outside ambiguity, no internal boundary artifacts.

**Implement:**
- `SDFShape(grid, cell_size, origin)`
- Sample surface of A, evaluate B's SDF, follow gradient
- Returns 1–2 deep contact points

---

### Phase 10 — Two-Stage Pipeline

**Goal:** SDF for deep penetrations, GJK/MPR + clipping for shallow/touching.

**Implement:**
1. SDF probe: estimate penetration depth
2. Route: deep → SDF gradient descent, shallow → GJK/MPR + contact patch
3. Merge: collect all contacts, deduplicate

---

## Known Issues / TODO

- **Nesterov acceleration** disabled — produces wrong normals for highly
  asymmetric shape pairs (e.g., large ground box vs small box). Investigate.
- **clip2x2 crossing edges** — rejected entirely, falls back to single point.
  Could be improved with proper edge-edge closest point computation.
- **clipNxN face plane projection** — can amplify depth for faces nearly
  perpendicular to the axis (~3x for 8° misalignment). Works in practice
  because contactFace thresholds prevent extreme cases.
