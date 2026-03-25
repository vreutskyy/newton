# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

###########################################################################
# Example Basic XCol
#
# Boxes dropping onto a ground plane. Demonstrates the experimental xcol
# collision pipeline feeding contacts into Newton's XPBD solver.
#
# Command: uv run -m newton.examples basic_xcol
#
###########################################################################

import warp as wp

import newton
import newton.examples
import xcol as xc
from newton._src.geometry.types import GeoType


# -----------------------------------------------------------------------
# Warp kernel: compute xcol shape transforms from Newton body state
# -----------------------------------------------------------------------
@wp.kernel
def _compute_xcol_transforms(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    xcol_shape_map: wp.array(dtype=int),
    out_transforms: wp.array(dtype=wp.transform),
):
    xcol_idx = wp.tid()
    newton_shape_idx = xcol_shape_map[xcol_idx]
    body_idx = shape_body[newton_shape_idx]
    shape_xform = shape_transform[newton_shape_idx]
    if body_idx < 0:
        out_transforms[xcol_idx] = shape_xform
    else:
        body_xform = body_q[body_idx]
        out_transforms[xcol_idx] = wp.transform_multiply(shape_xform, body_xform)


# -----------------------------------------------------------------------
# Warp kernel: convert xcol contacts to Newton contacts
# -----------------------------------------------------------------------
@wp.kernel
def _convert_contacts(
    xcol_count: wp.array(dtype=int),
    xcol_shape_a: wp.array(dtype=int),
    xcol_shape_b: wp.array(dtype=int),
    xcol_point: wp.array(dtype=wp.vec3),
    xcol_normal: wp.array(dtype=wp.vec3),
    xcol_depth: wp.array(dtype=float),
    xcol_shape_map: wp.array(dtype=int),
    # Newton model data
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    # Newton contact outputs
    newton_count: wp.array(dtype=int),
    newton_max: int,
    out_shape0: wp.array(dtype=int),
    out_shape1: wp.array(dtype=int),
    out_point0: wp.array(dtype=wp.vec3),
    out_point1: wp.array(dtype=wp.vec3),
    out_offset0: wp.array(dtype=wp.vec3),
    out_offset1: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= xcol_count[0]:
        return

    idx = wp.atomic_add(newton_count, 0, 1)
    if idx >= newton_max:
        return

    # Map xcol shape indices back to Newton shape indices
    ns0 = xcol_shape_map[xcol_shape_a[i]]
    ns1 = xcol_shape_map[xcol_shape_b[i]]

    out_shape0[idx] = ns0
    out_shape1[idx] = ns1

    contact_world = xcol_point[i]
    normal = xcol_normal[i]
    depth = xcol_depth[i]  # signed: positive=gap, negative=penetration

    # Transform contact point to body-local frames
    body0 = shape_body[ns0]
    body1 = shape_body[ns1]

    X_bw_a = wp.transform_identity()
    X_bw_b = wp.transform_identity()
    if body0 >= 0:
        X_bw_a = wp.transform_inverse(body_q[body0])
    if body1 >= 0:
        X_bw_b = wp.transform_inverse(body_q[body1])

    # contact_world is the clipped polygon point (near B's surface).
    # Place p0 on A's surface, p1 at the contact point (B's surface).
    # Solver: d = dot(n, bx_b - bx_a) = dot(n, contact - (contact + n*pen)) = -pen = depth.
    pen = -depth  # positive = overlap, negative = gap
    half_pen = pen * 0.5
    point_a_world = contact_world + normal * half_pen
    point_b_world = contact_world - normal * half_pen

    out_point0[idx] = wp.transform_point(X_bw_a, point_a_world)
    out_point1[idx] = wp.transform_point(X_bw_b, point_b_world)
    out_offset0[idx] = wp.vec3(0.0, 0.0, 0.0)
    out_offset1[idx] = wp.vec3(0.0, 0.0, 0.0)
    out_normal[idx] = normal


# -----------------------------------------------------------------------
# XColPipeline: adapter between xcol and Newton
# -----------------------------------------------------------------------
class XColPipeline:
    """Adapts xcol collision pipeline for use with Newton's XPBD solver.

    Maps Newton shapes to xcol shapes, runs xcol collision detection,
    and converts the results back to Newton's contact format.
    """

    def __init__(self, newton_model: newton.Model) -> None:
        self._newton_model = newton_model
        self._collider = xc.create_collider()

        # Build xcol model from Newton shapes
        builder = xc.Builder()

        newton_shape_types = newton_model.shape_type.numpy()
        newton_shape_scales = newton_model.shape_scale.numpy()
        newton_shape_worlds = newton_model.shape_world.numpy() if newton_model.shape_world is not None else None

        # xcol_shape_map[xcol_idx] = newton_shape_idx
        shape_map_list = []

        for ni in range(newton_model.shape_count):
            geo_type = int(newton_shape_types[ni])
            scale = newton_shape_scales[ni]

            if geo_type == int(GeoType.BOX):
                xcol_type = xc.SHAPE_BOX
                params = (float(scale[0]), float(scale[1]), float(scale[2]))
                margin = 0.0
            elif geo_type == int(GeoType.SPHERE):
                xcol_type = xc.SHAPE_POINT
                params = (0.0, 0.0, 0.0)
                margin = float(scale[0])
            elif geo_type == int(GeoType.CAPSULE):
                xcol_type = xc.SHAPE_SEGMENT
                params = (0.0, 0.0, float(scale[1]))
                margin = float(scale[0])
            else:
                continue

            world = int(newton_shape_worlds[ni]) if newton_shape_worlds is not None else 0
            builder.add_shape(xcol_type, params=params, margin=margin, world=world)
            shape_map_list.append(ni)

        self._xcol_model = builder.finalize()
        self._xcol_shape_map = wp.array(shape_map_list, dtype=int)

    def contacts(self) -> newton.Contacts:
        """Allocate a Newton Contacts buffer sized for this pipeline."""
        return newton.Contacts(
            rigid_contact_max=self._xcol_model.max_contacts,
            soft_contact_max=0,
        )

    def collide(self, state, contacts: newton.Contacts) -> None:
        """Update transforms, run xcol collision, convert to Newton contacts."""
        contacts.clear()

        # Compute xcol transforms from Newton body state
        wp.launch(
            _compute_xcol_transforms,
            dim=self._xcol_model.shape_count,
            inputs=[
                state.body_q,
                self._newton_model.shape_body,
                self._newton_model.shape_transform,
                self._xcol_shape_map,
            ],
            outputs=[self._xcol_model.shape_transforms],
        )

        self._collider.collide(self._xcol_model, contact_distance=0.2)

        # Clear Newton contact count before converting
        contacts.rigid_contact_count.zero_()

        # Convert xcol contacts to Newton format
        wp.launch(
            _convert_contacts,
            dim=self._xcol_model.max_contacts,
            inputs=[
                self._xcol_model.contact_count,
                self._xcol_model.contact_shape_a,
                self._xcol_model.contact_shape_b,
                self._xcol_model.contact_point,
                self._xcol_model.contact_normal,
                self._xcol_model.contact_depth,
                self._xcol_shape_map,
                self._newton_model.shape_body,
                state.body_q,
                contacts.rigid_contact_count,
                contacts.rigid_contact_max,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
            ],
        )


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # Ground box (static — body=-1 means fixed)
        builder.add_shape_box(body=-1, hx=5.0, hy=5.0, hz=0.1)

        # Drop 5 boxes from increasing heights with different rotations
        box_half = 0.25
        rotations = [
            wp.quat_identity(),
            wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.3),
            wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.5),
            wp.quat_from_axis_angle(wp.vec3(1, 1, 0), 0.7),
            wp.quat_from_axis_angle(wp.vec3(0, 1, 1), 1.0),
        ]
        for i in range(5):
            x = 0  # (i - 2) * 1.0
            z = 1.0 + i * 1.1
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x, 0.0, z), q=rotations[i]),
                label=f"box_{i}",
            )
            builder.add_shape_box(body, hx=box_half, hy=box_half, hz=box_half)

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        use_xcol = not hasattr(args, "no_xcol") or not args.no_xcol
        if use_xcol:
            self.xcol_pipeline = XColPipeline(self.model)
            self.contacts = self.xcol_pipeline.contacts()
        else:
            self.xcol_pipeline = None
            self.contacts = self.model.contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.0, -8.0, 4.0),
            pitch=-20.0,
            yaw=90.0,
        )

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.xcol_pipeline:
                self.xcol_pipeline.collide(self.state_0, self.contacts)
            else:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # All boxes should be resting near z = ground_top + box_half = 0.1 + 0.25 = 0.35
        for i in range(self.model.body_count):
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                f"box_{i} resting on ground",
                lambda q, qd: abs(q[2] - 0.35) < 0.1,
                [i],
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--no-xcol", action="store_true", help="Use Newton's built-in collision instead of xcol")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
