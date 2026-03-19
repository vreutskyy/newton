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

"""Integration tests: xcol collision pipeline + Newton XPBD solver.

Verifies that xcol-generated contacts produce stable, physically correct
simulations when fed into Newton's solver.
"""

import unittest

import warp as wp

import newton
from newton.examples.basic.example_basic_xcol import XColPipeline


def _run_simulation(model, xcol_pipeline, num_frames=300, substeps=32, iterations=10):
    """Run a simulation and return final body positions."""
    solver = newton.solvers.SolverXPBD(model, iterations=iterations)
    contacts = xcol_pipeline.contacts()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    dt = 1.0 / (60.0 * substeps)

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            xcol_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy()


class TestXColSimulation(unittest.TestCase):
    """Test xcol pipeline integrated with Newton's XPBD solver."""

    def test_box_stack_on_ground(self):
        """A stack of 3 cubes on a static ground should come to rest."""
        box_half = 0.25
        ground_hz = 0.1

        builder = newton.ModelBuilder()
        builder.add_shape_box(body=-1, hx=5.0, hy=5.0, hz=ground_hz)

        # Stack 3 boxes directly on top of each other, starting at rest
        expected_z = []
        for i in range(3):
            z = ground_hz + box_half + i * (2.0 * box_half)
            expected_z.append(z)
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()),
            )
            builder.add_shape_box(body, hx=box_half, hy=box_half, hz=box_half)

        model = builder.finalize()
        xcol = XColPipeline(model)

        body_q = _run_simulation(model, xcol, num_frames=300)

        for i in range(3):
            z = float(body_q[i][2])
            self.assertAlmostEqual(
                z,
                expected_z[i],
                delta=0.05,
                msg=f"Box {i} at z={z:.4f}, expected ~{expected_z[i]:.4f}",
            )

    def test_single_box_drop(self):
        """A single box dropped from height should settle on ground."""
        builder = newton.ModelBuilder()
        builder.add_shape_box(body=-1, hx=5.0, hy=5.0, hz=0.1)
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
        )
        builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
        model = builder.finalize()
        xcol = XColPipeline(model)

        body_q = _run_simulation(model, xcol, num_frames=600)

        z = float(body_q[0][2])
        self.assertAlmostEqual(z, 0.35, delta=0.15, msg=f"Box z={z:.4f}, expected ~0.35")


if __name__ == "__main__":
    unittest.main()
