# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent per-interface Schur references for the VBD tendon metric."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.vbd.tendon_kernels import TendonForceElementAdjacencyInfo
from newton._src.solvers.vbd.tendon_modes import TendonMaterialMode, prepare_tendon_material_modes
from newton._src.solvers.vbd.tendon_sparse_kernels import (
    accumulate_sparse_tendon_diagonal,
    assemble_sparse_tendon_cross_blocks,
)


class TestInterfaceCurvature(unittest.TestCase):
    device = "cpu"

    def test_distinct_friction_secants(self):
        """Eliminate each transfer independently without changing the body forces."""
        rng = np.random.default_rng(791)
        for case in range(20):
            count = (3, 6, 12, 24)[case % 4]
            body_count = 4
            stiffness = (10.0 ** rng.uniform(2, 7, count)).astype(np.float32)
            compliance = (1.0 / stiffness).astype(np.float32)
            tension = rng.uniform(0.1, 20, count).astype(np.float32)
            stretch = (tension * compliance).astype(np.float32)
            cap = rng.uniform(1.01, 3, count + 1).astype(np.float32)
            transfer = (10.0 ** rng.uniform(-7, -2, count + 1)).astype(np.float32)
            if case % 5 == 0:
                cap[:] = 1.0
            if case % 5 == 1:
                transfer[1::2] = 0.0
            points = np.cumsum(rng.normal(size=(count + 1, 3)), axis=0).astype(np.float32)
            link_body = rng.integers(-1, body_count, count + 1)
            xyz = rng.normal(size=(body_count, 3)).astype(np.float32)
            poses = np.c_[xyz, np.zeros((body_count, 3)), np.ones(body_count)].astype(np.float32)

            def arr(values, dtype=int):
                return wp.array(values, dtype=dtype, device=self.device)

            modes = wp.zeros(count, dtype=TendonMaterialMode, device=self.device)
            wp.launch(
                prepare_tendon_material_modes,
                dim=1,
                inputs=[
                    arr([0, count + 1]),
                    arr(np.r_[-1, np.arange(count - 1), -1]),
                    arr(np.r_[-1, np.arange(1, count), -1]),
                    arr(cap, float),
                    arr(1.0 - stretch, float),
                    arr(np.ones(count), float),
                    arr(stretch, float),
                    arr(compliance, float),
                    arr(np.zeros(count), float),
                    arr(np.zeros(count), float),
                    arr(np.ones(count)),
                    0.01,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    arr(transfer, float),
                    1,
                ],
                outputs=[modes],
                device=self.device,
            )
            material = modes.numpy()
            k = material["stiffness"].astype(float)
            t = material["tension"].astype(float)
            jacobian = np.zeros((count, 6 * body_count))
            segments, offsets = [], [0]
            for body in range(body_count):
                for seg in range(count):
                    if body not in (link_body[seg], link_body[seg + 1]):
                        continue
                    segments.append(seg)
                    if link_body[seg] == link_body[seg + 1]:
                        continue
                    direction = points[seg + 1].astype(float) - points[seg]
                    direction /= np.linalg.norm(direction)
                    for link, sign in ((seg, 1), (seg + 1, -1)):
                        if link_body[link] == body:
                            jacobian[seg, 6 * body : 6 * body + 3] += sign * direction
                            jacobian[seg, 6 * body + 3 : 6 * body + 6] += sign * np.cross(
                                points[link] - xyz[body], direction
                            )
                offsets.append(len(segments))
            free = [seg for seg in range(count - 1) if cap[seg + 1] == 1 or transfer[seg + 1] > 0]
            incidence = np.zeros((count, len(free)))
            friction = np.zeros(len(free))
            for index, seg in enumerate(free):
                incidence[seg, index], incidence[seg + 1, index] = 1, -1
                ratio = float(cap[seg + 1])
                if ratio > 1:
                    friction[index] = (ratio - 1) / (ratio + 1) * (t[seg] + t[seg + 1]) / transfer[seg + 1]
            system = incidence.T @ (k[:, None] * incidence) + np.diag(friction)
            slip = np.linalg.solve(system, incidence.T @ (k[:, None] * jacobian))
            elastic = jacobian - incidence @ slip
            expected = elastic.T @ (k[:, None] * elastic) + slip.T @ (friction[:, None] * slip)

            adjacency = TendonForceElementAdjacencyInfo()
            adjacency.body_adj_segments, adjacency.body_adj_segments_offsets = arr(segments), arr(offsets)
            adjacency.modes = modes
            adjacency.attachment_l, adjacency.attachment_r = arr(points[:-1], wp.vec3), arr(points[1:], wp.vec3)
            pose, com = arr(poses, wp.transform), wp.zeros(body_count, dtype=wp.vec3, device=self.device)
            active, bodies = arr(np.ones(count)), arr(link_body)
            left, right = arr(np.arange(count)), arr(np.arange(1, count + 1))
            output = [
                wp.zeros(body_count, dtype=dtype, device=self.device)
                for dtype in (wp.vec3, wp.vec3, wp.mat33, wp.mat33, wp.mat33)
            ]
            wp.launch(
                accumulate_sparse_tendon_diagonal,
                dim=body_count,
                inputs=[arr(np.arange(body_count)), pose, com, adjacency, bodies, active, left, right],
                outputs=output,
                device=self.device,
            )
            rows, cols = np.tril_indices(body_count, -1)
            cross = wp.zeros(len(rows) * 36, dtype=float, device=self.device)
            wp.launch(
                assemble_sparse_tendon_cross_blocks,
                dim=len(rows),
                inputs=[
                    arr(rows),
                    arr(cols),
                    arr(np.arange(len(rows))),
                    pose,
                    com,
                    arr(np.ones(body_count), float),
                    adjacency,
                    bodies,
                    active,
                    left,
                    right,
                ],
                outputs=[cross],
                device=self.device,
            )
            measured = np.zeros_like(expected)
            ll, al, aa = [value.numpy() for value in output[2:]]
            for body in range(body_count):
                measured[6 * body : 6 * body + 6, 6 * body : 6 * body + 6] = np.block(
                    [[ll[body], al[body].T], [al[body], aa[body]]]
                )
            for row, col, block in zip(rows, cols, cross.numpy().reshape(-1, 6, 6), strict=True):
                measured[6 * row : 6 * row + 6, 6 * col : 6 * col + 6] = block
                measured[6 * col : 6 * col + 6, 6 * row : 6 * row + 6] = block.T
            with self.subTest(case=case):
                error = np.linalg.norm(measured - expected) / max(np.linalg.norm(expected), 1e-30)
                self.assertLess(error, 3e-6)
                force = np.c_[output[0].numpy(), output[1].numpy()].ravel()
                np.testing.assert_allclose(force, jacobian.T @ t, atol=5e-5, rtol=2e-5)


class TestInterfaceCurvatureCUDA(TestInterfaceCurvature):
    def setUp(self):
        """Run the same independent matrix checks on CUDA when available."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA device unavailable")
        self.device = "cuda:0"


if __name__ == "__main__":
    wp.init()
    unittest.main()
