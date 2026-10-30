# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent checks of the VBD tendon friction-secant stiffness."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.vbd.tendon_kernels import TendonForceElementAdjacencyInfo
from newton._src.solvers.vbd.tendon_modes import TendonMaterialMode, prepare_tendon_material_modes
from newton._src.solvers.vbd.tendon_sparse_kernels import (
    accumulate_sparse_tendon_diagonal,
    assemble_sparse_tendon_cross_blocks,
)


class TestSecantModes(unittest.TestCase):
    device = "cpu"

    def modes(self, cap, slip, enabled=True, compliances=(0.1, 0.2)):
        """Evaluate a two-span capstan with fixed, explicitly specified forces."""
        device = self.device

        def arr(values, dtype=float):
            return wp.array(values, dtype=dtype, device=device)

        modes = wp.zeros(2, dtype=TendonMaterialMode, device=device)
        wp.launch(
            prepare_tendon_material_modes,
            dim=1,
            inputs=[
                arr([0, 3], int),
                arr([-1, 0, -1], int),
                arr([-1, 1, -1], int),
                arr([1, cap, 1]),
                arr([1.0 - 2.0 * compliances[0], 1.0 - compliances[1]]),
                arr([1.0, 1.0]),
                arr([2.0 * compliances[0], compliances[1]]),
                arr(compliances),
                arr([0.0, 0.0]),
                arr([0.0, 0.0]),
                arr([1, 1], int),
                0.01,
                0.0,
                0.0,
                0.0,
                0.0,
                arr([0.0, slip, 0.0]),
                int(enabled),
            ],
            outputs=[modes],
            device=device,
        )
        return modes.numpy()

    def test_disabled_preserves_sticking(self):
        """Keep the baseline stiffness when the experimental option is disabled."""
        m = self.modes(2.0, 0.01, False)
        np.testing.assert_array_equal(m["first"], [0, 1])
        np.testing.assert_array_equal(m["tension"], [2.0, 1.0])

    def test_no_slip_preserves_sticking(self):
        """Retain independent segment stiffness without material transfer."""
        m = self.modes(2.0, 0.0)
        np.testing.assert_array_equal(m["first"], [0, 1])

    def test_frictionless_limit(self):
        """Recover the existing series-spring stiffness at unit cap ratio."""
        m = self.modes(1.0, 0.0)
        np.testing.assert_array_equal(m["first"], [0, 0])
        np.testing.assert_array_equal(m["transfer_weight"], [0.0, 0.0])

    def test_secant_limits_and_force_invariance(self):
        """Check positive curvature, limiting weights, and unchanged forces."""
        previous = 2.0
        for slip in [1e-8, 1e-6, 1e-4, 0.01, 1.0, 100.0]:
            m = self.modes(2.0, slip)
            np.testing.assert_allclose(m["tension"], [2.0, 1.0], atol=1e-7)
            alpha = m["transfer_weight"][0]
            self.assertGreaterEqual(alpha, 0.0)
            self.assertLessEqual(alpha, 1.0)
            self.assertLessEqual(alpha, previous)
            previous = alpha
            expected = 1.0 / (1.0 + slip * (10.0 + 5.0))
            self.assertAlmostEqual(float(alpha), expected, places=6)
            h = alpha * np.diag([10.0, 5.0]) + (1 - alpha) / 0.3 * np.ones((2, 2))
            self.assertGreater(np.linalg.eigvalsh(h).min(), 0.0)
            # Eliminate one transfer coordinate from an independent quadratic:
            # E = (x-Bs)^T D (x-Bs)/2 + k_f*s^2/2, with B = [1,-1].
            d = np.diag([10.0, 5.0])
            b = np.array([1.0, -1.0])
            friction_stiffness = 1.0 / slip
            schur = d - np.outer(d @ b, d @ b) / (b @ d @ b + friction_stiffness)
            np.testing.assert_allclose(h, schur, atol=1e-6, rtol=1e-6)

    def test_transfer_stiffness_with_unequal_spans(self):
        """Match transfer-coordinate elimination across large stiffness ratios."""
        for compliance in ((0.1, 0.1), (1e-5, 0.1), (0.1, 1e-5)):
            stiffness = 1.0 / np.array(compliance)
            slip = 1.0 / stiffness.sum()
            with self.subTest(compliance=compliance):
                mode = self.modes(2.0, slip, compliances=compliance)
                np.testing.assert_allclose(mode["tension"], [2.0, 1.0], atol=1e-6)
                np.testing.assert_allclose(mode["transfer_weight"], [0.5, 0.0], atol=1e-6)

    def test_diagonal_and_cross_blocks(self):
        """Compare every body block against an independent dense spring matrix."""
        device = self.device

        def arr(values, dtype=int):
            return wp.array(values, dtype=dtype, device=device)

        m = self.modes(2.0, 0.5)
        modes = arr(m, TendonMaterialMode)
        pose = arr([[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1]], wp.transform)
        com = wp.zeros(3, dtype=wp.vec3, device=device)
        adj = TendonForceElementAdjacencyInfo()
        adj.body_adj_segments = arr([0, 0, 1, 1])
        adj.body_adj_segments_offsets = arr([0, 1, 3, 4])
        adj.modes = modes
        adj.attachment_l = arr([[0, 0, 0], [1, 0, 0]], wp.vec3)
        adj.attachment_r = arr([[1, 0, 0], [2, 0, 0]], wp.vec3)
        active, link_body, left, right = arr([1, 1]), arr([0, 1, 2]), arr([0, 1]), arr([1, 2])
        outputs = [wp.zeros(3, dtype=t, device=device) for t in (wp.vec3, wp.vec3, wp.mat33, wp.mat33, wp.mat33)]
        wp.launch(
            accumulate_sparse_tendon_diagonal,
            dim=3,
            inputs=[arr([0, 1, 2]), pose, com, adj, link_body, active, left, right],
            outputs=outputs,
            device=device,
        )
        cross = wp.zeros(108, dtype=float, device=device)
        wp.launch(
            assemble_sparse_tendon_cross_blocks,
            dim=3,
            inputs=[
                arr([0, 0, 1]),
                arr([1, 2, 2]),
                arr([0, 1, 2]),
                pose,
                com,
                arr([1.0, 1.0, 1.0], float),
                adj,
                link_body,
                active,
                left,
                right,
            ],
            outputs=[cross],
            device=device,
        )
        alpha = float(m["transfer_weight"][0])
        k = alpha * np.diag([10.0, 5.0]) + (1 - alpha) / 0.3 * np.ones((2, 2))
        j = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        expected = j.T @ k @ j
        measured = np.diag(outputs[2].numpy()[:, 0, 0])
        for value, row, col in zip(cross.numpy().reshape(3, 6, 6)[:, 0, 0], [0, 0, 1], [1, 2, 2], strict=True):
            measured[row, col] = measured[col, row] = value
        np.testing.assert_allclose(measured, expected, atol=2e-6)
        np.testing.assert_allclose(outputs[0].numpy()[:, 0], j.T @ [2.0, 1.0], atol=1e-6)


class TestSecantModesCUDA(TestSecantModes):
    def setUp(self):
        """Run the same stiffness checks on CUDA when available."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA device unavailable")
        self.device = "cuda:0"


if __name__ == "__main__":
    wp.init()
    unittest.main()
