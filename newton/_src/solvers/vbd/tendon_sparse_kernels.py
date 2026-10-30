# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Condensed tendon loads and cross-body stiffness for sparse VBD articulations."""

import warp as wp

from .rigid_sparse_articulation_kernels import _scalar_block_index
from .tendon_kernels import TendonForceElementAdjacencyInfo
from .tendon_modes import evaluate_condensed_tendon_force_hessians, tendon_mode_body_pair_hessians


@wp.kernel
def accumulate_sparse_tendon_diagonal(
    bodies: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    adjacency: TendonForceElementAdjacencyInfo,
    link_body: wp.array[int],
    active: wp.array[int],
    link_l: wp.array[int],
    link_r: wp.array[int],
    forces: wp.array[wp.vec3],
    torques: wp.array[wp.vec3],
    h_ll: wp.array[wp.mat33],
    h_al: wp.array[wp.mat33],
    h_aa: wp.array[wp.mat33],
):
    body = bodies[wp.tid()]
    f, t, ll, al, aa = evaluate_condensed_tendon_force_hessians(
        body,
        body_q,
        body_com,
        adjacency.body_adj_segments,
        adjacency.body_adj_segments_offsets,
        adjacency.modes,
        link_body,
        adjacency.attachment_l,
        adjacency.attachment_r,
        active,
        link_l,
        link_r,
    )
    forces[body] += f
    torques[body] += t
    h_ll[body] += ll
    h_al[body] += al
    h_aa[body] += aa


@wp.kernel
def assemble_sparse_tendon_cross_blocks(
    pair_body_row: wp.array[int],
    pair_body_col: wp.array[int],
    pair_slot: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    adjacency: TendonForceElementAdjacencyInfo,
    link_body: wp.array[int],
    active: wp.array[int],
    link_l: wp.array[int],
    link_r: wp.array[int],
    values: wp.array[float],
):
    pair = wp.tid()
    row = pair_body_row[pair]
    col = pair_body_col[pair]
    # Fixed/kinematic bodies have constrained rows in the sparse solve. Their pose
    # still contributes to the tendon force but must not couple to a free delta.
    if body_inv_mass[row] == 0.0 or body_inv_mass[col] == 0.0:
        return
    com_row = wp.transform_point(body_q[row], body_com[row])
    com_col = wp.transform_point(body_q[col], body_com[col])
    h_ll = wp.mat33(0.0)
    h_la = wp.mat33(0.0)
    h_al = wp.mat33(0.0)
    h_aa = wp.mat33(0.0)
    previous = int(-1)
    for cursor in range(adjacency.body_adj_segments_offsets[row], adjacency.body_adj_segments_offsets[row + 1]):
        seg = adjacency.body_adj_segments[cursor]
        if active[seg] == 0:
            continue
        mode = adjacency.modes[seg]
        if mode.first == previous:
            continue
        previous = mode.first
        ll, la, al, aa = tendon_mode_body_pair_hessians(
            row,
            col,
            com_row,
            com_col,
            mode.first,
            adjacency.modes,
            link_body,
            adjacency.attachment_l,
            adjacency.attachment_r,
            link_l,
            link_r,
        )
        h_ll += ll
        h_la += la
        h_al += al
        h_aa += aa
    # One thread owns a body pair, including contributions from multiple tendons.
    slot = pair_slot[pair]
    for i in range(3):
        for j in range(3):
            values[_scalar_block_index(slot, i, j)] += h_ll[i, j]
            values[_scalar_block_index(slot, i, j + 3)] += h_la[i, j]
            values[_scalar_block_index(slot, i + 3, j)] += h_al[i, j]
            values[_scalar_block_index(slot, i + 3, j + 3)] += h_aa[i, j]
