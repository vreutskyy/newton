# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packed workspace and route assembly for direct tendon material transfer."""

import warp as wp

from ..sim.tendon import TendonLinkType
from .tendon_material import solve_tendon_material_component


@wp.struct
class TendonMaterialState:
    enabled: bool
    raw_compliance: wp.array[float]
    initial: wp.array[float]
    compliance: wp.array[float]
    cap: wp.array[float]
    upper: wp.array[float]
    output: wp.array[float]
    status: wp.array[int]
    faces: wp.array[int]
    valid: wp.array[int]
    pieces: wp.array[int]
    count: wp.array[int]
    ids: wp.array[int]
    edge_ids: wp.array[int]
    next_link: wp.array[int]
    incoming: wp.array[int]
    failure: wp.array[int]
    failure_component: wp.array[int]


@wp.func
def fail_tendon_material(state: TendonMaterialState, code: int, tendon: int, component: int):
    # One thread owns a tendon. Latch the first failure, including during graph replay.
    if state.failure[tendon] == 0:
        state.failure[tendon] = code
        state.failure_component[tendon] = component
        wp.printf(
            "ERROR: direct tendon material solve failed (tendon=%d, component=%d, status=%d). "
            "Discard this simulation state and call solver.check_tendon_material() for details.\n",
            tendon,
            component,
            code,
        )


@wp.func
def project_tendon_component(state: TendonMaterialState, row: int, count: int, tendon: int):
    if state.count[row] != count:
        state.valid[row] = 0
    state.count[row] = count
    # Each component packs into its original segment interval; inactive route slots
    # need no padding. Storage remains O(total segments), not 32 slots per segment.
    solve_tendon_material_component(
        state.initial,
        state.compliance,
        state.cap,
        state.upper,
        state.output,
        state.status,
        state.faces,
        state.valid,
        state.pieces,
        count,
        1,
        row,
        1,
        1,
    )
    if state.status[row] < 0:
        fail_tendon_material(state, state.status[row], tendon, row)


@wp.func
def transfer_tendon_material_direct(
    state: TendonMaterialState,
    tendon: int,
    link_start: int,
    num_links: int,
    seg_offset: int,
    num_segs: int,
    tendon_link_type: wp.array[int],
    tendon_link_active: wp.array[bool],
    cone_seg_l: wp.array[int],
    cone_seg_r: wp.array[int],
    cap_ratio: wp.array[float],
    seg_active: wp.array[int],
    seg_compliance: wp.array[float],
    seg_damping: wp.array[float],
    seg_length: wp.array[float],
    seg_stretch: wp.array[float],
    rolling_delta_l: wp.array[float],
    rolling_delta_r: wp.array[float],
    apply_rolling_transfer: int,
    apply_pinhole_slip: int,
    min_rest: float,
) -> bool:
    for s in range(num_segs):
        seg = seg_offset + s
        # Check authored inputs too: VBD's per-span floor and route merging
        # must not turn invalid coefficients into an apparently valid solve.
        raw_compliance = state.raw_compliance[seg]
        if not wp.isfinite(raw_compliance) or raw_compliance < 1.0e-25:
            fail_tendon_material(state, -1, tendon, seg_offset)
            return False
        state.next_link[seg] = -1
        state.incoming[seg] = -1

    for i in range(1, num_links - 1):
        link = link_start + i
        kind = tendon_link_type[link]
        rolling = kind == int(TendonLinkType.ROLLING) and tendon_link_active[link]
        pinhole = kind == int(TendonLinkType.PINHOLE)
        if not ((apply_rolling_transfer != 0 and rolling) or (apply_pinhole_slip != 0 and pinhole)):
            continue
        left = cone_seg_l[link]
        right = cone_seg_r[link]
        if left < 0 or right < 0:
            continue
        if left < seg_offset or right >= seg_offset + num_segs or left >= right:
            fail_tendon_material(state, -102, tendon, seg_offset)
            return False
        if seg_active[left] == 0 or seg_active[right] == 0 or state.next_link[left] >= 0 or state.incoming[right] >= 0:
            fail_tendon_material(state, -102, tendon, seg_offset)
            return False
        for skipped in range(left + 1, right):
            if seg_active[skipped] != 0:
                fail_tendon_material(state, -102, tendon, seg_offset)
                return False
        state.next_link[left] = link
        state.incoming[right] = link
        if rolling:
            # The no-slip trial carries the full rolling motion, including material
            # exchanged with the arc. Project friction AFTER constructing this trial.
            # Keep the two modes separate to match the validated float32 evaluation.
            common = 0.5 * (rolling_delta_r[left] + rolling_delta_l[right])
            differential = 0.5 * (rolling_delta_r[left] - rolling_delta_l[right])
            seg_stretch[left] = seg_stretch[left] - common
            seg_stretch[right] = seg_stretch[right] - common
            seg_stretch[left] = seg_stretch[left] - differential
            seg_stretch[right] = seg_stretch[right] + differential

    row = int(-1)
    count = int(0)
    previous = int(-1)
    for s in range(num_segs):
        seg = seg_offset + s
        if seg_active[seg] == 0:
            continue
        link = int(-1)
        if previous >= 0:
            link = state.next_link[previous]
        if link < 0:
            if count > 0:
                project_tendon_component(state, row, count, tendon)
                if state.failure[tendon] != 0:
                    return False
            row = seg
            count = 0
        if count >= 32:
            fail_tendon_material(state, -103, tendon, row)
            return False
        index = row + count
        if state.ids[index] != seg:
            state.valid[row] = 0
        state.ids[index] = seg
        state.initial[index] = seg_stretch[seg] + seg_compliance[seg] * seg_damping[seg]
        state.compliance[index] = seg_compliance[seg]
        state.upper[index] = (seg_length[seg] - min_rest) + seg_compliance[seg] * seg_damping[seg]
        if count > 0:
            edge = index - 1
            if state.edge_ids[edge] != link:
                state.valid[row] = 0
            state.edge_ids[edge] = link
            state.cap[edge] = cap_ratio[link]
        count += 1
        previous = seg
    if count > 0:
        project_tendon_component(state, row, count, tendon)
        if state.failure[tendon] != 0:
            return False

    # Publish only after every component on this tendon has passed its certificate.
    row = int(-1)
    count = int(0)
    previous = int(-1)
    for s in range(num_segs):
        seg = seg_offset + s
        if seg_active[seg] == 0:
            continue
        if previous < 0:
            row = seg
            count = 0
        elif state.next_link[previous] < 0:
            row = seg
            count = 0
        seg_stretch[seg] = state.output[row + count] - seg_compliance[seg] * seg_damping[seg]
        count += 1
        previous = seg
    return True
