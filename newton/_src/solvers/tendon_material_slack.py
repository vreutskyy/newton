# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact zero-force allocation for linear, finite-friction material components."""

import warp as wp


@wp.struct
class TendonSlackBlock:
    """One block of the isotonic projection's per-component stack."""

    total: wp.float64
    error: wp.float64
    count: int
    end: int


@wp.func
def _add(total: wp.float64, error: wp.float64, value: wp.float64):
    # Large opposite transfers can cancel while leaving a small slack reserve.
    updated = total + value
    if wp.abs(total) >= wp.abs(value):
        error += (total - updated) + value
    else:
        error += (value - updated) + total
    return updated, error


@wp.func_native(
    "return ::fma(a, b, c);",
    adj_snippet="adj_a += b * adj_ret; adj_b += a * adj_ret; adj_c += adj_ret;",
)
def _fma(a: wp.float64, b: wp.float64, c: wp.float64) -> wp.float64:
    """Retain the division remainder when comparing compensated block means."""
    ...


@wp.func
def _mean(block: TendonSlackBlock):
    value = block.total / wp.float64(block.count)
    error = (_fma(-value, wp.float64(block.count), block.total) + block.error) / wp.float64(block.count)
    return value, error


@wp.func
def _limit(length: float, compliance: float, damping_tension: float) -> float:
    return wp.min(-compliance * wp.min(damping_tension, 0.0), length - 1.0e-6)


@wp.func
def project_slack_component(
    first: int,
    last: int,
    active: wp.array[int],
    length: wp.array[float],
    compliance: wp.array[float],
    damping: wp.array[float],
    stretch: wp.array[float],
    right_link: wp.array[int],
    transfer: wp.array[float],
    slack: wp.array[int],
    blocks: wp.array[TendonSlackBlock],
):
    """Find the zero-force allocation with minimum squared net material transfer.

    Pairwise cone projections can approach zero only asymptotically. A tiny
    positive residual then activates the full elastic stiffness in VBD. For a
    connected component whose total material permits zero force, solve that
    branch explicitly, including signed damping and the minimum rest length.
    No force threshold is used; taut components are left to the friction sweeps.

    With d_i = trial_stretch_i - zero_force_limit_i, cumulative transfer q_j
    gives final stretch_i = trial_stretch_i + q_(i-1) - q_i. Writing
    q_j = sum(d_0..d_j) + z_j reduces feasibility to nondecreasing z in
    [0, -sum(d)]. Minimizing sum(q_j^2) is an isotonic least-squares fit of
    -prefix(d). Pool adjacent violating blocks in O(n) time and O(n) storage.
    Inactive slots are skipped; a component cannot cross a disabled guide.
    """
    capacity = wp.float64(0.0)
    capacity_error = wp.float64(0.0)
    unchanged = bool(True)
    for seg in range(first, last + 1):
        if active[seg] == 0:
            continue
        if compliance[seg] <= 0.0 or not wp.isfinite(compliance[seg]):
            return
        limit = _limit(length[seg], compliance[seg], damping[seg])
        if not wp.isfinite(limit) or not wp.isfinite(stretch[seg]):
            return
        capacity, capacity_error = _add(capacity, capacity_error, wp.float64(limit))
        capacity, capacity_error = _add(capacity, capacity_error, -wp.float64(stretch[seg]))
        unchanged = unchanged and stretch[seg] <= limit
    if capacity + capacity_error < wp.float64(0.0):
        return
    if unchanged:
        for seg in range(first, last + 1):
            if active[seg] != 0:
                slack[seg] = 1
        return

    size = int(0)
    prefix = wp.float64(0.0)
    prefix_error = wp.float64(0.0)
    for seg in range(first, last):
        if active[seg] == 0:
            continue
        limit = _limit(length[seg], compliance[seg], damping[seg])
        prefix, prefix_error = _add(prefix, prefix_error, wp.float64(stretch[seg]))
        prefix, prefix_error = _add(prefix, prefix_error, -wp.float64(limit))
        block = TendonSlackBlock()
        block.total = -prefix
        block.error = -prefix_error
        block.count = 1
        block.end = seg
        while size > 0:
            previous = blocks[first + size - 1]
            left, left_error = _mean(previous)
            right, right_error = _mean(block)
            if (left - right) + (left_error - right_error) <= wp.float64(0.0):
                break
            total, error = _add(block.total, block.error, previous.total)
            total, error = _add(total, error, previous.error)
            block.total = total
            block.error = error
            block.count += previous.count
            size -= 1
        blocks[first + size] = block
        size += 1

    prefix = wp.float64(0.0)
    prefix_error = wp.float64(0.0)
    previous_fit = wp.float64(0.0)
    previous_error = wp.float64(0.0)
    begin = first
    for index in range(size):
        block = blocks[first + index]
        fit, fit_error = _mean(block)
        if fit + fit_error < wp.float64(0.0):
            fit, fit_error = wp.float64(0.0), wp.float64(0.0)
        elif (fit - capacity) + (fit_error - capacity_error) > wp.float64(0.0):
            fit, fit_error = capacity, capacity_error
        for seg in range(begin, block.end + 1):
            if active[seg] == 0:
                continue
            limit = _limit(length[seg], compliance[seg], damping[seg])
            prefix, prefix_error = _add(prefix, prefix_error, wp.float64(stretch[seg]))
            prefix, prefix_error = _add(prefix, prefix_error, -wp.float64(limit))
            gap, gap_error = _add(fit, fit_error, -previous_fit)
            gap_error -= previous_error
            stretch[seg] = float(wp.float64(limit) - (gap + gap_error))
            slip, slip_error = _add(prefix, prefix_error, fit)
            transfer[right_link[seg]] = float(slip + (slip_error + fit_error))
            slack[seg] = 1
            previous_fit = fit
            previous_error = fit_error
        begin = block.end + 1
    gap, gap_error = _add(capacity, capacity_error, -previous_fit)
    gap_error -= previous_error
    stretch[last] = float(wp.float64(_limit(length[last], compliance[last], damping[last])) - (gap + gap_error))
    slack[last] = 1
