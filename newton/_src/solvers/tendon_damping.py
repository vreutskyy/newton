# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Engagement-aware damping shared by body rows and material transfer."""

import warp as wp


@wp.func
def tendon_damping_tension(stretch: float, raw_tension: float, damping: float, dt: float) -> float:
    """Exclude travel through slack from positive damping; retain unloading."""
    if dt <= 0.0:
        return 0.0
    return wp.min(raw_tension, damping * wp.max(stretch, 0.0) / dt)


@wp.func
def linear_damped_coefficients(stretch: float, compliance: float, raw_tension: float, damping_rate: float):
    """Represent one affine branch as T=max((stretch+offset)/compliance,0)."""
    c = wp.max(compliance, 1.0e-30)
    if raw_tension > 0.0 and raw_tension > damping_rate * wp.max(stretch, 0.0):
        return c / (1.0 + c * damping_rate), 0.0
    return c, c * raw_tension


@wp.func
def linear_damped_tension(stretch: float, compliance: float, raw_tension: float, damping_rate: float) -> float:
    """Keep the represented force-free endpoint exactly force-free."""
    c = wp.max(compliance, 1.0e-30)
    if stretch <= -c * wp.min(raw_tension, 0.0):
        return 0.0
    return wp.max(stretch / c + wp.min(raw_tension, damping_rate * wp.max(stretch, 0.0)), 0.0)


@wp.func
def linear_damped_transfer(
    high: float,
    c_high: float,
    b_high: float,
    d_high: float,
    low: float,
    c_low: float,
    b_low: float,
    d_low: float,
    ratio: float,
    bound: float,
) -> float:
    """Project a linear pair through its damping branches without iterative sweeps.

    Each span has two affine branches: engagement-limited or full damping.
    Test their four combinations and accept the one containing its solution.
    The first zero-force point caps the transfer, preserving slack history.
    """
    upper = wp.max(wp.min(bound, high + c_high * wp.min(b_high, 0.0)), 0.0)
    for branch in range(4):
        ch = c_high
        cl = c_low
        oh = ch * b_high
        ol = cl * b_low
        limited_h = (branch & 1) != 0
        limited_l = (branch & 2) != 0
        if limited_h:
            if b_high <= 0.0:
                continue
            ch = ch / (1.0 + ch * d_high)
            oh = 0.0
        if limited_l:
            if b_low <= 0.0:
                continue
            cl = cl / (1.0 + cl * d_low)
            ol = 0.0
        delta = wp.clamp((cl * (high + oh) - ratio * ch * (low + ol)) / (cl + ratio * ch), 0.0, upper)
        eh = high - delta
        el = low + delta
        actual_h = b_high > 0.0 and b_high > d_high * wp.max(eh, 0.0)
        actual_l = b_low > 0.0 and b_low > d_low * wp.max(el, 0.0)
        if actual_h == limited_h and actual_l == limited_l:
            return delta
    # At a represented branch boundary, either neighboring affine root can
    # round to the opposite side. Bracket the monotone physical force instead.
    lower = float(0.0)
    for _ in range(32):
        middle = 0.5 * (lower + upper)
        if middle == lower or middle == upper:
            break
        residual = linear_damped_tension(high - middle, c_high, b_high, d_high) - ratio * linear_damped_tension(
            low + middle, c_low, b_low, d_low
        )
        if residual > 0.0:
            lower = middle
        else:
            upper = middle
    return lower
