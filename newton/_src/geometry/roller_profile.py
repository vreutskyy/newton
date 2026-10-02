# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental circular, elliptical, and circular-sector roller cross-sections."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import warp as wp


class _ProfileKind(IntEnum):
    POINT = 0
    ELLIPSE = 1
    SECTOR = 3
    CIRCLE = 4


@wp.struct
class ProfileData:
    """Private uniform device record; public shapes own only their parameters."""

    kind: int
    radii: wp.vec2
    period: float
    angle: float
    corner: wp.vec2


class RollerProfile(ABC):
    """Experimental immutable roller cross-section in local XY [m].

    Use a circle, ellipse, or exact circular sector. Shape-specific descriptions
    are packed privately for common support and signed boundary-length queries.
    Existing radius-only tendon links continue to describe circles.

    .. experimental::
    """

    @abstractmethod
    def _pack(self) -> ProfileData:
        """Create the private device representation."""


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class RollerProfileCircle(RollerProfile):
    """Circular roller with contact radius [m]."""

    radius: float

    def __post_init__(self):
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))

    def _pack(self) -> ProfileData:
        data = ProfileData()
        data.kind = int(_ProfileKind.CIRCLE)
        data.radii = wp.vec2(self.radius, self.radius)
        data.period = 2.0 * math.pi
        return data


@dataclass(frozen=True)
class RollerProfileEllipse(RollerProfile):
    """Elliptical roller with local X/Y semiaxes `a` and `b` [m]."""

    a: float
    b: float

    def __post_init__(self):
        object.__setattr__(self, "a", _positive(self.a, "a"))
        object.__setattr__(self, "b", _positive(self.b, "b"))

    def _pack(self) -> ProfileData:
        data = ProfileData()
        data.kind = int(_ProfileKind.ELLIPSE)
        data.radii = wp.vec2(self.a, self.b)
        data.period = 2.0 * math.pi
        return data


@dataclass(frozen=True)
class RollerProfileSector(RollerProfile):
    """Exact pizza slice with radius [m] and opening angle [rad] in (0, pi].

    The apex is at the origin and the slice faces local +X. Its boundary consists
    of two radial edges and a circular arc, not an approximating polygon.
    """

    radius: float
    angle: float

    def __post_init__(self):
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))
        angle = _positive(self.angle, "angle")
        if angle > math.pi:
            raise ValueError("A convex sector's angle must not exceed pi")
        object.__setattr__(self, "angle", angle)

    def _pack(self) -> ProfileData:
        data = ProfileData()
        data.kind = int(_ProfileKind.SECTOR)
        data.radii = wp.vec2(self.radius, self.radius)
        data.angle = self.angle
        data.corner = self.radius * wp.vec2(math.cos(self.angle / 2), math.sin(self.angle / 2))
        data.period = self.radius * (2.0 + self.angle)
        return data


def pack_profiles(profiles: list[RollerProfile | None], *, device=None) -> wp.array:
    """Pack roller descriptions; None denotes an anchor or pinhole point."""
    return wp.array(
        [ProfileData() if profile is None else profile._pack() for profile in profiles],
        dtype=ProfileData,
        device=device,
    )


@wp.func
def _support_point(
    profile: ProfileData,
    direction: wp.vec2,
) -> wp.vec2:
    """Evaluate support without computing an unused boundary coordinate."""
    point = wp.vec2(0.0)
    if profile.kind == int(_ProfileKind.CIRCLE):
        point = profile.radii[0] * wp.normalize(direction)
    elif profile.kind == int(_ProfileKind.ELLIPSE):
        a = profile.radii[0]
        b = profile.radii[1]
        scale = wp.length(wp.vec2(a * direction[0], b * direction[1]))
        if scale > 0.0:
            point = wp.vec2(a * a * direction[0], b * b * direction[1]) / scale
    elif profile.kind == int(_ProfileKind.SECTOR):
        radius = profile.radii[0]
        unit = wp.normalize(direction)
        candidate = wp.vec2(profile.corner[0], wp.sign(direction[1]) * profile.corner[1])
        if radius * unit[0] >= profile.corner[0]:
            candidate = radius * unit
        if wp.dot(candidate, direction) > 0.0:
            point = candidate
    return point


@wp.func
def profile_support(profile: ProfileData, direction: wp.vec2):
    """Return the supporting point and its boundary coordinate for a nonzero normal."""
    point = _support_point(profile, direction)
    parameter = float(0.0)
    if profile.kind == int(_ProfileKind.CIRCLE) or profile.kind == int(_ProfileKind.ELLIPSE):
        parameter = wp.atan2(point[1] / profile.radii[1], point[0] / profile.radii[0])
        if parameter < 0.0:
            parameter += 2.0 * wp.pi
    elif profile.kind == int(_ProfileKind.SECTOR) and wp.length_sq(point) > 0.0:
        parameter = profile.radii[0] * (1.0 + wp.atan2(point[1], point[0]) + 0.5 * profile.angle)
    return point, parameter


@wp.func
def profile_parameter_delta(profile: ProfileData, start: float, end: float) -> float:
    """Return the shortest signed boundary-coordinate displacement."""
    if profile.period <= 0.0:
        return 0.0
    delta = end - start
    return delta - profile.period * wp.floor(delta / profile.period + 0.5)


@wp.func
def profile_coordinate(
    profile: ProfileData,
    point: wp.vec2,
) -> float:
    """Recover the boundary coordinate of a stored local contact point."""
    parameter = float(0.0)
    if profile.kind == int(_ProfileKind.CIRCLE):
        parameter = wp.atan2(point[1], point[0])
        if parameter < 0.0:
            parameter += 2.0 * wp.pi
    elif profile.kind == int(_ProfileKind.ELLIPSE):
        parameter = wp.atan2(point[1] / profile.radii[1], point[0] / profile.radii[0])
        if parameter < 0.0:
            parameter += 2.0 * wp.pi
    elif profile.kind == int(_ProfileKind.SECTOR):
        radius = profile.radii[0]
        half = 0.5 * profile.angle
        angle = wp.clamp(wp.atan2(point[1], point[0]), -half, half)
        arc_point = radius * wp.vec2(wp.cos(angle), wp.sin(angle))
        best = wp.length_sq(point - arc_point)
        parameter = radius * (1.0 + angle + half)
        for side in range(2):
            edge_angle = -half if side == 0 else half
            direction = wp.vec2(wp.cos(edge_angle), wp.sin(edge_angle))
            t = wp.clamp(wp.dot(point, direction), 0.0, radius)
            distance = wp.length_sq(point - t * direction)
            if distance < best:
                best = distance
                parameter = t if side == 0 else profile.period - t
        if parameter >= profile.period:
            parameter = 0.0
    return parameter


@wp.func
def _ellipse_speed(radii: wp.vec2, t: float) -> float:
    return wp.length(wp.vec2(radii[0] * wp.sin(t), radii[1] * wp.cos(t)))


@wp.func
def profile_arc_length(profile: ProfileData, start: float, delta: float) -> float:
    """Integrate a signed boundary-coordinate interval, including seam crossings.

    Ellipses use eight-point Gauss quadrature on intervals no larger than pi/16.
    Short material transfers use a single interval, avoiding subtraction of two
    large accumulated lengths. Sector coordinates already measure arclength.
    """
    if profile.kind == int(_ProfileKind.CIRCLE):
        return profile.radii[0] * delta
    if profile.kind == int(_ProfileKind.SECTOR):
        return delta
    if profile.kind != int(_ProfileKind.ELLIPSE):
        return 0.0
    if profile.radii[0] == profile.radii[1]:
        return profile.radii[0] * delta
    panels = wp.max(1, int(wp.ceil(wp.abs(delta) * (16.0 / wp.pi))))
    step = delta / float(panels)
    length = float(0.0)
    for panel in range(panels):
        middle = start + (float(panel) + 0.5) * step
        half = 0.5 * step
        integral = float(0.0)
        for i in range(4):
            node = float(0.1834346424956498)
            weight = float(0.3626837833783620)
            if i == 1:
                node = 0.5255324099163290
                weight = 0.3137066458778873
            elif i == 2:
                node = 0.7966664774136267
                weight = 0.2223810344533745
            elif i == 3:
                node = 0.9602898564975363
                weight = 0.1012285362903763
            integral += weight * (
                _ellipse_speed(profile.radii, middle - half * node)
                + _ellipse_speed(profile.radii, middle + half * node)
            )
        length += half * integral
    return length


@wp.func
def _world_support(
    profile: ProfileData,
    center: wp.vec3,
    axis_x: wp.vec3,
    axis_y: wp.vec3,
    normal: wp.vec3,
):
    point, parameter = profile_support(profile, wp.vec2(wp.dot(normal, axis_x), wp.dot(normal, axis_y)))
    return center + point[0] * axis_x + point[1] * axis_y, parameter


@wp.func
def _world_support_point(profile: ProfileData, center: wp.vec3, axis_x: wp.vec3, axis_y: wp.vec3, normal: wp.vec3):
    point = _support_point(profile, wp.vec2(wp.dot(normal, axis_x), wp.dot(normal, axis_y)))
    return center + point[0] * axis_x + point[1] * axis_y


@wp.func
def profile_span_tangent(
    profile_l: ProfileData,
    profile_r: ProfileData,
    center_l: wp.vec3,
    center_r: wp.vec3,
    x_l: wp.vec3,
    y_l: wp.vec3,
    x_r: wp.vec3,
    y_r: wp.vec3,
    plane_normal: wp.vec3,
    orientation_l: int,
    orientation_r: int,
):
    """Find a common supporting tangent between two coplanar convex profiles.

    A point profile represents an anchor or pinhole. The two winding signs
    select an external or internal tangent. A false validity result reports
    nonexistent/degenerate spans rather than silently using a radial point.
    Frames must be orthonormal and share ``plane_normal``.
    """
    separation = center_r - center_l
    distance = wp.length(separation)
    point_l = center_l
    point_r = center_r
    parameter_l = float(0.0)
    parameter_r = float(0.0)
    normal = wp.vec3(0.0)
    valid = distance > 1.0e-10
    if valid:
        along = separation / distance
        across = wp.cross(plane_normal, along)
        circular = (profile_l.kind == int(_ProfileKind.POINT) or profile_l.kind == int(_ProfileKind.CIRCLE)) and (
            profile_r.kind == int(_ProfileKind.POINT) or profile_r.kind == int(_ProfileKind.CIRCLE)
        )
        if circular:
            # For circles the common-normal equation is linear in sin(angle).
            sine = (float(orientation_l) * profile_l.radii[0] - float(orientation_r) * profile_r.radii[0]) / distance
            valid = wp.abs(sine) < 1.0
            normal = sine * along - wp.sqrt(wp.max(0.0, 1.0 - sine * sine)) * across
            point_l, parameter_l = _world_support(profile_l, center_l, x_l, y_l, float(orientation_l) * normal)
            point_r, parameter_r = _world_support(profile_r, center_r, x_r, y_r, float(orientation_r) * normal)
        else:
            low = float(-0.5 * wp.pi)
            high = float(0.5 * wp.pi)
            angle = float(0.0)
            for _iteration in range(34):
                normal = wp.sin(angle) * along - wp.cos(angle) * across
                point_l = _world_support_point(profile_l, center_l, x_l, y_l, float(orientation_l) * normal)
                point_r = _world_support_point(profile_r, center_r, x_r, y_r, float(orientation_r) * normal)
                residual = wp.dot(point_r - point_l, normal)
                if wp.abs(residual) <= 2.0e-8 * distance:
                    break
                if residual < 0.0:
                    low = angle
                else:
                    high = angle
                # The support-point derivative is orthogonal to the normal.
                # Thus F'(angle) = span dot normal', including corner regions.
                derivative = wp.dot(point_r - point_l, wp.cos(angle) * along + wp.sin(angle) * across)
                candidate = 0.5 * (low + high)
                if derivative > 1.0e-8 * distance:
                    newton = angle - residual / derivative
                    if newton >= low and newton <= high:
                        candidate = newton
                if candidate == angle:
                    break
                angle = candidate
            # Boundary coordinates are needed only for the accepted tangent.
            point_l, parameter_l = _world_support(profile_l, center_l, x_l, y_l, float(orientation_l) * normal)
            point_r, parameter_r = _world_support(profile_r, center_r, x_r, y_r, float(orientation_r) * normal)
        span = point_r - point_l
        tangent = wp.cross(plane_normal, normal)
        scale = wp.max(distance, wp.length(span))
        valid = valid and (
            wp.abs(wp.dot(span, normal)) <= 2.0e-6 * scale
            and wp.dot(span, tangent) > 1.0e-8 * scale
            and wp.abs(wp.dot(span, plane_normal)) <= 2.0e-6 * scale
        )
    return point_l, point_r, parameter_l, parameter_r, normal, valid
