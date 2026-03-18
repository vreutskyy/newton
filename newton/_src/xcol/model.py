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

"""Builder and Model for xcol collision scenes."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .types import ContactResult


class Builder:
    """Accumulates shapes on the Python side, then finalizes into a :class:`Model`.

    Example::

        b = Builder()
        b.add_shape(xc.SHAPE_BOX, params=(1, 1, 1))
        b.add_shape(xc.SHAPE_POINT, margin=1.0, world=-1)
        model = b.finalize()
    """

    def __init__(self) -> None:
        self._types: list[int] = []
        self._params: list[tuple[float, float, float]] = []
        self._margins: list[float] = []
        self._worlds: list[int] = []
        self._transforms: list[tuple[float, ...]] = []

    @property
    def shape_count(self) -> int:
        return len(self._types)

    def add_shape(
        self,
        shape_type: int,
        params: tuple[float, float, float] = (0.0, 0.0, 0.0),
        margin: float = 0.0,
        world: int = 0,
        transform: tuple[float, ...] | None = None,
    ) -> int:
        """Add a shape.

        Args:
            shape_type: Shape type id (e.g. ``SHAPE_POINT``, ``SHAPE_BOX``).
            params: Core shape parameters.
            margin: Uniform inflation distance [m].
            world: World index. Same-world shapes collide.
                ``-1`` = global (collides with all worlds).
            transform: Initial ``(px, py, pz, qx, qy, qz, qw)``.
                Defaults to identity.

        Returns:
            Shape index.
        """
        idx = len(self._types)
        self._types.append(int(shape_type))
        self._params.append((float(params[0]), float(params[1]), float(params[2])))
        self._margins.append(float(margin))
        self._worlds.append(int(world))
        if transform is None:
            transform = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self._transforms.append(transform)
        return idx

    def finalize(self, max_contacts: int | None = None) -> Model:
        """Create a :class:`Model` from the accumulated shapes.

        Args:
            max_contacts: Maximum number of contact points to allocate.
                If ``None``, estimates as ``shape_count * (shape_count - 1) * 2``.

        Returns:
            A GPU-resident :class:`Model`.
        """
        n = len(self._types)

        shape_types = wp.array(self._types, dtype=int)
        shape_params = wp.array([wp.vec3(*p) for p in self._params], dtype=wp.vec3)
        shape_margins = wp.array(self._margins, dtype=float)
        shape_worlds = wp.array(self._worlds, dtype=int)
        shape_transforms = wp.array(
            [wp.transform(*t) for t in self._transforms], dtype=wp.transform
        )

        if max_contacts is None:
            max_contacts = max(n * (n - 1) * 2, 1)

        contact_count = wp.zeros(1, dtype=int)
        contact_shape_a = wp.zeros(max_contacts, dtype=int)
        contact_shape_b = wp.zeros(max_contacts, dtype=int)
        contact_point = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_normal = wp.zeros(max_contacts, dtype=wp.vec3)
        contact_depth = wp.zeros(max_contacts, dtype=float)

        return Model(
            shape_count=n,
            shape_types=shape_types,
            shape_params=shape_params,
            shape_margins=shape_margins,
            shape_worlds=shape_worlds,
            shape_transforms=shape_transforms,
            max_contacts=max_contacts,
            contact_count=contact_count,
            contact_shape_a=contact_shape_a,
            contact_shape_b=contact_shape_b,
            contact_point=contact_point,
            contact_normal=contact_normal,
            contact_depth=contact_depth,
        )


@dataclass
class Model:
    """GPU-resident collision scene data.

    Created by :meth:`Builder.finalize`.  The user updates
    :attr:`shape_transforms` between simulation steps (e.g. from a Warp
    kernel or via ``assign()``).  After :meth:`Collider.collide` the
    contact arrays are filled.

    Shape arrays (read-only after finalize):
        shape_count, shape_types, shape_params, shape_margins, shape_worlds

    Transform array (user-writable):
        shape_transforms: ``wp.array(dtype=wp.transform)``

    Contact arrays (written by Collider.collide):
        contact_count: ``wp.array(dtype=int, shape=(1,))``
        contact_shape_a, contact_shape_b: ``wp.array(dtype=int)``
        contact_point: ``wp.array(dtype=wp.vec3)``
        contact_normal: ``wp.array(dtype=wp.vec3)``
        contact_depth: ``wp.array(dtype=float)``
    """

    shape_count: int
    shape_types: wp.array
    shape_params: wp.array
    shape_margins: wp.array
    shape_worlds: wp.array
    shape_transforms: wp.array

    max_contacts: int
    contact_count: wp.array
    contact_shape_a: wp.array
    contact_shape_b: wp.array
    contact_point: wp.array
    contact_normal: wp.array
    contact_depth: wp.array
