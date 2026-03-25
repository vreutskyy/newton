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

"""xcol — experimental extensible collision library.

Example::

    import xcol as xc

    # Register custom shapes (optional — built-ins auto-registered)
    # xc.register_shape("cone", support_fn=..., contact_face_fn=..., aabb_fn=...)

    # Compile kernels
    collider = xc.create_collider()

    # Build scene
    builder = xc.Builder()
    builder.add_shape(xc.SHAPE_BOX, params=(10, 1, 10))
    builder.add_shape(xc.SHAPE_POINT, margin=1.0, world=-1)
    model = builder.finalize()

    # Simulation loop
    model.shape_transforms.assign(my_transforms)
    collider.collide(model)
    # model.contact_count, model.contact_point, etc. are now filled
"""

from .model import Builder, Model
from .pipeline import Collider, create_collider
from .shapes import SHAPE_BOX, SHAPE_POINT, SHAPE_SEGMENT, ShapeEntry, register_shape
from .types import ContactFaceResult, ContactResult, GJKResult, ShapeData

__all__ = [
    "SHAPE_BOX",
    "SHAPE_POINT",
    "SHAPE_SEGMENT",
    "Builder",
    "Collider",
    "ContactFaceResult",
    "ContactResult",
    "GJKResult",
    "Model",
    "ShapeData",
    "ShapeEntry",
    "create_collider",
    "register_shape",
]
