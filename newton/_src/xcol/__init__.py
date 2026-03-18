# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

Public API:

.. code-block:: python

    from newton._src.xcol import (
        # Pipeline
        create_pipeline,
        # Shape registration
        register_shape,
        SHAPE_POINT,
        SHAPE_SEGMENT,
        SHAPE_BOX,
        # Types
        ShapeData,
        ContactResult,
        ContactFaceResult,
        GJKResult,
    )
"""

from .pipeline import Pipeline, create_pipeline
from .shapes import SHAPE_BOX, SHAPE_POINT, SHAPE_SEGMENT, ShapeEntry, register_shape
from .types import ContactFaceResult, ContactResult, GJKResult, ShapeData

__all__ = [
    "Pipeline",
    "create_pipeline",
    "register_shape",
    "ShapeEntry",
    "SHAPE_POINT",
    "SHAPE_SEGMENT",
    "SHAPE_BOX",
    "ShapeData",
    "ContactResult",
    "ContactFaceResult",
    "GJKResult",
]
