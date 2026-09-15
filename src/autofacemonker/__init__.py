"""AutoFaceMonker: automatic 3D facial template registration via MVMP + MeshMonk."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._register import AutoFaceMonker, meshmonk_supports_point_to_surface, register_template

__version__ = "0.3.0"
__all__ = ["AutoFaceMonker", "meshmonk_supports_point_to_surface", "register_template"]


def __getattr__(name: str):
    if name in __all__:
        from . import _register
        for public in __all__:
            globals()[public] = getattr(_register, public)
        return globals()[name]
    raise AttributeError(f"module 'autofacemonker' has no attribute {name!r}")
