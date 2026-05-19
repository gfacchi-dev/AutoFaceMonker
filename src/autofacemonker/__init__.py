"""AutoFaceMonker: automatic 3D facial template registration via MVMP + MeshMonk."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._register import AutoFaceMonker, register_template

__version__ = "0.1.2"
__all__ = ["AutoFaceMonker", "register_template"]


def __getattr__(name: str):
    if name in ("AutoFaceMonker", "register_template"):
        from ._register import AutoFaceMonker, register_template
        globals()["AutoFaceMonker"] = AutoFaceMonker
        globals()["register_template"] = register_template
        return globals()[name]
    raise AttributeError(f"module 'autofacemonker' has no attribute {name!r}")
