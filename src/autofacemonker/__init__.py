"""AutoFaceMonker: automatic 3D facial template registration via MVMP + MeshMonk."""

from ._register import AutoFaceMonker, register_template

__version__ = "0.1.0"
__all__ = ["AutoFaceMonker", "register_template"]
