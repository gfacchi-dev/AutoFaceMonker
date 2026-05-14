"""AutoFaceMonker: automatic 3D facial template registration via MVMP + MeshMonk."""

__version__ = "0.1.0"
__all__ = ["AutoFaceMonker", "register_template"]


def __getattr__(name: str):
    if name in ("AutoFaceMonker", "register_template"):
        from ._register import AutoFaceMonker, register_template
        globals()["AutoFaceMonker"] = AutoFaceMonker
        globals()["register_template"] = register_template
        return globals()[name]
    raise AttributeError(f"module 'autofacemonker' has no attribute {name!r}")
