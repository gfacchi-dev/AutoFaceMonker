"""
Block mvmp/pyrender from loading during unit tests.

pyrender imports OpenGL/EGL at module level; on headless CI runners there is
no EGL library, which causes a hard ImportError during test collection before
any @patch decorator has a chance to run.  Stubbing the modules here (before
any test file is imported) avoids that crash.  The actual Facemarker behaviour
is replaced per-test via @patch("autofacemonker._register.Facemarker").
"""
import sys
from unittest.mock import MagicMock

for _mod in [
    "mvmp", "mvmp.core", "mvmp.core.facemarker", "mvmp.core.predict",
    "meshmonk", "meshmonk._meshmonk_core",
]:
    sys.modules.setdefault(_mod, MagicMock())
