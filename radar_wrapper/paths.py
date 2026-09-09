"""Centralised legacy path management for radar_wrapper.

All ``sys.path`` backs for the legacy folders live here so that no other
warpper module needs to know where CIRgenerator / MX / RFtests live on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CIRGEN_ROOT = ROOT / "CIRgenerator"
MX_ROOT = ROOT / "MX_CodeV1_Mustafa"
RFTESTS_ROOT = ROOT / "RFtestsUCI2_0_Adam"

def _prepend_path(path: Path) -> None:
    path = path.resolve()
    path_str = str(path)

    # 이미 있으면 제거 후 맨 앞으로 이동
    sys.path[:] = [
        p for p in sys.path
        if str(Path(p).resolve()) != path_str
    ]

    sys.path.insert(0, path_str)

def add_legacy_paths() -> None:
    """Insert all legacy folder roots into ``sys.path`` if not already present.

    Call this *before* importing any legacy module.
    """
    candidates = [str(ROOT), str(CIRGEN_ROOT), str(MX_ROOT), str(RFTESTS_ROOT)]
    for p in candidates:
        if p not in sys.path:
            sys.path.insert(0, p)

def add_mx_paths() -> None:
    """Use this before importing MX_CodeV1_Mustafa modules.

    This intentionally gives MX_CodeV1_Mustafa priority for imports like:
        from test_modules.radar_utils import ...
    """
    _prepend_path(ROOT)
    _prepend_path(MX_ROOT)
    clear_test_modules_cache_if_not_under(MX_ROOT)

def add_rftests_paths() -> None:
    """Use this only inside RFtests adapter later."""
    _prepend_path(ROOT)
    _prepend_path(RFTESTS_ROOT)


def add_cirgen_paths() -> None:
    """Use this inside SyntheticCIRSource."""
    _prepend_path(ROOT)
    _prepend_path(CIRGEN_ROOT)

# radar_wrapper/paths.py

def clear_test_modules_cache_if_not_under(expected_root: Path) -> None:
    expected_root = expected_root.resolve()

    for name in list(sys.modules.keys()):
        if name == "test_modules" or name.startswith("test_modules."):
            mod = sys.modules.get(name)
            mod_file = getattr(mod, "__file__", None)

            if mod_file is None:
                continue

            try:
                mod_path = Path(mod_file).resolve()
            except Exception:
                continue

            if expected_root not in mod_path.parents:
                del sys.modules[name]