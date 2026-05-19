"""
Pytest configuration for tmnf tests.

Adds the tmnf source directory and the tests directory to sys.path so that:
  - bare imports like `from utils import Vec3` resolve correctly
  - `from helpers import ...` works across all test files
"""
import os
import sys

_tmnf_dir  = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_tests_dir = os.path.dirname(__file__)

for _p in (_tmnf_dir, _tests_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Disable the SC2 map-access gate by default in tests so unit tests that
# touch SC2Client._make_sc2_env don't sleep 5s of real wall-clock time
# (issue #254). Tests that exercise the gate itself override this via
# monkeypatch or pass gap_s explicitly.
os.environ.setdefault("GAMER_AI_SC2_MAP_GAP_S", "0")
