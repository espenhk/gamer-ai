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
