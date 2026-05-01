"""SC2 belief schema — defines what gets tracked by the belief + info-gain stack.

This module configures the :class:`~framework.belief.EWMABelief` and
:class:`~framework.info_gain.RegionStalenessTracker` for StarCraft 2
fog-of-war play.

Tracked belief slots
--------------------
The minimap is divided into an ``N×N`` grid (default 8×8 = 64 cells).
Each cell tracks:

* ``enemy_supply`` — last-known enemy supply count in that cell.

The belief vector is ``2 * N * N`` floats (value + confidence per cell)
and the staleness vector is ``N * N`` floats, for a total of
``3 * N * N`` extra observation dimensions when fog-of-war is enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from framework.belief import EWMABelief
from framework.info_gain import RegionStalenessTracker
from framework.obs_spec import ObsDim, ObsSpec


# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "region_grid": [8, 8],
    "decay_tau": 30.0,
    "scout_drive_weight": 0.1,
    "scout_horizon_s": 60.0,
    "stale_threshold": 0.5,
    "never_seen_bonus": 2.0,
}


def load_belief_config(path: str | Path) -> dict[str, Any]:
    """Load belief config from YAML, falling back to defaults for missing keys."""
    cfg = dict(_DEFAULTS)
    p = Path(path)
    if p.exists():
        with open(p) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_belief(cfg: dict[str, Any] | None = None) -> EWMABelief:
    """Create a belief module from a config dict."""
    cfg = cfg or dict(_DEFAULTS)
    grid = cfg.get("region_grid", _DEFAULTS["region_grid"])
    n_slots = grid[0] * grid[1]
    return EWMABelief(
        n_slots=n_slots,
        decay_tau=cfg.get("decay_tau", _DEFAULTS["decay_tau"]),
    )


def make_info_gain(cfg: dict[str, Any] | None = None) -> RegionStalenessTracker:
    """Create an info-gain module from a config dict."""
    cfg = cfg or dict(_DEFAULTS)
    grid = cfg.get("region_grid", _DEFAULTS["region_grid"])
    return RegionStalenessTracker(
        n_rows=grid[0],
        n_cols=grid[1],
        scout_horizon_s=cfg.get("scout_horizon_s", _DEFAULTS["scout_horizon_s"]),
        stale_threshold=cfg.get("stale_threshold", _DEFAULTS["stale_threshold"]),
        scout_drive_weight=cfg.get("scout_drive_weight", _DEFAULTS["scout_drive_weight"]),
        never_seen_bonus=cfg.get("never_seen_bonus", _DEFAULTS["never_seen_bonus"]),
    )


def belief_obs_dims(cfg: dict[str, Any] | None = None) -> list[ObsDim]:
    """Return the extra ObsDim entries added by belief + staleness tracking.

    These are appended to the base observation spec when fog-of-war is
    enabled.
    """
    cfg = cfg or dict(_DEFAULTS)
    grid = cfg.get("region_grid", _DEFAULTS["region_grid"])
    n_slots = grid[0] * grid[1]
    dims: list[ObsDim] = []
    for i in range(n_slots):
        dims.append(ObsDim(f"belief_val_{i}", 1.0, f"Belief value for region {i}"))
        dims.append(ObsDim(f"belief_conf_{i}", 1.0, f"Belief confidence for region {i}"))
    for i in range(n_slots):
        dims.append(ObsDim(f"staleness_{i}", 1.0, f"Staleness for region {i}"))
    return dims


def extend_obs_spec(base_spec: ObsSpec, cfg: dict[str, Any] | None = None) -> ObsSpec:
    """Return a new ObsSpec with belief + staleness dimensions appended."""
    extra = belief_obs_dims(cfg)
    return ObsSpec(list(base_spec.dims) + extra)
