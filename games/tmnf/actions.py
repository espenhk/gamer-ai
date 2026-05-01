"""TMNF-specific action definitions.

DISCRETE_ACTIONS  — 25×3 array covering
                    {full brake, half brake, coast, half accel, full accel}
                    × {full left, half left, straight, half right, full right}
PROBE_ACTIONS     — fixed-action episodes for cold-start evaluation
WARMUP_ACTION     — full-throttle straight used during episode warmup steps
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Discrete action set for Q-table policies (EpsilonGreedy, MCTS)
# ---------------------------------------------------------------------------
# Each row is a (3,) action: [steer, accel, brake]
# Steer values: -1 (full left), -0.5 (half left), 0 (straight),
#               0.5 (half right), 1 (full right)
# Throttle states (accel, brake) mutually exclusive:
#   full brake (0, 1), half brake (0, 0.5), coast (0, 0),
#   half accel (0.5, 0), full accel (1, 0)

DISCRETE_ACTIONS = np.array([
    # --- full brake (accel=0, brake=1) ---
    [-1.,  0.,  1. ],   #  0: full brake + full left
    [-0.5, 0.,  1. ],   #  1: full brake + half left
    [ 0.,  0.,  1. ],   #  2: full brake + straight
    [ 0.5, 0.,  1. ],   #  3: full brake + half right
    [ 1.,  0.,  1. ],   #  4: full brake + full right
    # --- half brake (accel=0, brake=0.5) ---
    [-1.,  0.,  0.5],   #  5: half brake + full left
    [-0.5, 0.,  0.5],   #  6: half brake + half left
    [ 0.,  0.,  0.5],   #  7: half brake + straight
    [ 0.5, 0.,  0.5],   #  8: half brake + half right
    [ 1.,  0.,  0.5],   #  9: half brake + full right
    # --- coast (accel=0, brake=0) ---
    [-1.,  0.,  0. ],   # 10: coast + full left
    [-0.5, 0.,  0. ],   # 11: coast + half left
    [ 0.,  0.,  0. ],   # 12: coast + straight
    [ 0.5, 0.,  0. ],   # 13: coast + half right
    [ 1.,  0.,  0. ],   # 14: coast + full right
    # --- half accel (accel=0.5, brake=0) ---
    [-1.,  0.5, 0. ],   # 15: half accel + full left
    [-0.5, 0.5, 0. ],   # 16: half accel + half left
    [ 0.,  0.5, 0. ],   # 17: half accel + straight
    [ 0.5, 0.5, 0. ],   # 18: half accel + half right
    [ 1.,  0.5, 0. ],   # 19: half accel + full right
    # --- full accel (accel=1, brake=0) ---
    [-1.,  1.,  0. ],   # 20: full accel + full left
    [-0.5, 1.,  0. ],   # 21: full accel + half left
    [ 0.,  1.,  0. ],   # 22: full accel + straight
    [ 0.5, 1.,  0. ],   # 23: full accel + half right
    [ 1.,  1.,  0. ],   # 24: full accel + full right
], dtype=np.float32)


def _action_to_idx(action: np.ndarray) -> int:
    """Map an action array back to its nearest index in DISCRETE_ACTIONS."""
    diffs = np.abs(DISCRETE_ACTIONS - action[np.newaxis, :]).sum(axis=1)
    return int(np.argmin(diffs))


def _normalize_weight_cfg(cfg: dict, names: list[str]) -> dict:
    """Return a weight config in the current steer/accel/brake format.

    Older configs may still use a single throttle head.  Those are mapped to
    accel=throttle and brake=-throttle so existing saved policies and tests
    continue to load.
    """
    normalized = {
        k: v for k, v in cfg.items()
        if k not in {"steer_weights", "accel_weights", "brake_weights", "throttle_weights"}
    }

    steer_weights = dict(cfg.get("steer_weights", {}))
    if "accel_weights" in cfg or "brake_weights" in cfg:
        accel_weights = dict(cfg.get("accel_weights", {}))
        brake_weights = dict(cfg.get("brake_weights", {}))
    else:
        throttle_weights = dict(cfg.get("throttle_weights", {}))
        accel_weights = {name: float(throttle_weights.get(name, 0.0)) for name in names}
        brake_weights = {name: float(-throttle_weights.get(name, 0.0)) for name in names}

    for name in names:
        steer_weights.setdefault(name, 0.0)
        accel_weights.setdefault(name, 0.0)
        brake_weights.setdefault(name, 0.0)

    normalized["steer_weights"] = {name: float(steer_weights[name]) for name in names}
    normalized["accel_weights"] = {name: float(accel_weights[name]) for name in names}
    normalized["brake_weights"] = {name: float(brake_weights[name]) for name in names}
    return normalized


# ---------------------------------------------------------------------------
# Probe actions — fixed action vectors for cold-start evaluation.
# Each entry is (action_array, description_string).
# ---------------------------------------------------------------------------

PROBE_ACTIONS: list[tuple[np.ndarray, str]] = [
    (np.array([-1., 0., 1.], dtype=np.float32), "brake left"),
    (np.array([ 0., 0., 1.], dtype=np.float32), "brake"),
    (np.array([ 1., 0., 1.], dtype=np.float32), "brake right"),
    (np.array([-1., 1., 0.], dtype=np.float32), "accel left"),
    (np.array([ 0., 1., 0.], dtype=np.float32), "accel"),
    (np.array([ 1., 1., 0.], dtype=np.float32), "accel right"),
]

# Forced action used during episode warmup (first N steps): full throttle, straight.
WARMUP_ACTION = np.array([0., 1., 0.], dtype=np.float32)
