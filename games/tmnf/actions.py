"""TMNF-specific action definitions.

DISCRETE_ACTIONS  — 9×3 array covering {brake, coast, accel} × {left, straight, right}
PROBE_ACTIONS     — fixed-action episodes for cold-start evaluation
WARMUP_ACTION     — full-throttle straight used during episode warmup steps
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Discrete action set for Q-table policies (EpsilonGreedy, MCTS)
# ---------------------------------------------------------------------------
# Each row is a (3,) action: [steer, accel, brake]
# Includes brake+accel combinations since both can be pressed simultaneously.

DISCRETE_ACTIONS = np.array([
    [-1., 0., 1.],   #  0: brake + left
    [ 0., 0., 1.],   #  1: brake + straight
    [ 1., 0., 1.],   #  2: brake + right
    [-1., 0., 0.],   #  3: coast + left
    [ 0., 0., 0.],   #  4: coast + straight
    [ 1., 0., 0.],   #  5: coast + right
    [-1., 1., 0.],   #  6: accel + left
    [ 0., 1., 0.],   #  7: accel + straight
    [ 1., 1., 0.],   #  8: accel + right
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
