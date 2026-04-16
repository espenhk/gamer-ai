"""Behavior cloning: fit WeightedLinearPolicy to SimplePolicy demonstrations."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from games.tmnf.simple_policy import SimplePolicy
from policies import WeightedLinearPolicy

logger = logging.getLogger(__name__)

N_DEMO_LAPS = 3


def collect_demos(env, n_laps: int) -> tuple[np.ndarray, np.ndarray]:
    """Drive n_laps with SimplePolicy, return (obs_matrix, action_matrix)."""
    expert = SimplePolicy()
    obs, _ = env.reset()
    obs_list, act_list = [], []
    laps_done = 0
    while laps_done < n_laps:
        action = expert(obs)
        obs_list.append(obs.copy())
        act_list.append(action.copy())
        obs, _, terminated, truncated, info = env.step(action)
        if (terminated or truncated) and info.get("finished", False):
            laps_done += 1
        if terminated or truncated:
            obs, _ = env.reset()
    return np.array(obs_list), np.array(act_list)


def fit_weighted_linear(obs_matrix, act_matrix, obs_spec) -> WeightedLinearPolicy:
    """Fit steer/accel/brake heads via lstsq. Returns a WeightedLinearPolicy."""
    scales = obs_spec.scales.astype(float)
    norm_obs = obs_matrix / scales[np.newaxis, :]
    w_steer, *_ = np.linalg.lstsq(norm_obs, act_matrix[:, 0], rcond=None)
    w_accel, *_ = np.linalg.lstsq(norm_obs, act_matrix[:, 1], rcond=None)
    w_brake, *_ = np.linalg.lstsq(norm_obs, act_matrix[:, 2], rcond=None)
    n_lidar = sum(1 for d in obs_spec.dims if d.name.startswith("lidar_"))
    return WeightedLinearPolicy.from_cfg(
        {
            "steer_weights": w_steer.tolist(),
            "accel_weights": w_accel.tolist(),
            "brake_weights": w_brake.tolist(),
        },
        n_lidar_rays=n_lidar,
    )


def run(env, experiment_dir: Path, obs_spec) -> None:
    """Collect demos and save pre-trained weights to experiment_dir."""
    logger.info("Collecting demos from SimplePolicy (%d laps)...", N_DEMO_LAPS)
    obs_m, act_m = collect_demos(env, N_DEMO_LAPS)
    policy = fit_weighted_linear(obs_m, act_m, obs_spec)
    weights_file = Path(experiment_dir) / "policy_weights.yaml"
    policy.save(str(weights_file))
    logger.info("Pre-trained weights saved to %s (%d steps)", weights_file, len(obs_m))
