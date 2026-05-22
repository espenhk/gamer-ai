"""Rocket League observation space definition.

The observation vector is 142 floats and includes:
- self car state (18 dims)
- ball state (9 dims)
- 2 friendly teammate car states (2 × 18 dims)
- 3 opponent car states (3 × 18 dims)
- relative features/distances (15 dims)
- nearest 10 boost pad availability flags (10 dims)
"""

from __future__ import annotations

import numpy as np

from framework.obs_spec import ObsDim, ObsSpec


def _car_dims(prefix: str, who: str) -> list[ObsDim]:
    return [
        ObsDim(f"{prefix}_pos_x",       4096.0, f"{who} X position (UU)"),
        ObsDim(f"{prefix}_pos_y",       5120.0, f"{who} Y position (UU)"),
        ObsDim(f"{prefix}_pos_z",       2044.0, f"{who} Z position (UU)"),
        ObsDim(f"{prefix}_vel_x",       2300.0, f"{who} X linear velocity (UU/s)"),
        ObsDim(f"{prefix}_vel_y",       2300.0, f"{who} Y linear velocity (UU/s)"),
        ObsDim(f"{prefix}_vel_z",       2300.0, f"{who} Z linear velocity (UU/s)"),
        ObsDim(f"{prefix}_ang_vel_x",   5.5,    f"{who} X angular velocity (rad/s)"),
        ObsDim(f"{prefix}_ang_vel_y",   5.5,    f"{who} Y angular velocity (rad/s)"),
        ObsDim(f"{prefix}_ang_vel_z",   5.5,    f"{who} Z angular velocity (rad/s)"),
        ObsDim(f"{prefix}_forward_x",   1.0,    f"{who} forward unit vector X"),
        ObsDim(f"{prefix}_forward_y",   1.0,    f"{who} forward unit vector Y"),
        ObsDim(f"{prefix}_forward_z",   1.0,    f"{who} forward unit vector Z"),
        ObsDim(f"{prefix}_up_x",        1.0,    f"{who} up unit vector X"),
        ObsDim(f"{prefix}_up_y",        1.0,    f"{who} up unit vector Y"),
        ObsDim(f"{prefix}_up_z",        1.0,    f"{who} up unit vector Z"),
        ObsDim(f"{prefix}_on_ground",   1.0,    f"1.0 if {who.lower()} is on ground"),
        ObsDim(f"{prefix}_has_flip",    1.0,    f"1.0 if {who.lower()} has a flip available"),
        ObsDim(f"{prefix}_boost",       1.0,    f"{who} boost fuel remaining [0, 1]"),
    ]


_RL_DIMS: list[ObsDim] = [
    # --- Self car state (indices 0–17) ---
    ObsDim("car_pos_x",       4096.0,  "Self car X position (UU)"),
    ObsDim("car_pos_y",       5120.0,  "Self car Y position (UU)"),
    ObsDim("car_pos_z",       2044.0,  "Self car Z position (UU)"),
    ObsDim("car_vel_x",       2300.0,  "Self car X linear velocity (UU/s)"),
    ObsDim("car_vel_y",       2300.0,  "Self car Y linear velocity (UU/s)"),
    ObsDim("car_vel_z",       2300.0,  "Self car Z linear velocity (UU/s)"),
    ObsDim("car_ang_vel_x",   5.5,     "Self car X angular velocity (rad/s)"),
    ObsDim("car_ang_vel_y",   5.5,     "Self car Y angular velocity (rad/s)"),
    ObsDim("car_ang_vel_z",   5.5,     "Self car Z angular velocity (rad/s)"),
    ObsDim("car_forward_x",   1.0,     "Self car forward unit vector X"),
    ObsDim("car_forward_y",   1.0,     "Self car forward unit vector Y"),
    ObsDim("car_forward_z",   1.0,     "Self car forward unit vector Z"),
    ObsDim("car_up_x",        1.0,     "Self car up unit vector X"),
    ObsDim("car_up_y",        1.0,     "Self car up unit vector Y"),
    ObsDim("car_up_z",        1.0,     "Self car up unit vector Z"),
    ObsDim("is_on_ground",    1.0,     "1.0 if car is on ground, else 0.0"),
    ObsDim("has_flip",        1.0,     "1.0 if car has a flip available"),
    ObsDim("boost_amount",    1.0,     "Boost fuel remaining [0, 1]"),

    # --- Ball state (indices 18–26) ---
    ObsDim("ball_pos_x",      4096.0,  "Ball X position (UU)"),
    ObsDim("ball_pos_y",      5120.0,  "Ball Y position (UU)"),
    ObsDim("ball_pos_z",      2044.0,  "Ball Z position (UU)"),
    ObsDim("ball_vel_x",      6000.0,  "Ball X linear velocity (UU/s)"),
    ObsDim("ball_vel_y",      6000.0,  "Ball Y linear velocity (UU/s)"),
    ObsDim("ball_vel_z",      6000.0,  "Ball Z linear velocity (UU/s)"),
    ObsDim("ball_ang_vel_x",  6.0,     "Ball X angular velocity (rad/s)"),
    ObsDim("ball_ang_vel_y",  6.0,     "Ball Y angular velocity (rad/s)"),
    ObsDim("ball_ang_vel_z",  6.0,     "Ball Z angular velocity (rad/s)"),

    # --- Friendly teammates (indices 27–62) ---
    *_car_dims("mate1", "Friendly teammate 1"),
    *_car_dims("mate2", "Friendly teammate 2"),

    # --- Opponents (indices 63–116) ---
    *_car_dims("opp1", "Opponent 1"),
    *_car_dims("opp2", "Opponent 2"),
    *_car_dims("opp3", "Opponent 3"),

    # --- Relative features (indices 117–131) ---
    ObsDim("rel_ball_pos_x",  4096.0,  "ball_pos_x − car_pos_x"),
    ObsDim("rel_ball_pos_y",  5120.0,  "ball_pos_y − car_pos_y"),
    ObsDim("rel_ball_pos_z",  2044.0,  "ball_pos_z − car_pos_z"),
    ObsDim("rel_ball_vel_x",  6000.0,  "ball_vel_x − car_vel_x"),
    ObsDim("rel_ball_vel_y",  6000.0,  "ball_vel_y − car_vel_y"),
    ObsDim("rel_ball_vel_z",  6000.0,  "ball_vel_z − car_vel_z"),
    ObsDim("dist_to_ball",    13272.0, "Euclidean distance from car to ball (UU)"),
    ObsDim("vel_towards_ball", 2300.0, "Signed velocity component towards ball (UU/s)"),
    ObsDim("ball_to_opp_goal_dist",  10240.0, "Ball distance to opponent goal (UU)"),
    ObsDim("ball_to_own_goal_dist",  10240.0, "Ball distance to own goal (UU)"),
    ObsDim("car_to_opp_goal_dist",   13272.0, "Car distance to opponent goal (UU)"),
    ObsDim("car_to_own_goal_dist",   13272.0, "Car distance to own goal (UU)"),
    ObsDim("rel_opp1_pos_x",   4096.0, "opp1_pos_x − car_pos_x"),
    ObsDim("rel_opp1_pos_y",   5120.0, "opp1_pos_y − car_pos_y"),
    ObsDim("rel_opp1_pos_z",   2044.0, "opp1_pos_z − car_pos_z"),

    # --- Boost pad availability (indices 132–141, nearest 10 pads) ---
    ObsDim("boost_pad_0",     1.0, "Nearest boost pad 0 availability (0 or 1)"),
    ObsDim("boost_pad_1",     1.0, "Nearest boost pad 1 availability (0 or 1)"),
    ObsDim("boost_pad_2",     1.0, "Nearest boost pad 2 availability (0 or 1)"),
    ObsDim("boost_pad_3",     1.0, "Nearest boost pad 3 availability (0 or 1)"),
    ObsDim("boost_pad_4",     1.0, "Nearest boost pad 4 availability (0 or 1)"),
    ObsDim("boost_pad_5",     1.0, "Nearest boost pad 5 availability (0 or 1)"),
    ObsDim("boost_pad_6",     1.0, "Nearest boost pad 6 availability (0 or 1)"),
    ObsDim("boost_pad_7",     1.0, "Nearest boost pad 7 availability (0 or 1)"),
    ObsDim("boost_pad_8",     1.0, "Nearest boost pad 8 availability (0 or 1)"),
    ObsDim("boost_pad_9",     1.0, "Nearest boost pad 9 availability (0 or 1)"),
]

assert len(_RL_DIMS) == 142, f"Expected 142 dims, got {len(_RL_DIMS)}"

ROCKET_LEAGUE_OBS_SPEC: ObsSpec = ObsSpec(_RL_DIMS)
BASE_OBS_DIM: int = ROCKET_LEAGUE_OBS_SPEC.dim
OBS_NAMES: list[str] = ROCKET_LEAGUE_OBS_SPEC.names
OBS_SCALES: np.ndarray = ROCKET_LEAGUE_OBS_SPEC.scales
OBS_SPEC: list[ObsDim] = _RL_DIMS
