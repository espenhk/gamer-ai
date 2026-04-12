"""TMNF-specific hand-coded PD baseline policy."""
from __future__ import annotations

import numpy as np


class SimplePolicy:
    """
    Hardcoded PD+heading policy mirroring AdaptiveClient's steering formula.

    Maps obs[1] (lateral offset) and obs[3] (yaw error) to a continuous
    steering value; always accelerates without braking.

    The D term approximates lateral velocity as Δlateral_offset per tick.
    """

    LATERAL_GAIN    = 16.0   # P: steer per metre off-centre (normalised)
    DERIVATIVE_GAIN =  8.0   # D: steer per m/tick of lateral drift
    HEADING_GAIN    =  5.0   # steer per radian of heading error

    def __init__(self) -> None:
        self._prev_lateral = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        lateral = obs[1]
        yaw     = obs[3]

        lateral_vel        = lateral - self._prev_lateral
        self._prev_lateral = lateral

        # Compute a signed steer value in units of "steer%" ÷ 100
        # (LATERAL_GAIN is tuned for the old steer_pct range, so divide by 100)
        steer_pct = (
            -lateral     * self.LATERAL_GAIN
            - lateral_vel  * self.DERIVATIVE_GAIN
            + yaw          * self.HEADING_GAIN
        )
        steer = float(np.clip(steer_pct / 100.0, -1.0, 1.0))
        return np.array([steer, 1.0, 0.0], dtype=np.float32)  # always accelerate
