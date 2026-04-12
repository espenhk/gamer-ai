"""Shared PD+heading steering controller.

Used by both AdaptiveClient (game-thread real-time control) and SimplePolicy
(observation-based policy that mirrors the same formula).  Having one class
means tuning the gains in one place propagates everywhere.
"""

import math


class PDHeadingController:
    """
    Three-term steering controller:

    P — position error (LATERAL_GAIN)
        Pulls the car back to center proportional to how far off it is.

    D — lateral velocity (DERIVATIVE_GAIN)
        Opposes sideways motion; prevents overshoot when correcting.

    Heading (HEADING_GAIN)
        Aligns the car's nose with the track forward direction.
        Makes the car anticipate curves rather than react to drift.

    Tuning guide — symptom → cause:
        Car corrects slowly on straights                   → P too low
        Smooth sine-wave oscillation, consistent amplitude → P too high
        Oscillation with growing amplitude                 → D too low
        Car sluggishly resists its own correction          → D too high
        Cuts corners / overshoots turns                    → Heading too low
        Constant twitching even when centred and aligned   → Heading too high
    """

    LATERAL_GAIN:    float = 16.0   # steer% per metre off-centre (P term)
    DERIVATIVE_GAIN: float =  8.0   # steer% per m/s lateral velocity (D term)
    HEADING_GAIN:    float =  5.0   # steer% per radian of heading error

    def compute_steer(self, lateral: float, lateral_vel: float, yaw_error: float) -> float:
        """Return a steer percentage in [-100, 100]."""
        return (
            -lateral     * self.LATERAL_GAIN
            - lateral_vel  * self.DERIVATIVE_GAIN
            + yaw_error    * self.HEADING_GAIN
        )


def angle_diff(target: float, current: float) -> float:
    """Signed angular difference target − current, wrapped to [−π, π]."""
    return (target - current + math.pi) % (2 * math.pi) - math.pi
