"""
Shared test helpers for tmnf tests.

conftest.py ensures tmnf/ and tmnf/tests/ are on sys.path before this
module is imported, so the source imports below resolve correctly.
"""
import numpy as np

from policies import WeightedLinearPolicy  # backward-compat shim → framework.policies
from games.tmnf.state import StateData


def make_game_state(
    position=(0.0, 0.0, 0.0),
    linear_speed=(10.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    angular_speed=(0.0, 0.0, 0.0),
    gear=3,
    turning_rate=0.0,
    wheel_contacts=(True, True, True, True),
    wheel_sliding=(False, False, False, False),
):
    """Build a plain Python object matching StateData's nested attribute expectations."""
    class _CS:
        pass
    class _Dyna:
        current_state = _CS()
    class _Engine:
        pass
    class _Mobil:
        engine = _Engine()
    class _State:
        pass

    s = _State()
    s.dyna = _Dyna()
    s.dyna.current_state.position = position
    # StateData: dyna = state.dyna.current_state → dyna.linear_speed, .quat, .angular_speed
    s.dyna.current_state.linear_speed = linear_speed
    s.dyna.current_state.quat = quat
    s.dyna.current_state.angular_speed = angular_speed
    s.scene_mobil = _Mobil()
    s.scene_mobil.engine.gear = gear
    s.scene_mobil.turning_rate = turning_rate

    def _wheel(contact, sliding):
        rts = type("_RTS", (), {"has_ground_contact": contact, "is_sliding": sliding})()
        return type("_Wheel", (), {"real_time_state": rts})()

    s.simulation_wheels = [_wheel(wheel_contacts[i], wheel_sliding[i]) for i in range(4)]
    return s


def make_state_data(
    position=(0.0, 0.0, 0.0),
    speed=(10.0, 0.0, 0.0),
    track_progress=0.5,
    lateral_offset=0.0,
    vertical_offset=0.0,
    wheel_contacts=(True, True, True, True),
):
    """Build a StateData with track fields pre-filled (bypasses real centerline projection)."""
    gs = make_game_state(position=position, linear_speed=speed, wheel_contacts=wheel_contacts)
    sd = StateData(gs)
    sd.track_progress = track_progress
    sd.lateral_offset = lateral_offset
    sd.vertical_offset = vertical_offset
    return sd


def zero_obs(n: int = 15) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def make_wlp(steer_weights=None, throttle_weights=None,
             steer_threshold=0.5, throttle_threshold=0.5):
    """Construct a WeightedLinearPolicy directly from weight vectors (no YAML file)."""
    names = WeightedLinearPolicy.OBS_NAMES
    n = len(names)
    sw = steer_weights    if steer_weights    is not None else np.zeros(n, dtype=np.float32)
    tw = throttle_weights if throttle_weights is not None else np.zeros(n, dtype=np.float32)
    cfg = {
        "steer_threshold":    steer_threshold,
        "throttle_threshold": throttle_threshold,
        "steer_weights": {names[i]: float(sw[i]) for i in range(n)},
        "accel_weights": {names[i]: float(tw[i]) for i in range(n)},
        "brake_weights": {names[i]: float(-tw[i]) for i in range(n)},
    }
    return WeightedLinearPolicy.from_cfg(cfg)
