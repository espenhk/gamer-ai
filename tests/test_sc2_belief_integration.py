"""Tests for the fog-of-war belief system wired into SC2Env (issue #111).

All tests mock SC2Client so no StarCraft 2 binary is required.
"""
import unittest
from unittest.mock import patch

import numpy as np

from games.sc2.belief_schema import belief_obs_dims
from games.sc2.env import SC2Env
from games.sc2.obs_spec import BASE_OBS_DIM, LADDER_OBS_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BELIEF_EXTRA_DIMS = len(belief_obs_dims())  # 192 for default 8×8 grid

# A 64×64 all-visible minimap (visibility_map value 2 = fully visible).
_FULL_VIS = np.full((64, 64), 2, dtype=np.float32)
# A 64×64 all-hidden minimap.
_NO_VIS = np.zeros((64, 64), dtype=np.float32)


def _make_belief_env(map_name: str = "MoveToBeacon", minimap_vis=None) -> SC2Env:
    """Return a SC2Env with enable_belief=True and a mocked client.

    If *minimap_vis* is provided it is returned in every step's info dict,
    simulating the client exposing the raw minimap visibility layer.
    """
    obs_dim = BASE_OBS_DIM if map_name == "MoveToBeacon" else LADDER_OBS_DIM

    with patch("games.sc2.env.SC2Client") as mock_cls:
        info_base = {
            "score": 0.0, "minerals": 50.0, "vespene": 0.0,
            "food_used": 1.0, "food_cap": 15.0, "army_count": 0.0,
            "game_loop": 0.0,
        }
        if minimap_vis is not None:
            info_base["minimap_vis"] = minimap_vis

        mock_cls.return_value.reset.return_value = (
            np.zeros(obs_dim, dtype=np.float32),
            dict(info_base),
        )
        mock_cls.return_value.step.return_value = (
            np.zeros(obs_dim, dtype=np.float32),
            0.0,
            False,
            dict(info_base, score=1.0, game_loop=8.0),
        )
        env = SC2Env(map_name=map_name, enable_belief=True)
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBeliefObsShape(unittest.TestCase):
    """Observation space and actual obs shape grow by BELIEF_EXTRA_DIMS."""

    def test_observation_space_shape_minigame(self):
        with patch("games.sc2.env.SC2Client"):
            env = SC2Env(map_name="MoveToBeacon", enable_belief=True)
        self.assertEqual(
            env.observation_space.shape,
            (BASE_OBS_DIM + _BELIEF_EXTRA_DIMS,),
        )

    def test_observation_space_shape_ladder(self):
        with patch("games.sc2.env.SC2Client"):
            env = SC2Env(map_name="Simple64", enable_belief=True)
        self.assertEqual(
            env.observation_space.shape,
            (LADDER_OBS_DIM + _BELIEF_EXTRA_DIMS,),
        )

    def test_belief_disabled_shape_unchanged(self):
        with patch("games.sc2.env.SC2Client"):
            env = SC2Env(map_name="MoveToBeacon", enable_belief=False)
        self.assertEqual(env.observation_space.shape, (BASE_OBS_DIM,))

    def test_reset_obs_has_extended_shape(self):
        env = _make_belief_env()
        with patch("games.sc2.env.SC2Client", env._client.__class__):
            pass
        # Patch the client directly on the already-constructed env.
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (BASE_OBS_DIM + _BELIEF_EXTRA_DIMS,))

    def test_step_obs_has_extended_shape(self):
        env = _make_belief_env()
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        self.assertEqual(obs.shape, (BASE_OBS_DIM + _BELIEF_EXTRA_DIMS,))


class TestBeliefResetState(unittest.TestCase):
    """After reset(), belief enc is all-zero; staleness is all-ones."""

    def test_reset_belief_enc_all_zero(self):
        env = _make_belief_env()
        obs, _ = env.reset()
        # belief enc occupies dims [BASE_OBS_DIM : BASE_OBS_DIM + 2*64]
        n_slots = 64
        belief_enc = obs[BASE_OBS_DIM: BASE_OBS_DIM + 2 * n_slots]
        np.testing.assert_array_equal(belief_enc, np.zeros(2 * n_slots, dtype=np.float32))

    def test_reset_staleness_all_ones(self):
        env = _make_belief_env()
        obs, _ = env.reset()
        n_slots = 64
        staleness = obs[BASE_OBS_DIM + 2 * n_slots:]
        np.testing.assert_array_almost_equal(staleness, np.ones(n_slots, dtype=np.float32))

    def test_second_reset_clears_belief(self):
        """Belief state accumulated during an episode is cleared on re-reset."""
        env = _make_belief_env(minimap_vis=_FULL_VIS)
        env.reset()
        # Step once so belief absorbs visible data.
        env.step(np.zeros(4, dtype=np.float32))
        # Reset again — belief enc should be all zeros again.
        obs, _ = env.reset()
        n_slots = 64
        belief_enc = obs[BASE_OBS_DIM: BASE_OBS_DIM + 2 * n_slots]
        np.testing.assert_array_equal(belief_enc, np.zeros(2 * n_slots, dtype=np.float32))


class TestBeliefStepAppendsObs(unittest.TestCase):
    """Step appends belief + staleness dims regardless of minimap_vis presence."""

    def test_step_without_minimap_vis_appends_zeros(self):
        env = _make_belief_env(minimap_vis=None)
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        self.assertEqual(obs.shape, (BASE_OBS_DIM + _BELIEF_EXTRA_DIMS,))

    def test_step_with_minimap_vis_enc_nonzero(self):
        """After a step with a fully-visible minimap, belief enc must be non-zero."""
        env = _make_belief_env(minimap_vis=_FULL_VIS)
        env.reset()
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        n_slots = 64
        belief_enc = obs[BASE_OBS_DIM: BASE_OBS_DIM + 2 * n_slots]
        # At least the confidence slots should be 1.0 for visible regions.
        self.assertTrue(np.any(belief_enc > 0.0))

    def test_step_staleness_drops_after_visible(self):
        """Staleness should decrease for regions that became visible this step."""
        env = _make_belief_env(minimap_vis=_FULL_VIS)
        env.reset()
        obs_after, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        n_slots = 64
        staleness = obs_after[BASE_OBS_DIM + 2 * n_slots:]
        # After seeing all regions, every staleness should be < 1.
        self.assertTrue(np.all(staleness < 1.0))


class TestBeliefScoutReward(unittest.TestCase):
    """Scouting reward is added to step reward when unscouted regions are seen."""

    def test_scout_reward_positive_on_first_visit(self):
        """First time all regions are scouted → scout reward > 0."""
        env = _make_belief_env(minimap_vis=_FULL_VIS)
        env.reset()
        _, reward, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        # scout_drive_weight=0.1 (default) × reward for first discovery > 0
        scout_component = info["episode_reward_components"].get("scout", 0.0)
        self.assertGreater(scout_component, 0.0)

    def test_scout_reward_zero_when_no_vis(self):
        """No visible regions → no staleness-driven reward."""
        env = _make_belief_env(minimap_vis=_NO_VIS)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        scout_component = info["episode_reward_components"].get("scout", 0.0)
        self.assertEqual(scout_component, 0.0)

    def test_scout_component_in_episode_reward_components(self):
        """episode_reward_components always contains the 'scout' key when belief enabled."""
        env = _make_belief_env(minimap_vis=_FULL_VIS)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertIn("scout", info["episode_reward_components"])

    def test_scout_reward_accumulates_across_steps(self):
        """scout component in episode_reward_components grows after a visible step."""
        env = _make_belief_env(minimap_vis=_FULL_VIS)
        env.reset()
        _, _, _, _, info1 = env.step(np.zeros(4, dtype=np.float32))
        _, _, _, _, info2 = env.step(np.zeros(4, dtype=np.float32))
        # First step scouted first-time; both should be non-negative.
        self.assertGreaterEqual(
            info2["episode_reward_components"].get("scout", 0.0),
            info1["episode_reward_components"].get("scout", 0.0),
        )


if __name__ == "__main__":
    unittest.main()
