"""Verify framework algorithms produce byte-identical outputs to their TMNF game counterparts.

Each test:
  1. Creates both the framework version and the TMNF game version with identical
     seeds and hyperparameters.
  2. Copies weights from the framework policy into the TMNF policy (or vice versa),
     ensuring they share exactly the same parameter values regardless of
     initialisation order.
  3. Asserts that forward-pass outputs are bit-for-bit identical.

If any test here fails it means the framework port introduced a numerical
discrepancy in the core computation (forward pass, Q-value computation, etc.).
"""
import unittest

import numpy as np

from framework.dqn import DQNPolicy
from framework.lstm import LSTMCore
from framework.reinforce import REINFORCEPolicy
from games.tmnf.actions import DISCRETE_ACTIONS
from games.tmnf.obs_spec import BASE_OBS_DIM, TMNF_OBS_SPEC
from games.tmnf.policies import LSTMPolicy as TmnfLSTMPolicy
from games.tmnf.policies import NeuralDQNPolicy as TmnfDQNPolicy
from games.tmnf.policies import REINFORCEPolicy as TmnfREINFORCEPolicy

_OBS_SPEC  = TMNF_OBS_SPEC
_N         = BASE_OBS_DIM
_N_ACTIONS = len(DISCRETE_ACTIONS)
_ACTION_DEC = lambda i: DISCRETE_ACTIONS[i]  # noqa: E731


def _zero_obs() -> np.ndarray:
    return np.zeros(_N, dtype=np.float32)


def _rand_obs(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(_N).astype(np.float32)


# ---------------------------------------------------------------------------
# DQNPolicy ↔ NeuralDQNPolicy
# ---------------------------------------------------------------------------

class TestDQNEquivalence(unittest.TestCase):
    """Framework DQNPolicy and TMNF NeuralDQNPolicy produce identical Q-values
    when given the same weights and the same normalised observation."""

    def _make_pair(self, hidden_sizes, seed):
        fw = DQNPolicy(
            _OBS_SPEC, DISCRETE_ACTIONS,
            hidden_sizes=hidden_sizes,
            replay_buffer_size=100,
            min_replay_size=10,
            seed=seed,
        )
        tmnf = TmnfDQNPolicy(
            hidden_sizes=hidden_sizes,
            replay_buffer_size=100,
            min_replay_size=10,
            n_lidar_rays=0,
            seed=seed,
        )
        # Copy framework weights into TMNF policy so both share identical parameters.
        for i in range(len(fw._online["weights"])):
            tmnf._online["weights"][i] = fw._online["weights"][i].copy()
            tmnf._online["biases"][i]  = fw._online["biases"][i].copy()
        return fw, tmnf

    def test_q_values_zero_obs(self):
        fw, tmnf = self._make_pair([8, 8], seed=7)
        obs_norm = _zero_obs() / fw._scales
        fw_q   = fw._q_values(fw._online,     obs_norm)
        tmnf_q = tmnf._q_values(tmnf._online, obs_norm)
        np.testing.assert_array_equal(fw_q, tmnf_q)

    def test_q_values_random_obs(self):
        fw, tmnf = self._make_pair([16], seed=3)
        obs_norm = _rand_obs(seed=5) / fw._scales
        fw_q   = fw._q_values(fw._online,     obs_norm)
        tmnf_q = tmnf._q_values(tmnf._online, obs_norm)
        np.testing.assert_array_equal(fw_q, tmnf_q)

    def test_q_values_shape(self):
        fw, _ = self._make_pair([8], seed=0)
        obs_norm = _zero_obs() / fw._scales
        q = fw._q_values(fw._online, obs_norm)
        self.assertEqual(q.shape, (_N_ACTIONS,))

    def test_greedy_action_matches(self):
        """Greedy action index must agree when both networks share weights."""
        fw, tmnf = self._make_pair([8], seed=11)
        obs = _rand_obs(seed=22)
        fw._eps   = 0.0
        tmnf._eps = 0.0
        act_fw   = fw(obs)
        act_tmnf = tmnf(obs)
        np.testing.assert_array_equal(act_fw, act_tmnf)

    def test_same_seed_produces_same_initial_weights(self):
        """Both policies seeded identically should initialise to identical weights."""
        seed = 42
        fw   = DQNPolicy(_OBS_SPEC, DISCRETE_ACTIONS, hidden_sizes=[8], seed=seed)
        tmnf = TmnfDQNPolicy(hidden_sizes=[8], n_lidar_rays=0, seed=seed)
        for i in range(len(fw._online["weights"])):
            np.testing.assert_array_equal(fw._online["weights"][i],
                                          tmnf._online["weights"][i])
            np.testing.assert_array_equal(fw._online["biases"][i],
                                          tmnf._online["biases"][i])


# ---------------------------------------------------------------------------
# REINFORCEPolicy ↔ REINFORCEPolicy (TMNF)
# ---------------------------------------------------------------------------

class TestREINFORCEEquivalence(unittest.TestCase):
    """Framework REINFORCEPolicy and TMNF REINFORCEPolicy produce identical
    softmax distributions when given the same weights and the same observation."""

    def _make_pair(self, hidden_sizes, seed):
        fw = REINFORCEPolicy(
            _OBS_SPEC, _ACTION_DEC,
            output_dim=_N_ACTIONS,
            hidden_sizes=hidden_sizes,
            entropy_coeff=0.0,
            seed=seed,
        )
        tmnf = TmnfREINFORCEPolicy(
            hidden_sizes=hidden_sizes,
            n_lidar_rays=0,
            entropy_coeff=0.0,
            seed=seed,
        )
        # Copy framework weights into TMNF policy.
        for i in range(len(fw._weights)):
            tmnf._weights[i] = fw._weights[i].copy()
            tmnf._biases[i]  = fw._biases[i].copy()
        return fw, tmnf

    def _fw_probs(self, fw, obs):
        probs, _, _ = fw._forward(obs / fw._scales)
        return probs

    def _tmnf_probs(self, tmnf, obs):
        probs, _, _ = tmnf._forward(obs / tmnf._scales)
        return probs

    def test_softmax_probs_zero_obs(self):
        fw, tmnf = self._make_pair([8, 8], seed=5)
        obs = _zero_obs()
        np.testing.assert_array_equal(
            self._fw_probs(fw, obs),
            self._tmnf_probs(tmnf, obs),
        )

    def test_softmax_probs_random_obs(self):
        fw, tmnf = self._make_pair([16], seed=9)
        obs = _rand_obs(seed=3)
        np.testing.assert_array_equal(
            self._fw_probs(fw, obs),
            self._tmnf_probs(tmnf, obs),
        )

    def test_probs_sum_to_one(self):
        fw, _ = self._make_pair([8], seed=0)
        probs = self._fw_probs(fw, _rand_obs())
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_same_seed_produces_same_initial_weights(self):
        """Both policies seeded identically should initialise to identical weights."""
        seed = 13
        fw   = REINFORCEPolicy(_OBS_SPEC, _ACTION_DEC, output_dim=_N_ACTIONS,
                               hidden_sizes=[8], seed=seed)
        tmnf = TmnfREINFORCEPolicy(hidden_sizes=[8], n_lidar_rays=0, seed=seed)
        for i in range(len(fw._weights)):
            np.testing.assert_array_equal(fw._weights[i], tmnf._weights[i])
            np.testing.assert_array_equal(fw._biases[i],  tmnf._biases[i])


# ---------------------------------------------------------------------------
# LSTMCore ↔ LSTMPolicy (TMNF)
# ---------------------------------------------------------------------------

class TestLSTMEquivalence(unittest.TestCase):
    """Framework LSTMCore and TMNF LSTMPolicy produce identical hidden states
    and output actions when given the same weights and the same observations."""

    def _make_pair(self, hidden_size, seed):
        fw   = LSTMCore(_OBS_SPEC, hidden_size=hidden_size, seed=seed)
        tmnf = TmnfLSTMPolicy(hidden_size=hidden_size, n_lidar_rays=0, seed=seed)
        # Copy framework weights into TMNF policy.
        for attr in ("_W_f", "_b_f", "_W_i", "_b_i",
                     "_W_g", "_b_g", "_W_o", "_b_o",
                     "_W_steer", "_W_accel", "_W_brake"):
            setattr(tmnf, attr, getattr(fw, attr).copy())
        return fw, tmnf

    def test_hidden_state_identical_after_one_step(self):
        fw, tmnf = self._make_pair(8, seed=0)
        obs = _rand_obs(seed=1)
        fw(obs)
        tmnf(obs)
        np.testing.assert_array_equal(fw._h, tmnf._h)
        np.testing.assert_array_equal(fw._c, tmnf._c)

    def test_action_identical_after_one_step(self):
        fw, tmnf = self._make_pair(8, seed=2)
        obs = _rand_obs(seed=4)
        act_fw   = fw(obs)
        act_tmnf = tmnf(obs)
        np.testing.assert_array_equal(act_fw, act_tmnf)

    def test_hidden_state_identical_after_sequence(self):
        fw, tmnf = self._make_pair(16, seed=3)
        for i in range(10):
            obs = _rand_obs(seed=i)
            fw(obs)
            tmnf(obs)
        np.testing.assert_array_equal(fw._h, tmnf._h)
        np.testing.assert_array_equal(fw._c, tmnf._c)

    def test_episode_reset_in_sync(self):
        fw, tmnf = self._make_pair(8, seed=5)
        for _ in range(5):
            fw(_rand_obs(seed=0))
            tmnf(_rand_obs(seed=0))
        fw.on_episode_end()
        tmnf.on_episode_end()
        np.testing.assert_array_equal(fw._h, tmnf._h)
        np.testing.assert_array_equal(fw._c, tmnf._c)

    def test_flat_roundtrip_produces_same_output(self):
        """to_flat() / with_flat() must preserve weights to float32 precision."""
        fw = LSTMCore(_OBS_SPEC, hidden_size=8, seed=6)
        flat = fw.to_flat()
        fw2  = fw.with_flat(flat)
        obs  = _rand_obs(seed=7)
        np.testing.assert_array_almost_equal(fw(obs), fw2(obs), decimal=5)

    def test_same_seed_produces_same_initial_weights(self):
        """Both policies seeded identically should initialise to identical weights."""
        seed = 17
        fw   = LSTMCore(_OBS_SPEC, hidden_size=8, seed=seed)
        tmnf = TmnfLSTMPolicy(hidden_size=8, n_lidar_rays=0, seed=seed)
        for attr in ("_W_f", "_b_f", "_W_i", "_b_i",
                     "_W_g", "_b_g", "_W_o", "_b_o",
                     "_W_steer", "_W_accel", "_W_brake"):
            np.testing.assert_array_equal(
                getattr(fw, attr), getattr(tmnf, attr),
                err_msg=f"{attr} mismatch between framework and TMNF",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
