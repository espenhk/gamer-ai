"""Convergence regression smoke test (issue #484).

The rest of the suite thoroughly covers math/config/save-load correctness but
never checks that ``train_rl()`` actually improves a policy end-to-end
against a real environment. CHANGELOG issue #157 ("Four root causes of poor
SC2 genetic policy improvement") showed that kind of regression can ship and
go unnoticed without an automated signal.

This module runs a short tabular Q-learning (``epsilon_greedy``) session
against the real CarRacing-v3 Box2D env for a small, fixed episode budget and
asserts the best episode reward seen crosses a conservative threshold — well
below CarRacing's published "solved" benchmark (avg reward >= 900 over 100
episodes), just enough to prove the reward signal is actually being learned
from rather than the policy sitting at (or below) a "never accelerate"
floor.

``epsilon_greedy`` was chosen over a from-scratch continuous-weight search
(``hill_climbing`` / ``genetic`` / ``cmaes``) because CarRacing's action
space includes a self-referential ``brake`` observation feature: independent
random-initialised ``accel``/``brake`` linear heads tend to lock into an
"always braking" attractor that small-budget evolutionary search cannot
reliably escape, whereas discrete Q-learning selects directly from
``DISCRETE_ACTIONS`` (one of which is plain "accelerate straight") and
reliably learns to prefer it within a few dozen episodes.

CarRacing's observation now includes track-relative perception features
(lateral offset, heading error, progress — see games/car_racing/obs_spec.py)
in addition to car-physics features. That is real signal ``epsilon_greedy``
needs to converge at all (a policy that cannot see the track cannot learn to
follow it), but the *full* production observation also carries a lookahead
schedule that pushes the tabular state space (``n_bins ** obs_dim``) well
past what a few dozen episodes can visit — a known limitation of tabular
methods at higher dimensionality (see the policy-selection guidance in
CLAUDE.md). This test therefore builds a lookahead-free, coarser-binned
ObsSpec (``build_car_racing_obs_spec_from_steps([])``, ``n_bins=2`` — 12
dims, 4096 states) that still includes the core track-perception features,
keeping the table tractable while still exercising real perception signal.
Empirically this reliably scores in the tens-to-hundreds within 45 episodes
once epsilon decays enough to exploit, while a "never learns" control (ε
permanently pinned at 1.0, i.e. pure random action selection) tops out
around -18. The threshold below (``0.0``) sits with a wide margin on both
sides of that split.

Marked ``integration`` (like the rest of ``tests/integration/``) so it is
excluded from the fast unit-test suite and only runs where
``gymnasium[box2d]`` is installed — the ``car-racing`` job in
``.github/workflows/integration-tests.yml``.
"""

from __future__ import annotations

import tempfile
import time
import unittest

import pytest

try:
    import gymnasium as gym  # noqa: F401
    from gymnasium.envs.box2d import CarRacing  # noqa: F401

    _BOX2D_AVAILABLE = True
except ImportError:
    _BOX2D_AVAILABLE = False

pytestmark = pytest.mark.integration

_skip_no_box2d = pytest.mark.skipif(
    not _BOX2D_AVAILABLE,
    reason="gymnasium[box2d] not installed",
)

# Best reward observed over the run must clear this to pass. Empirically,
# learning runs land in the tens-to-hundreds once epsilon decays enough to
# exploit; a never-learns control (epsilon permanently 1.0) tops out around
# -18. See module docstring for the full margin analysis.
_BEST_REWARD_THRESHOLD = 0.0
_N_EPISODES = 45
_MAX_EPISODE_STEPS = 200


@_skip_no_box2d
class TestCarRacingConvergenceSmoke(unittest.TestCase):
    """A short real training run must actually improve, not just idle."""

    def test_epsilon_greedy_crosses_reward_threshold(self):
        from framework.policies import EpsilonGreedyPolicy
        from framework.training import _greedy_loop_q_learning
        from games.car_racing.actions import DISCRETE_ACTIONS
        from games.car_racing.env import CarRacingEnv
        from games.car_racing.obs_spec import build_car_racing_obs_spec_from_steps

        # Lookahead-free ObsSpec (12 dims: car-physics + lateral_offset_m /
        # yaw_error_rad / track_progress) so n_bins=2 keeps the tabular state
        # space (4096 states) visitable within _N_EPISODES. See module
        # docstring for why the full production ObsSpec isn't tractable here.
        obs_spec = build_car_racing_obs_spec_from_steps([])

        env = CarRacingEnv(max_episode_steps=_MAX_EPISODE_STEPS, lookahead_steps=[])
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                weights_file = f"{tmpdir}/policy_weights.yaml"
                policy = EpsilonGreedyPolicy(
                    obs_spec=obs_spec,
                    discrete_actions=DISCRETE_ACTIONS,
                    n_bins=2,
                    epsilon=1.0,
                    epsilon_decay=0.88,
                    epsilon_min=0.05,
                    alpha=0.3,
                    gamma=0.95,
                )
                loop = _greedy_loop_q_learning(
                    env,
                    policy,
                    n_episodes=_N_EPISODES,
                    weights_file=weights_file,
                )
        finally:
            env.close()

        elapsed = time.time() - t0
        self.assertEqual(len(loop.greedy_sims), _N_EPISODES)
        self.assertGreater(
            loop.best_reward,
            _BEST_REWARD_THRESHOLD,
            f"Best reward over {_N_EPISODES} episodes ({loop.best_reward:.1f}) did not "
            f"clear the convergence-smoke threshold ({_BEST_REWARD_THRESHOLD}) — the "
            "training loop may not be learning from the reward signal. "
            f"(ran in {elapsed:.1f}s)",
        )


if __name__ == "__main__":
    unittest.main()
