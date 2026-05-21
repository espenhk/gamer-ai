import numpy as np

from framework.training import _run_episode


class _DummyMultiAgentEnv:
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32))
        next_obs = np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
        return next_obs, 1.0, True, False, {}


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.updates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def on_episode_start(self, **kwargs) -> None:
        return

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.calls.append(np.asarray(obs, dtype=np.float32))
        return np.array([float(obs[0]), 1.0, 0.0], dtype=np.float32)

    def update(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        self.updates.append(
            (
                np.asarray(obs, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                np.asarray(next_obs, dtype=np.float32),
            )
        )


def test_run_episode_multi_agent_uses_shared_policy_per_agent_obs():
    env = _DummyMultiAgentEnv()
    policy = _RecordingPolicy()
    obs = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    total_reward, info, throttle_counts, total_steps, _trace = _run_episode(
        env=env,
        policy=policy,
        obs=obs,
    )

    assert total_reward == 1.0
    assert info == {}
    assert throttle_counts == [0, 0, 1]
    assert total_steps == 1

    assert len(policy.calls) == 2
    np.testing.assert_array_equal(policy.calls[0], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(policy.calls[1], np.array([2.0, 0.0], dtype=np.float32))

    assert len(env.actions) == 1
    assert env.actions[0].shape == (2, 3)
    np.testing.assert_array_equal(
        env.actions[0],
        np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32),
    )

    assert len(policy.updates) == 2
    np.testing.assert_array_equal(policy.updates[0][0], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(policy.updates[1][0], np.array([2.0, 0.0], dtype=np.float32))
