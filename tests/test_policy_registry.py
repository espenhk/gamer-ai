"""Tests for the POLICY_REGISTRY and BasePolicy registry-related machinery."""
import pytest

from framework.policies import (
    POLICY_REGISTRY,
    BasePolicy,
    EpsilonGreedyPolicy,
    GeneticPolicy,
    MCTSPolicy,
    NeuralNetPolicy,
    WeightedLinearPolicy,
    register_policy,
)


_VALID_LOOP_TYPES = {"hill_climbing", "q_learning", "cmaes", "genetic"}
_EXPECTED_BUILT_INS = {"hill_climbing", "neural_net", "epsilon_greedy", "mcts", "genetic"}


def test_all_five_built_ins_registered():
    assert _EXPECTED_BUILT_INS <= set(POLICY_REGISTRY)


def test_registered_loop_types_are_valid():
    for name, cls in POLICY_REGISTRY.items():
        assert cls.LOOP_TYPE in _VALID_LOOP_TYPES, (
            f"{name}: LOOP_TYPE={cls.LOOP_TYPE!r} not in {_VALID_LOOP_TYPES}"
        )


def test_built_in_policy_types():
    assert POLICY_REGISTRY["hill_climbing"] is WeightedLinearPolicy
    assert POLICY_REGISTRY["neural_net"] is NeuralNetPolicy
    assert POLICY_REGISTRY["epsilon_greedy"] is EpsilonGreedyPolicy
    assert POLICY_REGISTRY["mcts"] is MCTSPolicy
    assert POLICY_REGISTRY["genetic"] is GeneticPolicy


def test_register_policy_raises_on_duplicate():
    class _Dup(BasePolicy):
        POLICY_TYPE = "hill_climbing"
        def __call__(self, obs): ...
        def to_cfg(self): ...

    with pytest.raises(ValueError, match="Duplicate policy_type"):
        register_policy(_Dup)


def test_register_policy_raises_on_empty_policy_type():
    class _Empty(BasePolicy):
        POLICY_TYPE = ""
        def __call__(self, obs): ...
        def to_cfg(self): ...

    with pytest.raises(ValueError, match="must set POLICY_TYPE"):
        register_policy(_Empty)


def test_validate_params_raises_on_unknown_key():
    class _ValidatedPolicy(BasePolicy):
        POLICY_TYPE = "_test_validated"
        VALID_POLICY_PARAMS = frozenset({"lr", "gamma"})
        def __call__(self, obs): ...
        def to_cfg(self): ...

    with pytest.raises(ValueError, match="no effect"):
        _ValidatedPolicy._validate_params({"lr": 0.01, "unknown_key": 99})


def test_validate_params_noop_on_empty_valid_set():
    class _UnvalidatedPolicy(BasePolicy):
        POLICY_TYPE = "_test_unvalidated"
        VALID_POLICY_PARAMS = frozenset()
        def __call__(self, obs): ...
        def to_cfg(self): ...

    _UnvalidatedPolicy._validate_params({"any_key": 1, "another": 2})


def test_validate_params_accepts_valid_keys():
    class _StrictPolicy(BasePolicy):
        POLICY_TYPE = "_test_strict"
        VALID_POLICY_PARAMS = frozenset({"lr", "gamma"})
        def __call__(self, obs): ...
        def to_cfg(self): ...

    _StrictPolicy._validate_params({"lr": 0.001})
    _StrictPolicy._validate_params({})


def test_make_policy_uses_registry_for_hill_climbing(tmp_path):
    """_make_policy('hill_climbing', ...) returns a WeightedLinearPolicy via registry."""
    from framework.training import _make_policy
    from framework.obs_spec import ObsSpec, ObsDim
    import numpy as np

    obs_spec = ObsSpec([
        ObsDim("speed", 50.0, "speed in m/s"),
        ObsDim("offset", 5.0, "lateral offset in m"),
    ])
    discrete_actions = np.zeros((9, 3), dtype=np.float32)
    weights_file = str(tmp_path / "weights.yaml")

    policy = _make_policy(
        policy_type="hill_climbing",
        obs_spec=obs_spec,
        head_names=["steer", "accel", "brake"],
        discrete_actions=discrete_actions,
        weights_file=weights_file,
        policy_params={},
        re_initialize=False,
    )
    assert isinstance(policy, WeightedLinearPolicy)
