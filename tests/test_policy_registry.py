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
        assert cls.LOOP_TYPE in _VALID_LOOP_TYPES, f"{name}: LOOP_TYPE={cls.LOOP_TYPE!r} not in {_VALID_LOOP_TYPES}"


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
    import numpy as np

    from framework.obs_spec import ObsDim, ObsSpec
    from framework.training import _make_policy

    obs_spec = ObsSpec(
        [
            ObsDim("speed", 50.0, "speed in m/s"),
            ObsDim("offset", 5.0, "lateral offset in m"),
        ]
    )
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


def test_make_policy_unknown_type_raises(tmp_path):
    import numpy as np

    from framework.obs_spec import ObsDim, ObsSpec
    from framework.training import _make_policy

    obs_spec = ObsSpec([ObsDim("a", 1.0, ""), ObsDim("b", 1.0, "")])
    with pytest.raises(ValueError, match="Unknown policy_type"):
        _make_policy(
            policy_type="nope_not_real",
            obs_spec=obs_spec,
            head_names=["steer", "accel", "brake"],
            discrete_actions=np.zeros((9, 3), dtype=np.float32),
            weights_file=str(tmp_path / "w.yaml"),
            policy_params={},
            re_initialize=False,
        )


# ---------------------------------------------------------------------------
# Game/policy compatibility (moved from tests/test_game_adapter.py after the
# SC2 incompatibility check became the class-level compatible_with hook)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_type,expected_hint",
    [
        ("hill_climbing", "sc2_genetic"),
        ("genetic", "sc2_genetic"),
        ("neural_net", "sc2_neural_net"),
    ],
)
def test_sc2_rejects_incompatible_framework_policy(tmp_path, bad_type, expected_hint):
    """Continuous-action framework policies must fail fast against an SC2 game."""
    import numpy as np

    import games.sc2.adapter  # noqa: F401 — registers "sc2" action-encoding incompatibility
    from framework.obs_spec import ObsDim, ObsSpec
    from framework.training import _make_policy

    obs_spec = ObsSpec([ObsDim("a", 1.0, ""), ObsDim("b", 1.0, "")])
    with pytest.raises(ValueError) as exc_info:
        _make_policy(
            policy_type=bad_type,
            obs_spec=obs_spec,
            head_names=["steer", "accel", "brake"],
            discrete_actions=np.zeros((9, 3), dtype=np.float32),
            weights_file=str(tmp_path / "w.yaml"),
            policy_params={},
            re_initialize=False,
            game_name="sc2",
        )
    msg = str(exc_info.value)
    assert bad_type in msg
    assert expected_hint in msg


def test_compatible_framework_policy_allowed_on_non_sc2_game(tmp_path):
    """The same policies are fine on a non-SC2 game."""
    import numpy as np

    from framework.obs_spec import ObsDim, ObsSpec
    from framework.training import _make_policy

    obs_spec = ObsSpec([ObsDim("a", 1.0, ""), ObsDim("b", 1.0, "")])
    policy = _make_policy(
        policy_type="hill_climbing",
        obs_spec=obs_spec,
        head_names=["steer", "accel", "brake"],
        discrete_actions=np.zeros((9, 3), dtype=np.float32),
        weights_file=str(tmp_path / "w.yaml"),
        policy_params={},
        re_initialize=False,
        game_name="tmnf",
    )
    assert isinstance(policy, WeightedLinearPolicy)


def test_default_compatible_with_allows_all():
    assert BasePolicy.compatible_with("anything") == (True, None)


def test_registered_policies_reject_unknown_policy_params():
    """Every registered policy with a non-empty VALID_POLICY_PARAMS rejects a bogus key."""
    bogus = "definitely.not.a.real.param"
    checked = 0
    for name, cls in POLICY_REGISTRY.items():
        if not cls.VALID_POLICY_PARAMS:
            continue
        checked += 1
        with pytest.raises(ValueError, match="no effect"):
            cls._validate_params({bogus: 1})
    assert checked > 0


# ---------------------------------------------------------------------------
# SC2 policy registration + per-type param validation.
#
# These replace the SC2-specific cases that previously lived in
# tests/test_game_adapter.py and exercised the now-deleted
# SC2Adapter.build_extras / _SC2_VALID_POLICY_PARAMS path.  After Phase D the
# SC2 policy types are resolved through POLICY_REGISTRY and validated by their
# own VALID_POLICY_PARAMS, so the registry must actually contain them.
# ---------------------------------------------------------------------------


def _import_all_game_policies() -> None:
    """Side-effect imports populate POLICY_REGISTRY with every game's policies.

    Re-importing is a no-op (modules are cached), so this is safe to call from
    multiple tests without triggering duplicate-registration errors.
    """
    import games.sc2.cnn_policy  # noqa: F401
    import games.sc2.policies  # noqa: F401
    import games.sc2.sc2_policies  # noqa: F401
    import games.tmnf.policies  # noqa: F401


@pytest.mark.parametrize(
    "policy_type,loop_type",
    [
        ("sc2_cnn", "cmaes"),
        ("sc2_neural_net", "hill_climbing"),
        ("sc2_neural_dqn", "q_learning"),
    ],
)
def test_migrated_sc2_policies_registered(policy_type, loop_type):
    """The three SC2 policies migrated in Phase D resolve via the registry."""
    _import_all_game_policies()
    assert policy_type in POLICY_REGISTRY
    assert POLICY_REGISTRY[policy_type].LOOP_TYPE == loop_type


@pytest.mark.parametrize(
    "policy_type",
    [
        "sc2_genetic",
        "sc2_neural_net",
        "sc2_neural_dqn",
        "sc2_cnn",
    ],
)
def test_sc2_native_policies_require_sc2_game(policy_type):
    """SC2-native policy classes must reject non-SC2 game names."""
    _import_all_game_policies()
    cls = POLICY_REGISTRY[policy_type]
    assert cls.compatible_with("sc2") == (True, None)
    ok, hint = cls.compatible_with("tmnf")
    assert ok is False
    assert hint is not None and "SC2-specific" in hint


@pytest.mark.parametrize(
    "policy_type,bad_param",
    [
        ("sc2_genetic", "hidden_sizes"),
        ("sc2_neural_net", "population_size"),
        ("sc2_cmaes", "learning_rate"),
        ("sc2_lstm", "mutation_scale"),
        ("sc2_reinforce", "population_size"),
        ("sc2_neural_dqn", "mutation_scale"),
        ("cmaes", "mutation_scale"),
    ],
)
def test_sc2_policy_rejects_invalid_policy_params(policy_type, bad_param):
    """Per-class VALID_POLICY_PARAMS rejects unknown keys for SC2/TMNF policies."""
    _import_all_game_policies()
    cls = POLICY_REGISTRY[policy_type]
    with pytest.raises(ValueError, match=bad_param):
        cls._validate_params({bad_param: 0.1})


@pytest.mark.parametrize(
    "policy_type,good_params",
    [
        ("sc2_genetic", {"population_size": 10, "elite_k": 3}),
        ("sc2_neural_net", {"hidden_sizes": [16, 64, 64, 16]}),
        ("sc2_cnn", {"population_size": 8, "initial_sigma": 0.02}),
        ("sc2_neural_dqn", {"replay_buffer_size": 50000, "gamma": 0.995}),
    ],
)
def test_sc2_policy_accepts_valid_policy_params(policy_type, good_params):
    """Valid policy_params (and empty) must not raise for SC2 policies."""
    _import_all_game_policies()
    cls = POLICY_REGISTRY[policy_type]
    cls._validate_params(good_params)
    cls._validate_params({})
