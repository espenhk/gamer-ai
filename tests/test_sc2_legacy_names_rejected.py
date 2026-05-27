"""Regression tests for removed SC2 bare-name policy types."""

from __future__ import annotations

import numpy as np
import pytest

from framework.training import _make_policy
from games.sc2.actions import DISCRETE_ACTIONS
from games.sc2.obs_spec import SC2_MINIGAME_OBS_SPEC


def _import_policy_modules() -> None:
    import games.sc2.adapter  # noqa: F401 — registers "sc2" action-encoding incompatibility
    import games.sc2.cnn_policy  # noqa: F401
    import games.sc2.sc2_policies  # noqa: F401
    import games.tmnf.policies  # noqa: F401


@pytest.mark.parametrize(
    ("legacy_name", "expected_sc2_name"),
    [
        ("cmaes", "sc2_cmaes"),
        ("reinforce", "sc2_reinforce"),
        ("lstm", "sc2_lstm"),
        ("neural_dqn", "sc2_neural_dqn"),
    ],
)
def test_sc2_bare_name_rejected(tmp_path, legacy_name: str, expected_sc2_name: str):
    """TMNF bare-name policy types should fail with an incompatibility error for SC2."""
    _import_policy_modules()
    with pytest.raises(ValueError) as exc:
        _make_policy(
            legacy_name,
            obs_spec=SC2_MINIGAME_OBS_SPEC,
            head_names=["fn_idx", "x", "y", "queue"],
            discrete_actions=np.array(DISCRETE_ACTIONS, dtype=np.float32),
            weights_file=str(tmp_path / "weights.yaml"),
            policy_params={},
            re_initialize=True,
            game_name="sc2",
        )
    msg = str(exc.value)
    assert "not compatible with game" in msg
    assert expected_sc2_name in msg
