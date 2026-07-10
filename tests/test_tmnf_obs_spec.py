"""Tests for configurable TMNF lookahead observation (issue #493)."""

from unittest.mock import MagicMock

import numpy as np

from games.tmnf.obs_spec import (
    LOOKAHEAD_STEPS,
    TMNF_OBS_SPEC,
    build_lookahead_steps,
    build_tmnf_obs_spec,
    build_tmnf_obs_spec_from_steps,
)
from games.tmnf.state import StateData
from tests.helpers import make_game_state


class TestBuildLookaheadSteps:
    def test_defaults_to_legacy_list(self):
        assert build_lookahead_steps() == [10, 25, 50]
        assert build_lookahead_steps() == LOOKAHEAD_STEPS

    def test_n_points_only_uses_default_spacing(self):
        assert build_lookahead_steps(n_lookahead_points=5) == [20, 40, 60, 80, 100]

    def test_spacing_only_uses_default_n(self):
        assert build_lookahead_steps(lookahead_step_spacing=10) == [10, 20, 30]

    def test_both_args(self):
        assert build_lookahead_steps(n_lookahead_points=4, lookahead_step_spacing=15) == [15, 30, 45, 60]

    def test_single_point(self):
        assert build_lookahead_steps(n_lookahead_points=1, lookahead_step_spacing=100) == [100]


class TestBuildTmnfObsSpec:
    def test_default_matches_legacy_module_spec(self):
        spec = build_tmnf_obs_spec()
        assert spec.dim == TMNF_OBS_SPEC.dim == 21
        assert spec.names == TMNF_OBS_SPEC.names
        assert spec.names[-6:] == [
            "lookahead_10_lat",
            "lookahead_10_yaw",
            "lookahead_25_lat",
            "lookahead_25_yaw",
            "lookahead_50_lat",
            "lookahead_50_yaw",
        ]

    def test_custom_schedule_changes_dim_and_names(self):
        spec = build_tmnf_obs_spec(n_lookahead_points=5, lookahead_step_spacing=20)
        # 15 base dims + 5 * 2 lookahead dims
        assert spec.dim == 25
        assert spec.names[-2:] == ["lookahead_100_lat", "lookahead_100_yaw"]

    def test_scales_are_finite_and_positive(self):
        spec = build_tmnf_obs_spec(n_lookahead_points=8, lookahead_step_spacing=5)
        scales = spec.scales
        assert np.all(np.isfinite(scales))
        assert np.all(scales > 0)

    def test_from_steps_matches_equivalent_n_and_spacing(self):
        by_steps = build_tmnf_obs_spec_from_steps([7, 14, 21])
        by_n_spacing = build_tmnf_obs_spec(n_lookahead_points=3, lookahead_step_spacing=7)
        assert by_steps.names == by_n_spacing.names
        assert by_steps.dim == by_n_spacing.dim

    def test_zero_lookahead_points(self):
        spec = build_tmnf_obs_spec_from_steps([])
        assert spec.dim == 15
        assert not any(n.startswith("lookahead_") for n in spec.names)


class TestStateDataLookaheadWiring:
    def _centerline(self, project_ahead_side_effect=None):
        centerline = MagicMock()
        centerline.project_with_forward.return_value = MagicMock(
            progress=0.5,
            lateral_offset=0.0,
            vertical_offset=0.0,
            forward=np.array([1.0, 0.0, 0.0]),
            nearest_idx=3,
        )
        centerline.project_ahead.side_effect = project_ahead_side_effect or (
            lambda pos, idx, steps: (float(steps), float(steps) * 2.0)
        )
        return centerline

    def test_default_lookahead_steps_used_when_unset(self):
        centerline = self._centerline()
        gs = make_game_state()
        sd = StateData(gs, centerline=centerline)
        assert len(sd.lookahead) == len(LOOKAHEAD_STEPS)
        called_steps = [c.args[2] for c in centerline.project_ahead.call_args_list]
        assert called_steps == LOOKAHEAD_STEPS

    def test_custom_lookahead_steps_passed_through(self):
        centerline = self._centerline()
        gs = make_game_state()
        custom_steps = [5, 15, 30, 60]
        sd = StateData(gs, centerline=centerline, lookahead_steps=custom_steps)
        assert len(sd.lookahead) == len(custom_steps)
        called_steps = [c.args[2] for c in centerline.project_ahead.call_args_list]
        assert called_steps == custom_steps
        assert sd.lookahead == [(float(s), float(s) * 2.0) for s in custom_steps]

    def test_no_centerline_defaults_to_zero_tuples_matching_steps_len(self):
        gs = make_game_state()
        sd = StateData(gs, centerline=None, lookahead_steps=[1, 2, 3, 4, 5])
        assert sd.lookahead == [(0.0, 0.0)] * 5
