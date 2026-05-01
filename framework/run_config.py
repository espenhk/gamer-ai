"""Config bundles consumed by train_rl.

Each dataclass groups related settings so the training loop signature stays
small.  Game adapters build a GameSpec once per experiment; RunConfig carries
algorithm-level knobs that apply to every game.  The optional ProbeSpec,
WarmupSpec, and PolicyExtras handle TMNF-specific phases and custom policy
wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Game / track binding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameSpec:
    """Game/track-specific binding, built once per experiment by the adapter."""

    experiment_name: str
    track: str                                  # label used in results/dir naming
    make_env_fn: Callable[[], Any]              # factory returning a BaseGameEnv
    obs_spec: Any                               # ObsSpec instance
    head_names: list[str]                        # e.g. ["steer","accel","brake"]
    discrete_actions: Any                        # np.ndarray
    weights_file: str
    reward_config_file: str
    save_results_fn: Callable | None = None     # optional callable(data, results_dir)


# ---------------------------------------------------------------------------
# Algorithm-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """Algorithm-level config that applies to all games."""

    n_sims: int
    in_game_episode_s: float
    speed: float = 1.0
    mutation_scale: float = 0.05
    mutation_share: float = 1.0
    adaptive_mutation: bool = True
    do_pretrain: bool = False
    patience: int = 0
    policy_type: str = "hill_climbing"
    policy_params: dict = field(default_factory=dict)
    training_params: dict = field(default_factory=dict)   # raw YAML for record-keeping

    @classmethod
    def from_training_params(cls, p: dict) -> RunConfig:
        """Build a RunConfig from a training-params dict with safe defaults."""
        return cls(
            n_sims=p["n_sims"],
            in_game_episode_s=p["in_game_episode_s"],
            speed=p.get("speed", 1.0),
            mutation_scale=p.get("mutation_scale", 0.05),
            mutation_share=p.get("mutation_share", 1.0),
            adaptive_mutation=p.get("adaptive_mutation", True),
            do_pretrain=p.get("do_pretrain", False),
            patience=p.get("patience", 0),
            policy_type=p.get("policy_type", "hill_climbing"),
            policy_params=p.get("policy_params", {}),
            training_params=p,
        )


# ---------------------------------------------------------------------------
# TMNF-only optional specs (pass None to skip)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeSpec:
    """Probe + cold-start phase config."""

    actions: list           # list of (action_array, name) tuples
    probe_in_game_s: float
    cold_start_restarts: int
    cold_start_sims: int


@dataclass(frozen=True)
class WarmupSpec:
    """Forced-action warmup at episode start."""

    action: Any             # np.ndarray
    steps: int


@dataclass(frozen=True)
class PolicyExtras:
    """Game-specific extra policy types."""

    factories: dict         # maps policy_type name -> zero-arg callable
    loop_dispatch: dict     # maps policy_type name -> loop type string
