"""Game-agnostic behaviour-cloning protocol and orchestrator.

Behaviour cloning (BC) pre-trains a policy from demonstration trajectories
before the live RL loop starts.  StarCraft 2 has shipped its own end-to-end
BC pipeline (``games/sc2/replay_bc.py``) since #353; this module lifts the
generic parts of that pipeline — workflow shape, summary schema, NPZ I/O,
weight + trainer-state saving — into the framework so other games can plug
in their own replay parsers and per-target fitters.

Phase 1 (#393) introduces the seam.  No game wires its ``BCAdapter`` yet;
phase 2 (#394) ports SC2 onto it, phase 3 (#395) adds TMNF.

Per-game extension point
========================

Implement :class:`BCAdapter` for your game (one module per game, conventionally
``games/<game>/bc.py``).  Expose the instance via the game's ``GameAdapter.bc``
attribute.  The :func:`run` orchestrator drives the whole flow:

#. ``bc_adapter.validate_replay_dir(replay_dir, race=...)`` — fail fast on a
   bad source directory.
#. ``bc_adapter.build_dataset(replay_dir, save_path, ...)`` — replays →
   ``demos.npz`` (schema in :mod:`framework.bc_io`).
#. ``bc_adapter.fit_bc(dataset, obs_spec, target=...)`` — return a trained
   policy + final BC loss.
#. Save ``policy_weights.yaml`` (+ ``trainer_state.npz`` when the policy
   carries one) and ``bc_summary.json`` into the experiment directory.

The orchestrator never reads game-specific config; everything game-specific
is owned by the ``BCAdapter`` implementation, which is free to inspect the
``training_params`` dict it receives.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile
from typing import Any, Protocol

from framework.bc_io import load_dataset, save_summary
from framework.obs_spec import ObsSpec

logger = logging.getLogger(__name__)


class BCAdapter(Protocol):
    """Per-game behaviour-cloning extension.

    A concrete adapter owns its game's replay format, dataset assembly, and
    per-target fit logic.  The framework :func:`run` orchestrator calls into
    this interface to drive the BC flow.

    Attributes
    ----------
    name :
        Game identifier, matching the ``GameAdapter.name`` of the same game.
    supported_targets :
        ``policy_type`` strings this adapter knows how to BC into.  The
        framework :func:`run` orchestrator validates ``target`` against this
        tuple before dispatching to :meth:`fit_bc`.
    default_target :
        Default value when neither CLI nor config specifies ``bc_target``.
        Must appear in :attr:`supported_targets`.
    """

    name: str
    supported_targets: tuple[str, ...]
    default_target: str

    def validate_replay_dir(
        self,
        replay_dir: str | pathlib.Path | None,
        *,
        race: str | None = None,
    ) -> Any:
        """Fail fast if *replay_dir* cannot be used as a BC source.

        Implementations may return any metadata that helps logging (e.g. the
        list of replay paths found).  The orchestrator does not inspect the
        return value — it only cares whether this call raises.
        """
        ...

    def build_dataset(
        self,
        replay_dir: str | pathlib.Path | None,
        save_path: str | pathlib.Path,
        *,
        obs_spec: ObsSpec,
        training_params: dict,
        race: str | None = None,
        max_replays: int | None = None,
    ) -> dict:
        """Parse *replay_dir* → ``demos.npz`` at *save_path*.

        The saved file must be readable by :func:`framework.bc_io.load_dataset`
        (schema: ``obs``, ``actions``, ``episode_starts``, ``episode_lengths``,
        ``episode_id``, ``meta``).

        Implementations read any extra knobs they need (player id, step_mul,
        screen size, …) from *training_params*.

        Returns
        -------
        dict
            Metadata dict; at minimum ``n_episodes`` and ``n_steps``.  Also
            embedded in the ``.npz`` under the ``meta`` key.
        """
        ...

    def fit_bc(
        self,
        dataset: dict,
        obs_spec: ObsSpec,
        *,
        target: str,
        training_params: dict,
    ) -> tuple[Any, float]:
        """Pre-train a *target* policy on *dataset*.

        Returns
        -------
        (policy, final_bc_loss)
            ``policy`` must have a ``save(path)`` method.  When it is a
            :class:`framework.policies.BasePolicy`, the orchestrator also
            calls ``save_trainer_state``.
        """
        ...

    def summary_extras(
        self,
        dataset: dict,
        meta: dict,
        *,
        target: str,
        training_params: dict,
    ) -> dict:
        """Optional: per-game stats merged into ``bc_summary.json["extras"]``.

        Called by the orchestrator after :meth:`fit_bc` has returned, with
        the full loaded dataset still in scope.  Default implementation
        returns an empty dict — implement when your game wants to record
        game-specific stats (e.g. SC2's ``fn_idx_histogram``).
        """
        return {}


def run(
    bc_adapter: BCAdapter,
    replay_dir: str | pathlib.Path | None,
    experiment_dir: str | pathlib.Path,
    *,
    obs_spec: ObsSpec,
    target: str,
    training_params: dict,
    race: str | None = None,
    max_replays: int | None = None,
) -> dict:
    """Drive a full BC run end to end via *bc_adapter*.

    Workflow
    --------
    #. :meth:`BCAdapter.validate_replay_dir`.
    #. :meth:`BCAdapter.build_dataset` into a temp ``demos.npz``, then load
       it back via :func:`framework.bc_io.load_dataset`.
    #. :meth:`BCAdapter.fit_bc`.
    #. Save ``policy_weights.yaml`` and (when the policy has one)
       ``trainer_state.npz`` into *experiment_dir*.  Save ``bc_summary.json``
       via :func:`framework.bc_io.save_summary`.

    Parameters
    ----------
    bc_adapter :
        The game-specific BC implementation.
    replay_dir :
        Directory the *bc_adapter* knows how to read.  May be ``None`` for
        adapters whose source is live demonstrations rather than replay files.
    experiment_dir :
        Where ``policy_weights.yaml``, ``trainer_state.npz``, and
        ``bc_summary.json`` are written.
    obs_spec :
        Active observation spec for the experiment.
    target :
        Policy type to BC into.  Must be in
        :attr:`BCAdapter.supported_targets`.
    training_params :
        The full ``training_params.yaml`` dict.  Forwarded to *bc_adapter*
        unchanged so it can read game-specific knobs.
    race :
        Optional race filter forwarded to *bc_adapter*.
    max_replays :
        Optional cap on replays processed.

    Returns
    -------
    dict
        BC summary (same content written to ``bc_summary.json``).
    """
    from framework.policies import BasePolicy, trainer_state_path

    if target not in bc_adapter.supported_targets:
        raise ValueError(
            f"BC target {target!r} not supported by {bc_adapter.name!r} "
            f"BCAdapter; supported: {bc_adapter.supported_targets}"
        )

    experiment_dir = pathlib.Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Normalise race: "any"/empty → None
    race_filter: str | None = None
    if race and race.lower() not in ("", "any"):
        race_filter = race.lower()

    bc_adapter.validate_replay_dir(replay_dir, race=race_filter)

    with tempfile.TemporaryDirectory() as tmp:
        demos_path = pathlib.Path(tmp) / "demos.npz"
        meta = bc_adapter.build_dataset(
            replay_dir,
            demos_path,
            obs_spec=obs_spec,
            training_params=training_params,
            race=race_filter,
            max_replays=max_replays,
        )
        dataset = load_dataset(demos_path)

    policy, bc_loss = bc_adapter.fit_bc(
        dataset,
        obs_spec,
        target=target,
        training_params=training_params,
    )

    # Save weights.  Champion-based policies whose save() writes metadata
    # only (e.g. SC2CMAESPolicy) expose ``._champion`` — save that directly
    # so a subsequent fine-tune can reload it.
    weights_path = str(experiment_dir / "policy_weights.yaml")
    champion = getattr(policy, "_champion", None)
    if champion is not None and hasattr(champion, "save") and callable(champion.save):
        champion.save(weights_path)
    else:
        policy.save(weights_path)
    # CNN-style policies may redirect .yaml → .npz inside save(); log whichever exists.
    _npz = weights_path.replace(".yaml", ".npz")
    _logged = _npz if not pathlib.Path(weights_path).exists() and pathlib.Path(_npz).exists() else weights_path
    logger.info("BC: saved weights → %s", _logged)

    if isinstance(policy, BasePolicy):
        ts_path = trainer_state_path(weights_path)
        policy.save_trainer_state(ts_path)
        logger.info("BC: saved trainer state → %s", ts_path)

    extras: dict = dict(meta.get("summary_extras", {}))
    # Optional post-fit hook: lets the adapter compute stats that need the
    # loaded dataset (e.g. SC2's fn_idx_histogram).  Adapters that don't
    # implement it inherit the no-op default from the Protocol.
    extras_hook = getattr(bc_adapter, "summary_extras", None)
    if callable(extras_hook):
        extras.update(extras_hook(dataset, meta, target=target, training_params=training_params))

    summary = {
        "game": bc_adapter.name,
        "bc_target": target,
        "n_episodes": int(meta.get("n_episodes", 0)),
        "n_pairs": int(meta.get("n_steps", 0)),
        "bc_race": race_filter or "any",
        "final_bc_loss": float(bc_loss),
        "extras": extras,
    }

    save_summary(experiment_dir, summary)
    return summary
