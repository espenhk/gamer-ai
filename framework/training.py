"""Game-agnostic RL training loop.

Entry point: train_rl()

All game-specific details are injected as parameters:
    make_env_fn     — zero-argument factory returning a BaseGameEnv
    probe_actions   — list of (action_array, name) for the probe phase
    warmup_action   — forced action during episode warmup steps
    warmup_steps    — how many steps to force the warmup action

Policy construction is also parameterised so the loop never imports from games/.
"""

from __future__ import annotations

import datetime
import logging
import os
import time
from collections import deque
from typing import Any, Callable

import numpy as np
import yaml

from framework.analytics import (
    RunTrace,
    ProbeResult,
    ColdStartSimResult,
    ColdStartRestartResult,
    GreedySimResult,
    ExperimentData,
)
from framework.policies import (
    BasePolicy,
    WeightedLinearPolicy,
    NeuralNetPolicy,
    EpsilonGreedyPolicy,
    MCTSPolicy,
    GeneticPolicy,
    POLICY_REGISTRY,
    trainer_state_path as _trainer_state_path_canonical,
)
from framework.obs_spec import ObsSpec
from framework.run_config import GameSpec, RunConfig, ProbeSpec, WarmupSpec, PolicyExtras
from framework.version import code_version

logger = logging.getLogger(__name__)

_TRACE_SAMPLE_EVERY = 2   # record position every N steps


def _trainer_state_path(weights_file: str) -> str:
    return _trainer_state_path_canonical(weights_file)


# ---------------------------------------------------------------------------
# Constant-action policy (probe phase)
# ---------------------------------------------------------------------------

class _ConstantPolicy:
    """Always returns the same action — used during cold-start probing."""
    def __init__(self, action: np.ndarray) -> None:
        self._action = action
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._action
    def update(self, *_, **__) -> None:
        pass
    def on_episode_start(self, **kwargs) -> None:
        pass


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def _make_policy(
    policy_type: str,
    obs_spec: ObsSpec,
    head_names: list[str],
    discrete_actions: np.ndarray,
    weights_file: str,
    policy_params: dict,
    re_initialize: bool,
    extra_policy_types: dict[str, Callable[[], BasePolicy]] | None = None,
) -> BasePolicy:
    """Construct the appropriate policy given type, obs_spec, and hyperparams.

    Consults POLICY_REGISTRY first; falls back to extra_policy_types for
    game-registered policies not yet migrated to the registry.
    """
    cls = POLICY_REGISTRY.get(policy_type)
    if cls is not None:
        return cls.make(
            obs_spec=obs_spec, head_names=head_names,
            discrete_actions=discrete_actions, weights_file=weights_file,
            policy_params=policy_params, re_initialize=re_initialize,
        )
    if extra_policy_types and policy_type in extra_policy_types:
        return extra_policy_types[policy_type]()
    raise ValueError(
        f"Unknown policy_type: {policy_type!r}. "
        f"Registered: {sorted(POLICY_REGISTRY)}; "
        f"extras: {sorted(extra_policy_types or [])}"
    )


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def _run_episode(
    env,
    policy,
    obs: np.ndarray,
    warmup_action: np.ndarray | None = None,
    warmup_steps: int = 0,
    reset_info: dict | None = None,
) -> tuple[float, dict, list[int], int, RunTrace]:
    """Run one episode from *obs* until terminated/truncated.

    Parameters
    ----------
    reset_info :
        The info dict returned by ``env.reset()`` for this episode.
        Forwarded to ``policy.on_episode_start(info=reset_info)`` so
        that policies can prime state (e.g. available-actions masks)
        before the first ``policy(obs)`` call.

    Returns:
        total_reward    — float
        info            — final step info dict from env
        throttle_counts — [brake_steps, coast_steps, accel_steps]
        total_steps     — int
        trace           — RunTrace
    """
    total_reward   = 0.0
    steps          = 0
    info: dict     = {}
    throttle_counts = [0, 0, 0]
    turning_steps   = 0
    pos_x: list[float] = []
    pos_z: list[float] = []
    throttle_state: list = []
    prev_obs = obs

    policy.on_episode_start(info=reset_info or {})

    while True:
        in_warmup = (warmup_action is not None) and (steps < warmup_steps)
        action    = warmup_action if in_warmup else policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if not in_warmup:
            policy.update(prev_obs, action, reward, next_obs, terminated or truncated,
                          info=info)

        prev_obs = next_obs
        obs      = next_obs

        # Throttle classification (works for any action with accel@[1], brake@[2])
        if len(action) >= 3:
            accel_on = float(action[1]) >= 0.5
            brake_on = float(action[2]) >= 0.5
            if brake_on and not accel_on:
                t = 0
            elif accel_on:
                t = 2
            else:
                t = 1
            throttle_counts[t] += 1
            throttle_state.append((float(action[1]), float(action[2])))
            if abs(float(action[0])) > 0.05:
                turning_steps += 1

        if steps % _TRACE_SAMPLE_EVERY == 0:
            pos_x.append(float(info.get("pos_x", 0.0)))
            pos_z.append(float(info.get("pos_z", 0.0)))

        if terminated or truncated:
            _print_episode_summary(info, steps, total_reward, truncated)
            if len(action) >= 3:
                _print_action_stats(throttle_counts, turning_steps, steps)
            break

    trace = RunTrace(pos_x=pos_x, pos_z=pos_z,
                     throttle_state=throttle_state, total_reward=total_reward)
    return total_reward, info, throttle_counts, steps, trace


def _print_episode_summary(info: dict, steps: int, total_reward: float,
                            truncated: bool) -> None:
    finished = bool(info.get("finished", False))
    outcome  = "truncated" if truncated else ("finished" if finished else "terminated")
    skipped_frames = _episode_skipped_frames(info)
    skipped_suffix = (
        f"  skipped_frames={skipped_frames}"
        if skipped_frames is not None else ""
    )
    sc2_outcome = info.get("player_outcome")
    sc2_raw_reward = info.get("raw_reward")
    if sc2_outcome is not None or sc2_raw_reward is not None:
        if sc2_outcome is not None:
            sc2_outcome_v = float(sc2_outcome)
            if sc2_outcome_v > 0:
                sc2_outcome_s = "win"
            elif sc2_outcome_v < 0:
                sc2_outcome_s = "loss"
            else:
                sc2_outcome_s = "draw"
        else:
            sc2_outcome_s = str(info.get("termination_reason", outcome))
        reward_v = float(sc2_raw_reward if sc2_raw_reward is not None else 0.0)
        score = float(info.get("score", 0.0))
        logger.info(
            "ep end  %s  r=%+.1f  steps=%d  outcome=%s  reward=%+.1f  score=%+.1f%s",
            outcome, total_reward, steps, sc2_outcome_s, reward_v, score, skipped_suffix,
        )
        return
    logger.info("ep end  %s  r=%+.1f  steps=%d%s", outcome, total_reward, steps, skipped_suffix)


def _episode_skipped_frames(info: dict) -> int | None:
    """Return non-negative episode skipped-frame count when present in info."""
    skipped = info.get("episode_skipped_frames")
    if skipped is None:
        return None
    try:
        return max(0, int(skipped))
    except (TypeError, ValueError):
        return None


def _log_new_best_details(info: dict, prev_best_info: dict | None) -> None:
    """Log expanded breakdown for a newly-achieved best reward.

    Logged at INFO level immediately after the NEW BEST headline.  Each metric
    is compared with the previous best in parentheses where *prev_best_info* is
    available.  Groups emitted (only when the relevant data are present):

    1. Reward-component breakdown (``episode_reward_components``)
    2. Action-frequency breakdown (``episode_action_counts``)
    3. TMNF task metrics: track progress, mean lateral offset, finish time
    4. SC2 kill stats: enemy units/structures destroyed
    5. SC2 game-state averages: army size, enemy screen presence
    """
    prev: dict = prev_best_info or {}

    # 1. Reward-component breakdown -----------------------------------------
    rc = info.get("episode_reward_components")
    if rc:
        prev_rc: dict = prev.get("episode_reward_components") or {}
        terminal = float(rc.get("terminal", 0.0))
        prev_terminal = float(prev_rc.get("terminal", 0.0))
        for k, v in sorted(rc.items()):
            if k == "terminal":
                continue
            if abs(v) > 0.001 or k == "score":
                pv = prev_rc.get(k)
                cmp_s = f" (prev {pv:+.1f})" if pv is not None else ""
                logger.info("    %s=%+.1f%s", k, v, cmp_s)

        win_bonus = terminal if terminal > 0 else 0.0
        loss_penalty = terminal if terminal < 0 else 0.0
        prev_win_bonus = prev_terminal if prev_terminal > 0 else 0.0
        prev_loss_penalty = prev_terminal if prev_terminal < 0 else 0.0
        if "terminal" in prev_rc:
            logger.info("    win_bonus=%+.1f (prev %+.1f)", win_bonus, prev_win_bonus)
            logger.info("    loss_penalty=%+.1f (prev %+.1f)", loss_penalty, prev_loss_penalty)
        else:
            logger.info("    win_bonus=%+.1f", win_bonus)
            logger.info("    loss_penalty=%+.1f", loss_penalty)

    # 2. Action-frequency breakdown (SC2Env only) ----------------------------
    ac = info.get("episode_action_counts")
    if ac:
        total = sum(ac.values())
        if total > 0:
            prev_ac: dict = prev.get("episode_action_counts") or {}
            prev_total = sum(prev_ac.values()) if prev_ac else 0
            try:
                from games.sc2.actions import FUNCTION_IDS as _FNIDS  # noqa: PLC0415
            except ImportError:
                # SC2 extras not installed; fall back to numeric fn{idx} names.
                logger.debug("games.sc2.actions unavailable; action names shown as fn{idx}")
                _FNIDS: dict = {}
            for fn_idx, count in sorted(ac.items(), key=lambda x: -x[1]):
                name = _FNIDS.get(int(fn_idx), f"fn{fn_idx}")
                pct = 100.0 * count / total
                if prev_total > 0:
                    # Keys are ints from env.step(); str fallback handles any
                    # JSON-deserialised prev_best_info where keys became strings.
                    ppct = 100.0 * prev_ac.get(fn_idx, prev_ac.get(str(fn_idx), 0)) / prev_total
                    cmp_s = f" (prev {ppct:.1f}%)"
                else:
                    cmp_s = ""
                logger.info("    %s=%.1f%%%s", name, pct, cmp_s)

    # 3. TMNF task metrics ---------------------------------------------------
    progress = info.get("track_progress")
    if progress is not None:
        prev_progress = prev.get("track_progress")
        cmp_s = f" (prev {100.0*prev_progress:.1f}%)" if prev_progress is not None else ""
        lat = info.get("mean_abs_lateral_offset")
        lat_s = ""
        if lat is not None:
            prev_lat = prev.get("mean_abs_lateral_offset")
            lat_cmp = f" (prev {prev_lat:.2f}m)" if prev_lat is not None else ""
            lat_s = f"  mean_lateral={lat:.2f}m{lat_cmp}"
        finish_t = info.get("elapsed_s") if info.get("finished") else None
        t_s = ""
        if finish_t is not None:
            prev_ft = prev.get("elapsed_s") if prev.get("finished") else None
            ft_cmp = f" (prev {prev_ft:.1f}s)" if prev_ft is not None else ""
            t_s = f"  finish_time={finish_t:.1f}s{ft_cmp}"
        logger.info("    progress=%.1f%%%s%s%s", 100.0 * progress, cmp_s, lat_s, t_s)

    # 4. SC2 kill stats — suppress when no kills occurred --------------------
    kills = info.get("episode_killed_value_units")
    if kills is not None:
        struct_kills = info.get("episode_killed_value_structures", 0.0)
        if kills > 0.5 or struct_kills > 0.5:
            prev_kills = prev.get("episode_killed_value_units")
            cmp_s = f" (prev {prev_kills:.0f})" if prev_kills is not None else ""
            prev_struct = prev.get("episode_killed_value_structures")
            struct_cmp_s = f" (prev {prev_struct:.0f})" if prev_struct is not None else ""
            logger.info("    kills: units=%d%s  structures=%d%s",
                        int(kills), cmp_s, int(struct_kills), struct_cmp_s)

    # 5. SC2 game-state averages ---------------------------------------------
    obs_avgs = info.get("episode_obs_averages")
    if obs_avgs:
        prev_avgs: dict = prev.get("episode_obs_averages") or {}
        for k in ("army_count", "food_used", "screen_enemy_count"):
            v = obs_avgs.get(k)
            if v is not None and abs(v) > 0.001:
                pv = prev_avgs.get(k)
                cmp_s = f" (prev {pv:.1f})" if pv is not None else ""
                logger.info("    %s=%.1f%s", k, v, cmp_s)


def _print_action_stats(throttle_counts: list[int], turning_steps: int,
                         steps: int) -> None:
    if steps == 0:
        return
    b, c, a = throttle_counts
    logger.debug(
        "throttle — brake: %4.1f%%  coast: %4.1f%%  accel: %4.1f%%"
        "    steer — straight: %4.1f%%  turning: %4.1f%%",
        100*b/steps, 100*c/steps, 100*a/steps,
        100*(steps-turning_steps)/steps, 100*turning_steps/steps,
    )


def _scaled_episode_time(sim: int, n_total: int, max_time_s: float) -> float:
    """4-step schedule: 25%→50%→75%→100% of max_time_s by quarter of training."""
    quarter = n_total / 4
    if sim <= quarter:
        return max_time_s * 0.25
    elif sim <= 2 * quarter:
        return max_time_s * 0.5
    elif sim <= 3 * quarter:
        return max_time_s * 0.75
    return max_time_s


# ---------------------------------------------------------------------------
# Probe phase
# ---------------------------------------------------------------------------

def _run_probes(
    env,
    probe_actions: list[tuple[np.ndarray, str]],
    probe_in_game_s: float,
    speed: float,
    warmup_action: np.ndarray | None = None,
    warmup_steps: int = 0,
) -> tuple[float, list[ProbeResult]]:
    saved_limit = env.get_episode_time_limit()
    env.set_episode_time_limit(probe_in_game_s / speed)

    logger.info(
        "No weights file — running %d probe episodes (%ss each) to establish baseline.",
        len(probe_actions), probe_in_game_s,
    )

    results: dict[int, float] = {}
    probe_results: list[ProbeResult] = []

    try:
        for i, (action_arr, action_name) in enumerate(probe_actions):
            logger.info("Probe %d/%d: %s", i + 1, len(probe_actions), action_name)
            obs, reset_info = env.reset()
            reward, _, throttle_counts, total_steps, trace = _run_episode(
                env, _ConstantPolicy(action_arr), obs,
                warmup_action=warmup_action, warmup_steps=warmup_steps,
                reset_info=reset_info,
            )
            results[i] = reward
            probe_results.append(
                ProbeResult(action_idx=i, action_name=action_name, reward=reward, trace=trace)
            )
    finally:
        if saved_limit is not None:
            env.set_episode_time_limit(saved_limit)

    best_idx = max(results, key=lambda i: results[i])
    logger.info("Probe results:")
    for i, r in results.items():
        marker = " <-- best" if i == best_idx else ""
        logger.info("  action %d (%-15s)  reward=%+.1f%s",
                    i, probe_actions[i][1], r, marker)
    logger.info("Probe best: %+.1f", results[best_idx])
    time.sleep(1)
    return results[best_idx], probe_results


# ---------------------------------------------------------------------------
# Cold-start search
# ---------------------------------------------------------------------------

def _cold_start_search(
    env,
    obs_spec: ObsSpec,
    head_names: list[str],
    probe_best_reward: float,
    weights_file: str,
    mutation_scale: float,
    mutation_share: float = 1.0,
    n_restarts: int = 5,
    sims_per_restart: int = 10,
    warmup_action: np.ndarray | None = None,
    warmup_steps: int = 0,
) -> tuple[WeightedLinearPolicy, float, list[ColdStartRestartResult]]:
    overall_best_policy = None
    overall_best_reward = float("-inf")
    restart_results: list[ColdStartRestartResult] = []

    logger.info(
        "=== Cold-start — up to %d restarts × %d sims  target: %+.1f ===",
        n_restarts, sims_per_restart, probe_best_reward,
    )

    for restart in range(1, n_restarts + 1):
        logger.info("-- Restart %d/%d: random init --", restart, n_restarts)

        rng  = np.random.default_rng()
        names = obs_spec.names
        random_cfg = {
            f"{h}_weights": {n: float(rng.standard_normal()) for n in names}
            for h in head_names
        }
        local_best_policy = WeightedLinearPolicy.from_cfg(random_cfg, obs_spec, head_names)
        local_best_reward = float("-inf")
        sim_results: list[ColdStartSimResult] = []

        for sim in range(1, sims_per_restart + 1):
            candidate = local_best_policy.mutated(scale=mutation_scale, share=mutation_share)
            obs, reset_info = env.reset()
            reward, info, tc, total_steps, trace = _run_episode(
                env, candidate, obs,
                warmup_action=warmup_action, warmup_steps=warmup_steps,
                reset_info=reset_info,
            )
            sim_results.append(ColdStartSimResult(
                sim=sim, reward=reward,
                throttle_counts=list(tc), total_steps=total_steps, trace=trace,
                termination_reason=info.get("termination_reason"),
            ))
            if reward > local_best_reward:
                local_best_reward = reward
                local_best_policy = candidate
            if reward > overall_best_reward:
                overall_best_reward = reward
                overall_best_policy = candidate

        beat = local_best_reward > probe_best_reward
        logger.info(
            "Restart %d best: %+.1f  (%s probe floor %+.1f)",
            restart, local_best_reward, "beats" if beat else "below", probe_best_reward,
        )
        restart_results.append(ColdStartRestartResult(
            restart=restart, sims=sim_results,
            best_reward=local_best_reward, beat_probe_floor=beat,
        ))
        if overall_best_policy is not None:
            overall_best_policy.save(weights_file)
        if beat:
            logger.info("Beat probe floor — ending cold-start early.")
            break

    if overall_best_policy is None:
        overall_best_policy = WeightedLinearPolicy(obs_spec, head_names, weights_file)
        overall_best_policy.save(weights_file)

    logger.info("Cold-start complete — best: %+.1f  → %s",
                overall_best_reward, weights_file)
    return overall_best_policy, overall_best_reward, restart_results


# ---------------------------------------------------------------------------
# Replay saving (SC2 only — no-op for all other games)
# ---------------------------------------------------------------------------

def _replay_dir(weights_file: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(weights_file)), "replays")


def _next_best_prefix(weights_file: str, replay_dir: str) -> str:
    """Return the next sequential experiment_best-N prefix."""
    experiment_name = os.path.basename(os.path.dirname(os.path.abspath(weights_file)))
    n = 0
    if os.path.isdir(replay_dir):
        n = sum(
            1 for f in os.listdir(replay_dir)
            if f.endswith(".SC2Replay") and not f.startswith("_")
        )
    return f"{experiment_name}_best-{n + 1:02d}"


def _try_save_replay(env, weights_file: str) -> None:
    """Save the current episode's replay for single-episode-per-sim loops.

    Used by NeuralNetPolicy and Q-learning loops where exactly one episode
    runs per sim, so the episode that triggered the new-best is still current
    at the point of the call.  Multi-episode loops (ES, genetic, CMA-ES) use
    _save_candidate_replay / _finalize_candidate_replay instead.
    """
    if not hasattr(env, "save_replay"):
        return
    rdir = _replay_dir(weights_file)
    prefix = _next_best_prefix(weights_file, rdir)
    try:
        saved = env.save_replay(rdir, prefix=prefix)
        if saved:
            logger.info("  [replay] saved → %s", saved)
        else:
            logger.info("  [replay] save_replay returned None (skipped).")
    except Exception as exc:
        logger.warning("  [replay] save failed: %s", exc)


def _save_candidate_replay(env, weights_file: str) -> str | None:
    """Speculatively save the current episode to a temp candidate file.

    Called in multi-episode loops immediately after a potentially-best episode
    and before the next env.reset() overwrites the SC2 replay.  Returns the
    candidate path (``<replay_dir>/_candidate.SC2Replay``) or None when the
    env does not support replay saving or the save fails.
    """
    if not hasattr(env, "save_replay"):
        return None
    rdir = _replay_dir(weights_file)
    try:
        return env.save_replay(rdir, prefix="_candidate")
    except Exception as exc:
        logger.warning("  [replay] candidate save failed: %s", exc)
        return None


def _finalize_candidate_replay(candidate_path: str | None, weights_file: str) -> None:
    """Rename a candidate replay to the next sequential best-N name.

    Called after the winner of a multi-episode sim/generation is confirmed.
    No-op when candidate_path is None or the file no longer exists.
    """
    if not candidate_path:
        return
    if not os.path.exists(candidate_path):
        logger.debug("  [replay] candidate file missing — skipping finalize.")
        return
    rdir = os.path.dirname(candidate_path)
    dest = os.path.join(rdir, _next_best_prefix(weights_file, rdir) + ".SC2Replay")
    try:
        os.rename(candidate_path, dest)
        logger.info("  [replay] saved → %s", dest)
    except Exception as exc:
        logger.warning("  [replay] finalize failed: %s", exc)


def _discard_candidate_replay(candidate_path: str | None) -> None:
    """Delete a temporary candidate replay file. No-op if None or missing."""
    if candidate_path and os.path.exists(candidate_path):
        try:
            os.remove(candidate_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Greedy loops
# ---------------------------------------------------------------------------

def _greedy_loop(
    env,
    policy: BasePolicy,
    n_sims: int,
    mutation_scale: float,
    weights_file: str,
    best_reward: float = float("-inf"),
    learning_rate: float = 0.01,
    mutation_share: float = 1.0,
    adaptive_mutation: bool = True,
    warmup_action: np.ndarray | None = None,
    warmup_steps: int = 0,
    patience: int = 0,
) -> tuple[BasePolicy, float, list[GreedySimResult], bool, int | None]:
    """ES gradient-estimation loop for WeightedLinearPolicy / NeuralNetPolicy."""
    ADAPT_WINDOW = 20
    ADAPT_UP     = 1.2
    ADAPT_DOWN   = 0.85
    SCALE_MIN    = 0.001
    SCALE_MAX    = 1.0
    current_scale = mutation_scale
    improvement_history: deque[bool] = deque(maxlen=ADAPT_WINDOW)

    best_policy  = policy
    greedy_sims: list[GreedySimResult] = []

    if isinstance(best_policy, NeuralNetPolicy):
        # Single-candidate hill-climbing for neural_net
        full_episode_time_s = env.get_episode_time_limit()
        no_improve_streak = 0
        early_stopped     = False
        early_stop_sim    = None
        best_info: dict   = {}
        try:
            for sim in range(1, n_sims + 1):
                candidate = best_policy.mutated(scale=current_scale)
                logger.info("--- Sim %d/%d ---", sim, n_sims)
                if full_episode_time_s is not None:
                    env.set_episode_time_limit(
                        _scaled_episode_time(sim, n_sims, full_episode_time_s)
                    )
                obs, reset_info = env.reset()
                reward, info, tc, total_steps, trace = _run_episode(
                    env, candidate, obs,
                    warmup_action=warmup_action, warmup_steps=warmup_steps,
                    reset_info=reset_info,
                )
                improved = reward > best_reward
                if improved:
                    prev_best   = best_reward
                    best_reward = reward
                    best_policy = candidate
                    best_policy.save(weights_file)
                    best_policy.save_trainer_state(_trainer_state_path(weights_file))
                    _try_save_replay(env, weights_file)
                    verdict = f"NEW BEST  {reward:+.1f}  (was {prev_best:+.1f})"
                    logger.info("  >> %s", verdict)
                    _log_new_best_details(info, best_info)
                    best_info = info
                else:
                    verdict = f"no improvement  r={reward:+.1f}  best={best_reward:+.1f}"
                    logger.info("  >> %s", verdict)
                improvement_history.append(improved)
                _maybe_adapt_scale(improvement_history, current_scale, sim,
                                   ADAPT_WINDOW, ADAPT_UP, ADAPT_DOWN,
                                   SCALE_MIN, SCALE_MAX, adaptive_mutation,
                                   _set := [])
                if _set:
                    current_scale = _set[0]
                greedy_sims.append(GreedySimResult(
                    sim=sim, reward=reward, improved=improved,
                    throttle_counts=list(tc), total_steps=total_steps, trace=trace,
                    weights=candidate.to_cfg(),
                    final_track_progress=info.get("track_progress", 0.0),
                    laps_completed=info.get("laps_completed", 0),
                    mutation_scale=current_scale,
                    termination_reason=info.get("termination_reason"),
                    finish_time_s=info.get("elapsed_s") if info.get("finished") else None,
                    mean_abs_lateral_offset=info.get("mean_abs_lateral_offset"),
                    reward_components=info.get("episode_reward_components"),
                    action_counts=info.get("episode_action_counts"),
                    obs_averages=info.get("episode_obs_averages"),
                    xy_hist=info.get("episode_xy_hist"),
                    skipped_frames=_episode_skipped_frames(info),
                    supply_capped_fraction=info.get("episode_supply_capped_fraction"),
                    build_order=info.get("episode_build_order"),
                    army_count_series=info.get("episode_army_series"),
                    resource_series=info.get("episode_resource_series"),
                ))
                no_improve_streak = 0 if improved else no_improve_streak + 1
                if patience > 0 and no_improve_streak >= patience:
                    logger.info(
                        "Early stopping: no improvement in last %d sims (best=%.1f). "
                        "Stopping at sim %d/%d.",
                        patience, best_reward, sim, n_sims,
                    )
                    early_stopped  = True
                    early_stop_sim = sim
                    break
        except KeyboardInterrupt:
            logger.warning("Training interrupted.")
        return best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim

    if not isinstance(best_policy, WeightedLinearPolicy):
        raise TypeError(
            f"Unsupported policy for _greedy_loop: {type(best_policy).__name__}"
        )

    # ES mirrored-perturbation loop
    rng   = np.random.default_rng()
    theta = best_policy.to_flat()
    full_episode_time_s = env.get_episode_time_limit()
    has_episode_time_limit = full_episode_time_s is not None
    no_improve_streak = 0
    early_stopped     = False
    early_stop_sim    = None
    best_info_logged: dict = {}

    try:
        for sim in range(1, n_sims + 1):
            eps = rng.standard_normal(len(theta)).astype(np.float32) * current_scale
            policy_plus  = best_policy.with_flat(theta + eps)
            policy_minus = best_policy.with_flat(theta - eps)

            if has_episode_time_limit:
                env.set_episode_time_limit(
                    _scaled_episode_time(sim, n_sims, full_episode_time_s)
                )

            obs, reset_info = env.reset()
            r_plus, info_plus, tc_plus, steps_plus, trace_plus = _run_episode(
                env, policy_plus, obs,
                warmup_action=warmup_action, warmup_steps=warmup_steps,
                reset_info=reset_info,
            )
            # Capture the plus replay before env.reset() starts the minus episode
            # and overwrites it in the SC2 process.
            _plus_candidate = (
                _save_candidate_replay(env, weights_file)
                if r_plus > best_reward else None
            )
            obs, reset_info = env.reset()
            r_minus, info_minus, tc_minus, steps_minus, trace_minus = _run_episode(
                env, policy_minus, obs,
                warmup_action=warmup_action, warmup_steps=warmup_steps,
                reset_info=reset_info,
            )

            theta += learning_rate * (r_plus - r_minus) * eps

            if r_plus >= r_minus:
                best_r, best_info, best_tc, best_steps, best_trace, best_cand = (
                    r_plus, info_plus, tc_plus, steps_plus, trace_plus, policy_plus)
                # Minus episode is now current in SC2 but plus won — use the
                # pre-saved plus candidate; minus replay is not needed.
                _winner_candidate = _plus_candidate
            else:
                best_r, best_info, best_tc, best_steps, best_trace, best_cand = (
                    r_minus, info_minus, tc_minus, steps_minus, trace_minus, policy_minus)
                # Minus won and its replay is still current — save it now.
                _discard_candidate_replay(_plus_candidate)
                _winner_candidate = (
                    _save_candidate_replay(env, weights_file)
                    if r_minus > best_reward else None
                )

            improved = False
            if best_r > best_reward:
                prev_best   = best_reward
                best_reward = best_r
                best_policy = best_cand
                best_policy.save(weights_file)
                best_policy.save_trainer_state(_trainer_state_path(weights_file))
                _finalize_candidate_replay(_winner_candidate, weights_file)
                improved    = True
                verdict = (f"NEW BEST  {best_r:+.1f}  (was {prev_best:+.1f})"
                           f"  gradient={r_plus - r_minus:+.1f}")
                logger.info("  >> %s", verdict)
                _log_new_best_details(best_info, best_info_logged)
                best_info_logged = best_info
            else:
                _discard_candidate_replay(_winner_candidate)
                verdict = (f"no improvement  +ε={r_plus:+.1f}  −ε={r_minus:+.1f}"
                           f"  best={best_reward:+.1f}")
                logger.info("  >> %s", verdict)

            improvement_history.append(improved)
            if adaptive_mutation and len(improvement_history) == ADAPT_WINDOW and sim % ADAPT_WINDOW == 0:
                p = sum(improvement_history) / ADAPT_WINDOW
                prev_scale = current_scale
                if p > 1 / 5:
                    current_scale = min(current_scale * ADAPT_UP, SCALE_MAX)
                elif p < 1 / 5:
                    current_scale = max(current_scale * ADAPT_DOWN, SCALE_MIN)
                if current_scale != prev_scale:
                    logger.info("  [adaptive] scale %.4f → %.4f  (success_rate=%.2f)",
                                prev_scale, current_scale, p)

            greedy_sims.append(GreedySimResult(
                sim=sim, reward=best_r, improved=improved,
                throttle_counts=list(best_tc), total_steps=best_steps,
                trace=best_trace, weights=best_cand.to_cfg(),
                final_track_progress=best_info.get("track_progress", 0.0),
                laps_completed=best_info.get("laps_completed", 0),
                mutation_scale=current_scale,
                termination_reason=best_info.get("termination_reason"),
                finish_time_s=best_info.get("elapsed_s") if best_info.get("finished") else None,
                mean_abs_lateral_offset=best_info.get("mean_abs_lateral_offset"),
                reward_components=best_info.get("episode_reward_components"),
                action_counts=best_info.get("episode_action_counts"),
                obs_averages=best_info.get("episode_obs_averages"),
                xy_hist=best_info.get("episode_xy_hist"),
                skipped_frames=_episode_skipped_frames(best_info),
                supply_capped_fraction=best_info.get("episode_supply_capped_fraction"),
                build_order=best_info.get("episode_build_order"),
                army_count_series=best_info.get("episode_army_series"),
                resource_series=best_info.get("episode_resource_series"),
            ))
            no_improve_streak = 0 if improved else no_improve_streak + 1
            if patience > 0 and no_improve_streak >= patience:
                logger.info(
                    "Early stopping: no improvement in last %d sims (best=%.1f). "
                    "Stopping at sim %d/%d.",
                    patience, best_reward, sim, n_sims,
                )
                early_stopped  = True
                early_stop_sim = sim
                break
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")

    return best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim


def _maybe_adapt_scale(history, current, sim, window, up, down, mn, mx, enabled, out):
    """Mutates *out* to contain the new scale if adaptation fires."""
    if not enabled or len(history) < window or sim % window != 0:
        return
    p = sum(history) / window
    new = current
    if p > 1/5:
        new = min(current * up, mx)
    elif p < 1/5:
        new = max(current * down, mn)
    if new != current:
        logger.info("  [adaptive] scale %.4f → %.4f  (success_rate=%.2f)",
                    current, new, p)
    out.append(new)


def _greedy_loop_cmaes(
    env,
    policy,                # CMAESPolicy duck type: sample_population / update_distribution
    n_generations: int,
    weights_file: str,
    warmup_action: np.ndarray | None = None,
    warmup_steps: int = 0,
    patience: int = 0,
) -> tuple[Any, float, list[GreedySimResult], bool, int | None]:
    """CMA-ES loop: sample λ offspring, evaluate each for eval_episodes episodes, update distribution."""
    pop_size          = policy.population_size
    eval_episodes     = getattr(policy, "_eval_episodes", 1)
    best_reward       = policy.champion_reward
    greedy_sims: list[GreedySimResult] = []
    full_episode_time_s = env.get_episode_time_limit()
    no_improve_streak = 0
    early_stopped     = False
    early_stop_sim    = None

    logger.info(
        "[CMA-ES] population_size=%d, eval_episodes=%d, total episodes = %d × %d × %d = %d",
        pop_size, eval_episodes,
        n_generations, pop_size, eval_episodes,
        n_generations * pop_size * eval_episodes,
    )

    try:
        best_info_logged: dict = {}
        for gen in range(1, n_generations + 1):
            if full_episode_time_s is not None:
                env.set_episode_time_limit(_scaled_episode_time(
                    gen, n_generations, full_episode_time_s,
                ))
            offspring    = policy.sample_population()
            rewards      = []
            total_steps  = 0
            info: dict   = {}
            trace        = None
            # Track the replay for the best individual seen this generation.
            # Each candidate save overwrites _candidate.SC2Replay; we save
            # before the next env.reset() discards the current episode's replay.
            _gen_candidate: str | None = None
            _gen_candidate_reward: float = policy.champion_reward

            for individual in offspring:
                ep_rewards: list[float] = []
                for _ in range(eval_episodes):
                    obs, reset_info = env.reset()
                    reward, info, _, steps, trace = _run_episode(
                        env, individual, obs,
                        warmup_action=warmup_action, warmup_steps=warmup_steps,
                        reset_info=reset_info,
                    )
                    ep_rewards.append(reward)
                    total_steps += steps
                ind_avg = sum(ep_rewards) / len(ep_rewards)
                rewards.append(ind_avg)
                # Save replay for this individual if it beats the current best
                # candidate.  Must happen before the next env.reset().
                if ind_avg > _gen_candidate_reward:
                    _discard_candidate_replay(_gen_candidate)
                    _gen_candidate = _save_candidate_replay(env, weights_file)
                    _gen_candidate_reward = ind_avg

            improved = policy.update_distribution(rewards)
            gen_best = max(rewards)
            if gen_best > best_reward:
                best_reward = gen_best
            if improved:
                policy.save(weights_file)
                policy.save_trainer_state(_trainer_state_path(weights_file))
                _finalize_candidate_replay(_gen_candidate, weights_file)
                _gen_candidate = None
                verdict = (f"NEW BEST champion  reward={policy.champion_reward:+.1f}"
                           f"  sigma={policy.sigma:.4f}")
                logger.info("  >> %s", verdict)
                _log_new_best_details(info, best_info_logged)
                best_info_logged = info
            else:
                _discard_candidate_replay(_gen_candidate)
                verdict = (f"no improvement  gen_best={gen_best:+.1f}"
                           f"  champion={policy.champion_reward:+.1f}"
                           f"  sigma={policy.sigma:.4f}")
                logger.info("  >> %s", verdict)

            greedy_sims.append(GreedySimResult(
                sim=gen, reward=gen_best, improved=improved,
                throttle_counts=[0, 0, 0], total_steps=total_steps, trace=trace,
                weights=policy.to_cfg(),
                final_track_progress=info.get("track_progress", 0.0),
                laps_completed=info.get("laps_completed", 0),
                termination_reason=info.get("termination_reason"),
                finish_time_s=info.get("elapsed_s") if info.get("finished") else None,
                mean_abs_lateral_offset=info.get("mean_abs_lateral_offset"),
                reward_components=info.get("episode_reward_components"),
                action_counts=info.get("episode_action_counts"),
                obs_averages=info.get("episode_obs_averages"),
                xy_hist=info.get("episode_xy_hist"),
                skipped_frames=_episode_skipped_frames(info),
                supply_capped_fraction=info.get("episode_supply_capped_fraction"),
                build_order=info.get("episode_build_order"),
                army_count_series=info.get("episode_army_series"),
                resource_series=info.get("episode_resource_series"),
            ))
            no_improve_streak = 0 if improved else no_improve_streak + 1
            if patience > 0 and no_improve_streak >= patience:
                logger.info(
                    "Early stopping: no improvement in last %d gens (best=%.1f). "
                    "Stopping at gen %d/%d.",
                    patience, best_reward, gen, n_generations,
                )
                early_stopped  = True
                early_stop_sim = gen
                break
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")

    return policy, best_reward, greedy_sims, early_stopped, early_stop_sim


def _greedy_loop_q_learning(
    env,
    policy: BasePolicy,
    n_episodes: int,
    weights_file: str,
    warmup_action: np.ndarray | None = None,
    warmup_steps: int = 0,
    patience: int = 0,
) -> tuple[BasePolicy, float, list[GreedySimResult], bool, int | None]:
    """Q-learning greedy loop for epsilon_greedy and mcts policy types."""
    best_reward       = float("-inf")
    greedy_sims: list[GreedySimResult] = []
    full_episode_time_s = env.get_episode_time_limit()
    no_improve_streak = 0
    early_stopped     = False
    early_stop_sim    = None

    try:
        best_info_logged: dict = {}
        for episode in range(1, n_episodes + 1):
            if full_episode_time_s is not None:
                env.set_episode_time_limit(_scaled_episode_time(
                    episode, n_episodes, full_episode_time_s,
                ))
            obs, reset_info = env.reset()
            reward, info, tc, total_steps, trace = _run_episode(
                env, policy, obs,
                warmup_action=warmup_action, warmup_steps=warmup_steps,
                reset_info=reset_info,
            )
            policy.on_episode_end()

            improved = reward > best_reward
            if improved:
                prev_best   = best_reward
                best_reward = reward
                policy.save(weights_file)
                policy.save_trainer_state(_trainer_state_path(weights_file))
                _try_save_replay(env, weights_file)
                verdict = f"NEW BEST  {reward:+.1f}  (was {prev_best:+.1f})"
            else:
                verdict = f"no improvement  r={reward:+.1f}  best={best_reward:+.1f}"
            cfg = policy.to_cfg()
            logger.info("  >> %s  [states visited: %s]",
                        verdict, cfg.get("n_states_visited", "?"))
            if improved:
                _log_new_best_details(info, best_info_logged)
                best_info_logged = info

            greedy_sims.append(GreedySimResult(
                sim=episode, reward=reward, improved=improved,
                throttle_counts=list(tc), total_steps=total_steps, trace=trace,
                weights=cfg,
                final_track_progress=info.get("track_progress", 0.0),
                laps_completed=info.get("laps_completed", 0),
                termination_reason=info.get("termination_reason"),
                finish_time_s=info.get("elapsed_s") if info.get("finished") else None,
                mean_abs_lateral_offset=info.get("mean_abs_lateral_offset"),
                reward_components=info.get("episode_reward_components"),
                action_counts=info.get("episode_action_counts"),
                obs_averages=info.get("episode_obs_averages"),
                xy_hist=info.get("episode_xy_hist"),
                skipped_frames=_episode_skipped_frames(info),
                supply_capped_fraction=info.get("episode_supply_capped_fraction"),
                build_order=info.get("episode_build_order"),
                army_count_series=info.get("episode_army_series"),
                resource_series=info.get("episode_resource_series"),
            ))
            no_improve_streak = 0 if improved else no_improve_streak + 1
            if patience > 0 and no_improve_streak >= patience:
                logger.info(
                    "Early stopping: no improvement in last %d episodes (best=%.1f). "
                    "Stopping at episode %d/%d.",
                    patience, best_reward, episode, n_episodes,
                )
                early_stopped  = True
                early_stop_sim = episode
                break
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")

    return policy, best_reward, greedy_sims, early_stopped, early_stop_sim


def _greedy_loop_genetic(
    env,
    policy: GeneticPolicy,
    n_generations: int,
    weights_file: str,
    warmup_action: np.ndarray | None = None,
    warmup_steps: int = 0,
    patience: int = 0,
    adaptive_mutation: bool = True,
) -> tuple[GeneticPolicy, float, list[GreedySimResult], bool, int | None]:
    """Genetic algorithm loop: N_pop episodes per generation.

    When *adaptive_mutation* is True, the population's mutation scale is
    adjusted every ``ADAPT_WINDOW`` generations using a 1/5 success rule
    (same heuristic as the hill-climbing loop): if the champion improved in
    more than 1/5 of the recent window, scale up; if less, scale down.
    """
    ADAPT_WINDOW = 10
    ADAPT_UP     = 1.2
    ADAPT_DOWN   = 0.85
    SCALE_MIN    = 0.001
    SCALE_MAX    = 2.0
    improvement_history: deque[bool] = deque(maxlen=ADAPT_WINDOW)

    pop_size          = len(policy.population)
    eval_episodes     = getattr(policy, "_eval_episodes", 1)
    best_reward       = policy.champion_reward
    greedy_sims: list[GreedySimResult] = []
    full_episode_time_s = env.get_episode_time_limit()
    no_improve_streak = 0
    early_stopped     = False
    early_stop_sim    = None

    logger.info(
        "[Genetic] population_size=%d, eval_episodes=%d, total episodes = %d × %d × %d = %d",
        pop_size, eval_episodes,
        n_generations, pop_size, eval_episodes,
        n_generations * pop_size * eval_episodes,
    )
    if full_episode_time_s is None:
        logger.info("[Genetic] environment has no adjustable episode time limit; "
                    "skipping per-generation time scaling.")

    try:
        best_info_logged: dict = {}
        for gen in range(1, n_generations + 1):
            if full_episode_time_s is not None:
                env.set_episode_time_limit(_scaled_episode_time(
                    gen, n_generations, full_episode_time_s,
                ))
            rewards      = []
            total_steps  = 0
            trace        = None
            info: dict   = {}
            # Track the replay for the best individual seen this generation.
            _gen_candidate: str | None = None
            _gen_candidate_reward: float = policy.champion_reward

            for idx, individual in enumerate(policy.population):
                ep_rewards: list[float] = []
                for _ in range(eval_episodes):
                    obs, reset_info = env.reset()
                    reward, info, _, steps, trace = _run_episode(
                        env, individual, obs,
                        warmup_action=warmup_action, warmup_steps=warmup_steps,
                        reset_info=reset_info,
                    )
                    ep_rewards.append(reward)
                    total_steps += steps
                ind_avg = sum(ep_rewards) / len(ep_rewards)
                rewards.append(ind_avg)
                # Save replay for this individual if it beats the current best
                # candidate.  Must happen before the next env.reset().
                if ind_avg > _gen_candidate_reward:
                    _discard_candidate_replay(_gen_candidate)
                    _gen_candidate = _save_candidate_replay(env, weights_file)
                    _gen_candidate_reward = ind_avg

            improved = policy.evaluate_and_evolve(rewards)
            gen_best = max(rewards)
            if gen_best > best_reward:
                best_reward = gen_best
            if improved:
                policy.save(weights_file)
                policy.save_trainer_state(_trainer_state_path(weights_file))
                _finalize_candidate_replay(_gen_candidate, weights_file)
                _gen_candidate = None
                verdict = f"NEW BEST champion  reward={policy.champion_reward:+.1f}"
                logger.info("  >> %s", verdict)
                _log_new_best_details(info, best_info_logged)
                best_info_logged = info
            else:
                _discard_candidate_replay(_gen_candidate)
                verdict = (f"no improvement  gen_best={gen_best:+.1f}"
                           f"  champion={policy.champion_reward:+.1f}")
                logger.info("  >> %s", verdict)

            # --- adaptive mutation (1/5 success rule over recent window) ---
            improvement_history.append(improved)
            if (adaptive_mutation
                    and len(improvement_history) == ADAPT_WINDOW
                    and gen % ADAPT_WINDOW == 0):
                p = sum(improvement_history) / ADAPT_WINDOW
                prev_scale = policy.mutation_scale
                if p > 1 / 5:
                    new_scale = min(prev_scale * ADAPT_UP, SCALE_MAX)
                elif p < 1 / 5:
                    new_scale = max(prev_scale * ADAPT_DOWN, SCALE_MIN)
                else:
                    new_scale = prev_scale
                if new_scale != prev_scale:
                    logger.info(
                        "  [adaptive] mutation_scale %.4f → %.4f  (success_rate=%.2f)",
                        prev_scale, new_scale, p,
                    )
                    policy.mutation_scale = new_scale

            greedy_sims.append(GreedySimResult(
                sim=gen, reward=gen_best, improved=improved,
                throttle_counts=[0, 0, 0], total_steps=total_steps, trace=trace,
                weights=policy.to_cfg(),
                mutation_scale=policy.mutation_scale,
                final_track_progress=info.get("track_progress", 0.0),
                laps_completed=info.get("laps_completed", 0),
                termination_reason=info.get("termination_reason"),
                finish_time_s=info.get("elapsed_s") if info.get("finished") else None,
                mean_abs_lateral_offset=info.get("mean_abs_lateral_offset"),
                reward_components=info.get("episode_reward_components"),
                action_counts=info.get("episode_action_counts"),
                obs_averages=info.get("episode_obs_averages"),
                xy_hist=info.get("episode_xy_hist"),
                skipped_frames=_episode_skipped_frames(info),
                supply_capped_fraction=info.get("episode_supply_capped_fraction"),
                build_order=info.get("episode_build_order"),
                army_count_series=info.get("episode_army_series"),
                resource_series=info.get("episode_resource_series"),
            ))
            no_improve_streak = 0 if improved else no_improve_streak + 1
            if patience > 0 and no_improve_streak >= patience:
                logger.info(
                    "Early stopping: no improvement in last %d gens (best=%.1f). "
                    "Stopping at gen %d/%d.",
                    patience, best_reward, gen, n_generations,
                )
                early_stopped  = True
                early_stop_sim = gen
                break
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")

    return policy, best_reward, greedy_sims, early_stopped, early_stop_sim


# ---------------------------------------------------------------------------
# Top-level train_rl
# ---------------------------------------------------------------------------

def train_rl(
    game: GameSpec,
    config: RunConfig,
    *,
    probe: ProbeSpec | None = None,
    warmup: WarmupSpec | None = None,
    extras: PolicyExtras | None = None,
    no_interrupt: bool = False,
    re_initialize: bool = False,
) -> ExperimentData:
    """
    Train a policy via the selected algorithm.

    Parameters
    ----------
    game : GameSpec
        Game/track binding (env factory, obs spec, etc.).
    config : RunConfig
        Algorithm-level settings (n_sims, mutation_scale, policy_type, etc.).
    probe : ProbeSpec | None
        Probe + cold-start config.  None skips both phases.
    warmup : WarmupSpec | None
        Forced-action warmup at episode start.  None skips warmup.
    extras : PolicyExtras | None
        Game-specific extra policy types and loop dispatch.
    """

    # ── unpack bundles into local scalars for internal helpers ────────
    experiment_name  = game.experiment_name
    make_env_fn      = game.make_env_fn
    obs_spec         = game.obs_spec
    head_names       = game.head_names
    discrete_actions = game.discrete_actions
    weights_file     = game.weights_file
    reward_config_file = game.reward_config_file
    save_results_fn  = game.save_results_fn
    track            = game.track

    speed             = config.speed
    n_sims            = config.n_sims
    in_game_episode_s = config.in_game_episode_s
    mutation_scale    = config.mutation_scale
    mutation_share    = config.mutation_share
    adaptive_mutation = config.adaptive_mutation
    do_pretrain       = config.do_pretrain
    patience          = config.patience
    policy_type       = config.policy_type
    policy_params     = dict(config.policy_params)
    training_params   = config.training_params

    if probe is not None:
        probe_actions       = probe.actions
        probe_in_game_s     = probe.probe_in_game_s
        cold_start_restarts = probe.cold_start_restarts
        cold_start_sims     = probe.cold_start_sims
    else:
        probe_actions       = []
        probe_in_game_s     = 0.0
        cold_start_restarts = 0
        cold_start_sims     = 0

    if warmup is not None:
        warmup_action = warmup.action
        warmup_steps  = warmup.steps
    else:
        warmup_action = None
        warmup_steps  = 0

    if extras is not None:
        extra_policy_types  = extras.factories
        extra_loop_dispatch = extras.loop_dispatch
    else:
        extra_policy_types  = None
        extra_loop_dispatch = None

    policy_params = policy_params or {}
    probe_actions = probe_actions or []
    t_start       = datetime.datetime.now()

    cold_start = (
        (not os.path.exists(weights_file) or re_initialize)
        and policy_type == "hill_climbing"
        and len(probe_actions) > 0
    )

    _will_pretrain = (
        do_pretrain
        and policy_type == "hill_climbing"
        and not os.path.exists(weights_file)
        and not re_initialize
    )

    if _will_pretrain and not no_interrupt:
        input("\n  [PRE-TRAIN]  Press Enter to connect and start behavior cloning from SimplePolicy...")
    elif cold_start and not no_interrupt:
        input("\n  [PROBE PHASE]  Press Enter to connect and start probe runs...")

    logger.info("Connecting to game...")
    env = make_env_fn()

    pretrained = False
    if _will_pretrain:
        from rl.pretrain import run as _pretrain_run
        _pretrain_run(env, experiment_dir=os.path.dirname(os.path.abspath(weights_file)), obs_spec=obs_spec)
        pretrained = True

    probe_results: list[ProbeResult]             = []
    cold_start_data: list[ColdStartRestartResult] = []
    probe_best    = None
    t_after_probe = t_after_cold = None

    if cold_start and not pretrained:
        probe_best, probe_results = _run_probes(
            env, probe_actions, probe_in_game_s, speed,
            warmup_action=warmup_action, warmup_steps=warmup_steps,
        )
        t_after_probe = datetime.datetime.now()

        if not no_interrupt:
            input("\n  [COLD-START SEARCH]  Press Enter to start random-restart search...")
        time.sleep(1)
        best_policy, best_reward, cold_start_data = _cold_start_search(
            env, obs_spec, head_names, probe_best, weights_file,
            mutation_scale, mutation_share=mutation_share,
            n_restarts=cold_start_restarts, sims_per_restart=cold_start_sims,
            warmup_action=warmup_action, warmup_steps=warmup_steps,
        )
        t_after_cold = datetime.datetime.now()
    else:
        best_policy = _make_policy(
            policy_type    = policy_type,
            obs_spec       = obs_spec,
            head_names     = head_names,
            discrete_actions = discrete_actions,
            weights_file   = weights_file,
            policy_params  = {**policy_params,
                              "_mutation_scale_fallback": mutation_scale},
            re_initialize  = re_initialize,
            extra_policy_types = extra_policy_types,
        )
        best_reward = float("-inf")

    logger.info(
        "=== Training — %d sims, speed=%sx, episode=%ss in-game ===",
        n_sims, speed, in_game_episode_s,
    )
    logger.info("    policy_type=%s  mutation_scale=%s  weights → %s",
                policy_type, mutation_scale, weights_file)

    if not no_interrupt:
        input("\n  [GREEDY PHASE]  Press Enter to start optimisation...\n")
    time.sleep(1)
    t_greedy_start = datetime.datetime.now()

    kw = dict(warmup_action=warmup_action, warmup_steps=warmup_steps)

    _extra_dispatch = extra_loop_dispatch or {}

    loop_type = (best_policy.LOOP_TYPE if best_policy.POLICY_TYPE
                 else _extra_dispatch.get(policy_type))

    if loop_type == "hill_climbing" or loop_type is None:
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop(
            env=env, policy=best_policy, n_sims=n_sims,
            mutation_scale=mutation_scale, mutation_share=mutation_share,
            best_reward=best_reward, weights_file=weights_file,
            adaptive_mutation=adaptive_mutation, patience=patience, **kw,
        )
    elif loop_type == "q_learning":
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop_q_learning(
            env=env, policy=best_policy, n_episodes=n_sims,
            weights_file=weights_file, patience=patience, **kw,
        )
    elif loop_type == "cmaes":
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop_cmaes(
            env=env, policy=best_policy,
            n_generations=n_sims, weights_file=weights_file, patience=patience, **kw,
        )
    elif loop_type == "genetic":
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop_genetic(
            env=env, policy=best_policy,  # type: ignore[arg-type]
            n_generations=n_sims, weights_file=weights_file,
            patience=patience, adaptive_mutation=adaptive_mutation, **kw,  # type: ignore[arg-type]
        )
    else:
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop(
            env=env, policy=best_policy, n_sims=n_sims,
            mutation_scale=mutation_scale, mutation_share=mutation_share,
            best_reward=best_reward, weights_file=weights_file,
            adaptive_mutation=adaptive_mutation, patience=patience, **kw,
        )

    env.close()

    logger.info("=== Training complete — best total reward: %+.1f ===", best_reward)

    t_end  = datetime.datetime.now()
    fmt    = "%Y-%m-%d %H:%M:%S"
    timings = {
        "start":        t_start.strftime(fmt),
        "end":          t_end.strftime(fmt),
        "total_s":      (t_end - t_start).total_seconds(),
        "probe_s":      (t_after_probe - t_start).total_seconds()       if t_after_probe else None,
        "cold_start_s": (t_after_cold - t_after_probe).total_seconds()  if t_after_cold and t_after_probe else None,
        "greedy_s":     (t_end - t_greedy_start).total_seconds(),
    }

    data = ExperimentData(
        experiment_name    = experiment_name,
        probe_results      = probe_results,
        cold_start_restarts = cold_start_data,
        greedy_sims        = greedy_sims,
        probe_floor        = probe_best,
        weights_file       = weights_file,
        reward_config_file = reward_config_file,
        training_params    = training_params or {},
        timings            = timings,
        track              = track,
        early_stopped      = early_stopped,
        early_stop_sim     = early_stop_sim,
        code_version       = code_version(),
    )

    if save_results_fn is not None:
        experiment_dir = os.path.dirname(weights_file)
        save_results_fn(data, os.path.join(experiment_dir, "results"))

    return data
