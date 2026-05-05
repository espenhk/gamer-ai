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
)
from framework.obs_spec import ObsSpec
from framework.run_config import GameSpec, RunConfig, ProbeSpec, WarmupSpec, PolicyExtras

logger = logging.getLogger(__name__)

_TRACE_SAMPLE_EVERY = 2   # record position every N steps


def _trainer_state_path(weights_file: str) -> str:
    """Return the trainer-state checkpoint path alongside the weights file.

    e.g. experiments/a03/my_run/policy_weights.yaml
         → experiments/a03/my_run/trainer_state.npz
    """
    return os.path.join(os.path.dirname(os.path.abspath(weights_file)), "trainer_state.npz")


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

    extra_policy_types maps policy_type names to zero-arg factory callables.
    This lets game-specific policy types (e.g. 'neural_dqn', 'cmaes') be
    injected without the framework importing from games/.
    """

    if policy_type == "hill_climbing":
        if re_initialize and os.path.exists(weights_file):
            os.remove(weights_file)
            logger.info("[WeightedLinearPolicy] removed existing weights file for re-initialization: %s",
                        weights_file)
        return WeightedLinearPolicy(obs_spec, head_names, weights_file)

    elif policy_type == "neural_net":
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as f:
                loaded_cfg = yaml.safe_load(f)
            cfg = loaded_cfg if isinstance(loaded_cfg, dict) else {}
            if cfg.get("policy_type") == "neural_net":
                logger.info("[NeuralNetPolicy] loaded from %s", weights_file)
                return NeuralNetPolicy.from_cfg(cfg, obs_spec)
        hidden = policy_params.get("hidden_sizes", [16, 16])
        logger.info("[NeuralNetPolicy] initialised random weights (hidden=%s)", hidden)
        return NeuralNetPolicy(obs_spec, action_dim=len(head_names),
                               hidden_sizes=hidden)

    elif policy_type == "epsilon_greedy":
        epsilon = policy_params.get("epsilon", 1.0)
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as f:
                saved_cfg = yaml.safe_load(f) or {}
            epsilon = float(saved_cfg.get("epsilon", epsilon))
        policy = EpsilonGreedyPolicy(
            obs_spec=obs_spec,
            discrete_actions=discrete_actions,
            n_bins=policy_params.get("n_bins", 3),
            epsilon=epsilon,
            epsilon_decay=policy_params.get("epsilon_decay", 0.995),
            epsilon_min=policy_params.get("epsilon_min", 0.05),
            alpha=policy_params.get("alpha", 0.1),
            gamma=policy_params.get("gamma", 0.99),
        )
        if os.path.exists(weights_file) and not re_initialize:
            policy._load_table(weights_file)
        return policy

    elif policy_type == "mcts":
        policy = MCTSPolicy(
            obs_spec=obs_spec,
            discrete_actions=discrete_actions,
            c=policy_params.get("c", 1.41),
            alpha=policy_params.get("alpha", 0.1),
            gamma=policy_params.get("gamma", 0.99),
            n_bins=policy_params.get("n_bins", 3),
        )
        if os.path.exists(weights_file) and not re_initialize:
            policy._load_table(weights_file)
        return policy

    elif policy_type == "genetic":
        pop_size = policy_params.get("population_size", 10)
        elite_k  = policy_params.get("elite_k", 3)
        policy   = GeneticPolicy(
            obs_spec       = obs_spec,
            head_names     = head_names,
            population_size = pop_size,
            elite_k        = elite_k,
            mutation_scale = policy_params.get("mutation_scale",
                             policy_params.get("_mutation_scale_fallback", 0.1)),
            mutation_share = policy_params.get("mutation_share",
                             policy_params.get("_mutation_share_fallback", 1.0)),
            eval_episodes  = policy_params.get("eval_episodes", 1),
        )
        if os.path.exists(weights_file) and not re_initialize:
            champion = WeightedLinearPolicy(obs_spec, head_names, weights_file)
            policy.initialize_from_champion(champion)
            logger.info("[GeneticPolicy] seeded population from champion at %s",
                        weights_file)
        else:
            policy.initialize_random()
            logger.info("[GeneticPolicy] random population of %d", pop_size)
        return policy

    elif extra_policy_types and policy_type in extra_policy_types:
        return extra_policy_types[policy_type]()

    else:
        raise ValueError(
            f"Unknown policy_type: {policy_type!r}. "
            "Choose from: hill_climbing, neural_net, epsilon_greedy, mcts, genetic"
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
    progress = 100 * float(info.get("track_progress", 0.0))
    laps     = int(info.get("laps_completed", 0))
    finished = bool(info.get("finished", False))
    outcome  = "truncated" if truncated else ("finished" if finished else "terminated")
    logger.info(
        "episode end — %s  steps=%d  reward=%+.1f  progress=%5.1f%%  laps=%d",
        outcome, steps, total_reward, progress, laps,
    )


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
                    verdict = f"NEW BEST  {reward:+.1f}  (was {prev_best:+.1f})"
                else:
                    verdict = f"no improvement  candidate={reward:+.1f}  best={best_reward:+.1f}"
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
            else:
                best_r, best_info, best_tc, best_steps, best_trace, best_cand = (
                    r_minus, info_minus, tc_minus, steps_minus, trace_minus, policy_minus)

            improved = False
            if best_r > best_reward:
                prev_best   = best_reward
                best_reward = best_r
                best_policy = best_cand
                best_policy.save(weights_file)
                best_policy.save_trainer_state(_trainer_state_path(weights_file))
                improved    = True
                verdict = (f"NEW BEST  {best_r:+.1f}  (was {prev_best:+.1f})"
                           f"  gradient={r_plus - r_minus:+.1f}")
            else:
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
                rewards.append(sum(ep_rewards) / len(ep_rewards))

            improved = policy.update_distribution(rewards)
            gen_best = max(rewards)
            if gen_best > best_reward:
                best_reward = gen_best
            if improved:
                policy.save(weights_file)
                policy.save_trainer_state(_trainer_state_path(weights_file))
                verdict = (f"NEW BEST champion  reward={policy.champion_reward:+.1f}"
                           f"  sigma={policy.sigma:.4f}")
            else:
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
                verdict = f"NEW BEST  {reward:+.1f}  (was {prev_best:+.1f})"
            else:
                verdict = f"no improvement  episode={reward:+.1f}  best={best_reward:+.1f}"
            cfg = policy.to_cfg()
            logger.info("  >> %s  [states visited: %s]",
                        verdict, cfg.get("n_states_visited", "?"))

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
) -> tuple[GeneticPolicy, float, list[GreedySimResult], bool, int | None]:
    """Genetic algorithm loop: N_pop episodes per generation."""
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
        for gen in range(1, n_generations + 1):
            if full_episode_time_s is not None:
                env.set_episode_time_limit(_scaled_episode_time(
                    gen, n_generations, full_episode_time_s,
                ))
            rewards      = []
            total_steps  = 0
            trace        = None
            info: dict   = {}

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
                rewards.append(sum(ep_rewards) / len(ep_rewards))

            improved = policy.evaluate_and_evolve(rewards)
            gen_best = max(rewards)
            if gen_best > best_reward:
                best_reward = gen_best
            if improved:
                policy.save(weights_file)
                policy.save_trainer_state(_trainer_state_path(weights_file))
                verdict = f"NEW BEST champion  reward={policy.champion_reward:+.1f}"
            else:
                verdict = (f"no improvement  gen_best={gen_best:+.1f}"
                           f"  champion={policy.champion_reward:+.1f}")
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

    if policy_type in ("hill_climbing", "neural_net"):
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop(
            env=env, policy=best_policy, n_sims=n_sims,
            mutation_scale=mutation_scale, mutation_share=mutation_share,
            best_reward=best_reward, weights_file=weights_file,
            adaptive_mutation=adaptive_mutation, patience=patience, **kw,
        )
    elif policy_type in ("epsilon_greedy", "mcts"):
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop_q_learning(
            env=env, policy=best_policy, n_episodes=n_sims,
            weights_file=weights_file, patience=patience, **kw,
        )
    elif policy_type == "genetic":
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop_genetic(
            env=env, policy=best_policy,  # type: ignore[arg-type]
            n_generations=n_sims, weights_file=weights_file, patience=patience, **kw,
        )
    elif _extra_dispatch.get(policy_type) == "q_learning":
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop_q_learning(
            env=env, policy=best_policy, n_episodes=n_sims,
            weights_file=weights_file, patience=patience, **kw,
        )
    elif _extra_dispatch.get(policy_type) == "cmaes":
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop_cmaes(
            env=env, policy=best_policy,
            n_generations=n_sims, weights_file=weights_file, patience=patience, **kw,
        )
    elif _extra_dispatch.get(policy_type) == "genetic":
        best_policy, best_reward, greedy_sims, early_stopped, early_stop_sim = _greedy_loop_genetic(
            env=env, policy=best_policy,  # type: ignore[arg-type]
            n_generations=n_sims, weights_file=weights_file, patience=patience, **kw,  # type: ignore[arg-type]
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
    )

    if save_results_fn is not None:
        experiment_dir = os.path.dirname(weights_file)
        save_results_fn(data, os.path.join(experiment_dir, "results"))

    return data
