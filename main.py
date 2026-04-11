from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from collections import deque
from typing import Any
import yaml
import numpy as np
import datetime

logger = logging.getLogger(__name__)


from clients.rl_client import ACTIONS
from constants import N_ACTIONS
from policies import (
    BasePolicy,
    WeightedLinearPolicy,
    NeuralNetPolicy,
    EpsilonGreedyPolicy,
    MCTSPolicy,
    GeneticPolicy,
    NeuralDQNPolicy,
    CMAESPolicy,
)
from rl.env import TMNFEnv, make_env
from rl.reward import RewardConfig
from analytics import (
    ProbeResult,
    RunTrace,
    ColdStartSimResult,
    ColdStartRestartResult,
    GreedySimResult,
    ExperimentData,
    save_experiment_results
)


# ---------------------------------------------------------------------------
# Probe actions — fixed action vectors for cold-start evaluation.
# Each is (action_array, description).  Coast actions are skipped (same as before).
# ---------------------------------------------------------------------------

_PROBE_ACTIONS: list[tuple[np.ndarray, str]] = [
    (np.array([-1., 0., 1.], dtype=np.float32), "brake left"),
    (np.array([ 0., 0., 1.], dtype=np.float32), "brake"),
    (np.array([ 1., 0., 1.], dtype=np.float32), "brake right"),
    (np.array([-1., 1., 0.], dtype=np.float32), "accel left"),
    (np.array([ 0., 1., 0.], dtype=np.float32), "accel"),
    (np.array([ 1., 1., 0.], dtype=np.float32), "accel right"),
]


# ---------------------------------------------------------------------------
# Constant-action policy (used by probe phase)
# ---------------------------------------------------------------------------

class _ConstantPolicy:
    """Always returns the same action — used during cold-start probing."""
    def __init__(self, action: np.ndarray) -> None:
        self._action = action
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._action
    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        pass


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def _make_policy(
    policy_type: str,
    weights_file: str,
    n_lidar_rays: int,
    policy_params: dict,
    re_initialize: bool,
) -> BasePolicy:
    """Construct the appropriate policy given type, file, and hyperparams."""

    if policy_type == "hill_climbing":
        if os.path.exists(weights_file) and not re_initialize:
            return WeightedLinearPolicy(weights_file, n_lidar_rays)
        rng = np.random.default_rng()
        obs_names = WeightedLinearPolicy.get_obs_names(n_lidar_rays)
        cfg = {
            "steer_weights": {n: float(rng.standard_normal()) for n in obs_names},
            "accel_weights": {n: float(rng.standard_normal()) for n in obs_names},
            "brake_weights": {n: float(rng.standard_normal()) for n in obs_names},
        }
        return WeightedLinearPolicy.from_cfg(cfg, n_lidar_rays)

    elif policy_type == "neural_net":
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as f:
                cfg = yaml.safe_load(f)
            if cfg.get("policy_type") == "neural_net":
                logger.info("[NeuralNetPolicy] loaded from %s", weights_file)
                return NeuralNetPolicy.from_cfg(cfg, n_lidar_rays)
        hidden = policy_params.get("hidden_sizes", [16, 16])
        logger.info("[NeuralNetPolicy] initialised random weights (hidden=%s)", hidden)
        return NeuralNetPolicy(hidden_sizes=hidden, n_lidar_rays=n_lidar_rays)

    elif policy_type == "epsilon_greedy":
        return EpsilonGreedyPolicy.from_cfg(policy_params, n_lidar_rays)

    elif policy_type == "mcts":
        return MCTSPolicy.from_cfg(policy_params, n_lidar_rays)

    elif policy_type == "genetic":
        pop_size = policy_params.get("population_size", 10)
        elite_k  = policy_params.get("elite_k", 3)
        policy   = GeneticPolicy(
            population_size = pop_size,
            elite_k         = elite_k,
            mutation_scale  = policy_params.get("mutation_scale",
                              policy_params.get("_mutation_scale_fallback", 0.1)),
            mutation_share  = policy_params.get("mutation_share",
                              policy_params.get("_mutation_share_fallback", 1.0)),
            n_lidar_rays    = n_lidar_rays,
        )
        if os.path.exists(weights_file) and not re_initialize:
            champion = WeightedLinearPolicy(weights_file, n_lidar_rays)
            policy.initialize_from_champion(champion)
            logger.info("[GeneticPolicy] seeded population from champion at %s", weights_file)
        else:
            policy.initialize_random()
            logger.info("[GeneticPolicy] random population of %d", pop_size)
        return policy

    elif policy_type == "neural_dqn":
        if os.path.exists(weights_file) and not re_initialize:
            with open(weights_file) as f:
                cfg = yaml.safe_load(f)
            if cfg.get("policy_type") == "neural_dqn":
                logger.info("[NeuralDQNPolicy] loaded from %s", weights_file)
                return NeuralDQNPolicy.from_cfg(cfg, n_lidar_rays)
        hidden = policy_params.get("hidden_sizes", [64, 64])
        logger.info("[NeuralDQNPolicy] initialised new network (hidden=%s)", hidden)
        return NeuralDQNPolicy(
            hidden_sizes        = hidden,
            replay_buffer_size  = policy_params.get("replay_buffer_size",  10000),
            batch_size          = policy_params.get("batch_size",          64),
            min_replay_size     = policy_params.get("min_replay_size",     500),
            target_update_freq  = policy_params.get("target_update_freq",  200),
            learning_rate       = policy_params.get("learning_rate",       0.001),
            epsilon_start       = policy_params.get("epsilon_start",       1.0),
            epsilon_end         = policy_params.get("epsilon_end",         0.05),
            epsilon_decay_steps = policy_params.get("epsilon_decay_steps", 5000),
            gamma               = policy_params.get("gamma",               0.99),
            n_lidar_rays        = n_lidar_rays,
        )

    else:
        raise ValueError(f"Unknown policy_type: {policy_type!r}. "
                         f"Choose from: hill_climbing, neural_net, epsilon_greedy, mcts, genetic, neural_dqn")
    elif policy_type == "cmaes":
        pop_size = policy_params.get("population_size", 20)
        sigma    = policy_params.get("initial_sigma", 0.3)
        policy   = CMAESPolicy(
            population_size = pop_size,
            initial_sigma   = sigma,
            n_lidar_rays    = n_lidar_rays,
        )
        if os.path.exists(weights_file) and not re_initialize:
            champion = WeightedLinearPolicy(weights_file, n_lidar_rays)
            policy.initialize_from_champion(champion)
            logger.info("[CMAESPolicy] seeded mean from champion at %s", weights_file)
        else:
            policy.initialize_random()
            logger.info("[CMAESPolicy] initialised with zero mean, σ=%.3f", sigma)
        return policy

    else:
        raise ValueError(f"Unknown policy_type: {policy_type!r}. "
                         f"Choose from: hill_climbing, neural_net, epsilon_greedy, mcts, genetic, cmaes")


# ---------------------------------------------------------------------------
# Probe phase: run each of the 9 actions for probe_in_game_s seconds,
# return the best reward as a baseline floor for hill-climbing.
# ---------------------------------------------------------------------------

def _run_probes(env: TMNFEnv, probe_in_game_s: float, speed: float) -> tuple[float, list[ProbeResult]]:
    saved_limit = env._max_episode_time_s
    env._max_episode_time_s = probe_in_game_s / speed

    logger.info("No weights file found — running %d probe episodes (%ss each) to establish a baseline.",
                N_ACTIONS, probe_in_game_s)

    results = {}  # probe_idx -> reward
    probe_results = []
    for i, (action_arr, action_name) in enumerate(_PROBE_ACTIONS):
        logger.info("Probe %d/%d: %s", i + 1, len(_PROBE_ACTIONS), action_name)
        obs, _ = env.reset()
        reward, _, throttle_counts, total_steps, trace = _run_episode(env, _ConstantPolicy(action_arr), obs)
        results[i] = reward
        probe_results.append(ProbeResult(action_idx=i, action_name=action_name, reward=reward, trace=trace))

    env._max_episode_time_s = saved_limit

    best_idx = max(results, key=lambda i: results[i])
    logger.info("Probe results:")
    for i, r in results.items():
        marker = " <-- best" if i == best_idx else ""
        logger.info("  action %d (%-15s)  reward=%+.1f%s", i, ACTIONS[i][3], r, marker)
    logger.info("Using probe best (%+.1f) as initial reward floor.", results[best_idx])

    time.sleep(1)

    return results[best_idx], probe_results


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

_TRACE_SAMPLE_EVERY = 2   # record position every N steps
_WARMUP_STEPS = 100        # 1 in-game second of forced straight acceleration at episode start
_WARMUP_ACTION = np.array([0., 1., 0.], dtype=np.float32)   # accel + straight, no brake


def _run_episode(
    env: TMNFEnv,
    policy: BasePolicy | _ConstantPolicy,
    obs: np.ndarray,
) -> tuple[float, dict[str, Any], list[int], int, RunTrace]:
    """Run one episode from *obs* until terminated/truncated.

    Calls policy.update() after each post-warmup step so that online policies
    (EpsilonGreedyPolicy, MCTSPolicy) can update their Q-tables in real time.

    Returns:
        total_reward    — float
        info            — final step info dict from env
        throttle_counts — [brake_steps, coast_steps, accel_steps]
        total_steps     — int
        trace           — RunTrace
    """
    total_reward = 0.0
    steps = 0
    info: dict[str, Any] = {}
    throttle_counts = [0, 0, 0]
    turning_steps = 0
    pos_x: list[float] = []
    pos_z: list[float] = []
    throttle_state: list[tuple[float, float]] = []
    prev_obs = obs

    while True:
        in_warmup = steps < _WARMUP_STEPS
        action = _WARMUP_ACTION if in_warmup else policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if not in_warmup:
            policy.update(prev_obs, action, reward, next_obs, terminated or truncated)

        prev_obs = next_obs
        obs      = next_obs

        accel_on = float(action[1]) >= 0.5
        brake_on = float(action[2]) >= 0.5
        if brake_on and not accel_on:
            t = 0   # brake only
        elif accel_on:
            t = 2   # accel (including accel+brake)
        else:
            t = 1   # coast
        throttle_counts[t] += 1
        throttle_state.append((float(action[1]), float(action[2])))
        if abs(float(action[0])) > 0.05:
            turning_steps += 1

        if steps % _TRACE_SAMPLE_EVERY == 0:
            pos_x.append(info["pos_x"])
            pos_z.append(info["pos_z"])

        if terminated or truncated:
            _print_episode_summary(info, steps, total_reward, truncated)
            _print_action_stats(throttle_counts, turning_steps, steps)
            break

    trace = RunTrace(pos_x=pos_x, pos_z=pos_z,
                     throttle_state=throttle_state, total_reward=total_reward)
    return total_reward, info, throttle_counts, steps, trace


def _scaled_episode_time(sim: int, n_total: int, max_time_s: float) -> float:
    """Return episode time for sim (1-indexed) using a 4-step schedule.

    First 25%  → 1/4 of max_time_s
    Next  25%  → 1/2
    Next  25%  → 3/4
    Final 25%  → full max_time_s
    """
    quarter = n_total / 4
    if sim <= quarter:
        return max_time_s * 0.25
    elif sim <= 2 * quarter:
        return max_time_s * 0.5
    elif sim <= 3 * quarter:
        return max_time_s * 0.75
    else:
        return max_time_s


def _print_episode_summary(
    info: dict[str, Any],
    steps: int,
    total_reward: float,
    truncated: bool,
) -> None:
    progress = 100 * float(info.get("track_progress", 0.0))
    laps_completed = int(info.get("laps_completed", 0))
    finished = bool(info.get("finished", False))
    if truncated:
        outcome = "truncated"
    elif finished:
        outcome = "finished"
    else:
        outcome = "terminated"
    logger.info(
        "episode end — %s  steps=%d  reward=%+.1f  progress=%5.1f%%  laps=%d",
        outcome, steps, total_reward, progress, laps_completed
    )


def _print_action_stats(throttle_counts: list[int], turning_steps: int, steps: int) -> None:
    if steps == 0:
        return
    b, c, a = throttle_counts
    logger.debug(
        "throttle — brake: %4.1f%%  coast: %4.1f%%  accel: %4.1f%%"
        "    steer — straight: %4.1f%%  turning: %4.1f%%",
        100*b/steps, 100*c/steps, 100*a/steps,
        100*(steps-turning_steps)/steps, 100*turning_steps/steps,
    )


# ---------------------------------------------------------------------------
# Watch mode: run indefinitely, resetting every in_game_episode_s seconds
# ---------------------------------------------------------------------------

def run_rl_policy(speed: float, policy: BasePolicy, in_game_episode_s: float = 20.0,
                  reward_config_file: str = "config/reward_config.yaml",
                  centerline_file: str = "tracks/a03_centerline.npy") -> None:
    """
    Repeatedly drive the track with *policy*, resetting every
    *in_game_episode_s* in-game seconds.  Ctrl+C to stop.
    """
    env = TMNFEnv(
        centerline_file=centerline_file,
        speed=speed,
        reward_config=RewardConfig.from_yaml(reward_config_file),
        max_episode_time_s=in_game_episode_s / speed,
    )
    time.sleep(1)

    run = 0
    try:
        while True:
            run += 1
            logger.info("--- Run %d --- (respawning)", run)
            obs, _ = env.reset()
            _run_episode(env, policy, obs)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Cold-start search: random restarts until a policy beats the probe floor
# ---------------------------------------------------------------------------

def _cold_start_search(
    env: TMNFEnv,
    probe_best_reward: float,
    weights_file: str,
    mutation_scale: float,
    mutation_share: float = 1.0,
    n_restarts: int = 5,
    sims_per_restart: int = 10,
    n_lidar_rays: int = 0,
) -> tuple[WeightedLinearPolicy, float, list[ColdStartRestartResult]]:
    """
    Try up to n_restarts random policy initializations.
    Each restart runs sims_per_restart hill-climb sims.
    Stops early if a policy beats probe_best_reward.
    Returns (best_policy, best_reward, restart_results) across all restarts.
    """

    overall_best_policy = None
    overall_best_reward = float("-inf")
    restart_results = []

    logger.info("=== Cold-start search — up to %d restarts × %d sims  target: %+.1f ===",
                n_restarts, sims_per_restart, probe_best_reward)

    for restart in range(1, n_restarts + 1):
        logger.info("-- Restart %d/%d: random init --", restart, n_restarts)

        rng = np.random.default_rng()
        obs_names = WeightedLinearPolicy.get_obs_names(n_lidar_rays)
        random_cfg = {
            "steer_weights": {n: float(rng.standard_normal()) for n in obs_names},
            "accel_weights": {n: float(rng.standard_normal()) for n in obs_names},
            "brake_weights": {n: float(rng.standard_normal()) for n in obs_names},
        }
        local_best_policy = WeightedLinearPolicy.from_cfg(random_cfg, n_lidar_rays=n_lidar_rays)
        local_best_reward = float("-inf")
        sim_results = []

        for sim in range(1, sims_per_restart + 1):
            candidate = local_best_policy.mutated(scale=mutation_scale, share=mutation_share)
            logger.debug("Restart %d sim %d/%d (respawning)", restart, sim, sims_per_restart)
            obs, _ = env.reset()
            reward, _, throttle_counts, total_steps, trace = _run_episode(env, candidate, obs)

            sim_results.append(ColdStartSimResult(
                sim=sim, reward=reward,
                throttle_counts=list(throttle_counts), total_steps=total_steps,
                trace=trace,
            ))

            if reward > local_best_reward:
                local_best_reward = reward
                local_best_policy = candidate

            if reward > overall_best_reward:
                overall_best_reward = reward
                overall_best_policy = candidate

        beat = local_best_reward > probe_best_reward
        logger.info("Restart %d best: %+.1f  (%s probe floor %+.1f)",
                    restart, local_best_reward, "beats" if beat else "below", probe_best_reward)

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
        overall_best_policy = WeightedLinearPolicy(weights_file)
        overall_best_policy.save(weights_file)
    logger.info("Cold-start complete — best reward: %+.1f  Weights saved to %s",
                overall_best_reward, weights_file)
    return overall_best_policy, overall_best_reward, restart_results


# ---------------------------------------------------------------------------
# Unified greedy loop (hill_climbing, neural_net, epsilon_greedy, mcts)
#
# hill_climbing / neural_net:  mutate the current best, keep if improved
# epsilon_greedy / mcts:       run the policy as-is; it updates its Q-table
#                              in-place via policy.update() inside _run_episode
# ---------------------------------------------------------------------------

_MUTATION_POLICIES = {"hill_climbing", "neural_net"}
_ONLINE_POLICIES   = {"epsilon_greedy", "mcts"}


def _greedy_loop(
    env: TMNFEnv,
    policy: BasePolicy,
    n_sims: int,
    mutation_scale: float,
    weights_file: str,
    best_reward: float = float("-inf"),
    learning_rate: float = 0.01,
    mutation_share: float = 1.0,
    adaptive_mutation: bool = True,
) -> tuple[BasePolicy, float, list[GreedySimResult]]:
    """
    ES gradient-estimation loop for WeightedLinearPolicy (hill_climbing policy type).

    Each sim evaluates two mirrored perturbations (+ε, −ε) and updates the weight
    vector using the reward difference as a gradient signal:

        θ += lr * (R⁺ − R⁻) * ε

    This means every episode pair contributes to the update, even when neither
    candidate beats the current best. Based on OpenAI Evolution Strategies
    (Salimans et al. 2017).

    Falls back to single-candidate greedy for policies without flat-weight support
    (neural_net).

    Returns (best_policy, best_reward, greedy_sims).
    """
    # 1/5th success rule constants
    ADAPT_WINDOW = 20
    ADAPT_UP     = 1.2
    ADAPT_DOWN   = 0.85
    SCALE_MIN    = 0.001
    SCALE_MAX    = 1.0
    current_scale = mutation_scale
    improvement_history: deque[bool] = deque(maxlen=ADAPT_WINDOW)

    best_policy = policy
    if isinstance(best_policy, NeuralNetPolicy):
        # Fallback: single-candidate greedy for neural_net
        greedy_sims = []
        try:
            for sim in range(1, n_sims + 1):
                candidate = best_policy.mutated(scale=current_scale)
                logger.info("--- Sim %d/%d --- (respawning)", sim, n_sims)
                obs, _ = env.reset()
                reward, info, throttle_counts, total_steps, trace = _run_episode(env, candidate, obs)
                improved = reward > best_reward
                if improved:
                    prev_best   = best_reward
                    best_reward = reward
                    best_policy = candidate
                    best_policy.save(weights_file)
                    verdict = f"NEW BEST  {reward:+.1f}  (was {prev_best:+.1f})"
                else:
                    verdict = f"no improvement  candidate={reward:+.1f}  best={best_reward:+.1f}"
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
                    sim=sim, reward=reward, improved=improved,
                    throttle_counts=list(throttle_counts), total_steps=total_steps,
                    trace=trace, weights=candidate.to_cfg(),
                    final_track_progress=info.get("track_progress", 0.0),
                    laps_completed=info.get("laps_completed", 0),
                    mutation_scale=current_scale,
                ))
        except KeyboardInterrupt:
            logger.warning("Training interrupted.")
        return best_policy, best_reward, greedy_sims

    if not isinstance(best_policy, WeightedLinearPolicy):
        raise TypeError(f"Unsupported policy for _greedy_loop: {type(best_policy).__name__}")

    # ES gradient update loop
    rng = np.random.default_rng()
    theta = best_policy.to_flat()
    greedy_sims = []
    full_episode_time_s = env._max_episode_time_s
    try:
        for sim in range(1, n_sims + 1):
            eps = rng.standard_normal(len(theta)).astype(np.float32) * current_scale

            policy_plus  = best_policy.with_flat(theta + eps)
            policy_minus = best_policy.with_flat(theta - eps)

            logger.debug("--- Sim %d/%d (+) --- (respawning)", sim, n_sims)
            env._max_episode_time_s = _scaled_episode_time(sim, n_sims, full_episode_time_s)
            candidate = best_policy.mutated(scale=current_scale, share=mutation_share)

            logger.debug("--- Sim %d/%d --- (respawning, episode_time=%.1fs)", sim, n_sims, env._max_episode_time_s)
            obs, _ = env.reset()
            r_plus, info_plus, tc_plus, steps_plus, trace_plus = _run_episode(env, policy_plus, obs)

            logger.debug("--- Sim %d/%d (-) --- (respawning)", sim, n_sims)
            obs, _ = env.reset()
            r_minus, info_minus, tc_minus, steps_minus, trace_minus = _run_episode(env, policy_minus, obs)

            # Gradient step — always update regardless of improvement
            theta += learning_rate * (r_plus - r_minus) * eps

            # Track best-seen candidate for checkpointing
            improved = False
            if r_plus >= r_minus:
                best_r, best_info, best_tc, best_steps, best_trace, best_candidate = (
                    r_plus, info_plus, tc_plus, steps_plus, trace_plus, policy_plus)
            else:
                best_r, best_info, best_tc, best_steps, best_trace, best_candidate = (
                    r_minus, info_minus, tc_minus, steps_minus, trace_minus, policy_minus)

            if best_r > best_reward:
                prev_best   = best_reward
                best_reward = best_r
                best_policy = best_candidate
                best_policy.save(weights_file)
                improved = True
                verdict = f"NEW BEST  {best_r:+.1f}  (was {prev_best:+.1f})  gradient_signal={r_plus - r_minus:+.1f}"
            else:
                verdict = (f"no improvement  +ε={r_plus:+.1f}  −ε={r_minus:+.1f}  "
                           f"best={best_reward:+.1f}  gradient_signal={r_plus - r_minus:+.1f}")

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
                trace=best_trace,
                weights=best_candidate.to_cfg(),
                final_track_progress=best_info.get("track_progress", 0.0),
                laps_completed=best_info.get("laps_completed", 0),
                mutation_scale=current_scale,
            ))
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")

    return best_policy, best_reward, greedy_sims


def _greedy_loop_q_learning(
    env: TMNFEnv,
    policy: BasePolicy,
    n_episodes: int,
    weights_file: str,
) -> tuple[BasePolicy, float, list[GreedySimResult]]:
    """
    Q-learning greedy loop for epsilon_greedy and mcts policy types.
    The policy updates its Q-table in-place via policy.update() inside _run_episode().
    No mutation is performed; the policy itself is the state that improves over time.
    Returns (policy, best_reward, greedy_sims).
    """

    best_reward = float("-inf")
    greedy_sims = []
    full_episode_time_s = env._max_episode_time_s
    try:
        for episode in range(1, n_episodes + 1):
            env._max_episode_time_s = _scaled_episode_time(episode, n_episodes, full_episode_time_s)
            logger.info("--- Episode %d/%d --- (respawning, episode_time=%.1fs)", episode, n_episodes, env._max_episode_time_s)
            obs, _ = env.reset()
            reward, info, throttle_counts, total_steps, trace = _run_episode(env, policy, obs)
            policy.on_episode_end()

            improved = reward > best_reward
            if improved:
                prev_best   = best_reward
                best_reward = reward
                policy.save(weights_file)
                verdict = f"NEW BEST  {reward:+.1f}  (was {prev_best:+.1f})"
            else:
                verdict = f"no improvement  episode={reward:+.1f}  best={best_reward:+.1f}"

            cfg = policy.to_cfg()
            logger.info("  >> %s  [states visited: %s]", verdict, cfg.get("n_states_visited", "?"))
            greedy_sims.append(GreedySimResult(
                sim=episode, reward=reward, improved=improved,
                throttle_counts=list(throttle_counts), total_steps=total_steps,
                trace=trace,
                weights=cfg,
                final_track_progress=info.get("track_progress", 0.0),
                laps_completed=info.get("laps_completed", 0),
            ))
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")

    return policy, best_reward, greedy_sims


# ---------------------------------------------------------------------------
# Genetic greedy loop (structurally different: N_pop episodes per generation)
# ---------------------------------------------------------------------------

def _greedy_loop_genetic(
    env: TMNFEnv,
    policy: GeneticPolicy,
    n_generations: int,
    weights_file: str,
) -> tuple[GeneticPolicy, float, list[GreedySimResult]]:
    """
    Genetic algorithm loop.
    Each "sim" is one generation: evaluate all population members, then evolve.
    Total episodes = n_generations × population_size.
    """
    pop_size    = len(policy.population)
    best_reward = policy.champion_reward
    greedy_sims: list[GreedySimResult] = []
    full_episode_time_s = env._max_episode_time_s

    logger.info("[Genetic] population_size=%d, total episodes = %d × %d = %d",
                pop_size, n_generations, pop_size, n_generations * pop_size)

    try:
        for gen in range(1, n_generations + 1):
            env._max_episode_time_s = _scaled_episode_time(gen, n_generations, full_episode_time_s)
            logger.info("--- Generation %d/%d --- evaluating %d individuals (episode_time=%.1fs)",
                        gen, n_generations, pop_size, env._max_episode_time_s)
            rewards = []
            total_steps = 0
            trace = None
            info: dict[str, Any] = {}
            for idx, individual in enumerate(policy.population):
                logger.debug("Individual %d/%d (respawning)", idx + 1, pop_size)
                obs, _ = env.reset()
                reward, info, _, steps, trace = _run_episode(env, individual, obs)
                rewards.append(reward)
                total_steps += steps

            improved = policy.evaluate_and_evolve(rewards)
            gen_best = max(rewards)
            if gen_best > best_reward:
                best_reward = gen_best

            if improved:
                policy.save(weights_file)
                verdict = f"NEW BEST champion  reward={policy.champion_reward:+.1f}"
            else:
                verdict = f"no improvement  gen_best={gen_best:+.1f}  champion={policy.champion_reward:+.1f}"

            logger.info("  >> %s", verdict)
            greedy_sims.append(GreedySimResult(
                sim=gen, reward=gen_best, improved=improved,
                throttle_counts=[0, 0, 0], total_steps=total_steps,
                trace=trace,
                weights=policy.to_cfg(),
                final_track_progress=info.get("track_progress", 0.0),
                laps_completed=info.get("laps_completed", 0),
            ))
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")

    return policy, best_reward, greedy_sims


# ---------------------------------------------------------------------------
# CMA-ES greedy loop (structurally similar to genetic: N_pop episodes per generation)
# ---------------------------------------------------------------------------

def _greedy_loop_cmaes(
    env: TMNFEnv,
    policy: CMAESPolicy,
    n_generations: int,
    weights_file: str,
) -> tuple[CMAESPolicy, float, list[GreedySimResult]]:
    """
    CMA-ES training loop.

    Each "sim" is one generation:
      1. sample_population() → draw λ offspring from N(mean, σ²·C)
      2. Evaluate each offspring for one episode
      3. update_distribution(rewards) → update mean, σ, C, paths

    Total episodes = n_generations × population_size.
    """
    best_reward = policy.champion_reward
    greedy_sims: list[GreedySimResult] = []
    full_episode_time_s = env._max_episode_time_s

    logger.info(
        "[CMAES] population_size=%d, total episodes = %d × %d = %d",
        policy.population_size, n_generations, policy.population_size,
        n_generations * policy.population_size,
    )

    try:
        for gen in range(1, n_generations + 1):
            env._max_episode_time_s = _scaled_episode_time(gen, n_generations, full_episode_time_s)
            population = policy.sample_population()
            pop_size   = len(population)
            logger.info(
                "--- Generation %d/%d --- evaluating %d individuals (σ=%.4f, episode_time=%.1fs)",
                gen, n_generations, pop_size, policy.sigma, env._max_episode_time_s,
            )

            rewards     = []
            total_steps = 0
            trace       = None
            info: dict[str, Any] = {}

            for idx, individual in enumerate(population):
                logger.debug("Individual %d/%d (respawning)", idx + 1, pop_size)
                obs, _ = env.reset()
                reward, info, _, steps, trace = _run_episode(env, individual, obs)
                rewards.append(reward)
                total_steps += steps

            improved = policy.update_distribution(rewards)
            gen_best = max(rewards)
            if gen_best > best_reward:
                best_reward = gen_best

            if improved:
                policy.save(weights_file)
                verdict = (
                    f"NEW BEST champion  reward={policy.champion_reward:+.1f}"
                    f"  σ={policy.sigma:.4f}"
                )
            else:
                verdict = (
                    f"no improvement  gen_best={gen_best:+.1f}"
                    f"  champion={policy.champion_reward:+.1f}  σ={policy.sigma:.4f}"
                )

            logger.info("  >> %s", verdict)
            greedy_sims.append(GreedySimResult(
                sim=gen, reward=gen_best, improved=improved,
                throttle_counts=[0, 0, 0], total_steps=total_steps,
                trace=trace,
                weights=policy.to_cfg(),
                final_track_progress=info.get("track_progress", 0.0),
                laps_completed=info.get("laps_completed", 0),
            ))
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")

    return policy, best_reward, greedy_sims


# ---------------------------------------------------------------------------
# Training orchestrator
# ---------------------------------------------------------------------------

def train_rl(
    experiment_name: str,
    speed: float,
    n_sims: int = 10,
    in_game_episode_s: float = 20.0,
    weights_file: str = "config/policy_weights.yaml",
    reward_config_file: str = "config/reward_config.yaml",
    mutation_scale: float = 0.1,
    mutation_share: float = 1.0,
    probe_in_game_s: float = 8.0,
    cold_start_restarts: int = 5,
    cold_start_sims: int = 10,
    training_params: dict[str, Any] | None = None,
    no_interrupt: bool = False,
    n_lidar_rays: int = 0,
    re_initialize: bool = False,
    policy_type: str = "hill_climbing",
    policy_params: dict[str, Any] | None = None,
    track: str = "",
    adaptive_mutation: bool = True,
) -> ExperimentData:
    """
    Train a driving policy via the selected algorithm.

    policy_type options:
      hill_climbing  — random weight mutation, keep if improved (default)
      neural_net     — MLP policy, same hill-climbing loop
      epsilon_greedy — tabular Q-learning with epsilon-greedy exploration
      mcts           — UCT-style online Q-learner with UCB1 action selection
      genetic        — population of linear policies, evolutionary selection
    """

    policy_params = policy_params or {}
    t_start = datetime.datetime.now()

    cold_start = (not os.path.exists(weights_file) or re_initialize)
    cold_start = cold_start and (policy_type == "hill_climbing")

    if cold_start and not no_interrupt:
        input("\n  [PROBE PHASE]  Press Enter to connect and start probe runs...")

    logger.info("Connecting to game...")
    experiment_dir = os.path.dirname(weights_file)
    env = make_env(
        experiment_dir=experiment_dir,
        speed=speed,
        in_game_episode_s=in_game_episode_s,
        n_lidar_rays=n_lidar_rays,
    )

    probe_results: list[ProbeResult] = []
    cold_start_data: list[ColdStartRestartResult] = []
    probe_best = None
    t_after_probe = t_after_cold = None

    if cold_start:
        probe_best, probe_results = _run_probes(env, probe_in_game_s=probe_in_game_s, speed=speed)
        t_after_probe = datetime.datetime.now()

        if not no_interrupt:
            input("\n  [COLD-START SEARCH]  Press Enter to start random-restart search...")
        time.sleep(1)
        best_policy, best_reward, cold_start_data = _cold_start_search(
            env, probe_best, weights_file, mutation_scale,
            mutation_share=mutation_share,
            n_restarts=cold_start_restarts, sims_per_restart=cold_start_sims,
            n_lidar_rays=n_lidar_rays,
        )
        t_after_cold = datetime.datetime.now()
    else:
        best_policy = _make_policy(
            policy_type    = policy_type,
            weights_file   = weights_file,
            n_lidar_rays   = n_lidar_rays,
            policy_params  = {**policy_params,
                              "_mutation_scale_fallback": mutation_scale},
            re_initialize  = re_initialize,
        )
        best_reward = float("-inf")

    logger.info("=== Training — %d sims/generations, speed=%sx, episode=%ss in-game ===",
                n_sims, speed, in_game_episode_s)
    logger.info("    policy_type=%s  mutation_scale=%s  mutation_share=%s  weights → %s",
                policy_type, mutation_scale, mutation_share, weights_file)
    if not no_interrupt:
        input("\n  [GREEDY PHASE]  Press Enter to start optimisation...\n")
    time.sleep(1)
    t_greedy_start = datetime.datetime.now()

    # Dispatch to the appropriate greedy loop
    if policy_type in ("hill_climbing", "neural_net"):
        best_policy, best_reward, greedy_sims = _greedy_loop(
            env=env,
            policy=best_policy,
            n_sims=n_sims,
            mutation_scale=mutation_scale,
            mutation_share=mutation_share,
            best_reward=best_reward,
            weights_file=weights_file,
            adaptive_mutation=adaptive_mutation,
        )
    elif policy_type in ("epsilon_greedy", "mcts"):
        best_policy, best_reward, greedy_sims = _greedy_loop_q_learning(
            env=env,
            policy=best_policy,
            n_episodes=n_sims,
            weights_file=weights_file
        )
    elif policy_type == "genetic":
        best_policy, best_reward, greedy_sims = _greedy_loop_genetic(
            env=env,
            policy=best_policy, # type: ignore[arg-type]
            n_generations=n_sims,
            weights_file=weights_file
        )
    elif policy_type == "cmaes":
        best_policy, best_reward, greedy_sims = _greedy_loop_cmaes(
            env=env,
            policy=best_policy,  # type: ignore[arg-type]
            n_generations=n_sims,
            weights_file=weights_file,
        )
    else:
        best_policy, best_reward, greedy_sims = _greedy_loop(
            env=env,
            policy=best_policy,
            n_sims=n_sims,
            mutation_scale=mutation_scale,
            mutation_share=mutation_share,
            weights_file=weights_file,
            best_reward=best_reward,
            adaptive_mutation=adaptive_mutation,
        )

    env.close()

    logger.info("=== Training complete — best total reward: %+.1f ===", best_reward)
    logger.info("  %4s  %8s  Result", "Sim", "Reward")
    for s in greedy_sims:
        tag = "NEW BEST" if s.improved else ""
        logger.info("  %4d  %8.1f  %s", s.sim, s.reward, tag)

    t_end = datetime.datetime.now()
    fmt = "%Y-%m-%d %H:%M:%S"
    timings = {
        "start":        t_start.strftime(fmt),
        "end":          t_end.strftime(fmt),
        "total_s":      (t_end - t_start).total_seconds(),
        "probe_s":      (t_after_probe - t_start).total_seconds()        if t_after_probe else None,
        "cold_start_s": (t_after_cold - t_after_probe).total_seconds()   if t_after_cold and t_after_probe else None,
        "greedy_s":     (t_end - t_greedy_start).total_seconds(),
    }

    return ExperimentData(
        experiment_name=experiment_name,
        probe_results=probe_results,
        cold_start_restarts=cold_start_data,
        greedy_sims=greedy_sims,
        probe_floor=probe_best,
        weights_file=weights_file,
        reward_config_file=reward_config_file,
        training_params=training_params or {},
        timings=timings,
        track=track,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TMNF RL training")
    parser.add_argument("experiment", help="Experiment name — files stored in experiments/<track>/<name>/")
    parser.add_argument("--no-interrupt", action="store_true",
                        help="Skip all 'Press Enter' prompts and run all phases automatically")
    parser.add_argument("--re-initialize", action="store_true",
                        help="Ignore any existing weights file and restart from scratch, "
                             "including probe and cold-start phases.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Bootstrap: read track from master config before the experiment dir exists,
    # then re-read the experiment-local copy once it has been created.
    with open("config/training_params.yaml") as f:
        master_p = yaml.safe_load(f)
    track = master_p.get("track", "a03_centerline")

    experiment_dir  = f"experiments/{track}/{args.experiment}"
    weights_file    = f"{experiment_dir}/policy_weights.yaml"
    reward_cfg_file = f"{experiment_dir}/reward_config.yaml"

    training_params_file = f"{experiment_dir}/training_params.yaml"

    os.makedirs(experiment_dir, exist_ok=True)
    if not os.path.exists(reward_cfg_file):
        shutil.copy("config/reward_config.yaml", reward_cfg_file)
        logger.info("Copied master reward config → %s", reward_cfg_file)
    if not os.path.exists(training_params_file):
        shutil.copy("config/training_params.yaml", training_params_file)
        logger.info("Copied master training params → %s", training_params_file)

    with open(training_params_file) as f:
        p = yaml.safe_load(f)

    # Allow per-experiment overrides of track (if the copied config was edited).
    track = p.get("track", track)

    data = train_rl(
        experiment_name=args.experiment,
        speed=p["speed"],
        n_sims=p["n_sims"],
        in_game_episode_s=p["in_game_episode_s"],
        weights_file=weights_file,
        reward_config_file=reward_cfg_file,
        mutation_scale=p["mutation_scale"],
        mutation_share=p.get("mutation_share", 1.0),
        probe_in_game_s=p["probe_s"],
        cold_start_restarts=p["cold_restarts"],
        cold_start_sims=p["cold_sims"],
        training_params=p,
        no_interrupt=args.no_interrupt,
        n_lidar_rays=p.get("n_lidar_rays", 0),
        re_initialize=args.re_initialize,
        policy_type=p.get("policy_type", "hill_climbing"),
        policy_params=p.get("policy_params") or {},
        track=track,
        adaptive_mutation=p.get("adaptive_mutation", True),
    )

    save_experiment_results(data, results_dir=f"{experiment_dir}/results")


if __name__ == "__main__":
    main()
