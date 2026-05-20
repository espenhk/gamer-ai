# Experiment: gs_sc2_cmaes_v1__arzerg__pop25__sigma0.5

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-12 13:17:20
- **End:** 2026-05-12 19:33:05
- **Total runtime:** 6h 15m 45.0s

| Phase | Duration |
|-------|----------|
| Greedy | 6h 15m 44.0s |

## Run Parameters

### Training

| Parameter | Value |
|-----------|-------|
| track | sc2_DefeatZerglingsAndBanelings |
| map_name | DefeatZerglingsAndBanelings |
| in_game_episode_s | 120.0 |
| step_mul | 8 |
| screen_size | 64 |
| minimap_size | 64 |
| max_apm | 300 |
| agent_race | zerg |
| n_sims | 50 |
| policy_type | cmaes |
| obs_spec_preset | rich |
| enable_belief | True |
| population_size | 25 |
| initial_sigma | 0.5 |
| policy_params | {'eval_episodes': 5, 'population_size': 25, 'initial_sigma': 0.5} |

### Reward Config

| Parameter | Value |
|-----------|-------|
| score_weight | 10.0 |
| win_bonus | 1000.0 |
| loss_penalty | -100.0 |
| step_penalty | -0.001 |
| idle_penalty | 0.0 |
| idle_bonus | 0.5 |
| move_exploration_bonus | 1.0 |
| move_repeat_penalty | -0.05 |
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+651.5**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +324.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +561.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +448.4 | 0.000    | —           | —       | finish       |  |
|    4 |   +290.3 | 0.000    | —           | —       | finish       |  |
|    5 |   +361.3 | 0.000    | —           | —       | finish       |  |
|    6 |   +373.4 | 0.000    | —           | —       | finish       |  |
|    7 |   +327.8 | 0.000    | —           | —       | finish       |  |
|    8 |   +535.1 | 0.000    | —           | —       | finish       |  |
|    9 |   +374.1 | 0.000    | —           | —       | finish       |  |
|   10 |   +466.0 | 0.000    | —           | —       | finish       |  |
|   11 |   +406.3 | 0.000    | —           | —       | finish       |  |
|   12 |   +424.0 | 0.000    | —           | —       | finish       |  |
|   13 |   +364.5 | 0.000    | —           | —       | finish       |  |
|   14 |   +349.4 | 0.000    | —           | —       | finish       |  |
|   15 |   +415.4 | 0.000    | —           | —       | finish       |  |
|   16 |   +430.4 | 0.000    | —           | —       | finish       |  |
|   17 |   +367.5 | 0.000    | —           | —       | finish       |  |
|   18 |   +439.4 | 0.000    | —           | —       | finish       |  |
|   19 |   +416.5 | 0.000    | —           | —       | finish       |  |
|   20 |   +439.5 | 0.000    | —           | —       | finish       |  |
|   21 |   +585.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   22 |   +486.9 | 0.000    | —           | —       | finish       |  |
|   23 |   +459.1 | 0.000    | —           | —       | finish       |  |
|   24 |   +551.4 | 0.000    | —           | —       | finish       |  |
|   25 |   +375.5 | 0.000    | —           | —       | finish       |  |
|   26 |   +322.1 | 0.000    | —           | —       | finish       |  |
|   27 |   +431.0 | 0.000    | —           | —       | finish       |  |
|   28 |   +402.0 | 0.000    | —           | —       | finish       |  |
|   29 |   +562.1 | 0.000    | —           | —       | finish       |  |
|   30 |   +462.3 | 0.000    | —           | —       | finish       |  |
|   31 |   +459.6 | 0.000    | —           | —       | finish       |  |
|   32 |   +457.5 | 0.000    | —           | —       | finish       |  |
|   33 |   +537.1 | 0.000    | —           | —       | finish       |  |
|   34 |   +385.8 | 0.000    | —           | —       | finish       |  |
|   35 |   +401.8 | 0.000    | —           | —       | finish       |  |
|   36 |   +553.2 | 0.000    | —           | —       | finish       |  |
|   37 |   +568.4 | 0.000    | —           | —       | finish       |  |
|   38 |   +623.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   39 |   +590.8 | 0.000    | —           | —       | finish       |  |
|   40 |   +494.3 | 0.000    | —           | —       | finish       |  |
|   41 |   +460.9 | 0.000    | —           | —       | finish       |  |
|   42 |   +541.4 | 0.000    | —           | —       | finish       |  |
|   43 |   +391.4 | 0.000    | —           | —       | finish       |  |
|   44 |   +479.6 | 0.000    | —           | —       | finish       |  |
|   45 |   +651.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   46 |   +441.4 | 0.000    | —           | —       | finish       |  |
|   47 |   +576.2 | 0.000    | —           | —       | finish       |  |
|   48 |   +565.7 | 0.000    | —           | —       | finish       |  |
|   49 |   +517.6 | 0.000    | —           | —       | finish       |  |
|   50 |   +558.7 | 0.000    | —           | —       | finish       |  |

![Greedy rewards](greedy_rewards.png)


![Reward components](reward_components.png)


![Action frequency](action_frequency.png)


![Game-state averages](obs_averages.png)


![Spatial target heatmap](spatial_heatmap.png)


![Outcome breakdown](outcome_breakdown.png)


![Skipped frames](skipped_frames.png)


![Time supply-capped](supply_capped.png)


![Resources available over time](resource_series.png)


![Army count over time](army_count.png)


![Reward trajectory](reward_trajectory.png)

