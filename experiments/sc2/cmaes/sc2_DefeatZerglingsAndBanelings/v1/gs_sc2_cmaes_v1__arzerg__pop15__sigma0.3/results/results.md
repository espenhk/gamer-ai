# Experiment: gs_sc2_cmaes_v1__arzerg__pop15__sigma0.3

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-11 17:49:02
- **End:** 2026-05-11 21:38:55
- **Total runtime:** 3h 49m 52.4s

| Phase | Duration |
|-------|----------|
| Greedy | 3h 49m 51.4s |

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
| population_size | 15 |
| initial_sigma | 0.3 |
| policy_params | {'eval_episodes': 5, 'population_size': 15, 'initial_sigma': 0.3} |

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

Best reward: **+737.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +288.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |    +77.1 | 0.000    | —           | —       | finish       |  |
|    3 |   +275.4 | 0.000    | —           | —       | finish       |  |
|    4 |   +296.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |   +105.5 | 0.000    | —           | —       | finish       |  |
|    6 |   +249.9 | 0.000    | —           | —       | finish       |  |
|    7 |   +285.1 | 0.000    | —           | —       | finish       |  |
|    8 |   +415.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |   +519.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   10 |   +389.1 | 0.000    | —           | —       | finish       |  |
|   11 |   +328.7 | 0.000    | —           | —       | finish       |  |
|   12 |   +342.7 | 0.000    | —           | —       | finish       |  |
|   13 |   +390.3 | 0.000    | —           | —       | finish       |  |
|   14 |   +259.1 | 0.000    | —           | —       | finish       |  |
|   15 |   +370.6 | 0.000    | —           | —       | finish       |  |
|   16 |   +403.6 | 0.000    | —           | —       | finish       |  |
|   17 |   +243.9 | 0.000    | —           | —       | finish       |  |
|   18 |   +391.8 | 0.000    | —           | —       | finish       |  |
|   19 |   +437.1 | 0.000    | —           | —       | finish       |  |
|   20 |   +344.7 | 0.000    | —           | —       | finish       |  |
|   21 |   +518.8 | 0.000    | —           | —       | finish       |  |
|   22 |   +304.2 | 0.000    | —           | —       | finish       |  |
|   23 |   +464.9 | 0.000    | —           | —       | finish       |  |
|   24 |   +427.9 | 0.000    | —           | —       | finish       |  |
|   25 |   +293.1 | 0.000    | —           | —       | finish       |  |
|   26 |   +411.8 | 0.000    | —           | —       | finish       |  |
|   27 |   +552.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   28 |   +430.6 | 0.000    | —           | —       | finish       |  |
|   29 |   +427.9 | 0.000    | —           | —       | finish       |  |
|   30 |   +467.9 | 0.000    | —           | —       | finish       |  |
|   31 |   +407.0 | 0.000    | —           | —       | finish       |  |
|   32 |   +479.7 | 0.000    | —           | —       | finish       |  |
|   33 |   +505.3 | 0.000    | —           | —       | finish       |  |
|   34 |   +349.7 | 0.000    | —           | —       | finish       |  |
|   35 |   +483.2 | 0.000    | —           | —       | finish       |  |
|   36 |   +430.1 | 0.000    | —           | —       | finish       |  |
|   37 |   +380.6 | 0.000    | —           | —       | finish       |  |
|   38 |   +324.1 | 0.000    | —           | —       | finish       |  |
|   39 |   +414.9 | 0.000    | —           | —       | finish       |  |
|   40 |   +472.9 | 0.000    | —           | —       | finish       |  |
|   41 |   +322.5 | 0.000    | —           | —       | finish       |  |
|   42 |   +462.3 | 0.000    | —           | —       | finish       |  |
|   43 |   +373.6 | 0.000    | —           | —       | finish       |  |
|   44 |   +402.8 | 0.000    | —           | —       | finish       |  |
|   45 |   +391.3 | 0.000    | —           | —       | finish       |  |
|   46 |   +717.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   47 |   +345.9 | 0.000    | —           | —       | finish       |  |
|   48 |   +737.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   49 |   +389.8 | 0.000    | —           | —       | finish       |  |
|   50 |   +539.7 | 0.000    | —           | —       | finish       |  |

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

