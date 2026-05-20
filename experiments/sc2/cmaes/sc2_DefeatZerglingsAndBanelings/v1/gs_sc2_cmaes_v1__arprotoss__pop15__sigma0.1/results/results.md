# Experiment: gs_sc2_cmaes_v1__arprotoss__pop15__sigma0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-14 07:53:55
- **End:** 2026-05-14 11:12:08
- **Total runtime:** 3h 18m 13.6s

| Phase | Duration |
|-------|----------|
| Greedy | 3h 18m 12.5s |

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
| agent_race | protoss |
| n_sims | 50 |
| policy_type | cmaes |
| obs_spec_preset | rich |
| enable_belief | True |
| population_size | 15 |
| initial_sigma | 0.1 |
| policy_params | {'eval_episodes': 5, 'population_size': 15, 'initial_sigma': 0.1} |

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

Best reward: **+532.3**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +214.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +155.2 | 0.000    | —           | —       | finish       |  |
|    3 |   +229.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |   +268.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |    +36.7 | 0.000    | —           | —       | finish       |  |
|    6 |   +195.3 | 0.000    | —           | —       | finish       |  |
|    7 |   +131.4 | 0.000    | —           | —       | finish       |  |
|    8 |   +207.1 | 0.000    | —           | —       | finish       |  |
|    9 |   +306.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   10 |   +221.7 | 0.000    | —           | —       | finish       |  |
|   11 |   +254.7 | 0.000    | —           | —       | finish       |  |
|   12 |   +241.4 | 0.000    | —           | —       | finish       |  |
|   13 |   +295.6 | 0.000    | —           | —       | finish       |  |
|   14 |   +363.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   15 |   +205.3 | 0.000    | —           | —       | finish       |  |
|   16 |   +297.2 | 0.000    | —           | —       | finish       |  |
|   17 |   +310.4 | 0.000    | —           | —       | finish       |  |
|   18 |   +401.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   19 |   +312.8 | 0.000    | —           | —       | finish       |  |
|   20 |   +367.1 | 0.000    | —           | —       | finish       |  |
|   21 |   +199.6 | 0.000    | —           | —       | finish       |  |
|   22 |   +220.5 | 0.000    | —           | —       | finish       |  |
|   23 |   +294.0 | 0.000    | —           | —       | finish       |  |
|   24 |   +473.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   25 |   +284.5 | 0.000    | —           | —       | finish       |  |
|   26 |   +337.1 | 0.000    | —           | —       | finish       |  |
|   27 |   +165.8 | 0.000    | —           | —       | finish       |  |
|   28 |   +274.2 | 0.000    | —           | —       | finish       |  |
|   29 |   +357.2 | 0.000    | —           | —       | finish       |  |
|   30 |   +532.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   31 |   +330.0 | 0.000    | —           | —       | finish       |  |
|   32 |   +291.2 | 0.000    | —           | —       | finish       |  |
|   33 |   +429.3 | 0.000    | —           | —       | finish       |  |
|   34 |   +364.6 | 0.000    | —           | —       | finish       |  |
|   35 |   +299.3 | 0.000    | —           | —       | finish       |  |
|   36 |   +446.0 | 0.000    | —           | —       | finish       |  |
|   37 |   +302.1 | 0.000    | —           | —       | finish       |  |
|   38 |   +394.6 | 0.000    | —           | —       | finish       |  |
|   39 |   +309.6 | 0.000    | —           | —       | finish       |  |
|   40 |   +444.8 | 0.000    | —           | —       | finish       |  |
|   41 |   +433.6 | 0.000    | —           | —       | finish       |  |
|   42 |   +404.6 | 0.000    | —           | —       | finish       |  |
|   43 |   +433.3 | 0.000    | —           | —       | finish       |  |
|   44 |   +425.8 | 0.000    | —           | —       | finish       |  |
|   45 |   +397.9 | 0.000    | —           | —       | finish       |  |
|   46 |   +509.7 | 0.000    | —           | —       | finish       |  |
|   47 |   +427.8 | 0.000    | —           | —       | finish       |  |
|   48 |   +516.1 | 0.000    | —           | —       | finish       |  |
|   49 |   +303.4 | 0.000    | —           | —       | finish       |  |
|   50 |   +475.8 | 0.000    | —           | —       | finish       |  |

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

