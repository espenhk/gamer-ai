# Experiment: gs_sc2_lstm_v3__move_exploration_bonus0.5__move_repeat_penaltyn0.05__move_self_penaltyn0.02

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 04:27:08
- **End:** 2026-05-08 05:14:25
- **Total runtime:** 47m 17.0s

| Phase | Duration |
|-------|----------|
| Greedy | 47m 16.0s |

## Run Parameters

### Training

| Parameter | Value |
|-----------|-------|
| track | sc2_DefeatRoaches |
| map_name | DefeatRoaches |
| in_game_episode_s | 120.0 |
| step_mul | 8 |
| screen_size | 64 |
| minimap_size | 64 |
| max_apm | 300 |
| agent_race | random |
| n_sims | 50 |
| policy_type | lstm |
| obs_spec_preset | rich |
| enable_belief | True |
| hidden_size | 128 |
| initial_sigma | 0.1 |
| policy_params | {'population_size': 20, 'hidden_size': 128, 'initial_sigma': 0.1} |

### Reward Config

| Parameter | Value |
|-----------|-------|
| score_weight | 1.0 |
| win_bonus | 20.0 |
| loss_penalty | 0.0 |
| step_penalty | -0.001 |
| idle_penalty | 0.0 |
| idle_bonus | 1.0 |
| move_exploration_bonus | 0.5 |
| move_repeat_penalty | -0.05 |
| move_self_penalty | -0.02 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1734.9**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1169.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1718.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1023.3 | 0.000    | —           | —       | finish       |  |
|    4 |  +1126.0 | 0.000    | —           | —       | finish       |  |
|    5 |  +1102.1 | 0.000    | —           | —       | finish       |  |
|    6 |   +805.9 | 0.000    | —           | —       | finish       |  |
|    7 |  +1079.0 | 0.000    | —           | —       | finish       |  |
|    8 |  +1729.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |  +1702.5 | 0.000    | —           | —       | finish       |  |
|   10 |  +1725.7 | 0.000    | —           | —       | finish       |  |
|   11 |  +1718.1 | 0.000    | —           | —       | finish       |  |
|   12 |  +1703.2 | 0.000    | —           | —       | finish       |  |
|   13 |  +1587.0 | 0.000    | —           | —       | finish       |  |
|   14 |  +1702.1 | 0.000    | —           | —       | finish       |  |
|   15 |  +1235.6 | 0.000    | —           | —       | finish       |  |
|   16 |  +1438.5 | 0.000    | —           | —       | finish       |  |
|   17 |  +1719.8 | 0.000    | —           | —       | finish       |  |
|   18 |  +1717.7 | 0.000    | —           | —       | finish       |  |
|   19 |  +1734.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   20 |  +1710.9 | 0.000    | —           | —       | finish       |  |
|   21 |  +1702.1 | 0.000    | —           | —       | finish       |  |
|   22 |  +1691.4 | 0.000    | —           | —       | finish       |  |
|   23 |  +1491.0 | 0.000    | —           | —       | finish       |  |
|   24 |  +1196.0 | 0.000    | —           | —       | finish       |  |
|   25 |  +1339.0 | 0.000    | —           | —       | finish       |  |
|   26 |  +1243.9 | 0.000    | —           | —       | finish       |  |
|   27 |  +1718.1 | 0.000    | —           | —       | finish       |  |
|   28 |  +1726.9 | 0.000    | —           | —       | finish       |  |
|   29 |  +1726.9 | 0.000    | —           | —       | finish       |  |
|   30 |  +1726.9 | 0.000    | —           | —       | finish       |  |
|   31 |  +1678.1 | 0.000    | —           | —       | finish       |  |
|   32 |  +1707.0 | 0.000    | —           | —       | finish       |  |
|   33 |  +1707.0 | 0.000    | —           | —       | finish       |  |
|   34 |  +1290.5 | 0.000    | —           | —       | finish       |  |
|   35 |  +1412.2 | 0.000    | —           | —       | finish       |  |
|   36 |  +1325.7 | 0.000    | —           | —       | finish       |  |
|   37 |  +1709.3 | 0.000    | —           | —       | finish       |  |
|   38 |  +1733.9 | 0.000    | —           | —       | finish       |  |
|   39 |  +1734.1 | 0.000    | —           | —       | finish       |  |
|   40 |  +1726.9 | 0.000    | —           | —       | finish       |  |
|   41 |  +1710.9 | 0.000    | —           | —       | finish       |  |
|   42 |  +1706.7 | 0.000    | —           | —       | finish       |  |
|   43 |  +1714.7 | 0.000    | —           | —       | finish       |  |
|   44 |  +1674.5 | 0.000    | —           | —       | finish       |  |
|   45 |  +1450.7 | 0.000    | —           | —       | finish       |  |
|   46 |  +1467.2 | 0.000    | —           | —       | finish       |  |
|   47 |   +705.8 | 0.000    | —           | —       | finish       |  |
|   48 |  +1428.0 | 0.000    | —           | —       | finish       |  |
|   49 |  +1709.8 | 0.000    | —           | —       | finish       |  |
|   50 |  +1734.9 | 0.000    | —           | —       | finish       | **NEW BEST** |

![Greedy rewards](greedy_rewards.png)


![Reward components](reward_components.png)


![Action frequency](action_frequency.png)


![Game-state averages](obs_averages.png)


![Spatial target heatmap](spatial_heatmap.png)


![Outcome breakdown](outcome_breakdown.png)


![Time supply-capped](supply_capped.png)


![Resources available over time](resource_series.png)


![Army count over time](army_count.png)


![Reward trajectory](reward_trajectory.png)

