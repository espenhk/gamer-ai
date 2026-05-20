# Experiment: gs_sc2_lstm_v3__move_exploration_bonus1__move_repeat_penaltyn0.05__move_self_penaltyn0.05

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 10:08:21
- **End:** 2026-05-08 10:50:31
- **Total runtime:** 42m 10.4s

| Phase | Duration |
|-------|----------|
| Greedy | 42m 09.4s |

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
| move_exploration_bonus | 1.0 |
| move_repeat_penalty | -0.05 |
| move_self_penalty | -0.05 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+1728.1**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +577.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1726.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1716.9 | 0.000    | —           | —       | finish       |  |
|    4 |  +1150.5 | 0.000    | —           | —       | finish       |  |
|    5 |  +1251.3 | 0.000    | —           | —       | finish       |  |
|    6 |  +1684.5 | 0.000    | —           | —       | finish       |  |
|    7 |  +1705.3 | 0.000    | —           | —       | finish       |  |
|    8 |  +1728.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |   +732.3 | 0.000    | —           | —       | finish       |  |
|   10 |   +830.9 | 0.000    | —           | —       | finish       |  |
|   11 |   +968.3 | 0.000    | —           | —       | finish       |  |
|   12 |   +839.7 | 0.000    | —           | —       | finish       |  |
|   13 |  +1401.1 | 0.000    | —           | —       | finish       |  |
|   14 |  +1271.9 | 0.000    | —           | —       | finish       |  |
|   15 |  +1294.5 | 0.000    | —           | —       | finish       |  |
|   16 |  +1509.3 | 0.000    | —           | —       | finish       |  |
|   17 |  +1681.7 | 0.000    | —           | —       | finish       |  |
|   18 |  +1222.9 | 0.000    | —           | —       | finish       |  |
|   19 |  +1487.3 | 0.000    | —           | —       | finish       |  |
|   20 |  +1152.3 | 0.000    | —           | —       | finish       |  |
|   21 |  +1320.3 | 0.000    | —           | —       | finish       |  |
|   22 |  +1454.5 | 0.000    | —           | —       | finish       |  |
|   23 |  +1085.7 | 0.000    | —           | —       | finish       |  |
|   24 |   +993.9 | 0.000    | —           | —       | finish       |  |
|   25 |   +538.9 | 0.000    | —           | —       | finish       |  |
|   26 |   +641.1 | 0.000    | —           | —       | finish       |  |
|   27 |  +1498.9 | 0.000    | —           | —       | finish       |  |
|   28 |  +1049.5 | 0.000    | —           | —       | finish       |  |
|   29 |  +1142.2 | 0.000    | —           | —       | finish       |  |
|   30 |  +1208.9 | 0.000    | —           | —       | finish       |  |
|   31 |  +1236.9 | 0.000    | —           | —       | finish       |  |
|   32 |  +1647.9 | 0.000    | —           | —       | finish       |  |
|   33 |   +982.5 | 0.000    | —           | —       | finish       |  |
|   34 |  +1356.5 | 0.000    | —           | —       | finish       |  |
|   35 |  +1047.8 | 0.000    | —           | —       | finish       |  |
|   36 |    +93.7 | 0.000    | —           | —       | finish       |  |
|   37 |   +926.1 | 0.000    | —           | —       | finish       |  |
|   38 |  +1288.6 | 0.000    | —           | —       | finish       |  |
|   39 |  +1225.1 | 0.000    | —           | —       | finish       |  |
|   40 |  +1686.9 | 0.000    | —           | —       | finish       |  |
|   41 |  +1223.7 | 0.000    | —           | —       | finish       |  |
|   42 |  +1230.5 | 0.000    | —           | —       | finish       |  |
|   43 |   +951.9 | 0.000    | —           | —       | finish       |  |
|   44 |  +1438.1 | 0.000    | —           | —       | finish       |  |
|   45 |  +1470.5 | 0.000    | —           | —       | finish       |  |
|   46 |  +1718.9 | 0.000    | —           | —       | finish       |  |
|   47 |  +1254.1 | 0.000    | —           | —       | finish       |  |
|   48 |  +1081.3 | 0.000    | —           | —       | finish       |  |
|   49 |  +1262.1 | 0.000    | —           | —       | finish       |  |
|   50 |  +1349.3 | 0.000    | —           | —       | finish       |  |

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

