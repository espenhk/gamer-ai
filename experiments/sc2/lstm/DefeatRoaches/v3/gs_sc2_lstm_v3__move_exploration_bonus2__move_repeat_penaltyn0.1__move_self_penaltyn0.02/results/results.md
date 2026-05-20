# Experiment: gs_sc2_lstm_v3__move_exploration_bonus2__move_repeat_penaltyn0.1__move_self_penaltyn0.02

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 15:21:47
- **End:** 2026-05-08 15:58:07
- **Total runtime:** 36m 20.4s

| Phase | Duration |
|-------|----------|
| Greedy | 36m 19.4s |

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
| move_exploration_bonus | 2.0 |
| move_repeat_penalty | -0.1 |
| move_self_penalty | -0.02 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+1731.0**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +468.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +547.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1238.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |   +513.7 | 0.000    | —           | —       | finish       |  |
|    5 |   +504.5 | 0.000    | —           | —       | finish       |  |
|    6 |  +1191.6 | 0.000    | —           | —       | finish       |  |
|    7 |  +1481.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    8 |  +1147.3 | 0.000    | —           | —       | finish       |  |
|    9 |  +1510.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   10 |  +1465.8 | 0.000    | —           | —       | finish       |  |
|   11 |   +517.2 | 0.000    | —           | —       | finish       |  |
|   12 |  +1271.4 | 0.000    | —           | —       | finish       |  |
|   13 |   +689.0 | 0.000    | —           | —       | finish       |  |
|   14 |  +1454.9 | 0.000    | —           | —       | finish       |  |
|   15 |  +1662.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   16 |  +1672.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   17 |  +1543.9 | 0.000    | —           | —       | finish       |  |
|   18 |   +713.7 | 0.000    | —           | —       | finish       |  |
|   19 |  +1569.7 | 0.000    | —           | —       | finish       |  |
|   20 |  +1659.2 | 0.000    | —           | —       | finish       |  |
|   21 |  +1647.4 | 0.000    | —           | —       | finish       |  |
|   22 |   +807.4 | 0.000    | —           | —       | finish       |  |
|   23 |  +1705.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   24 |  +1730.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   25 |  +1731.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   26 |  +1723.0 | 0.000    | —           | —       | finish       |  |
|   27 |   +619.9 | 0.000    | —           | —       | finish       |  |
|   28 |  +1083.9 | 0.000    | —           | —       | finish       |  |
|   29 |  +1616.0 | 0.000    | —           | —       | finish       |  |
|   30 |  +1634.2 | 0.000    | —           | —       | finish       |  |
|   31 |  +1666.0 | 0.000    | —           | —       | finish       |  |
|   32 |  +1476.0 | 0.000    | —           | —       | finish       |  |
|   33 |  +1666.6 | 0.000    | —           | —       | finish       |  |
|   34 |  +1659.1 | 0.000    | —           | —       | finish       |  |
|   35 |   +450.5 | 0.000    | —           | —       | finish       |  |
|   36 |  +1388.9 | 0.000    | —           | —       | finish       |  |
|   37 |  +1715.0 | 0.000    | —           | —       | finish       |  |
|   38 |  +1691.1 | 0.000    | —           | —       | finish       |  |
|   39 |  +1713.0 | 0.000    | —           | —       | finish       |  |
|   40 |   +405.2 | 0.000    | —           | —       | finish       |  |
|   41 |   +563.8 | 0.000    | —           | —       | finish       |  |
|   42 |   +555.4 | 0.000    | —           | —       | finish       |  |
|   43 |   +487.1 | 0.000    | —           | —       | finish       |  |
|   44 |  +1639.4 | 0.000    | —           | —       | finish       |  |
|   45 |   +641.6 | 0.000    | —           | —       | finish       |  |
|   46 |  +1014.6 | 0.000    | —           | —       | finish       |  |
|   47 |  +1498.8 | 0.000    | —           | —       | finish       |  |
|   48 |  +1113.8 | 0.000    | —           | —       | finish       |  |
|   49 |  +1081.5 | 0.000    | —           | —       | finish       |  |
|   50 |  +1337.8 | 0.000    | —           | —       | finish       |  |

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

