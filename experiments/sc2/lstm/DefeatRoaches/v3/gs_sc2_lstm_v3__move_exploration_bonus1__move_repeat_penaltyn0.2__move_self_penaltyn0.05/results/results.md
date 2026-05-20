# Experiment: gs_sc2_lstm_v3__move_exploration_bonus1__move_repeat_penaltyn0.2__move_self_penaltyn0.05

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 06:01:15
- **End:** 2026-05-08 06:49:19
- **Total runtime:** 48m 04.5s

| Phase | Duration |
|-------|----------|
| Greedy | 48m 03.5s |

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
| move_repeat_penalty | -0.2 |
| move_self_penalty | -0.05 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1733.5**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1625.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1255.9 | 0.000    | —           | —       | finish       |  |
|    3 |  +1239.9 | 0.000    | —           | —       | finish       |  |
|    4 |  +1701.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |  +1725.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1559.6 | 0.000    | —           | —       | finish       |  |
|    7 |  +1714.1 | 0.000    | —           | —       | finish       |  |
|    8 |   +915.8 | 0.000    | —           | —       | finish       |  |
|    9 |  +1072.5 | 0.000    | —           | —       | finish       |  |
|   10 |  +1672.1 | 0.000    | —           | —       | finish       |  |
|   11 |  +1621.3 | 0.000    | —           | —       | finish       |  |
|   12 |  +1571.5 | 0.000    | —           | —       | finish       |  |
|   13 |  +1680.5 | 0.000    | —           | —       | finish       |  |
|   14 |  +1676.5 | 0.000    | —           | —       | finish       |  |
|   15 |  +1708.1 | 0.000    | —           | —       | finish       |  |
|   16 |  +1694.5 | 0.000    | —           | —       | finish       |  |
|   17 |  +1668.5 | 0.000    | —           | —       | finish       |  |
|   18 |  +1654.5 | 0.000    | —           | —       | finish       |  |
|   19 |  +1650.5 | 0.000    | —           | —       | finish       |  |
|   20 |  +1691.9 | 0.000    | —           | —       | finish       |  |
|   21 |  +1718.7 | 0.000    | —           | —       | finish       |  |
|   22 |  +1733.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   23 |  +1696.5 | 0.000    | —           | —       | finish       |  |
|   24 |  +1714.1 | 0.000    | —           | —       | finish       |  |
|   25 |  +1696.5 | 0.000    | —           | —       | finish       |  |
|   26 |  +1676.5 | 0.000    | —           | —       | finish       |  |
|   27 |  +1684.5 | 0.000    | —           | —       | finish       |  |
|   28 |  +1658.5 | 0.000    | —           | —       | finish       |  |
|   29 |  +1549.6 | 0.000    | —           | —       | finish       |  |
|   30 |  +1603.1 | 0.000    | —           | —       | finish       |  |
|   31 |  +1668.5 | 0.000    | —           | —       | finish       |  |
|   32 |  +1711.1 | 0.000    | —           | —       | finish       |  |
|   33 |  +1706.5 | 0.000    | —           | —       | finish       |  |
|   34 |  +1698.1 | 0.000    | —           | —       | finish       |  |
|   35 |  +1541.6 | 0.000    | —           | —       | finish       |  |
|   36 |  +1661.9 | 0.000    | —           | —       | finish       |  |
|   37 |  +1658.5 | 0.000    | —           | —       | finish       |  |
|   38 |  +1566.5 | 0.000    | —           | —       | finish       |  |
|   39 |  +1472.0 | 0.000    | —           | —       | finish       |  |
|   40 |  +1579.3 | 0.000    | —           | —       | finish       |  |
|   41 |  +1660.5 | 0.000    | —           | —       | finish       |  |
|   42 |  +1717.5 | 0.000    | —           | —       | finish       |  |
|   43 |  +1704.5 | 0.000    | —           | —       | finish       |  |
|   44 |  +1662.9 | 0.000    | —           | —       | finish       |  |
|   45 |  +1690.5 | 0.000    | —           | —       | finish       |  |
|   46 |  +1273.8 | 0.000    | —           | —       | finish       |  |
|   47 |  +1538.5 | 0.000    | —           | —       | finish       |  |
|   48 |  +1677.9 | 0.000    | —           | —       | finish       |  |
|   49 |  +1666.5 | 0.000    | —           | —       | finish       |  |
|   50 |  +1684.5 | 0.000    | —           | —       | finish       |  |

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

