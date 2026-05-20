# Experiment: gs_sc2_lstm_v2__hsize128__sigma0.05

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-07 20:44:44
- **End:** 2026-05-07 21:32:24
- **Total runtime:** 47m 39.8s

| Phase | Duration |
|-------|----------|
| Greedy | 47m 38.8s |

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
| initial_sigma | 0.05 |
| policy_params | {'population_size': 20, 'hidden_size': 128, 'initial_sigma': 0.05} |

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
| move_repeat_penalty | -0.2 |
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1735.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1649.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1183.5 | 0.000    | —           | —       | finish       |  |
|    3 |  +1720.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |  +1710.8 | 0.000    | —           | —       | finish       |  |
|    5 |  +1721.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |  +1734.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    7 |  +1719.1 | 0.000    | —           | —       | finish       |  |
|    8 |  +1726.2 | 0.000    | —           | —       | finish       |  |
|    9 |  +1698.9 | 0.000    | —           | —       | finish       |  |
|   10 |  +1735.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   11 |  +1726.2 | 0.000    | —           | —       | finish       |  |
|   12 |  +1731.2 | 0.000    | —           | —       | finish       |  |
|   13 |  +1730.5 | 0.000    | —           | —       | finish       |  |
|   14 |  +1722.4 | 0.000    | —           | —       | finish       |  |
|   15 |  +1734.4 | 0.000    | —           | —       | finish       |  |
|   16 |  +1734.2 | 0.000    | —           | —       | finish       |  |
|   17 |  +1734.3 | 0.000    | —           | —       | finish       |  |
|   18 |  +1721.4 | 0.000    | —           | —       | finish       |  |
|   19 |  +1729.4 | 0.000    | —           | —       | finish       |  |
|   20 |  +1724.9 | 0.000    | —           | —       | finish       |  |
|   21 |  +1734.4 | 0.000    | —           | —       | finish       |  |
|   22 |  +1732.8 | 0.000    | —           | —       | finish       |  |
|   23 |  +1732.7 | 0.000    | —           | —       | finish       |  |
|   24 |  +1732.8 | 0.000    | —           | —       | finish       |  |
|   25 |  +1731.1 | 0.000    | —           | —       | finish       |  |
|   26 |  +1724.6 | 0.000    | —           | —       | finish       |  |
|   27 |  +1715.0 | 0.000    | —           | —       | finish       |  |
|   28 |  +1729.4 | 0.000    | —           | —       | finish       |  |
|   29 |  +1720.5 | 0.000    | —           | —       | finish       |  |
|   30 |  +1721.3 | 0.000    | —           | —       | finish       |  |
|   31 |  +1732.7 | 0.000    | —           | —       | finish       |  |
|   32 |  +1732.8 | 0.000    | —           | —       | finish       |  |
|   33 |  +1727.9 | 0.000    | —           | —       | finish       |  |
|   34 |  +1722.4 | 0.000    | —           | —       | finish       |  |
|   35 |  +1730.4 | 0.000    | —           | —       | finish       |  |
|   36 |  +1723.1 | 0.000    | —           | —       | finish       |  |
|   37 |  +1729.5 | 0.000    | —           | —       | finish       |  |
|   38 |  +1722.9 | 0.000    | —           | —       | finish       |  |
|   39 |  +1706.9 | 0.000    | —           | —       | finish       |  |
|   40 |  +1724.7 | 0.000    | —           | —       | finish       |  |
|   41 |  +1731.9 | 0.000    | —           | —       | finish       |  |
|   42 |  +1724.7 | 0.000    | —           | —       | finish       |  |
|   43 |  +1724.7 | 0.000    | —           | —       | finish       |  |
|   44 |  +1723.2 | 0.000    | —           | —       | finish       |  |
|   45 |  +1730.4 | 0.000    | —           | —       | finish       |  |
|   46 |  +1724.8 | 0.000    | —           | —       | finish       |  |
|   47 |  +1721.5 | 0.000    | —           | —       | finish       |  |
|   48 |  +1715.0 | 0.000    | —           | —       | finish       |  |
|   49 |  +1730.4 | 0.000    | —           | —       | finish       |  |
|   50 |  +1732.7 | 0.000    | —           | —       | finish       |  |

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

