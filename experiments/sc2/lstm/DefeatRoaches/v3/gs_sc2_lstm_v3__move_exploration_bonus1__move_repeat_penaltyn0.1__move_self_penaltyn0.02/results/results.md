# Experiment: gs_sc2_lstm_v3__move_exploration_bonus1__move_repeat_penaltyn0.1__move_self_penaltyn0.02

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 08:36:44
- **End:** 2026-05-08 09:30:26
- **Total runtime:** 53m 41.9s

| Phase | Duration |
|-------|----------|
| Greedy | 53m 40.9s |

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
| move_repeat_penalty | -0.1 |
| move_self_penalty | -0.02 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1782.5**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1288.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1665.6 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1028.7 | 0.000    | —           | —       | finish       |  |
|    4 |  +1060.1 | 0.000    | —           | —       | finish       |  |
|    5 |  +1782.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |   +818.6 | 0.000    | —           | —       | finish       |  |
|    7 |  +1710.5 | 0.000    | —           | —       | finish       |  |
|    8 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|    9 |  +1730.5 | 0.000    | —           | —       | finish       |  |
|   10 |  +1716.1 | 0.000    | —           | —       | finish       |  |
|   11 |  +1728.9 | 0.000    | —           | —       | finish       |  |
|   12 |  +1724.9 | 0.000    | —           | —       | finish       |  |
|   13 |  +1736.7 | 0.000    | —           | —       | finish       |  |
|   14 |  +1723.9 | 0.000    | —           | —       | finish       |  |
|   15 |  +1729.7 | 0.000    | —           | —       | finish       |  |
|   16 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   17 |  +1741.7 | 0.000    | —           | —       | finish       |  |
|   18 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   19 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   20 |  +1731.9 | 0.000    | —           | —       | finish       |  |
|   21 |  +1732.9 | 0.000    | —           | —       | finish       |  |
|   22 |  +1737.7 | 0.000    | —           | —       | finish       |  |
|   23 |  +1738.5 | 0.000    | —           | —       | finish       |  |
|   24 |  +1737.7 | 0.000    | —           | —       | finish       |  |
|   25 |  +1737.7 | 0.000    | —           | —       | finish       |  |
|   26 |  +1738.5 | 0.000    | —           | —       | finish       |  |
|   27 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   28 |  +1732.1 | 0.000    | —           | —       | finish       |  |
|   29 |  +1731.9 | 0.000    | —           | —       | finish       |  |
|   30 |  +1732.1 | 0.000    | —           | —       | finish       |  |
|   31 |  +1716.1 | 0.000    | —           | —       | finish       |  |
|   32 |  +1738.5 | 0.000    | —           | —       | finish       |  |
|   33 |  +1748.9 | 0.000    | —           | —       | finish       |  |
|   34 |  +1772.9 | 0.000    | —           | —       | finish       |  |
|   35 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   36 |  +1724.1 | 0.000    | —           | —       | finish       |  |
|   37 |  +1732.1 | 0.000    | —           | —       | finish       |  |
|   38 |  +1732.9 | 0.000    | —           | —       | finish       |  |
|   39 |  +1736.7 | 0.000    | —           | —       | finish       |  |
|   40 |  +1738.5 | 0.000    | —           | —       | finish       |  |
|   41 |  +1732.9 | 0.000    | —           | —       | finish       |  |
|   42 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   43 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   44 |  +1715.6 | 0.000    | —           | —       | finish       |  |
|   45 |  +1732.1 | 0.000    | —           | —       | finish       |  |
|   46 |  +1729.7 | 0.000    | —           | —       | finish       |  |
|   47 |  +1729.7 | 0.000    | —           | —       | finish       |  |
|   48 |  +1732.9 | 0.000    | —           | —       | finish       |  |
|   49 |  +1738.5 | 0.000    | —           | —       | finish       |  |
|   50 |  +1732.9 | 0.000    | —           | —       | finish       |  |

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

