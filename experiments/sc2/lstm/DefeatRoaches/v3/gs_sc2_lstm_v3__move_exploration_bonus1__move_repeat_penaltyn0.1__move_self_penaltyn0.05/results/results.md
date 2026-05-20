# Experiment: gs_sc2_lstm_v3__move_exploration_bonus1__move_repeat_penaltyn0.1__move_self_penaltyn0.05

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 07:57:02
- **End:** 2026-05-08 08:36:32
- **Total runtime:** 39m 30.2s

| Phase | Duration |
|-------|----------|
| Greedy | 39m 29.2s |

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
| move_self_penalty | -0.05 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1775.7**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1049.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1101.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +398.1 | 0.000    | —           | —       | finish       |  |
|    4 |  +1493.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |   +987.7 | 0.000    | —           | —       | finish       |  |
|    6 |   +901.2 | 0.000    | —           | —       | finish       |  |
|    7 |  +1307.1 | 0.000    | —           | —       | finish       |  |
|    8 |  +1588.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    9 |  +1321.3 | 0.000    | —           | —       | finish       |  |
|   10 |  +1083.1 | 0.000    | —           | —       | finish       |  |
|   11 |  +1489.3 | 0.000    | —           | —       | finish       |  |
|   12 |  +1480.5 | 0.000    | —           | —       | finish       |  |
|   13 |  +1698.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   14 |  +1686.5 | 0.000    | —           | —       | finish       |  |
|   15 |  +1557.3 | 0.000    | —           | —       | finish       |  |
|   16 |  +1460.9 | 0.000    | —           | —       | finish       |  |
|   17 |  +1514.7 | 0.000    | —           | —       | finish       |  |
|   18 |  +1495.7 | 0.000    | —           | —       | finish       |  |
|   19 |   +980.5 | 0.000    | —           | —       | finish       |  |
|   20 |   +812.9 | 0.000    | —           | —       | finish       |  |
|   21 |  +1684.1 | 0.000    | —           | —       | finish       |  |
|   22 |  +1558.5 | 0.000    | —           | —       | finish       |  |
|   23 |  +1700.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   24 |  +1704.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   25 |  +1556.5 | 0.000    | —           | —       | finish       |  |
|   26 |  +1647.7 | 0.000    | —           | —       | finish       |  |
|   27 |  +1470.9 | 0.000    | —           | —       | finish       |  |
|   28 |  +1373.3 | 0.000    | —           | —       | finish       |  |
|   29 |  +1156.7 | 0.000    | —           | —       | finish       |  |
|   30 |   +921.4 | 0.000    | —           | —       | finish       |  |
|   31 |  +1127.3 | 0.000    | —           | —       | finish       |  |
|   32 |  +1296.9 | 0.000    | —           | —       | finish       |  |
|   33 |  +1725.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   34 |  +1500.1 | 0.000    | —           | —       | finish       |  |
|   35 |  +1747.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   36 |  +1712.9 | 0.000    | —           | —       | finish       |  |
|   37 |  +1730.5 | 0.000    | —           | —       | finish       |  |
|   38 |  +1043.0 | 0.000    | —           | —       | finish       |  |
|   39 |  +1255.7 | 0.000    | —           | —       | finish       |  |
|   40 |  +1379.4 | 0.000    | —           | —       | finish       |  |
|   41 |  +1117.7 | 0.000    | —           | —       | finish       |  |
|   42 |  +1440.5 | 0.000    | —           | —       | finish       |  |
|   43 |  +1433.1 | 0.000    | —           | —       | finish       |  |
|   44 |  +1313.2 | 0.000    | —           | —       | finish       |  |
|   45 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   46 |  +1659.5 | 0.000    | —           | —       | finish       |  |
|   47 |  +1423.5 | 0.000    | —           | —       | finish       |  |
|   48 |  +1765.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   49 |  +1775.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   50 |  +1651.9 | 0.000    | —           | —       | finish       |  |

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

