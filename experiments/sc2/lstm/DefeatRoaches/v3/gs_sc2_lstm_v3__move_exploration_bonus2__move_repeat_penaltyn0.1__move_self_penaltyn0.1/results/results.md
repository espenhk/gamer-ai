# Experiment: gs_sc2_lstm_v3__move_exploration_bonus2__move_repeat_penaltyn0.1__move_self_penaltyn0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 13:52:47
- **End:** 2026-05-08 14:42:30
- **Total runtime:** 49m 43.6s

| Phase | Duration |
|-------|----------|
| Greedy | 49m 42.5s |

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
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+1804.9**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1028.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |  +1680.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |  +1724.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    4 |  +1110.1 | 0.000    | —           | —       | finish       |  |
|    5 |  +1679.3 | 0.000    | —           | —       | finish       |  |
|    6 |  +1413.7 | 0.000    | —           | —       | finish       |  |
|    7 |  +1583.1 | 0.000    | —           | —       | finish       |  |
|    8 |  +1676.1 | 0.000    | —           | —       | finish       |  |
|    9 |  +1683.6 | 0.000    | —           | —       | finish       |  |
|   10 |  +1716.9 | 0.000    | —           | —       | finish       |  |
|   11 |  +1730.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   12 |  +1804.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   13 |  +1763.3 | 0.000    | —           | —       | finish       |  |
|   14 |  +1730.2 | 0.000    | —           | —       | finish       |  |
|   15 |  +1722.5 | 0.000    | —           | —       | finish       |  |
|   16 |  +1716.9 | 0.000    | —           | —       | finish       |  |
|   17 |  +1732.1 | 0.000    | —           | —       | finish       |  |
|   18 |  +1716.1 | 0.000    | —           | —       | finish       |  |
|   19 |  +1713.7 | 0.000    | —           | —       | finish       |  |
|   20 |  +1727.3 | 0.000    | —           | —       | finish       |  |
|   21 |  +1741.7 | 0.000    | —           | —       | finish       |  |
|   22 |  +1724.1 | 0.000    | —           | —       | finish       |  |
|   23 |  +1733.7 | 0.000    | —           | —       | finish       |  |
|   24 |  +1728.9 | 0.000    | —           | —       | finish       |  |
|   25 |  +1718.2 | 0.000    | —           | —       | finish       |  |
|   26 |  +1716.1 | 0.000    | —           | —       | finish       |  |
|   27 |  +1723.3 | 0.000    | —           | —       | finish       |  |
|   28 |  +1721.7 | 0.000    | —           | —       | finish       |  |
|   29 |  +1721.7 | 0.000    | —           | —       | finish       |  |
|   30 |  +1765.7 | 0.000    | —           | —       | finish       |  |
|   31 |  +1765.7 | 0.000    | —           | —       | finish       |  |
|   32 |  +1749.7 | 0.000    | —           | —       | finish       |  |
|   33 |  +1722.5 | 0.000    | —           | —       | finish       |  |
|   34 |  +1724.7 | 0.000    | —           | —       | finish       |  |
|   35 |  +1715.7 | 0.000    | —           | —       | finish       |  |
|   36 |  +1715.3 | 0.000    | —           | —       | finish       |  |
|   37 |  +1720.9 | 0.000    | —           | —       | finish       |  |
|   38 |  +1722.5 | 0.000    | —           | —       | finish       |  |
|   39 |  +1757.7 | 0.000    | —           | —       | finish       |  |
|   40 |  +1773.7 | 0.000    | —           | —       | finish       |  |
|   41 |  +1741.7 | 0.000    | —           | —       | finish       |  |
|   42 |  +1724.1 | 0.000    | —           | —       | finish       |  |
|   43 |  +1724.5 | 0.000    | —           | —       | finish       |  |
|   44 |  +1725.4 | 0.000    | —           | —       | finish       |  |
|   45 |  +1723.3 | 0.000    | —           | —       | finish       |  |
|   46 |  +1721.7 | 0.000    | —           | —       | finish       |  |
|   47 |  +1721.7 | 0.000    | —           | —       | finish       |  |
|   48 |  +1729.7 | 0.000    | —           | —       | finish       |  |
|   49 |  +1773.7 | 0.000    | —           | —       | finish       |  |
|   50 |  +1765.7 | 0.000    | —           | —       | finish       |  |

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

