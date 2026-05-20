# Experiment: gs_sc2_lstm_v3__move_exploration_bonus1__move_repeat_penaltyn0.05__move_self_penaltyn0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 09:30:39
- **End:** 2026-05-08 10:08:09
- **Total runtime:** 37m 29.8s

| Phase | Duration |
|-------|----------|
| Greedy | 37m 28.8s |

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
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+1798.9**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1714.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +632.9 | 0.000    | —           | —       | finish       |  |
|    3 |  +1444.9 | 0.000    | —           | —       | finish       |  |
|    4 |   +536.0 | 0.000    | —           | —       | finish       |  |
|    5 |  +1037.7 | 0.000    | —           | —       | finish       |  |
|    6 |  +1511.3 | 0.000    | —           | —       | finish       |  |
|    7 |  +1734.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    8 |  +1732.9 | 0.000    | —           | —       | finish       |  |
|    9 |  +1734.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   10 |  +1718.5 | 0.000    | —           | —       | finish       |  |
|   11 |  +1714.1 | 0.000    | —           | —       | finish       |  |
|   12 |  +1726.5 | 0.000    | —           | —       | finish       |  |
|   13 |  +1732.9 | 0.000    | —           | —       | finish       |  |
|   14 |  +1732.9 | 0.000    | —           | —       | finish       |  |
|   15 |  +1734.5 | 0.000    | —           | —       | finish       |  |
|   16 |  +1734.9 | 0.000    | —           | —       | finish       |  |
|   17 |  +1782.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   18 |  +1790.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   19 |  +1782.9 | 0.000    | —           | —       | finish       |  |
|   20 |  +1669.3 | 0.000    | —           | —       | finish       |  |
|   21 |  +1782.5 | 0.000    | —           | —       | finish       |  |
|   22 |  +1773.7 | 0.000    | —           | —       | finish       |  |
|   23 |  +1781.7 | 0.000    | —           | —       | finish       |  |
|   24 |  +1790.9 | 0.000    | —           | —       | finish       |  |
|   25 |  +1798.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   26 |  +1798.9 | 0.000    | —           | —       | finish       |  |
|   27 |  +1790.9 | 0.000    | —           | —       | finish       |  |
|   28 |  +1798.9 | 0.000    | —           | —       | finish       |  |
|   29 |  +1732.1 | 0.000    | —           | —       | finish       |  |
|   30 |  +1759.3 | 0.000    | —           | —       | finish       |  |
|   31 |  +1788.9 | 0.000    | —           | —       | finish       |  |
|   32 |  +1798.1 | 0.000    | —           | —       | finish       |  |
|   33 |  +1790.5 | 0.000    | —           | —       | finish       |  |
|   34 |  +1798.9 | 0.000    | —           | —       | finish       |  |
|   35 |  +1798.9 | 0.000    | —           | —       | finish       |  |
|   36 |  +1798.1 | 0.000    | —           | —       | finish       |  |
|   37 |  +1790.9 | 0.000    | —           | —       | finish       |  |
|   38 |  +1782.9 | 0.000    | —           | —       | finish       |  |
|   39 |  +1775.7 | 0.000    | —           | —       | finish       |  |
|   40 |  +1618.3 | 0.000    | —           | —       | finish       |  |
|   41 |  +1790.1 | 0.000    | —           | —       | finish       |  |
|   42 |  +1798.1 | 0.000    | —           | —       | finish       |  |
|   43 |  +1798.5 | 0.000    | —           | —       | finish       |  |
|   44 |  +1798.9 | 0.000    | —           | —       | finish       |  |
|   45 |  +1798.9 | 0.000    | —           | —       | finish       |  |
|   46 |  +1798.9 | 0.000    | —           | —       | finish       |  |
|   47 |  +1773.9 | 0.000    | —           | —       | finish       |  |
|   48 |  +1503.0 | 0.000    | —           | —       | finish       |  |
|   49 |  +1797.7 | 0.000    | —           | —       | finish       |  |
|   50 |  +1790.5 | 0.000    | —           | —       | finish       |  |

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

