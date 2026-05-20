# Experiment: gs_sc2_lstm_v3__move_exploration_bonus1__move_repeat_penaltyn0.2__move_self_penaltyn0.02

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-08 06:49:32
- **End:** 2026-05-08 07:30:46
- **Total runtime:** 41m 13.3s

| Phase | Duration |
|-------|----------|
| Greedy | 41m 12.3s |

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
| move_self_penalty | -0.02 |
| attack_move_bonus | 0.0 |
| click_attack_bonus | 0.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -5.0 |
| economy_weight | 0.0 |

## Greedy Phase

Best reward: **+1834.0**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |  +1725.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +573.3 | 0.000    | —           | —       | finish       |  |
|    3 |  +1049.8 | 0.000    | —           | —       | finish       |  |
|    4 |  +1724.1 | 0.000    | —           | —       | finish       |  |
|    5 |  +1703.1 | 0.000    | —           | —       | finish       |  |
|    6 |  +1678.5 | 0.000    | —           | —       | finish       |  |
|    7 |  +1707.9 | 0.000    | —           | —       | finish       |  |
|    8 |  +1709.2 | 0.000    | —           | —       | finish       |  |
|    9 |  +1725.4 | 0.000    | —           | —       | finish       |  |
|   10 |  +1834.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   11 |  +1796.1 | 0.000    | —           | —       | finish       |  |
|   12 |  +1704.7 | 0.000    | —           | —       | finish       |  |
|   13 |  +1716.7 | 0.000    | —           | —       | finish       |  |
|   14 |  +1724.9 | 0.000    | —           | —       | finish       |  |
|   15 |  +1715.3 | 0.000    | —           | —       | finish       |  |
|   16 |  +1720.6 | 0.000    | —           | —       | finish       |  |
|   17 |  +1712.7 | 0.000    | —           | —       | finish       |  |
|   18 |  +1718.8 | 0.000    | —           | —       | finish       |  |
|   19 |  +1720.4 | 0.000    | —           | —       | finish       |  |
|   20 |  +1728.6 | 0.000    | —           | —       | finish       |  |
|   21 |  +1722.5 | 0.000    | —           | —       | finish       |  |
|   22 |  +1728.4 | 0.000    | —           | —       | finish       |  |
|   23 |  +1717.0 | 0.000    | —           | —       | finish       |  |
|   24 |  +1709.0 | 0.000    | —           | —       | finish       |  |
|   25 |  +1718.6 | 0.000    | —           | —       | finish       |  |
|   26 |  +1712.7 | 0.000    | —           | —       | finish       |  |
|   27 |  +1712.7 | 0.000    | —           | —       | finish       |  |
|   28 |  +1720.6 | 0.000    | —           | —       | finish       |  |
|   29 |  +1720.6 | 0.000    | —           | —       | finish       |  |
|   30 |  +1712.7 | 0.000    | —           | —       | finish       |  |
|   31 |  +1712.7 | 0.000    | —           | —       | finish       |  |
|   32 |  +1730.3 | 0.000    | —           | —       | finish       |  |
|   33 |  +1730.3 | 0.000    | —           | —       | finish       |  |
|   34 |  +1730.2 | 0.000    | —           | —       | finish       |  |
|   35 |  +1716.9 | 0.000    | —           | —       | finish       |  |
|   36 |  +1719.0 | 0.000    | —           | —       | finish       |  |
|   37 |  +1719.0 | 0.000    | —           | —       | finish       |  |
|   38 |  +1719.0 | 0.000    | —           | —       | finish       |  |
|   39 |  +1720.7 | 0.000    | —           | —       | finish       |  |
|   40 |  +1720.7 | 0.000    | —           | —       | finish       |  |
|   41 |  +1720.7 | 0.000    | —           | —       | finish       |  |
|   42 |  +1712.7 | 0.000    | —           | —       | finish       |  |
|   43 |  +1712.7 | 0.000    | —           | —       | finish       |  |
|   44 |  +1730.5 | 0.000    | —           | —       | finish       |  |
|   45 |  +1767.0 | 0.000    | —           | —       | finish       |  |
|   46 |  +1783.0 | 0.000    | —           | —       | finish       |  |
|   47 |  +1784.7 | 0.000    | —           | —       | finish       |  |
|   48 |  +1776.6 | 0.000    | —           | —       | finish       |  |
|   49 |  +1776.7 | 0.000    | —           | —       | finish       |  |
|   50 |  +1776.6 | 0.000    | —           | —       | finish       |  |

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

