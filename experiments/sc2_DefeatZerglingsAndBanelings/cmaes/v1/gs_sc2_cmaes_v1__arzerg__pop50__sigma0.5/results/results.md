# Experiment: gs_sc2_cmaes_v1__arzerg__pop50__sigma0.5

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-13 19:36:38
- **End:** 2026-05-14 07:53:42
- **Total runtime:** 12h 17m 04.3s

| Phase | Duration |
|-------|----------|
| Greedy | 12h 17m 03.3s |

## Run Parameters

### Training

| Parameter | Value |
|-----------|-------|
| track | sc2_DefeatZerglingsAndBanelings |
| map_name | DefeatZerglingsAndBanelings |
| in_game_episode_s | 120.0 |
| step_mul | 8 |
| screen_size | 64 |
| minimap_size | 64 |
| max_apm | 300 |
| agent_race | zerg |
| n_sims | 50 |
| policy_type | cmaes |
| obs_spec_preset | rich |
| enable_belief | True |
| population_size | 50 |
| initial_sigma | 0.5 |
| policy_params | {'eval_episodes': 5, 'population_size': 50, 'initial_sigma': 0.5} |

### Reward Config

| Parameter | Value |
|-----------|-------|
| score_weight | 10.0 |
| win_bonus | 1000.0 |
| loss_penalty | -100.0 |
| step_penalty | -0.001 |
| idle_penalty | 0.0 |
| idle_bonus | 0.5 |
| move_exploration_bonus | 1.0 |
| move_repeat_penalty | -0.05 |
| move_self_penalty | -0.1 |
| attack_move_bonus | 0.5 |
| click_attack_bonus | 1.0 |
| click_attack_cooldown_steps | 8 |
| attack_friendly_penalty | -10.0 |
| economy_weight | 0.001 |

## Greedy Phase

Best reward: **+667.2**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +326.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +357.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +314.2 | 0.000    | —           | —       | finish       |  |
|    4 |   +322.6 | 0.000    | —           | —       | finish       |  |
|    5 |   +420.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    6 |   +550.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    7 |   +653.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    8 |   +394.8 | 0.000    | —           | —       | finish       |  |
|    9 |   +481.9 | 0.000    | —           | —       | finish       |  |
|   10 |   +438.5 | 0.000    | —           | —       | finish       |  |
|   11 |   +555.4 | 0.000    | —           | —       | finish       |  |
|   12 |   +463.3 | 0.000    | —           | —       | finish       |  |
|   13 |   +405.7 | 0.000    | —           | —       | finish       |  |
|   14 |   +411.6 | 0.000    | —           | —       | finish       |  |
|   15 |   +425.9 | 0.000    | —           | —       | finish       |  |
|   16 |   +517.8 | 0.000    | —           | —       | finish       |  |
|   17 |   +472.4 | 0.000    | —           | —       | finish       |  |
|   18 |   +451.2 | 0.000    | —           | —       | finish       |  |
|   19 |   +493.3 | 0.000    | —           | —       | finish       |  |
|   20 |   +495.6 | 0.000    | —           | —       | finish       |  |
|   21 |   +500.1 | 0.000    | —           | —       | finish       |  |
|   22 |   +498.6 | 0.000    | —           | —       | finish       |  |
|   23 |   +501.0 | 0.000    | —           | —       | finish       |  |
|   24 |   +635.6 | 0.000    | —           | —       | finish       |  |
|   25 |   +436.4 | 0.000    | —           | —       | finish       |  |
|   26 |   +447.7 | 0.000    | —           | —       | finish       |  |
|   27 |   +556.0 | 0.000    | —           | —       | finish       |  |
|   28 |   +455.3 | 0.000    | —           | —       | finish       |  |
|   29 |   +355.5 | 0.000    | —           | —       | finish       |  |
|   30 |   +501.7 | 0.000    | —           | —       | finish       |  |
|   31 |   +481.7 | 0.000    | —           | —       | finish       |  |
|   32 |   +473.0 | 0.000    | —           | —       | finish       |  |
|   33 |   +573.7 | 0.000    | —           | —       | finish       |  |
|   34 |   +649.6 | 0.000    | —           | —       | finish       |  |
|   35 |   +401.8 | 0.000    | —           | —       | finish       |  |
|   36 |   +581.6 | 0.000    | —           | —       | finish       |  |
|   37 |   +640.9 | 0.000    | —           | —       | finish       |  |
|   38 |   +537.8 | 0.000    | —           | —       | finish       |  |
|   39 |   +667.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   40 |   +653.0 | 0.000    | —           | —       | finish       |  |
|   41 |   +509.7 | 0.000    | —           | —       | finish       |  |
|   42 |   +624.1 | 0.000    | —           | —       | finish       |  |
|   43 |   +538.9 | 0.000    | —           | —       | finish       |  |
|   44 |   +586.5 | 0.000    | —           | —       | finish       |  |
|   45 |   +653.4 | 0.000    | —           | —       | finish       |  |
|   46 |   +634.6 | 0.000    | —           | —       | finish       |  |
|   47 |   +553.0 | 0.000    | —           | —       | finish       |  |
|   48 |   +590.3 | 0.000    | —           | —       | finish       |  |
|   49 |   +659.5 | 0.000    | —           | —       | finish       |  |
|   50 |   +610.4 | 0.000    | —           | —       | finish       |  |

![Greedy rewards](greedy_rewards.png)


![Reward components](reward_components.png)


![Action frequency](action_frequency.png)


![Game-state averages](obs_averages.png)


![Spatial target heatmap](spatial_heatmap.png)


![Outcome breakdown](outcome_breakdown.png)


![Skipped frames](skipped_frames.png)


![Time supply-capped](supply_capped.png)


![Resources available over time](resource_series.png)


![Army count over time](army_count.png)


![Reward trajectory](reward_trajectory.png)

