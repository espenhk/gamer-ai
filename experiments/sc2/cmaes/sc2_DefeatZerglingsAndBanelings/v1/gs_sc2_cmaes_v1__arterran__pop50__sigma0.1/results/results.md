# Experiment: gs_sc2_cmaes_v1__arterran__pop50__sigma0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-10 01:49:45
- **End:** 2026-05-10 13:38:01
- **Total runtime:** 11h 48m 15.1s

| Phase | Duration |
|-------|----------|
| Greedy | 11h 48m 14.1s |

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
| agent_race | terran |
| n_sims | 50 |
| policy_type | cmaes |
| obs_spec_preset | rich |
| enable_belief | True |
| population_size | 50 |
| initial_sigma | 0.1 |
| policy_params | {'eval_episodes': 5, 'population_size': 50, 'initial_sigma': 0.1} |

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

Best reward: **+896.2**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +225.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +267.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +246.5 | 0.000    | —           | —       | finish       |  |
|    4 |   +287.9 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    5 |   +277.0 | 0.000    | —           | —       | finish       |  |
|    6 |   +268.5 | 0.000    | —           | —       | finish       |  |
|    7 |   +400.8 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    8 |   +349.4 | 0.000    | —           | —       | finish       |  |
|    9 |   +337.6 | 0.000    | —           | —       | finish       |  |
|   10 |   +319.9 | 0.000    | —           | —       | finish       |  |
|   11 |   +391.5 | 0.000    | —           | —       | finish       |  |
|   12 |   +388.9 | 0.000    | —           | —       | finish       |  |
|   13 |   +605.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   14 |   +424.3 | 0.000    | —           | —       | finish       |  |
|   15 |   +438.8 | 0.000    | —           | —       | finish       |  |
|   16 |   +554.4 | 0.000    | —           | —       | finish       |  |
|   17 |   +451.0 | 0.000    | —           | —       | finish       |  |
|   18 |   +373.4 | 0.000    | —           | —       | finish       |  |
|   19 |   +591.9 | 0.000    | —           | —       | finish       |  |
|   20 |   +527.3 | 0.000    | —           | —       | finish       |  |
|   21 |   +557.3 | 0.000    | —           | —       | finish       |  |
|   22 |   +563.5 | 0.000    | —           | —       | finish       |  |
|   23 |   +587.9 | 0.000    | —           | —       | finish       |  |
|   24 |   +598.7 | 0.000    | —           | —       | finish       |  |
|   25 |   +560.3 | 0.000    | —           | —       | finish       |  |
|   26 |   +862.1 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   27 |   +631.3 | 0.000    | —           | —       | finish       |  |
|   28 |   +529.4 | 0.000    | —           | —       | finish       |  |
|   29 |   +553.0 | 0.000    | —           | —       | finish       |  |
|   30 |   +573.1 | 0.000    | —           | —       | finish       |  |
|   31 |   +630.1 | 0.000    | —           | —       | finish       |  |
|   32 |   +574.7 | 0.000    | —           | —       | finish       |  |
|   33 |   +526.4 | 0.000    | —           | —       | finish       |  |
|   34 |   +896.2 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   35 |   +767.2 | 0.000    | —           | —       | finish       |  |
|   36 |   +560.7 | 0.000    | —           | —       | finish       |  |
|   37 |   +608.9 | 0.000    | —           | —       | finish       |  |
|   38 |   +717.7 | 0.000    | —           | —       | finish       |  |
|   39 |   +532.5 | 0.000    | —           | —       | finish       |  |
|   40 |   +548.0 | 0.000    | —           | —       | finish       |  |
|   41 |   +647.7 | 0.000    | —           | —       | finish       |  |
|   42 |   +557.8 | 0.000    | —           | —       | finish       |  |
|   43 |   +586.7 | 0.000    | —           | —       | finish       |  |
|   44 |   +526.1 | 0.000    | —           | —       | finish       |  |
|   45 |   +601.0 | 0.000    | —           | —       | finish       |  |
|   46 |   +554.1 | 0.000    | —           | —       | finish       |  |
|   47 |   +615.0 | 0.000    | —           | —       | finish       |  |
|   48 |   +623.6 | 0.000    | —           | —       | finish       |  |
|   49 |   +637.6 | 0.000    | —           | —       | finish       |  |
|   50 |   +494.6 | 0.000    | —           | —       | finish       |  |

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

