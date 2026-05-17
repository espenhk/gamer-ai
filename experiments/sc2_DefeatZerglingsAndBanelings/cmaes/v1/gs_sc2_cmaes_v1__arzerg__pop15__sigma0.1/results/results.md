# Experiment: gs_sc2_cmaes_v1__arzerg__pop15__sigma0.1

**Game:** StarCraft 2

## Timings

- **Start:** 2026-05-11 14:30:08
- **End:** 2026-05-11 17:48:47
- **Total runtime:** 3h 18m 39.8s

| Phase | Duration |
|-------|----------|
| Greedy | 3h 18m 38.8s |

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
| population_size | 15 |
| initial_sigma | 0.1 |
| policy_params | {'eval_episodes': 5, 'population_size': 15, 'initial_sigma': 0.1} |

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

Best reward: **+619.0**

| Sim  | Reward   | Progress | Finish Time | Mean abs lat | Reason       | Result       |
|------|----------|----------|-------------|--------------|--------------|-------------|
|    1 |   +116.4 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    2 |   +384.7 | 0.000    | —           | —       | finish       | **NEW BEST** |
|    3 |   +159.2 | 0.000    | —           | —       | finish       |  |
|    4 |   +178.7 | 0.000    | —           | —       | finish       |  |
|    5 |   +241.4 | 0.000    | —           | —       | finish       |  |
|    6 |   +335.9 | 0.000    | —           | —       | finish       |  |
|    7 |   +231.1 | 0.000    | —           | —       | finish       |  |
|    8 |   +212.8 | 0.000    | —           | —       | finish       |  |
|    9 |   +183.1 | 0.000    | —           | —       | finish       |  |
|   10 |   +203.4 | 0.000    | —           | —       | finish       |  |
|   11 |   +234.7 | 0.000    | —           | —       | finish       |  |
|   12 |   +213.8 | 0.000    | —           | —       | finish       |  |
|   13 |   +418.5 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   14 |   +136.7 | 0.000    | —           | —       | finish       |  |
|   15 |   +261.3 | 0.000    | —           | —       | finish       |  |
|   16 |   +205.0 | 0.000    | —           | —       | finish       |  |
|   17 |   +184.6 | 0.000    | —           | —       | finish       |  |
|   18 |   +297.9 | 0.000    | —           | —       | finish       |  |
|   19 |   +288.0 | 0.000    | —           | —       | finish       |  |
|   20 |   +344.1 | 0.000    | —           | —       | finish       |  |
|   21 |   +362.5 | 0.000    | —           | —       | finish       |  |
|   22 |   +519.3 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   23 |   +619.0 | 0.000    | —           | —       | finish       | **NEW BEST** |
|   24 |   +356.1 | 0.000    | —           | —       | finish       |  |
|   25 |   +413.3 | 0.000    | —           | —       | finish       |  |
|   26 |   +268.9 | 0.000    | —           | —       | finish       |  |
|   27 |   +432.2 | 0.000    | —           | —       | finish       |  |
|   28 |   +287.2 | 0.000    | —           | —       | finish       |  |
|   29 |   +322.3 | 0.000    | —           | —       | finish       |  |
|   30 |   +257.8 | 0.000    | —           | —       | finish       |  |
|   31 |   +344.1 | 0.000    | —           | —       | finish       |  |
|   32 |   +235.5 | 0.000    | —           | —       | finish       |  |
|   33 |   +519.8 | 0.000    | —           | —       | finish       |  |
|   34 |   +465.6 | 0.000    | —           | —       | finish       |  |
|   35 |   +403.3 | 0.000    | —           | —       | finish       |  |
|   36 |   +305.3 | 0.000    | —           | —       | finish       |  |
|   37 |   +424.1 | 0.000    | —           | —       | finish       |  |
|   38 |   +352.8 | 0.000    | —           | —       | finish       |  |
|   39 |   +456.0 | 0.000    | —           | —       | finish       |  |
|   40 |   +372.1 | 0.000    | —           | —       | finish       |  |
|   41 |   +389.5 | 0.000    | —           | —       | finish       |  |
|   42 |   +523.5 | 0.000    | —           | —       | finish       |  |
|   43 |   +406.4 | 0.000    | —           | —       | finish       |  |
|   44 |   +334.3 | 0.000    | —           | —       | finish       |  |
|   45 |   +342.5 | 0.000    | —           | —       | finish       |  |
|   46 |   +556.6 | 0.000    | —           | —       | finish       |  |
|   47 |   +439.6 | 0.000    | —           | —       | finish       |  |
|   48 |   +453.5 | 0.000    | —           | —       | finish       |  |
|   49 |   +258.8 | 0.000    | —           | —       | finish       |  |
|   50 |   +430.3 | 0.000    | —           | —       | finish       |  |

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

