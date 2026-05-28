from __future__ import annotations

import re
from pathlib import Path

import games.tmnf.policies  # noqa: F401

from framework.policies import POLICY_REGISTRY

_CLAUDE_MD = Path(__file__).resolve().parents[1] / "CLAUDE.md"
_POLICY_BLOCK = (
    _CLAUDE_MD.read_text(encoding="utf-8").split("## Policies", 1)[1].split("## Training Phases", 1)[0]
)

_HEADING_TO_POLICY_TYPE = {
    "WeightedLinearPolicy": "hill_climbing",
    "NeuralNetPolicy": "neural_net",
    "EpsilonGreedyPolicy": "epsilon_greedy",
    "UCBQPolicy": "ucb_q",
    "GeneticPolicy": "genetic",
    "CMAESPolicy": "cmaes",
    "NeuralDQNPolicy": "neural_dqn",
    "REINFORCEPolicy": "reinforce",
    "LSTMEvolutionPolicy": "lstm",
    "AlphaZeroMCTSPolicy": "alphazero_mcts",
    "PPOPolicy": "ppo",
    "A2CPolicy": "a2c",
    "SACPolicy": "sac",
    "TD3Policy": "td3",
    "QRDQNPolicy": "qr_dqn",
    "RecurrentPPOPolicy": "recurrent_ppo",
}

_EXPECTED_DEFAULTS = {
    "WeightedLinearPolicy": {},
    "NeuralNetPolicy": {"hidden_sizes": "[16, 16]"},
    "EpsilonGreedyPolicy": {
        "epsilon": "1.0",
        "n_bins": "3",
        "epsilon_decay": "0.995",
        "epsilon_min": "0.05",
        "alpha": "0.1",
        "gamma": "0.99",
    },
    "UCBQPolicy": {
        "c": "1.41",
        "alpha": "0.1",
        "gamma": "0.99",
        "n_bins": "3",
    },
    "GeneticPolicy": {
        "population_size": "10",
        "elite_k": "3",
        "mutation_scale": "0.1",
        "mutation_share": "1.0",
        "eval_episodes": "1",
    },
    "CMAESPolicy": {
        "population_size": "20",
        "initial_sigma": "0.3",
        "eval_episodes": "1",
    },
    "NeuralDQNPolicy": {
        "hidden_sizes": "[64, 64]",
        "replay_buffer_size": "10000",
        "batch_size": "64",
        "min_replay_size": "500",
        "target_update_freq": "200",
        "learning_rate": "0.001",
        "epsilon_start": "1.0",
        "epsilon_end": "0.05",
        "epsilon_decay_steps": "5000",
        "gamma": "0.99",
        "double_dqn": "true",
        "dueling": "false",
        "huber_loss": "true",
        "huber_kappa": "1.0",
        "max_grad_norm": "10.0",
    },
    "REINFORCEPolicy": {
        "hidden_sizes": "[64, 64]",
        "learning_rate": "0.001",
        "gamma": "0.99",
        "entropy_coeff": "0.01",
        "baseline": '"running_mean"',
    },
    "LSTMEvolutionPolicy": {
        "hidden_size": "32",
        "population_size": "20",
        "initial_sigma": "0.05",
    },
    "AlphaZeroMCTSPolicy": {
        "n_simulations": "32",
        "c_puct": "1.5",
        "gamma": "0.99",
        "hidden_sizes": "[64, 64]",
        "learning_rate": "0.001",
        "temperature": "1.0",
        "dirichlet_alpha": "0.3",
        "dirichlet_frac": "0.25",
        "value_loss_coef": "1.0",
        "train_batch_size": "32",
        "seed": "null",
    },
    "PPOPolicy": {
        "total_timesteps": "n_sims × steps_per_sim",
        "steps_per_sim": "1000",
        "learning_rate": "SB3 default",
        "gamma": "SB3 default",
        "hidden_sizes": "SB3 default",
        "seed": "null",
        "verbose": "0",
        "n_steps": "SB3 default",
        "batch_size": "SB3 default",
        "n_epochs": "SB3 default",
        "gae_lambda": "SB3 default",
        "clip_range": "SB3 default",
        "ent_coef": "SB3 default",
        "vf_coef": "SB3 default",
    },
    "A2CPolicy": {
        "total_timesteps": "n_sims × steps_per_sim",
        "steps_per_sim": "1000",
        "learning_rate": "SB3 default",
        "gamma": "SB3 default",
        "hidden_sizes": "SB3 default",
        "seed": "null",
        "verbose": "0",
        "n_steps": "SB3 default",
        "gae_lambda": "SB3 default",
        "ent_coef": "SB3 default",
        "vf_coef": "SB3 default",
    },
    "SACPolicy": {
        "total_timesteps": "n_sims × steps_per_sim",
        "steps_per_sim": "1000",
        "learning_rate": "SB3 default",
        "gamma": "SB3 default",
        "hidden_sizes": "SB3 default",
        "seed": "null",
        "verbose": "0",
        "buffer_size": "SB3 default",
        "batch_size": "SB3 default",
        "tau": "SB3 default",
        "train_freq": "SB3 default",
        "learning_starts": "SB3 default",
        "ent_coef": "SB3 default",
    },
    "TD3Policy": {
        "total_timesteps": "n_sims × steps_per_sim",
        "steps_per_sim": "1000",
        "learning_rate": "SB3 default",
        "gamma": "SB3 default",
        "hidden_sizes": "SB3 default",
        "seed": "null",
        "verbose": "0",
        "buffer_size": "SB3 default",
        "batch_size": "SB3 default",
        "tau": "SB3 default",
        "train_freq": "SB3 default",
        "learning_starts": "SB3 default",
        "policy_delay": "SB3 default",
    },
    "QRDQNPolicy": {
        "total_timesteps": "n_sims × steps_per_sim",
        "steps_per_sim": "1000",
        "learning_rate": "SB3 default",
        "gamma": "SB3 default",
        "hidden_sizes": "SB3 default",
        "seed": "null",
        "verbose": "0",
        "buffer_size": "SB3 default",
        "batch_size": "SB3 default",
        "learning_starts": "SB3 default",
        "target_update_interval": "SB3 default",
        "train_freq": "SB3 default",
        "exploration_fraction": "SB3 default",
        "exploration_final_eps": "SB3 default",
        "n_quantiles": "SB3 default",
    },
    "RecurrentPPOPolicy": {
        "total_timesteps": "n_sims × steps_per_sim",
        "steps_per_sim": "1000",
        "learning_rate": "SB3 default",
        "gamma": "SB3 default",
        "hidden_sizes": "SB3 default",
        "seed": "null",
        "verbose": "0",
        "n_steps": "SB3 default",
        "batch_size": "SB3 default",
        "n_epochs": "SB3 default",
        "gae_lambda": "SB3 default",
        "clip_range": "SB3 default",
        "ent_coef": "SB3 default",
    },
}


def _section(title: str) -> str:
    match = re.search(rf"^### {re.escape(title)}\n(.*?)(?=^### |\Z)", _POLICY_BLOCK, flags=re.MULTILINE | re.DOTALL)
    assert match, f"missing CLAUDE.md section for {title}"
    return match.group(1)


def _clean_cell(text: str) -> str:
    return text.strip().strip("`")


def _hyperparams_for(title: str) -> dict[str, str]:
    body = _section(title)
    if "**Hyperparams** (in `policy_params`): none." in body:
        return {}
    lines = body.splitlines()
    try:
        header_idx = next(i for i, line in enumerate(lines) if line.strip() == "| Param | Default | Description |")
    except StopIteration as exc:  # pragma: no cover - test failure path
        raise AssertionError(f"missing hyperparameter table for {title}") from exc

    rows: dict[str, str] = {}
    for line in lines[header_idx + 2 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        cols = [_clean_cell(c) for c in stripped.strip("|").split("|")]
        if len(cols) < 3:
            continue
        rows[cols[0]] = cols[1]
    return rows


def test_claude_policy_tables_match_registered_policy_params():
    for heading, policy_type in _HEADING_TO_POLICY_TYPE.items():
        documented = _hyperparams_for(heading)
        assert set(documented) == set(POLICY_REGISTRY[policy_type].VALID_POLICY_PARAMS)


def test_claude_policy_defaults_match_audited_source_values():
    for heading, defaults in _EXPECTED_DEFAULTS.items():
        assert _hyperparams_for(heading) == defaults


def test_claude_includes_selection_and_budget_guides():
    assert "### Choosing a policy" in _POLICY_BLOCK
    assert "### Sizing a run" in _POLICY_BLOCK
    assert "total_episodes = n_sims × population_size × eval_episodes" in _POLICY_BLOCK
    assert "n_sims × steps_per_sim" in _POLICY_BLOCK
    assert "SC2's `[fn_idx, x, y, queue]` action encoding" in _POLICY_BLOCK
