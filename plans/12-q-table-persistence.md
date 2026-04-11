# Plan: Q-Table Persistence for Tabular Policies

## Problem

`EpsilonGreedyPolicy` and `MCTSPolicy` both inherit from `QTablePolicy`. They build up
Q-value estimates (`_q_table`) and visit counts (`_n_sa`, `_n_s`) over many episodes,
but `to_cfg()` (in `policies.py` lines ~550 and ~639) saves **only hyperparameters**
(alpha, gamma, epsilon, etc.) — the Q-table itself is discarded on exit.

Tabular methods need hundreds to thousands of episodes to converge. Losing the table on
every restart means the policy re-explores from scratch each time, making it impractical
for the iterative workflow (`python main.py <experiment>`).

## Proposed Solution

Pickle the three table dicts alongside the YAML config:

- YAML (`policy_weights.yaml`) — human-readable hyperparameters, unchanged
- Pickle (`policy_weights_qtable.pkl`) — binary `(q_table, n_sa, n_s)` tuple

On load, if the pickle exists, restore the tables automatically.

## Changes to `policies.py`

### `QTablePolicy` — add `save()` override and `_load_table()` classmethod

```python
MAX_QTABLE_ENTRIES = 500_000  # guard against saving enormous tables

def save(self, path: Path) -> None:
    """Save hyperparams to YAML and Q-table to sibling .pkl file."""
    super().save(path)  # writes YAML via to_cfg()
    if len(self._q_table) > MAX_QTABLE_ENTRIES:
        print(f"WARNING: Q-table has {len(self._q_table)} entries (>{MAX_QTABLE_ENTRIES}), skipping pickle.")
        return
    pkl_path = path.with_name(path.stem + "_qtable.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump((self._q_table, self._n_sa, self._n_s), f)
    print(f"Q-table saved: {len(self._q_table)} states → {pkl_path}")

@classmethod
def _load_table(cls, self_instance, path: Path) -> None:
    """Restore Q-table from sibling .pkl if it exists."""
    pkl_path = path.with_name(path.stem + "_qtable.pkl")
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            q_table, n_sa, n_s = pickle.load(f)
        self_instance._q_table = q_table
        self_instance._n_sa = n_sa
        self_instance._n_s = n_s
        print(f"Q-table loaded: {len(q_table)} states from {pkl_path}")
```

Add `import pickle` at the top of `policies.py`.

### `EpsilonGreedyPolicy.from_cfg()` (lines 507–517)
After the existing construction, call `_load_table()`:
```python
@classmethod
def from_cfg(cls, cfg: dict, weights_file: Path | None = None) -> "EpsilonGreedyPolicy":
    policy = cls(...)  # existing construction
    if weights_file is not None:
        cls._load_table(policy, weights_file)
    return policy
```

### `MCTSPolicy.from_cfg()` (lines 596–604)
Same pattern as `EpsilonGreedyPolicy.from_cfg()`.

## File Naming

| File | Contents |
|------|----------|
| `experiments/<name>/policy_weights.yaml` | Hyperparameters (unchanged) |
| `experiments/<name>/policy_weights_qtable.pkl` | `(q_table, n_sa, n_s)` pickle |

The `.pkl` file lives next to the YAML so it's automatically found by `_load_table()`.
Both files should be added to `.gitignore` under `experiments/` (already git-ignored).

## Size Concern

With `n_bins=5` and 15 obs dimensions, the theoretical table size is 5^15 ≈ 30 billion
states — but only **visited** states are stored (sparse dict). In practice, a 100-episode
session on a 13-second track at 10× speed visits ~100 × (13 × 100 steps) = 130,000 unique
state bins at most. The `MAX_QTABLE_ENTRIES = 500_000` guard prevents pathological cases
from producing huge pickle files.

## Files to Change

| File | Change |
|------|--------|
| `policies.py` | Add `save()` + `_load_table()` to `QTablePolicy`; update `from_cfg()` in `EpsilonGreedy` and `MCTS` |

## Testing

1. Run `python main.py my_qtable_exp` with `policy_type=epsilon_greedy` for 50 episodes
2. Confirm `experiments/my_qtable_exp/policy_weights_qtable.pkl` exists
3. Restart: `python main.py my_qtable_exp` — confirm the logged Q-table size matches
   (non-zero number of states loaded)
4. Confirm `epsilon` continues decaying from where it left off (saved in YAML)
5. Verify with `policy_type=mcts` also persists and restores correctly
