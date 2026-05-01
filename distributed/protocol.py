"""
Shared dataclasses and JSON serialisation for distributed grid search.

ComboSpec     — sent from coordinator to worker describing one work item
ResultPayload — sent from worker back to coordinator with the result

ExperimentData serialisation round-trips through dataclasses.asdict() + JSON.
Numpy arrays are converted to lists on the way out; reconstruction is done
via experiment_from_dict() which rebuilds the full dataclass tree.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ComboSpec:
    """One grid-search combination, ready for a worker to execute."""
    name: str
    track: str
    training_params: dict[str, Any]
    reward_params: dict[str, Any]
    game: str = "tmnf"


@dataclasses.dataclass
class ResultPayload:
    """Serialised experiment result posted by a worker to the coordinator."""
    name: str
    data_json: str   # ExperimentData serialised via experiment_to_json()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _default(obj: Any) -> Any:
    """Custom encoder: numpy arrays/scalars → plain Python, dataclasses → dict."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
    except ImportError:
        pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def experiment_to_json(data: Any) -> str:
    """Serialise an ExperimentData instance to a JSON string."""
    return json.dumps(dataclasses.asdict(data), default=_default)


def experiment_from_dict(d: dict[str, Any]) -> Any:
    """Reconstruct an ExperimentData instance from a plain dict (e.g. from JSON)."""
    from framework.analytics import (
        ExperimentData, RunTrace, ProbeResult,
        ColdStartSimResult, ColdStartRestartResult, GreedySimResult,
    )

    def _trace(t: dict | None) -> RunTrace | None:
        if t is None:
            return None
        return RunTrace(
            pos_x=t.get("pos_x", []),
            pos_z=t.get("pos_z", []),
            throttle_state=t.get("throttle_state", []),
            total_reward=t.get("total_reward", 0.0),
        )

    def _probe(p: dict) -> ProbeResult:
        return ProbeResult(
            action_idx=p["action_idx"],
            action_name=p["action_name"],
            reward=p["reward"],
            trace=_trace(p.get("trace")),
        )

    def _cs_sim(s: dict) -> ColdStartSimResult:
        return ColdStartSimResult(
            sim=s["sim"],
            reward=s["reward"],
            throttle_counts=s["throttle_counts"],
            total_steps=s["total_steps"],
            trace=_trace(s.get("trace")),
            termination_reason=s.get("termination_reason"),
        )

    def _cs_restart(r: dict) -> ColdStartRestartResult:
        return ColdStartRestartResult(
            restart=r["restart"],
            sims=[_cs_sim(s) for s in r.get("sims", [])],
            best_reward=r["best_reward"],
            beat_probe_floor=r["beat_probe_floor"],
        )

    def _greedy_sim(s: dict) -> GreedySimResult:
        return GreedySimResult(
            sim=s["sim"],
            reward=s["reward"],
            improved=s["improved"],
            throttle_counts=s["throttle_counts"],
            total_steps=s["total_steps"],
            trace=_trace(s.get("trace")),
            weights=s.get("weights"),
            final_track_progress=s.get("final_track_progress", 0.0),
            laps_completed=s.get("laps_completed", 0),
            mutation_scale=s.get("mutation_scale"),
            termination_reason=s.get("termination_reason"),
            finish_time_s=s.get("finish_time_s"),
            mean_abs_lateral_offset=s.get("mean_abs_lateral_offset"),
            reward_components=s.get("reward_components"),
        )

    return ExperimentData(
        experiment_name=d["experiment_name"],
        probe_results=[_probe(p) for p in d.get("probe_results", [])],
        cold_start_restarts=[_cs_restart(r) for r in d.get("cold_start_restarts", [])],
        greedy_sims=[_greedy_sim(s) for s in d.get("greedy_sims", [])],
        probe_floor=d.get("probe_floor"),
        weights_file=d.get("weights_file", ""),
        reward_config_file=d.get("reward_config_file", ""),
        training_params=d.get("training_params", {}),
        timings=d.get("timings", {}),
        track=d.get("track", ""),
        early_stopped=d.get("early_stopped", False),
        early_stop_sim=d.get("early_stop_sim"),
    )


def combo_to_dict(spec: ComboSpec) -> dict[str, Any]:
    return dataclasses.asdict(spec)


def combo_from_dict(d: dict[str, Any]) -> ComboSpec:
    return ComboSpec(**d)


def result_to_dict(payload: ResultPayload) -> dict[str, Any]:
    return dataclasses.asdict(payload)


def result_from_dict(d: dict[str, Any]) -> ResultPayload:
    return ResultPayload(**d)
