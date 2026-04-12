from __future__ import annotations

from dataclasses import dataclass
from games.tmnf.state import steer_percent


@dataclass
class Instruction:
    time_s: float   # seconds from start of run phase
    action: str     # e.g. "press up", "steer left"


@dataclass
class InputState:
    accelerate: bool = False
    brake: bool = False
    steer: int = 0  # -65536 to 65536


def parse_instructions(path: str) -> list[Instruction]:
    """Parse an instruction file into a sorted list of Instructions.

    File format (one instruction per line):
        <time_seconds> <action>

    Lines starting with '#' or empty lines are ignored.

    Supported actions:
        press up        — accelerate (release brake)
        press down      — brake (release accelerate)
        release         — release both accelerate and brake
        steer left      — full left steer (-100%)
        steer right     — full right steer (+100%)
        steer straight  — center steer (0%)
        steer <pct>     — steer to percentage, e.g. "steer -50" or "steer 75"
    """
    instructions = []
    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Line {lineno}: expected '<time> <action>', got: {raw!r}")
            try:
                time_s = float(parts[0])
            except ValueError:
                raise ValueError(f"Line {lineno}: invalid time {parts[0]!r}")
            instructions.append(Instruction(time_s=time_s, action=parts[1]))

    return sorted(instructions, key=lambda i: i.time_s)


def apply_action(action: str, state: InputState) -> None:
    """Mutate InputState according to the given action string."""

    match action:
        case "press up":
            state.accelerate = True
            state.brake = False
        case "press down":
            state.brake = True
            state.accelerate = False
        case "release":
            state.accelerate = False
            state.brake = False
        case "steer left":
            state.steer = steer_percent(-100)
        case "steer right":
            state.steer = steer_percent(100)
        case "steer straight":
            state.steer = 0
        case _:
            if action.startswith("steer "):
                try:
                    pct = int(action.split()[1])
                    state.steer = steer_percent(pct)
                    return
                except (IndexError, ValueError):
                    pass
            raise ValueError(f"Unknown action: {action!r}")
