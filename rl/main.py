"""Module alias for the top-level RL trainer.

Allows ``python -m rl.main --game assetto my_experiment`` (per issue #79
acceptance criterion 1) to drive the same entry point as ``python main.py``.
"""

from __future__ import annotations

from main import main


if __name__ == "__main__":
    main()
