"""
Debug script: accelerate straight until the finish line (or 60 s).

Usage
-----
1. In-game: place the car just before the finish line.
2. From tmnf/:
       python -m games.tmnf.tools.debug_straight

The InstructionClient will:
  - brake to a stop (BRAKING_START)
  - pause 2 s (PAUSE_START)
  - hold full throttle, straight (RUNNING) until 60 s elapsed
  - brake to a stop again (BRAKING_END)

Watch the console for phase transitions, periodic state prints, and any
finish/simulation events.
"""

import logging
import time

from tminterface.interface import TMInterface

logger = logging.getLogger(__name__)

from games.tmnf.clients.instruction_client import InstructionClient

INSTRUCTION_FILE = "runs/debug_straight.txt"
CENTERLINE_FILE = "tracks/a03_centerline.npy"
SPEED = 1.0  # real-time; increase if you want faster playback


def main():
    client = InstructionClient(
        instruction_file=INSTRUCTION_FILE,
        centerline_file=CENTERLINE_FILE,
        speed=SPEED,
    )
    iface = TMInterface()
    logger.info("Waiting for TMInterface connection...")
    iface.register(client)
    try:
        while iface.running:
            time.sleep(0.002)  # 2 ms yield — avoids busy-spin while waiting for TMInterface
    except KeyboardInterrupt:
        pass
    iface.close()


if __name__ == "__main__":
    main()
