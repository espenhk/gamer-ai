"""Atari 2600 game integration via ale-py / Gymnasium.

The canonical RL benchmark suite — 50+ Atari titles exposed through
``gymnasium.make("ALE/<Game>-v5")`` with ale-py shipping the MIT-licensed
ROMs.  Default observation mode is RAM (128 bytes), which is a fixed-size
flat vector compatible with every framework policy.

Stable-retro / NES/SNES/Genesis support is not implemented in this initial
integration and may be added later under the same adapter.
"""
