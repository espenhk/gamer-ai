"""StarCraft 2 game integration via PySC2.

Implements the framework's abstract interfaces using DeepMind's PySC2 API.
Two scopes are supported:

1. **Minigames** (``map_name`` is one of MoveToBeacon, CollectMineralShards,
   FindAndDefeatZerglings, DefeatRoaches, DefeatZerglingsAndBanelings,
   CollectMineralsAndGas, BuildMarines).  Fully observable, short episodes,
   small discrete action subset per map.

2. **Ladder game stub** (``map_name`` is e.g. ``Simple64``).  Builds an
   ``SC2Env`` against a built-in bot opponent so issue #91 can layer
   fog-of-war belief machinery on top.  No learning is claimed at this
   level — the env just runs without crashing.
"""
