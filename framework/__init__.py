"""Game-agnostic RL training framework.

This package contains all code that is independent of any specific game.
Game integrations live in games/<name>/ and depend on this package,
but this package must never import from games/.
"""
