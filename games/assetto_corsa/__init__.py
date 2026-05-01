"""Assetto Corsa game integration.

Implements the framework's abstract interfaces using the assetto-corsa-rl
Gymnasium wrapper. Unlike the TMNF integration, this package has no
Windows-only or game-process dependencies at import time — the underlying
gym env is imported lazily inside the client wrapper so the package can be
introspected and stubbed on any OS (including Linux CI).
"""
