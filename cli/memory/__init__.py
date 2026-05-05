"""CLI entry point for ``m memory`` — mellea's external memory tooling.

Subcommands live in ``commands.py``. Import ``memory_app`` when wiring into
``cli/m.py``.
"""

from .commands import memory_app

__all__ = ["memory_app"]
