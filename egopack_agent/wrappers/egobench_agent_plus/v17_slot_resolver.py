# -*- coding: utf-8 -*-
"""V17 slot resolver facade.

Current smoke5 implementation exposes the generic compiler repair entry point.
It is kept separate so later V17 work can add stronger slot-specific logic
without changing runner wiring.
"""

from __future__ import annotations

from .v17_process_compiler import repair_tool_output

__all__ = ["repair_tool_output"]
