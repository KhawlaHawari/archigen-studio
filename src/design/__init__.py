"""
Ported from the original ArchiGen backend (team project: rule-based furniture placement engine).

src/design/__init__.py
----------------------
Deterministic rule-based room design placement engine.
"""

from src.design.engine import RoomDesignEngine
from src.design.types import RoomDesignResult, PlacedItem, ValidationReport

__all__ = ["RoomDesignEngine", "RoomDesignResult", "PlacedItem", "ValidationReport"]
