"""Behavior analysis package."""

from typing import Any

__all__ = ["BehaviorAnalyzer"]


def __getattr__(name: str) -> Any:
    if name == "BehaviorAnalyzer":
        from .analyzer import BehaviorAnalyzer

        return BehaviorAnalyzer
    raise AttributeError(name)
