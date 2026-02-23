"""Theory figure generation utilities."""

from theory.figures._base import FigureGenerator
from theory.figures._helpers import clean_returns, prices_to_log_returns, prices_to_returns
from theory.figures._loader import PriceLoader
from theory.figures._runner import run_chapter

__all__ = [
    "FigureGenerator",
    "PriceLoader",
    "clean_returns",
    "prices_to_log_returns",
    "prices_to_returns",
    "run_chapter",
]
