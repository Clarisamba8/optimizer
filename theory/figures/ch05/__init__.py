"""Chapter 05 figure generators.

Portfolio Optimization and Robust Methods.
"""

from theory.figures.ch05._fig72 import Fig72EfficientFrontier
from theory.figures.ch05._fig74 import Fig74Regularization
from theory.figures.ch05._fig75 import Fig75ConstrainedFrontier
from theory.figures.ch05._fig76 import Fig76RobustEllipse
from theory.figures.ch05._fig78 import Fig78DRCVaR
from theory.figures.ch05._fig79 import Fig79VineCopula
from theory.figures.ch05._fig80 import Fig80StressFan
from theory.figures.ch05._fig81 import Fig81BenchmarkTracking
from theory.figures.ch05._fig82 import Fig82StrategyComparison
from theory.figures.ch05._fig83 import Fig83StackingSchematic
from theory.figures.ch05._registry import build_generators

__all__ = [
    "Fig72EfficientFrontier",
    "Fig74Regularization",
    "Fig75ConstrainedFrontier",
    "Fig76RobustEllipse",
    "Fig78DRCVaR",
    "Fig79VineCopula",
    "Fig80StressFan",
    "Fig81BenchmarkTracking",
    "Fig82StrategyComparison",
    "Fig83StackingSchematic",
    "build_generators",
]
