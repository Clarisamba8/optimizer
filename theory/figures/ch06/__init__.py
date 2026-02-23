"""Chapter 06 figure generators.

Validation, Model Selection, and Production Pipeline.
"""

from theory.figures.ch06._fig84 import Fig84WalkForward
from theory.figures.ch06._fig85 import Fig85RollingVsExpanding
from theory.figures.ch06._fig86 import Fig86CPCVFoldMatrix
from theory.figures.ch06._fig87 import Fig87CPCVSharpe
from theory.figures.ch06._fig89 import Fig89PerformanceRadar
from theory.figures.ch06._fig90 import Fig90GridSearch
from theory.figures.ch06._fig91 import Fig91Pipeline
from theory.figures.ch06._fig92 import Fig92RebalancingFrequency
from theory.figures.ch06._fig93 import Fig93TradeTriggers
from theory.figures.ch06._fig94 import Fig94GrossVsNet
from theory.figures.ch06._registry import build_generators

__all__ = [
    "Fig84WalkForward",
    "Fig85RollingVsExpanding",
    "Fig86CPCVFoldMatrix",
    "Fig87CPCVSharpe",
    "Fig89PerformanceRadar",
    "Fig90GridSearch",
    "Fig91Pipeline",
    "Fig92RebalancingFrequency",
    "Fig93TradeTriggers",
    "Fig94GrossVsNet",
    "build_generators",
]
