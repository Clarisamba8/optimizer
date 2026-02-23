"""Chapter 01 figure generators — Data Preparation and Universe Construction."""

from theory.figures.ch01._fig17 import Fig17ArithmeticVsLog
from theory.figures.ch01._fig19 import Fig19ScalingComparison
from theory.figures.ch01._fig21 import Fig21OutlierGroups
from theory.figures.ch01._fig23 import Fig23DataQuality
from theory.figures.ch01._fig24 import Fig24CorrelationDistribution
from theory.figures.ch01._fig25 import Fig25ParetoFront
from theory.figures.ch01._fig27 import Fig27PreselectionFunnel
from theory.figures.ch01._registry import build_generators

__all__ = [
    "Fig17ArithmeticVsLog",
    "Fig19ScalingComparison",
    "Fig21OutlierGroups",
    "Fig23DataQuality",
    "Fig24CorrelationDistribution",
    "Fig25ParetoFront",
    "Fig27PreselectionFunnel",
    "build_generators",
]
