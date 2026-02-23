"""Chapter 00 figure generators — Quantitative Stock Pre-Selection."""

from theory.figures.ch00._fig01 import Fig01HysteresisTurnover
from theory.figures.ch00._fig02 import Fig02CrossFactorCorrelation
from theory.figures.ch00._fig04 import Fig04RankNormalHistograms
from theory.figures.ch00._fig05 import Fig05SectorNeutralization
from theory.figures.ch00._fig06 import Fig06AlphaScoreDistribution
from theory.figures.ch00._fig07 import Fig07RollingIC
from theory.figures.ch00._fig08 import Fig08VIFBarChart
from theory.figures.ch00._fig09 import Fig09BufferTurnover
from theory.figures.ch00._fig10 import Fig10NetAlpha
from theory.figures.ch00._fig11 import Fig11SectorBalance
from theory.figures.ch00._fig12 import Fig12RegimeTiltHeatmap
from theory.figures.ch00._fig13 import Fig13TimeVaryingWeights
from theory.figures.ch00._fig14 import Fig14QuintileSpreads
from theory.figures.ch00._fig15 import Fig15ISvsOOSIC
from theory.figures.ch00._registry import build_generators

__all__ = [
    "Fig01HysteresisTurnover",
    "Fig02CrossFactorCorrelation",
    "Fig04RankNormalHistograms",
    "Fig05SectorNeutralization",
    "Fig06AlphaScoreDistribution",
    "Fig07RollingIC",
    "Fig08VIFBarChart",
    "Fig09BufferTurnover",
    "Fig10NetAlpha",
    "Fig11SectorBalance",
    "Fig12RegimeTiltHeatmap",
    "Fig13TimeVaryingWeights",
    "Fig14QuintileSpreads",
    "Fig15ISvsOOSIC",
    "build_generators",
]
