"""Chapter 04 figure generators.

Risk Measures, Diversification, and Hierarchical Methods.
"""

from theory.figures.ch04._fig56 import Fig56AnnotatedRiskMeasures
from theory.figures.ch04._fig57 import Fig57CVaRVsVaR
from theory.figures.ch04._fig59 import Fig59RiskRadar
from theory.figures.ch04._fig60 import Fig60RiskContributionPies
from theory.figures.ch04._fig62 import Fig62ERCComparison
from theory.figures.ch04._fig63 import Fig63DiversificationScatter
from theory.figures.ch04._fig64 import Fig64DistanceHeatmaps
from theory.figures.ch04._fig65 import Fig65WardDendrogram
from theory.figures.ch04._fig66 import Fig66HRPWeights
from theory.figures.ch04._fig68 import Fig68HERCvsHRP
from theory.figures.ch04._fig69 import Fig69NCOSchematic
from theory.figures.ch04._fig70 import Fig70RegimeRisk
from theory.figures.ch04._fig71 import Fig71RegimeBudgetRotation
from theory.figures.ch04._registry import build_generators

__all__ = [
    "Fig56AnnotatedRiskMeasures",
    "Fig57CVaRVsVaR",
    "Fig59RiskRadar",
    "Fig60RiskContributionPies",
    "Fig62ERCComparison",
    "Fig63DiversificationScatter",
    "Fig64DistanceHeatmaps",
    "Fig65WardDendrogram",
    "Fig66HRPWeights",
    "Fig68HERCvsHRP",
    "Fig69NCOSchematic",
    "Fig70RegimeRisk",
    "Fig71RegimeBudgetRotation",
    "build_generators",
]
