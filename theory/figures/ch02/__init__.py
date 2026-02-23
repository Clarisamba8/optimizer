"""Chapter 02 figure generators — Moment Estimation and Prior Construction."""

from theory.figures.ch02._fig28 import Fig28SampleMeanInstability
from theory.figures.ch02._fig30 import Fig30ShrinkageScatter
from theory.figures.ch02._fig32 import Fig32EquilibriumVsHistorical
from theory.figures.ch02._fig33 import Fig33LWEigenvalues
from theory.figures.ch02._fig34 import Fig34MarchenkoPastur
from theory.figures.ch02._fig36 import Fig36Detoning
from theory.figures.ch02._fig37 import Fig37GerberVsPearson
from theory.figures.ch02._fig38 import Fig38FactorModelParams
from theory.figures.ch02._fig39 import Fig39HMMRegimes
from theory.figures.ch02._fig40 import Fig40HMMBlended
from theory.figures.ch02._fig42 import Fig42DMMvsHMM
from theory.figures.ch02._registry import build_generators

__all__ = [
    "Fig28SampleMeanInstability",
    "Fig30ShrinkageScatter",
    "Fig32EquilibriumVsHistorical",
    "Fig33LWEigenvalues",
    "Fig34MarchenkoPastur",
    "Fig36Detoning",
    "Fig37GerberVsPearson",
    "Fig38FactorModelParams",
    "Fig39HMMRegimes",
    "Fig40HMMBlended",
    "Fig42DMMvsHMM",
    "build_generators",
]
