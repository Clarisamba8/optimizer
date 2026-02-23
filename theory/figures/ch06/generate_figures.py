"""Entry point: python theory/figures/ch06/generate_figures.py

Run from the repository root:
    python theory/figures/ch06/generate_figures.py

Or as a module (preferred — no sys.path manipulation needed):
    python -m theory.figures.ch06.generate_figures
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is on sys.path when invoked directly as a script
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from theory.figures._loader import PriceLoader  # noqa: E402
from theory.figures._runner import run_chapter  # noqa: E402
from theory.figures.ch06._registry import build_generators  # noqa: E402

_DB_URL = "postgresql://postgres:postgres@localhost:54320/optimizer_db"
_OUTPUT_DIR = Path(__file__).parent

if __name__ == "__main__":
    print(f"Output directory: {_OUTPUT_DIR}\n")
    print("Loading price data from database...")
    loader = PriceLoader(db_url=_DB_URL, min_days=1200)
    prices = loader.load()
    generators = build_generators(prices, _OUTPUT_DIR, _DB_URL)
    run_chapter(generators)
