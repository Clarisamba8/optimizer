"""Chapter runner — iterates generators and reports progress."""

from __future__ import annotations

from theory.figures._base import FigureGenerator


def run_chapter(generators: list[FigureGenerator]) -> None:
    """Execute every generator in order and print a summary.

    Parameters
    ----------
    generators:
        Ordered list of :class:`FigureGenerator` instances to execute.
    """
    for gen in generators:
        print(f"\n--- {gen.name} ---")
        gen.generate()
    print(f"\nAll {len(generators)} figures generated.")
