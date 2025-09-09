from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt


def plot_after_run(cfg, results: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Minimal example: plot count of configs executed
    num_points = sum(
        len(pairs) if isinstance(pairs, dict) else 0 for pairs in results.values()
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["runs"], [num_points])
    ax.set_title(f"{cfg.experiment.name}: Executed runs")
    fig.tight_layout()
    fig.savefig(output_dir / "summary.png", dpi=200)
    plt.close(fig)


def replot_from_saved(results_dir: Path, output_dir: Path, style_overrides: Dict[str, Any] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Placeholder for re-plotting logic with style overrides
    pass
