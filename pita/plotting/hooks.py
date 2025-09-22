from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict

import matplotlib.pyplot as plt


def plot_after_run(cfg, results: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics per (dataset, family) across algos and rounds
    # Input shape: results[algo][family][dataset][rN] -> {metrics: {...}}
    coll: DefaultDict[Tuple[str, str], Dict[str, Dict[int, Dict[str, float]]]] = (
        defaultdict(dict)
    )
    for algo_key, fam_map in results.items():
        if not isinstance(fam_map, dict):
            continue
        for family, ds_map in fam_map.items():
            if not isinstance(ds_map, dict):
                continue
            for dataset, rounds in ds_map.items():
                if not isinstance(rounds, dict):
                    continue
                series = coll[(dataset, family)].setdefault(algo_key, {})
                for rkey, payload in rounds.items():
                    if not isinstance(payload, dict):
                        continue
                    metrics = payload.get("metrics") or {}
                    if not isinstance(metrics, dict):
                        continue
                    try:
                        rnum = int(str(rkey).lstrip("r"))
                    except Exception:
                        continue
                    series[rnum] = {
                        "pass1": float(metrics.get("pass@1", 0.0) or 0.0),
                        "maj8": float(metrics.get("maj@8", 0.0) or 0.0),
                    }

    # Colors and labels
    color_map = {
        "QSharp": "#7b3fb3",  # purple
        "DPO": "#87ceeb",  # sky blue
        "PITA": "#888888",
        "QSharpHF": "#ff8c00",
    }

    for (dataset, family), algo_series in coll.items():
        if not algo_series:
            continue
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
        for algo_key, rmap in sorted(algo_series.items()):
            if not rmap:
                continue
            rounds = sorted(rmap.keys())
            y_pass1 = [rmap[r]["pass1"] for r in rounds]
            y_maj8 = [rmap[r]["maj8"] for r in rounds]
            c = color_map.get(algo_key, None)

            ax1.plot(
                rounds, y_pass1, marker="o", linewidth=1.8, label=algo_key, color=c
            )
            ax2.plot(rounds, y_maj8, marker="o", linewidth=1.8, label=algo_key, color=c)

        ax1.set_ylabel("pass@1")
        ax2.set_ylabel("maj@8")
        ax2.set_xlabel("Round")
        title = f"{cfg.experiment.name}: {dataset} · {family}"
        fig.suptitle(title, y=0.98)
        ax2.legend(loc="lower right", frameon=False, ncol=2)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f"{dataset}_{family}_metrics.png".replace("/", "-")
        fig.savefig(output_dir / fname, dpi=200)
        plt.close(fig)


def replot_from_saved(
    results_dir: Path, output_dir: Path, style_overrides: Dict[str, Any] | None = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Placeholder for re-plotting logic with style overrides
    pass
