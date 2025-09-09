# PITA Experiments Skeleton

A clean, modular experiment framework for math reasoning with Hydra configs.

## Conda setup
```bash
conda env create -f environment.yml
conda activate pita
```

## Run a sample experiment
```bash
python run.py --config-name run1
```
Hydra will create an output directory at `outputs/<date>/<experiment-name>-<time>/`.

## Config overview
- experiment.name (required): Name for the run; used in output paths.
- algos: Configure multiple algorithms (enable/disable and hyperparams).
- model_pairs: Reference/classifier model mappings.
- datasets: List of dataset names.

Override any config at the CLI, e.g.:
```bash
python run.py experiment.name=my_exp algos.PITA.enabled=true datasets="[AIME,IMDBGen]"
```

## Plotting
This repo provides a plotting hook to generate figures after runs, and a separate function to re-plot from saved results with style overrides for paper-ready figures.
