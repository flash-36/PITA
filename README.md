# PITA Experiments Skeleton

A clean, modular experiment framework for math reasoning with Hydra configs.

## Conda setup
```bash
conda env create -f environment.yml
conda activate pita
```

## Run a sample experiment

### Multi-GPU Parallel Execution (Recommended)
```bash
python run_parallel.py experiment.name=my_experiment
```

The parallel runner automatically:
- Detects available GPUs
- Parallelizes data generation across GPUs
- Distributes training jobs optimally
- Works seamlessly with 1 GPU or multiple GPUs

Hydra will create an output directory at `outputs/<date>/<experiment-name>-<time>/`.

## Config overview
- experiment.name (required): Name for the run; used in output paths.
- algos: Configure multiple algorithms (enable/disable and hyperparams).
- model_pairs: Reference/classifier model mappings.
- datasets: List of dataset names.
- system.use_parallel_execution (default: true): Enable multi-GPU parallel execution

Override any config at the CLI, e.g.:
```bash
python run_parallel.py experiment.name=my_exp algos.PITA.enabled=true datasets="[AIME,IMDBGen]"
```

### Monitor GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

For detailed GPU execution guide, see `GPU_EXECUTION_GUIDE.md`.

## Plotting
This repo provides a plotting hook to generate figures after runs, and a separate function to re-plot from saved results with style overrides for paper-ready figures.
