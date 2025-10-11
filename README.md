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
