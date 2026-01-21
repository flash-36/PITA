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

## Resume Failed/Interrupted Runs

If a run fails or is interrupted, you can resume from where it left off:

```bash
python run_parallel.py resume_from=outputs/2025-10-20/my_experiment-14-30-00
```

### To reproduce results from the paper:

```bash
sh run_experiments.sh
```


