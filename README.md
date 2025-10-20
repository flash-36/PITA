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

### Key Features

- **Same Directory**: All outputs continue in the original run directory
- **Smart Skipping**: Automatically skips completed work (data generation, training phases)
- **Config Overrides**: Change hyperparameters for incomplete jobs
  ```bash
  python run_parallel.py resume_from=outputs/.../run training.micro_batch_size=1
  ```
- **Fine-Grained Tracking**: For complex algorithms, tracks individual phases:
  - **QSharp-HF, GRPO-HF**: Proxy RM training → Data rescoring → Classifier/Policy training
  - **PITA, QSharp**: Classifier training → Evaluation
  
### Resume Behavior

The system checks for completion markers and skips finished work:
- **Data generation** (`.hf` files in `datasets/`)
- **Training phases** (markers in `models/`)  
- **Final results** (`result.json` in `results/`)

Example:
```bash
# Training completes, eval OOMs
python run_parallel.py experiment.name=test

# Resume with smaller eval batch - skips training!
python run_parallel.py resume_from=outputs/2025-10-20/test-14-30-00 evaluation.batch_size=32
```

For detailed phase tracking information, see [`PHASE_TRACKING.md`](PHASE_TRACKING.md).
