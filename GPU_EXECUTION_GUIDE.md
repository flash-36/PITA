# GPU Execution Guide

Complete guide for running PITA experiments with multi-GPU parallel execution.

## Quick Start

### Run an Experiment
```bash
# Automatic multi-GPU parallel execution
python run_parallel.py experiment.name=my_experiment

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi
```

The parallel runner automatically:
- Detects all available GPUs
- Parallelizes data generation across GPUs
- Distributes training jobs optimally
- Works seamlessly with 1 GPU or multiple GPUs (up to 8+)

### Test Your Setup
```bash
# Verify GPU detection and parallel functionality
python test_parallel_setup.py
```

## Architecture Overview

### Core Components

#### 1. GPU Manager (`pita/core/gpu_manager.py`)
Automatically detects and manages GPU resources:
- GPU detection and memory monitoring
- Optimal batch size estimation
- Work distribution across GPUs

#### 2. Parallel Data Generation (`pita/core/parallel_generation.py`)
Data generation parallelized across GPUs using multiprocessing:
- Each GPU processes a subset of examples independently
- Reward scoring uses larger batches (32 instead of 2)
- Automatically scales to available GPUs

#### 3. Parallel Job Execution (`pita/core/parallel_executor.py`)
Training loop parallelization:
- Multiple algorithm/dataset/model combinations run simultaneously
- Jobs grouped by round to respect dependencies
- GPU memory managed to prevent OOM

#### 4. Algorithm Base Classes (`pita/algos/base.py`)
All algorithms now include parallel execution capabilities:
- `ValueGuidedAlgorithms` - For PITA and QSharp
- `PostTrainingAlgorithms` - For DPO, GRPO, and GRPO-HF

### How It Works

#### Data Generation Phase
1. **GPU Detection**: `GPUManager` detects all available GPUs
2. **Job Distribution**: `ParallelDataGenerator` splits examples across GPUs
3. **Worker Setup**: Each worker process:
   - Gets assigned a specific GPU via `CUDA_VISIBLE_DEVICES`
   - Receives serialized config (plain Python dict)
   - Reconstructs OmegaConf object
   - Loads model on its GPU
   - Processes its batch of examples

#### Training Phase
1. **Job Queue**: All training jobs created upfront
2. **Round-based Scheduling**: Jobs grouped by round for dependency management
3. **Parallel Execution**: Jobs run in parallel across GPUs where dependencies allow
4. **Worker Isolation**: Each worker:
   - Has own GPU assignment
   - Receives serialized config
   - Imports necessary modules (algorithms, datasets)
   - Reconstructs config and paths
   - Runs assigned job

## Configuration

### Key Settings

Edit `conf/config.yaml`:

```yaml
# System settings
system:
  attn_impl: eager               # Attention implementation (eager/flash_attention_2)
  clear_cache_interval: 10       # Clear GPU cache every N steps
  use_parallel_execution: true   # Enable multi-GPU parallel execution

# Training settings
training:
  micro_batch_size: 4            # Gradient accumulation micro-batches
  gradient_checkpointing: true   # Save memory at cost of speed

# Data collection
data_collection:
  reward_batch_size: 32          # Reward model batch size (was 2)

# Algorithm settings
algos:
  PITA:
    enabled: true
    batch_size: 24               # Training batch size (was 16)
    num_workers: 4               # Data loader workers (was 2)
```

### GPU-Specific Configurations

#### For 40GB A100s
```yaml
algos:
  PITA:
    batch_size: 16               # Smaller batch
training:
  micro_batch_size: 2
  gradient_checkpointing: true   # Essential
```

#### For 80GB A100s
```yaml
algos:
  PITA:
    batch_size: 32               # Larger batch
training:
  micro_batch_size: 4
  gradient_checkpointing: false  # Can disable for speed
```

#### For 2 GPUs
- Parallel execution uses both GPUs
- Jobs distributed evenly
- Data generation uses both GPUs

#### For 8 GPUs
- Maximum parallelism across experiments
- Each GPU can run different algorithm/dataset
- Data generation uses all 8 GPUs

## Performance Optimizations

### Key Improvements

1. **Reward Model Batching**: 2 → 32 batch size (16x increase)
2. **Data Generation**: Parallel across GPUs (up to 8x speedup)
3. **Job Parallelism**: Multiple experiments simultaneously
4. **Optimized Batch Sizes**: Tuned for A100 GPUs (40-80GB)
5. **Memory Management**: Gradient checkpointing, micro-batching, cache clearing

### Expected Speedups

With multi-GPU parallelization:
- **2 GPUs**: ~1.8x speedup
- **4 GPUs**: ~3.5x speedup
- **8 GPUs**: ~6.5-7x speedup

Speedup isn't linear due to:
- Data loading overhead
- Model loading/unloading
- Synchronization between jobs
- Memory management

## Testing and Validation

### Test 1: GPU Detection
```bash
python test_parallel_setup.py
```

Verifies:
- GPU detection
- Parallel data generation
- Worker process GPU assignments
- Memory utilization

### Test 2: Small Experiment
```bash
python run_parallel.py experiment.name=gpu_test_run
```

Monitor with:
```bash
watch -n 1 nvidia-smi
```

Look for:
- Multiple GPUs showing activity
- Memory allocation on different GPUs
- Different PIDs on different GPUs

### Test 3: Check Logs
```bash
cat outputs/gpu_test_run/*/run.log | grep -i "gpu\|device\|cuda"
```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms**: CUDA out of memory errors

**Solutions**:
1. Reduce batch sizes:
   ```yaml
   algos:
     PITA:
       batch_size: 12  # Reduce from 24
   ```

2. Enable gradient checkpointing:
   ```yaml
   training:
     gradient_checkpointing: true
   ```

3. Reduce micro-batch size:
   ```yaml
   training:
     micro_batch_size: 2  # Reduce from 4
   ```

4. Use smaller models or fewer concurrent jobs

### Issue: Only One GPU Used

**Symptoms**: nvidia-smi shows only GPU 0 active

**Causes**:
1. Job dependencies prevent parallelization
2. Not enough jobs to saturate multiple GPUs
3. Worker processes not setting CUDA_VISIBLE_DEVICES correctly

**Solutions**:
1. Check job dependencies - ensure rounds allow parallelization
2. Increase `num_samples` in config to create more jobs
3. Verify `CUDA_VISIBLE_DEVICES` in worker logs
4. Check that `use_parallel_execution: true` in config

### Issue: Worker Process Crashes

**Symptoms**: Workers exit with pickling or import errors

**Solutions**:
1. Ensure all imports inside worker functions (already implemented)
2. Verify config serialization (already fixed)
3. Check run_root is passed correctly (already fixed)

### Issue: Flash Attention Not Available

**Symptoms**: Warning about flash_attention_2

**Solution**: Set explicitly to eager:
```yaml
system:
  attn_impl: eager
```

## Monitoring

### During Execution

The script automatically prints GPU summaries:
```
GPU Memory Summary:
  GPU 0: 45.23 GB allocated, 48.50 GB reserved, 80.00 GB total
  GPU 1: 42.18 GB allocated, 45.30 GB reserved, 80.00 GB total
```

### External Tools

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# If nvitop installed
nvitop

# Monitor specific experiment logs
tail -f outputs/my_experiment/*/run.log
```

### After Execution

```bash
# Check GPU utilization
grep "GPU utilization" outputs/my_experiment/*/run.log

# Check for errors
grep -i "error\|warning" outputs/my_experiment/*/run.log
```

## Diagnostic Commands

```bash
# Check GPU availability
nvidia-smi

# Check CUDA in Python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Verify imports
python -c "import pita.algos; import pita.datasets; print('Imports OK')"

# Test config serialization
python -c "from omegaconf import OmegaConf; from hydra import compose, initialize; initialize(config_path='conf', version_base=None); cfg = compose(config_name='config'); d = OmegaConf.to_container(cfg, resolve=True); print('Serialization OK')"
```

## Best Practices

1. **Start small**: Test with small experiments first
2. **Monitor memory**: Watch GPU usage during first run
3. **Adjust batch sizes**: Based on your specific GPU memory
4. **Use parallel execution**: For multiple experiments
5. **Clear cache**: Regularly for long experiments
6. **Save checkpoints**: Frequently to prevent data loss

## Configuration Tips

### Optimal Batch Sizes
```yaml
training:
  micro_batch_size: 4              # Per GPU
  gradient_checkpointing: true     # Save memory
  
generation:
  batch_size: 8                    # Per GPU for data generation
  
algos:
  PITA:
    batch_size: 32                 # For reward scoring
```

### Scaling to More GPUs

The system automatically scales. No config changes needed for 8 GPUs:
- Data generation uses all 8 GPUs in parallel
- Training jobs distributed across all 8 GPUs

## Advanced Usage

### Custom Job Scheduling

Edit `pita/core/parallel_executor.py` to customize:
```python
def estimate_job_priority(job: TrainingJob, cfg) -> int:
    # Higher priority jobs run first
    return job.round_idx * 100
```

### Distributed Data Parallel (DDP)

For even more speed (future feature):
```bash
torchrun --nproc_per_node=4 run_parallel.py experiment.name=my_experiment
```

The distributed wrapper in `pita/trainers/distributed_wrapper.py` provides utilities for this.

## Files Structure

### Core Files
- `run_parallel.py` - Main entry point for parallel execution
- `test_parallel_setup.py` - Setup validation script
- `conf/config.yaml` - Configuration file

### Parallel Execution Modules
- `pita/core/gpu_manager.py` - GPU detection and management
- `pita/core/parallel_generation.py` - Parallel data generation
- `pita/core/parallel_executor.py` - Job queue and execution
- `pita/trainers/distributed_wrapper.py` - DDP utilities

### Algorithm Files
- `pita/algos/base.py` - Base classes with parallel support
- `pita/algos/pita_algo.py` - PITA algorithm
- `pita/algos/qsharp_algo.py` - QSharp algorithm
- `pita/algos/dpo_algo.py` - DPO algorithm
- `pita/algos/grpo_algo.py` - GRPO algorithm
- `pita/algos/grpo_hf_algo.py` - GRPO-HF algorithm

### Evaluation
- `pita/eval/evaluate.py` - Standard evaluation (used by algorithms)
- `pita/eval/evaluate_parallel.py` - Batched evaluation utilities

## Support

If you encounter issues:
1. Check error traceback carefully
2. Verify GPU availability with `nvidia-smi`
3. Review logs in `outputs/` directory
4. Ensure config has valid settings
5. Run `test_parallel_setup.py` for diagnostics

## Future Enhancements

Potential improvements:
- Pipeline parallelism for very large models
- Model sharding with FSDP
- Asynchronous data generation
- Dynamic batch size adjustment
- Multi-node support
- Better load balancing
- Improved checkpointing and resume

