# Noise Ablation Experiments

## Why run noise ablations?
- Goal: stress-test active learning acquisition functions when a controllable fraction of the unlabeled pool is permanently corrupted.
- Setup: corrupt 0%, 20%, 40%, 60%, or 80% of training samples at load time with Gaussian, Uniform, or Poisson perturbations (strength defaults to 0.1 in normalized pixel space).
- Expectation: pool-based strategies will react differently—e.g., core-set style methods may avoid noisy outliers, while pure uncertainty methods might over-query them. This mirrors real-world data collection where sensors intermittently fail.

## What do we measure?
- Same metrics already logged by `demo.py`: per-round test accuracy, area under the learning curve (AUBC), and wall-clock acquisition time.
- Comparisons of those curves across noise levels highlight robustness (or brittleness) of each acquisition function.

## How to run
1. Submit the new Slurm job:
   ```bash
   sbatch job_scripts/run_noise_ablation.job
   ```
2. The script sweeps:
   - Datasets: `MNIST`, `CIFAR10`
   - Acquisition functions: `RandomSampling`, `EntropySampling`, `KCenterGreedy`
   - Noise types: `none`, `gaussian`, `uniform`, `poisson`
   - Noise fractions: `0.0, 0.2, 0.4, 0.6, 0.8`
3. All runs share seed `4666`, making corrupt sample selection reproducible via `--input_noise_seed`.

Tune `--input_noise_strength` or extend the dataset/strategy lists in `job_scripts/run_noise_ablation.job` to explore additional regimes.

## W&B Logging (optional)

By default, W&B logging is **enabled** (`--use_wandb True`). Disable it with `--use_wandb False` if running locally without network access or if you prefer local-only logging.

### Configuration
- `--use_wandb`: Enable/disable W&B logging (default: True).
- `--wandb_project`: W&B project name (default: `deepal-noise-ablation`).
- `--wandb_entity`: W&B entity/username (default: None; uses your default W&B workspace).

### Logged Metrics
**Per-round metrics** (logged at each AL round):
- `test_accuracy`: Test set accuracy after labeling and training.
- `num_labeled`: Number of samples in labeled pool.
- `acquisition_time_sec`: Time taken to query samples (seconds).
- `train_loss`: Average training loss per epoch (logged per epoch within a round).

**Run-level summaries** (logged at end of experiment):
- `mean_aubc`: Mean AUBC across all iterations.
- `std_aubc`: Std dev of AUBC.
- `mean_acquisition_time`: Mean wall-clock acquisition time.
- `final_accuracy`: Test accuracy at the end of the AL budget.

### Example: Disable W&B
```bash
python demo.py \
  -a RandomSampling \
  -d MNIST \
  -s 500 \
  -q 10000 \
  -b 250 \
  --seed 4666 \
  -t 1 \
  -g 0 \
  --use_wandb False
```

### Example: Custom W&B Project
```bash
python demo.py \
  -a EntropySampling \
  -d CIFAR10 \
  -s 1000 \
  -q 40000 \
  -b 500 \
  --seed 4666 \
  -t 1 \
  -g 0 \
  --use_wandb True \
  --wandb_project my-al-experiments \
  --wandb_entity my-team
```

### Dashboard Tips
- Use **parallel coordinates** plot in W&B to visualize how different strategies respond to noise levels.
- Compare `test_accuracy` curves across noise types to see robustness differences.
- Filter runs by `noise_fraction` to isolate specific corruption levels and compare strategy rankings.

