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

