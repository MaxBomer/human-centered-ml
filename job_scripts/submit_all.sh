#!/bin/bash

echo "Submitting all noise ablation jobs..."

sbatch job_scripts/run_noise_ablation.job
sbatch job_scripts/run_noise_ablation_kcenter_mnist.job
sbatch job_scripts/run_noise_ablation_kcenter_cifar_0.2.job
sbatch job_scripts/run_noise_ablation_kcenter_cifar_0.4.job
sbatch job_scripts/run_noise_ablation_kcenter_cifar_0.6.job
sbatch job_scripts/run_noise_ablation_kcenter_cifar_0.8.job

echo "All jobs submitted! Use 'squeue -u <your_username>' to monitor them."

