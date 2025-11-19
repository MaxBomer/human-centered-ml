#!/bin/bash

# ==========================================
# TEST SCRIPT FOR JOB FILES (FIXED)
# ==========================================

# Define the mock python function and export it so the subshells see it
function python() {
    echo "    [CMD] python $*"
}
export -f python

run_test() {
    local job_file=$1
    echo "========================================================"
    echo "TESTING: $job_file"
    echo "========================================================"
    
    # We filter out 'module' and 'source' commands using sed, then pipe to bash.
    # This allows the logic (loops, variables) to run while ignoring environment setup.
    # We also ensure #SBATCH directives are ignored (they are comments in bash anyway).
    
    sed '/^\s*module/d; /^\s*source/d' "$job_file" | bash
    
    echo ""
}

run_test "job_scripts/run_noise_ablation.job"
run_test "job_scripts/run_noise_ablation_kcenter_mnist.job"

for frac in 0.2 0.4 0.6 0.8; do
    run_test "job_scripts/run_noise_ablation_kcenter_cifar_${frac}.job"
done

echo "========================================================"
echo "VERIFICATION COMPLETE"
echo "If you see the correct python commands above, the logic is valid."
echo "========================================================"
