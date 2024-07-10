run_experiment.sh#!/bin/bash

# Predefined values
GPU_INDEX=0
MAX_ITERATIONS=2500
MODEL="convnetsmall"
CALLBACK_INTERVAL=$(( MAX_ITERATIONS / 10 ))
BATCH_SIZES=(1 2 4 8 16 32) # List of batch sizes
REPETITIONS=10         # Default value 1

# Experiment group name
EXPERIMENT_GROUP="fast_batch_size"

# List of optimizers
OPTIMIZERS=("SGD")

# Function to display usage
usage() {
    echo "Usage: $0 [-n] [-g GROUP_NAME]"
    echo "  -n               Do not run pytest before experiments"
    echo "  -g GROUP_NAME    Specify the experiment group name (default: group_1)"
    exit 1
}

# Parse command-line arguments
RUN_PYTEST=true
while getopts "ng:" opt; do
    case ${opt} in
        n ) RUN_PYTEST=false
            ;;
        g ) EXPERIMENT_GROUP=$OPTARG
            ;;
        * ) usage
            ;;
    esac
done

# Run pytest to ensure all tests pass, if required
if [ "$RUN_PYTEST" = true ]; then
    echo "Running pytest to ensure all tests pass..."
    pytest
    if [ $? -ne 0 ]; then
        echo "Tests failed. Aborting the experiment."
        exit 1
    fi
fi

# move to the results directory, so all files are created there
cd ../results

# Function to run the experiment
run_experiment() {
    local optimizer=$1
    local batch_size=$2
    local experiment_name=$3

    echo "Starting experiment with optimizer: $optimizer and batch size: $batch_size"
    nohup python3 ../breaching/tilo_experiment.py $GPU_INDEX $MAX_ITERATIONS $optimizer $MODEL $experiment_name --callback_interval $CALLBACK_INTERVAL --batch_size $batch_size --image_count $batch_size --repetitions $REPETITIONS > ${experiment_name}.out 2>&1 &

    # Wait for the current experiment to finish
    wait $!
    echo "Experiment $experiment_name finished."
}

# Loop through the list of optimizers and batch sizes, and run the experiment for each combination
for optimizer in "${OPTIMIZERS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        experiment_name="${EXPERIMENT_GROUP}_experiment_${optimizer}_batch_size_${batch_size}"
        run_experiment $optimizer $batch_size $experiment_name
    done
done

# Tar all of the folders at the end
tar -czf ${EXPERIMENT_GROUP}_results.tar.gz ${EXPERIMENT_GROUP}_experiment_*

echo "All experiments completed."