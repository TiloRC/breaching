#!/bin/bash

# move to the results directory, so all files are created there
cd ../results

# Predefined values
GPU_INDEX=0
MAX_ITERATIONS=10
MODEL="covnetsmall"
CALLBACK_INTERVAL=100  # Default value 100
BATCH_SIZE=1           # Default value 1
REPETITIONS=1          # Default value 1

# Experiment group name
EXPERIMENT_GROUP="group_1"

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

# Function to run the experiment
run_experiment() {
    local optimizer=$1
    local experiment_name=$2

    echo "Starting experiment with optimizer: $optimizer"
    nohup python3 ../breaching/tilo_experiment.py $GPU_INDEX $MAX_ITERATIONS $optimizer $MODEL $experiment_name --callback_interval $CALLBACK_INTERVAL --batch_size $BATCH_SIZE --repetitions $REPETITIONS > ${experiment_name}.out 2>&1 &

    # Wait for the current experiment to finish
    wait $!
    echo "Experiment $experiment_name finished."
}

# Loop through the list of optimizers and run the experiment for each one
for optimizer in "${OPTIMIZERS[@]}"; do
    experiment_name="${EXPERIMENT_GROUP}_experiment_${optimizer}"
    run_experiment $optimizer $experiment_name
done

echo "All experiments completed."
