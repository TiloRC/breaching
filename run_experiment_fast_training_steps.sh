#!/bin/bash

# Predefined values
GPU_INDEX=0
MAX_ITERATIONS=2500
MODEL="convnetsmall"
CALLBACK_INTERVAL=$(( MAX_ITERATIONS / 10 ))
BATCH_SIZES=(4 8) # List of batch sizes
IMAGE_COUNT=8
REPETITIONS=10         # Default value 1
EPOCH_COUNTS=(1 2 4 8 16)  # List of epoch counts

# Experiment group name
EXPERIMENT_GROUP="fast_training_step"

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
    local epoch_count=$3
    local experiment_name=$4

    echo "Starting experiment with optimizer: $optimizer, batch size: $batch_size, and epoch count: $epoch_count"
    nohup python3 ../breaching/tilo_experiment.py $GPU_INDEX $MAX_ITERATIONS $optimizer $MODEL $experiment_name --callback_interval $CALLBACK_INTERVAL --batch_size $batch_size --image_count $IMAGE_COUNT --repetitions $REPETITIONS --epoch_count $epoch_count > ${experiment_name}.out 2>&1 &

    # Wait for the current experiment to finish
    wait $!
    echo "Experiment $experiment_name finished."
}

# Loop through the list of optimizers, batch sizes, and epoch counts, and run the experiment for each combination
for optimizer in "${OPTIMIZERS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for epoch_count in "${EPOCH_COUNTS[@]}"; do
            experiment_name="${EXPERIMENT_GROUP}_experiment_${optimizer}_batch_size_${batch_size}_epochs_${epoch_count}"
            run_experiment $optimizer $batch_size $epoch_count $experiment_name
        done
    done
done

# Tar all of the folders at the end
tar -czf ${EXPERIMENT_GROUP}_results.tar.gz ${EXPERIMENT_GROUP}_experiment_*

echo "All experiments completed."