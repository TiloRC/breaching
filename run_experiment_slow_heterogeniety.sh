#!/bin/bash

# Predefined values
GPU_INDEX=0
MAX_ITERATIONS=3000
MODEL="resnet18"
CALLBACK_INTERVAL=$(( MAX_ITERATIONS / 10 ))
BATCH_SIZE=8        # Single batch size
REPETITIONS=10      # Default value 1
CLASSES_PER_BATCH=(1 2 3 4 5 6 7 8)

# Experiment group name
EXPERIMENT_GROUP="fast_heterogeneity"

# List of optimizers as an array
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
    local class_count=$2
    local experiment_name=$3

    echo "Starting experiment with optimizer: $optimizer and class count: $class_count"
    nohup python3 ../breaching/tilo_experiment.py $GPU_INDEX $MAX_ITERATIONS $optimizer $MODEL $experiment_name \
    --callback_interval $CALLBACK_INTERVAL --batch_size $BATCH_SIZE --image_count $BATCH_SIZE \
    --classes_per_batch $class_count --repetitions $REPETITIONS > ${experiment_name}.out 2>&1 &

    # Wait for the current experiment to finish
    wait $!
    echo "Experiment $experiment_name finished."
}

# Loop through the list of class counts, and run the experiment for each count
for class_count in "${CLASSES_PER_BATCH[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
        experiment_name="${EXPERIMENT_GROUP}_experiment_${optimizer}_classes_${class_count}"
        run_experiment $optimizer $class_count $experiment_name
    done
done

# Tar all of the folders at the end
tar -czf ${EXPERIMENT_GROUP}_results.tar.gz ${EXPERIMENT_GROUP}_experiment_*

echo "All experiments completed."
