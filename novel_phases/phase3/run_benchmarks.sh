#!/bin/bash

# This script runs the benchmark for all ablation modes and saves the results to a CSV file.

# Navigate to the script's directory to ensure correct relative paths
cd "$(dirname "$0")"

# Output CSV file
OUTPUT_FILE="energy_latency_results.csv"

# Write the header to the CSV file
echo "ablation_mode,spikes_per_inference,cpu_latency_ms" > $OUTPUT_FILE

# Run benchmark for each mode and append to the CSV
python3 benchmark.py --ablation_mode baseline >> $OUTPUT_FILE
python3 benchmark.py --ablation_mode pos_only >> $OUTPUT_FILE
python3 benchmark.py --ablation_mode router_only >> $OUTPUT_FILE
python3 benchmark.py --ablation_mode full >> $OUTPUT_FILE

echo "Benchmarking complete. Results saved to $OUTPUT_FILE"