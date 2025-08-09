#!/bin/bash

# SNN-DT vs DSF-DT Comparison Script

echo "=== SNN-DT vs DSF-DT Comparison ==="
echo "Starting comparison on CartPole-v1"

# Run the comparison
echo "Running comparison..."
python src/run_comparison.py --env CartPole-v1 --seed 42

# Check if comparison was successful
if [ $? -eq 0 ]; then
    echo "Comparison completed successfully!"
    
    # Generate plots
    echo "Generating plots..."
    python src/plot_comparison.py
    
    # Display results
    echo "Displaying results..."
    if [ -f comparison_results.csv ]; then
        echo "=== Numerical Results ==="
        cat comparison_results.csv
    fi
    
    if [ -f training_losses.csv ]; then
        echo "=== Training Losses ==="
        head -10 training_losses.csv
        echo "..."
        tail -5 training_losses.csv
    fi
    
    echo "=== Comparison Complete ==="
    echo "Results saved to:"
    echo "- comparison_results.csv"
    echo "- training_losses.csv"
    echo "- comparison_results.png"
else
    echo "Comparison failed!"
    exit 1
fi