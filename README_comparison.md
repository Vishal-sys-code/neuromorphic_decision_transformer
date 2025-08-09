# SNN-DT vs DSF-DT Comparison

This repository contains code for comparing SNN-DT (Spiking Neural Network Decision Transformer) and DSF-DT (Decision SpikeFormer Decision Transformer) on the CartPole-v1 environment.

## Overview

This comparison evaluates both models using matched hyperparameters on the same offline dataset to ensure a fair comparison. The evaluation metrics include:
- Average return over 10 episodes
- Standard deviation of returns
- Average inference latency
- Average spikes per episode
- Total number of parameters

## Files

- `src/run_comparison.py`: Main script to run the comparison
- `src/plot_comparison.py`: Script to visualize results
- `src/train_snn_dt.py`: SNN-DT training implementation
- `src/train_dsf_dt.py`: DSF-DT training implementation
- `src/run_benchmark.py`: Evaluation script
- `src/config.py`: Configuration file with hyperparameters

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Gym
- NumPy
- Pandas
- Matplotlib
- Transformers library

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Running the Comparison

### Full Comparison (Training + Evaluation)

```bash
python src/run_comparison.py --env CartPole-v1 --seed 42
```

This will:
1. Collect a shared offline dataset using a random policy
2. Train both SNN-DT and DSF-DT models on the same dataset
3. Evaluate both models on CartPole-v1
4. Save results to `comparison_results.csv`
5. Save training losses to `training_losses.csv`

### Skip Training (Evaluation Only)

If you already have trained models, you can skip training:

```bash
python src/run_comparison.py --env CartPole-v1 --seed 42 --skip-training
```

### Individual Model Training

You can also train models individually:

```bash
# Train SNN-DT
python src/train_snn_dt.py

# Train DSF-DT
python src/train_dsf_dt.py
```

### Individual Model Evaluation

Evaluate models individually:

```bash
# Evaluate SNN-DT
python src/run_benchmark.py --model snn-dt --env CartPole-v1

# Evaluate DSF-DT
python src/run_benchmark.py --model dsf-dt --env CartPole-v1
```

## Visualization

After running the comparison, you can visualize the results:

```bash
python src/plot_comparison.py
```

This will generate a plot comparing:
- Average returns
- Inference latency
- Spiking activity
- Training loss curves

## Expected Results

The comparison will generate:
- `comparison_results.csv`: Numerical results for both models
- `training_losses.csv`: Training loss curves
- `comparison_results.png`: Visualization of results

## Hyperparameters

Both models use the following matched hyperparameters:
- Environment: CartPole-v1
- Sequence length: 20
- Batch size: 64
- Training epochs: 10
- Offline steps: 5000
- Hidden size: 128
- Layers: 2
- Heads: 1
- Learning rate: 1e-4

Model-specific differences:
- SNN-DT: 5 spiking time steps
- DSF-DT: 4 spiking time steps

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `src/config.py`
2. **Training Instability**: Reduce learning rate in `src/config.py`
3. **Checkpoint Loading Errors**: Verify that models were trained successfully
4. **Environment Issues**: Ensure Gym is properly installed

### Debugging Steps

1. Check that all dependencies are installed
2. Verify environment is correctly set up
3. Confirm hyperparameters match between models
4. Ensure dataset is correctly formatted