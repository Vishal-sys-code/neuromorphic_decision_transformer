# Training Plan: SNN-DT and DSF-DT on CartPole-v1

## Overview
This document outlines the training procedure for both SNN-DT and DSF-DT models on the CartPole-v1 environment using matched hyperparameters.

## Data Collection Phase

### Shared Dataset Collection
Both models will use the same offline dataset collected from CartPole-v1 using a random policy.

```bash
# Collect 5000 steps of random policy data
python src/train_dsf_dt.py --collect-only --env CartPole-v1
```

This will generate `offline_data_CartPole-v1_dsf.pkl` which will be used for both models.

## DSF-DT Training

### Training Command
```bash
# Train DSF-DT with matched hyperparameters
python src/train_dsf_dt.py --env CartPole-v1
```

### Expected Training Configuration
- Environment: CartPole-v1
- Epochs: 10
- Batch Size: 64
- Sequence Length: 20
- Hidden Size: 128
- Layers: 2
- Heads: 1
- T (spiking steps): 4
- Learning Rate: 1e-4
- Offline Steps: 5000

### Checkpoints
Checkpoints will be saved to `checkpoints/offline_dsf_CartPole-v1_{epoch}.pt`

## SNN-DT Training

### Training Command
```bash
# Train SNN-DT with matched hyperparameters
python src/train_snn_dt.py --env CartPole-v1
```

### Expected Training Configuration
- Environment: CartPole-v1
- Epochs: 10
- Batch Size: 64
- Sequence Length: 20
- Hidden Size: 128
- Layers: 2
- Heads: 1
- Time Window (spiking steps): 5
- Learning Rate: 1e-4
- Offline Steps: 5000

### Checkpoints
Checkpoints will be saved to `checkpoints/offline_dt_CartPole-v1_{epoch}.pt`

## Training Monitoring

### Loss Tracking
Both training scripts will log:
- Epoch number
- Average offline loss
- Checkpoint saving

### Expected Training Time
- DSF-DT: ~4 hours
- SNN-DT: ~4 hours

## Validation

### Model Validation
After training, validate that:
1. Both models have saved checkpoints
2. Training logs show decreasing loss
3. Models can be loaded successfully

### Dataset Consistency
Ensure that:
1. Both models use the exact same offline dataset
2. Data preprocessing is identical
3. Sequence generation is consistent

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size to 32
2. **Training Instability**: Reduce learning rate to 5e-5
3. **Poor Convergence**: Increase epochs to 15
4. **Checkpoint Loading Issues**: Verify model configuration matches checkpoint

### Debugging Steps
1. Check that all dependencies are installed
2. Verify environment is correctly set up
3. Confirm hyperparameters match between models
4. Ensure dataset is correctly formatted