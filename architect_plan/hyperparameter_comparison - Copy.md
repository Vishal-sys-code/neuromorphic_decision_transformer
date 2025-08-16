# Hyperparameter Comparison: SNN-DT vs DSF-DT

## Shared Hyperparameters

| Parameter | Value | Description |
|----------|-------|-------------|
| Environment | CartPole-v1 | Gym environment |
| Sequence Length | 20 | Context window length |
| Batch Size | 64 | Training batch size |
| Discount Factor (gamma) | 0.99 | For return-to-go calculation |
| Learning Rate | 1e-4 | Optimizer learning rate |
| Offline Steps | 5000 | Random policy steps to collect |
| Training Epochs | 10 | Number of training epochs |

## DSF-DT Specific Hyperparameters

| Parameter | Value | Description |
|----------|-------|-------------|
| Hidden Size | 128 | Embedding dimension |
| Number of Layers | 2 | Transformer layers |
| Number of Heads | 1 | Attention heads |
| Time Window | N/A | Not applicable for DSF-DT |
| T | 4 | Spiking time steps |
| Norm Type | 1 | Layer normalization |
| Warmup Ratio | 0.1 | Learning rate warmup |
| Window Size | 8 | Local attention window |

## SNN-DT Specific Hyperparameters

| Parameter | Value | Description |
|----------|-------|-------------|
| Hidden Size | 128 | Embedding dimension |
| Number of Layers | 2 | Transformer layers |
| Number of Heads | 1 | Attention heads |
| Time Window | 5 | Spiking time steps |
| T | N/A | Not applicable for SNN-DT |
| Norm Type | N/A | Uses standard LayerNorm |
| Warmup Ratio | N/A | Not implemented |
| Window Size | N/A | Not implemented |

## Model Architecture Differences

### DSF-DT Architecture
- Based on Spiking Transformer blocks
- Uses LIF neurons in attention and feed-forward layers
- Implements PTNorm (Positional Temporal Normalization)
- Uses positional spiking attention mechanism
- Has dedicated spiking components for Q, K, V projections

### SNN-DT Architecture
- Based on standard Decision Transformer with spiking attention
- Uses LIF neurons for attention computation
- Implements rate encoding for spike generation
- Uses spiking multi-head attention with temporal integration
- Has modular Phase 3 components (positional encoder, dendritic router)

## Hyperparameter Matching Strategy

To ensure a fair comparison:

1. **Shared Configuration:**
   - Same environment (CartPole-v1)
   - Same sequence length (20)
   - Same batch size (64)
   - Same training epochs (10)
   - Same offline dataset (5000 steps)

2. **Model-Specific Adjustments:**
   - Hidden size: 128 for both
   - Layers: 2 for both
   - Heads: 1 for both
   - Time window: 5 for SNN-DT, 4 for DSF-DT (closest match)

3. **Evaluation Consistency:**
   - Same evaluation protocol (10 episodes)
   - Same metrics collected (return, latency, spikes)
   - Same hardware for timing comparisons