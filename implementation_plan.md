# Implementation Plan: SNN-DT vs DSF-DT Comparison

## Phase 1: Environment Setup and Data Collection

### Task 1.1: Verify Environment Setup
- Check that all required packages are installed:
  - gym (with CartPole-v1)
  - torch
  - numpy
  - pandas
  - transformers
- Verify CUDA availability if using GPU

### Task 1.2: Data Collection for CartPole-v1
- Use the existing `collect_trajectories` function from `train_dsf_dt.py` and `train_snn_dt.py`
- Collect 5000 steps of random policy data
- Save trajectories to a shared file for both models

## Phase 2: DSF-DT Implementation and Training

### Task 2.1: Verify DSF-DT Model Implementation
- Review `src/models/dsf_dt.py` for DecisionSpikeFormer implementation
- Check `src/train_dsf_dt.py` for training procedure
- Ensure model can be instantiated with correct hyperparameters

### Task 2.2: Train DSF-DT on CartPole-v1
- Run training script with default hyperparameters:
  - Environment: CartPole-v1
  - Sequence length: 20
  - Batch size: 64
  - Epochs: 10
  - Offline steps: 5000
- Save checkpoints during training
- Log training metrics (loss, etc.)

## Phase 3: SNN-DT Implementation and Training

### Task 3.1: Verify SNN-DT Model Implementation
- Review `src/models/snn_dt_patch.py` and related files
- Check `src/train_snn_dt.py` for training procedure
- Ensure model can be instantiated with matching hyperparameters

### Task 3.2: Configure SNN-DT Hyperparameters
- Match hyperparameters to DSF-DT as closely as possible:
  - Hidden size: 128
  - Number of layers: 2
  - Number of heads: 1
  - Sequence length: 20
  - Time window: 5
  - Batch size: 64
  - Epochs: 10

### Task 3.3: Train SNN-DT on CartPole-v1
- Run training script with matched hyperparameters
- Use the same offline dataset as DSF-DT
- Save checkpoints during training
- Log training metrics (loss, etc.)

## Phase 4: Evaluation and Benchmarking

### Task 4.1: Implement Evaluation Framework
- Use existing `src/run_benchmark.py` as a starting point
- Ensure both models can be evaluated with the same protocol
- Add logging for all required metrics:
  - Average return
  - Standard deviation of returns
  - Inference latency
  - Spike count
  - Validation loss
  - Parameter count

### Task 4.2: Evaluate DSF-DT
- Load best checkpoint
- Run evaluation on CartPole-v1 for 10 episodes
- Collect and log all metrics

### Task 4.3: Evaluate SNN-DT
- Load best checkpoint
- Run evaluation on CartPole-v1 for 10 episodes
- Collect and log all metrics

## Phase 5: Results Analysis and Reporting

### Task 5.1: Create Comparison Report
- Generate comparison table with all metrics
- Create visualizations for key metrics:
  - Return comparison
  - Latency comparison
  - Spike efficiency comparison
- Write analysis of results

### Task 5.2: Verify Results
- Check that both models achieve reasonable performance
- Verify that metrics are logged correctly
- Ensure results are reproducible

## Detailed Implementation Steps

### Step 1: Data Collection Script
```python
# Collect shared dataset for both models
python src/train_dsf_dt.py --collect-only --env CartPole-v1
```

### Step 2: DSF-DT Training
```python
# Train DSF-DT
python src/train_dsf_dt.py --env CartPole-v1
```

### Step 3: SNN-DT Training
```python
# Train SNN-DT
python src/train_snn_dt.py --env CartPole-v1
```

### Step 4: Evaluation
```python
# Evaluate both models
python src/run_benchmark.py --model dsf-dt --env CartPole-v1
python src/run_benchmark.py --model snn-dt --env CartPole-v1
```

## Expected Outcomes

1. Both models successfully trained on CartPole-v1
2. Comprehensive benchmark results for both models
3. Comparison report with visualizations
4. Reproducible results with logged metrics

## Risk Mitigation

1. If training is unstable, reduce learning rate or epochs
2. If models don't converge, check hyperparameter matching
3. If evaluation fails, verify checkpoint loading
4. If performance is poor, verify data collection and preprocessing