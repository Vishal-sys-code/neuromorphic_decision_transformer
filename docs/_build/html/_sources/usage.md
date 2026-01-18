# Usage Guide

This framework is designed to facilitate research into spiking decision transformers. It supports training, evaluation, and ablation studies across standard OpenAI Gym control tasks.

## Training

To train the SNN-DT model, use the `train.py` script. The pipeline handles data generation, offline trajectory preprocessing, and surrogate gradient training.

### Single Experiment

```bash
python snn-dt/scripts/train.py \
    --model snn_dt \
    --env "Pendulum-v1" \
    --save-dir "results/snn_dt_pendulum" \
    --max_iters 10000
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model` | architecture variant (`snn_dt` or `dt`) | `snn_dt` |
| `--env` | Gym environment ID | `Hopper-v3` |
| `--context_len` | Sequence length $K$ | `20` |

### Batch Reproduction

To reproduce the paper's results across all environments:

```bash
./run_all_experiments.sh
```

## Evaluation & Metrics

We provide tools to evaluate checkpoints for both reward performance and energy metrics (spike counts).

```bash
python snn-dt/scripts/eval_snn_dt.py \
    --env "Pendulum-v1" \
    --checkpoint_path "results/snn_dt_pendulum/best_model.pt" \
    --target_return -200
```

The script reports:
-   **Normalized Return**: Performance relative to expert.
-   **Spike Rate**: Average spikes per time step.
-   **Energy Estimate**: Total energy consumption based on AC/MAC counts.

## Comparison Baselines

To run the vanilla Decision Transformer (ANN) baseline:

```bash
python snn-dt/scripts/train.py --model dt --env "CartPole-v1"
```