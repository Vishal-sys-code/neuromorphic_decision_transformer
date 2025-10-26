# SNN-DT: Spiking Neural Network Decision Transformer

This repository contains the code for Phase 1 of the SNN-DT project, focusing on baselines and comparisons.

## Structure

- `data/`: Raw and processed datasets.
- `src/`: Library code for models, environments, training, evaluation, and utilities.
- `experiments/`: Experiment configurations and run scripts.
- `results/`: CSVs, model checkpoints, and figures.
- `notebooks/`: Rapid analysis and exploration.
- `scripts/`: CLI scripts for training, evaluation, dataset generation, and plotting.
- `docs/`: Project documentation.
- `tests/`: Unit tests.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Datasets

```bash
python scripts/make_dataset.py --env CartPole-v1 --seed 1234 --out data/CartPole-v1/dataset_v1.npz
```

### 2. Train a Model

```bash
python scripts/train.py --model dt --config experiments/configs/dt_cartpole.yaml --seed 42 --save-dir results/cartpole/dt_run1
```

### 3. Evaluate a Model

```bash
python scripts/eval.py --ckpt results/cartpole/dt_run1/best.pt --env CartPole-v1 --episodes 50
```

### 4. Plot Results

```bash
python scripts/plot_results.py --summary results/summary.csv --out results/figures/
```