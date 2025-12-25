# Phase 2: Ablation Study & Baseline Comparison

This directory contains the code and resources for the Phase 2 ablation study of the SNN-DT project, which now includes a comprehensive comparison against baseline models.

## 1. Installation

Ensure you have the required dependencies installed from the main project's `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

You will also need to install `stable-baselines3`, `pyyaml`, `tqdm`, and `gymnasium`:

```bash
pip install stable-baselines3 pyyaml tqdm gymnasium
```

## 2. Dataset Generation

The new dataset generation process is a multi-stage pipeline that produces high-quality, return-stratified datasets.

### Step 1: Generate Raw Trajectories

First, generate the raw trajectories from random, medium, and expert policies.

```bash
python scripts/A1_generate_trajectories.py
```

This will save the raw trajectories to `ablation_studies/datasets/raw`.

### Step 2: Process Datasets

Next, process the raw trajectories to create the final, stratified datasets.

```bash
python scripts/B1_process_datasets.py
```

This will create `stratified_dataset.npz` and `random_heavy_dataset.npz` in `ablation_studies/datasets/processed` for each environment.

### Step 3: Verify Datasets

Finally, verify the quality of the generated datasets.

```bash
python scripts/F1_verify_datasets.py
```

This will generate distribution plots in `ablation_studies/datasets/verification_plots` and print a spike sanity check to the console.

## 3. Run Experiments

To run an experiment, use the `run_experiment.py` script with the desired variant, environment, and seed.

### Ablation Variants

**Example:**

```bash
python run_experiment.py --variant full --env CartPole-v1 --seed 1001
```

### Baseline Models

**Example:**

```bash
python run_experiment.py --variant dt --env CartPole-v1 --seed 1001
```

### Full Experimental Run

You can run all experiments using a simple shell loop:

```bash
for variant in full no_phase no_routing no_plasticity dt snn_dt iql cql; do
  for env in CartPole-v1 Acrobot-v1 Pendulum-v1; do
    for seed in 1001 1002 1003; do
      echo "--- Running $variant on $env with seed $seed ---"
      python run_experiment.py --variant "$variant" --env "$env" --seed "$seed"
    done
  done
done
```

## 4. Post-process Results

After the experiments are complete, you can generate the plots and summary tables using the `post_process.py` script.

```bash
python scripts/post_process.py
```

This will save the figures to the `ablation_studies/figures` directory.