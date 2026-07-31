import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# D4RL Reference Scores for normalization (expert, random)
REF_SCORES = {
    "hopper-medium-v2": (3234.3, -20.27),
    "walker2d-medium-v2": (4592.3, 1.629),
    "halfcheetah-medium-v2": (12135.0, -280.178)
}

def normalize_score(env, score):
    if env not in REF_SCORES:
        return score
    expert, random = REF_SCORES[env]
    return (score - random) / (expert - random) * 100.0

def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_file = repo_root / "results/energy_benchmark_results.csv"
    
    if not results_file.exists():
        print(f"Results file not found at {results_file}")
        return
        
    df = pd.read_csv(results_file)
    
    # Calculate D4RL Normalized Score
    df['normalized_return'] = df.apply(lambda row: normalize_score(row['env'], row['eval_return_mean']), axis=1)
    
    # Calculate Formal Metric: Efficiency Score = Normalized Reward / Energy (Joules)
    df['efficiency_score_train'] = df['normalized_return'] / df['train_energy_joules']
    df['efficiency_score_eval'] = df['normalized_return'] / df['eval_energy_joules']
    
    # Aggregate stats over seeds
    stats = df.groupby(['env', 'model']).agg(
        normalized_return_mean=('normalized_return', 'mean'),
        normalized_return_std=('normalized_return', 'std'),
        train_energy_mean=('train_energy_joules', 'mean'),
        train_energy_std=('train_energy_joules', 'std'),
        train_time_mean=('train_time_s', 'mean'),
        train_time_std=('train_time_s', 'std'),
        eval_energy_mean=('eval_energy_joules', 'mean'),
        eval_energy_std=('eval_energy_joules', 'std'),
        efficiency_score_mean=('efficiency_score_train', 'mean'),
        params_mean=('params', 'mean')
    ).reset_index()

    # Calculate Energy to reach DT Performance (Heuristic based on 200 steps)
    # We compare the energy of SNN-DT to DT for the same configuration
    dt_stats = stats[stats['model'] == 'dt'].set_index('env')
    
    def calc_relative_efficiency(row):
        try:
            dt_perf = dt_stats.loc[row['env'], 'normalized_return_mean']
            # If SNN-DT achieves X% of DT's performance, how much energy did it use relative to DT?
            # It's an approximation since we don't have a dense curve.
            return (row['normalized_return_mean'] / max(dt_perf, 1e-5)) * 100.0
        except KeyError:
            return np.nan

    stats['perc_of_dt_perf'] = stats.apply(calc_relative_efficiency, axis=1)

    print("\n--- Summary Statistics (Mean ± Std) ---")
    for env in df['env'].unique():
        print(f"\nEnvironment: {env}")
        env_stats = stats[stats['env'] == env]
        for _, row in env_stats.iterrows():
            print(f"  Model: {row['model']:<10} | "
                  f"Score: {row['normalized_return_mean']:.1f} ± {row['normalized_return_std']:.1f} | "
                  f"Train Energy: {row['train_energy_mean']:.1f}J ± {row['train_energy_std']:.1f} | "
                  f"Eval Energy: {row['eval_energy_mean']:.1f}J ± {row['eval_energy_std']:.1f} | "
                  f"Train Time: {row['train_time_mean']:.1f}s | "
                  f"Efficiency (Score/J): {row['efficiency_score_mean']:.4f}")

    # Plotting
    plots_dir = repo_root / "results/plots"
    plots_dir.mkdir(exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Reward vs Train Energy Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='train_energy_joules', y='normalized_return', hue='model', style='env', s=150)
    plt.title("Reward vs Training Energy (200 steps)")
    plt.xlabel("Total Training Energy (Joules)")
    plt.ylabel("D4RL Normalized Score")
    plt.savefig(plots_dir / "reward_vs_train_energy.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Reward vs Train Time Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='train_time_s', y='normalized_return', hue='model', style='env', s=150)
    plt.title("Reward vs Training Wall-Clock Time (200 steps)")
    plt.xlabel("Total Training Time (Seconds)")
    plt.ylabel("D4RL Normalized Score")
    plt.savefig(plots_dir / "reward_vs_train_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Efficiency Score Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='env', y='efficiency_score_train', hue='model', capsize=0.1)
    plt.title("Training Efficiency Score (Reward / Joules)")
    plt.ylabel("Efficiency Score")
    plt.savefig(plots_dir / "efficiency_score_train.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Inference Energy Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='env', y='energy_per_eval_episode', hue='model', capsize=0.1)
    plt.title("Inference Energy per Episode")
    plt.ylabel("Energy (Joules)")
    plt.savefig(plots_dir / "inference_energy_per_episode.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {plots_dir}")

if __name__ == "__main__":
    main()
