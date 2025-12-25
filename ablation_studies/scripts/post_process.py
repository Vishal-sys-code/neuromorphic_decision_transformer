import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
FIGURES_DIR = Path(__file__).parent.parent / "figures"
RUNS_DIR = Path(__file__).parent.parent / "runs"
SPIKE_ENERGY_PJ = 5.0

# --- Plotting Style ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# --- Helper Functions ---
def parse_results():
    records = []
    for path in RUNS_DIR.rglob("metrics.jsonl"):
        run_name = path.parent.parent.parent.name
        seed = path.parent.parent.name.split('_')[1]
        env = path.parent.name
        
        with open(path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'val/mean_return' in record:
                        record['variant'] = run_name
                        record['seed'] = int(seed)
                        record['env'] = env
                        records.append(record)
                except (json.JSONDecodeError, KeyError):
                    continue
    return pd.DataFrame(records)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# --- Plotting Functions ---
def plot_learning_curves(df):
    for env in df['env'].unique():
        plt.figure(figsize=(10, 6))
        
        env_df = df[df['env'] == env].copy()
        
        smoothed_dfs = []
        for (variant, seed), group in env_df.groupby(['variant', 'seed']):
            group = group.sort_values('epoch')
            if len(group) >= 5:
                group['smoothed_return'] = moving_average(group['val/mean_return'].values, 5)
                group = group.iloc[4:] # Adjust for convolution window
                smoothed_dfs.append(group)
        
        if not smoothed_dfs: continue
        
        smoothed_df = pd.concat(smoothed_dfs)

        sns.lineplot(data=smoothed_df, x='epoch', y='smoothed_return', hue='variant', errorbar='sd')
        plt.title(f"Learning Curves for {env}")
        plt.xlabel("Epoch")
        plt.ylabel("Episodic Return (Smoothed)")
        plt.legend(title="Model/Variant")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{env}_learning_curves.pdf")
        plt.close()
    print("Learning curves plotted.")

def generate_summary_table(df):
    final_metrics = df.loc[df.groupby(['env', 'variant', 'seed'])['epoch'].idxmax()]
    
    final_metrics['energy_nJ'] = (final_metrics.get('val/spikes_per_inference', 0) * SPIKE_ENERGY_PJ) / 1000

    summary = final_metrics.groupby(['env', 'variant']).agg(
        mean_return=('val/mean_return', 'mean'),
        std_return=('val/mean_return', 'std'),
        spikes=('val/spikes_per_inference', 'mean'),
        energy=('energy_nJ', 'mean'),
        latency=('val/inference_latency_ms', 'mean')
    ).reset_index()
    
    summary.to_csv(FIGURES_DIR / "summary_metrics.csv", index=False)
    with open(FIGURES_DIR / "summary_metrics.tex", "w") as f:
        f.write(summary.to_latex(index=False, float_format="%.2f", caption="Summary of performance.", label="tab:summary"))
        
    print("Summary table generated.")
    return summary

def plot_pareto(summary_df):
    for env in summary_df['env'].unique():
        plt.figure(figsize=(8, 6))
        
        env_df = summary_df[summary_df['env'] == env]
        
        sns.scatterplot(data=env_df, x='energy', y='mean_return', hue='variant', s=150, style='variant')
        plt.title(f"Energy vs. Performance for {env}")
        plt.xlabel("Energy per Decision (nJ)")
        plt.ylabel("Mean Return")
        plt.legend(title="Model/Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{env}_pareto.pdf")
        plt.close()
    print("Pareto plots generated.")

# --- Main ---
def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    
    print("Parsing results...")
    results_df = parse_results()
    
    if results_df.empty:
        print("No results found. Exiting.")
        return
        
    plot_learning_curves(results_df)
    summary_df = generate_summary_table(results_df)
    plot_pareto(summary_df)
    
    print(f"All plots and tables saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()