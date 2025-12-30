
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import ast # literal_eval for parsing lists in CSV

# --- Configuration ---
EVAL_DIR = Path(__file__).parent.parent / "spike_eval"
FROZEN_DIR = Path(__file__).parent.parent / "datasets/frozen_v1"
FIGURES_DIR = Path(__file__).parent.parent / "figures/dataset_validation"
ENVS = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCar-v0"]

# Style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.dpi'] = 150

def parse_lists(df):
    """Parses string representations of lists in the dataframe."""
    for col in ["spikes_per_layer", "spikes_per_head", "spikes_per_timestep"]:
        df[col] = df[col].apply(ast.literal_eval)
    return df

def analyze_env(env_name):
    print(f"--- Analyzing {env_name} ---")
    env_eval_dir = EVAL_DIR / env_name
    env_fig_dir = FIGURES_DIR
    env_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    try:
        strat_df = pd.read_csv(env_eval_dir / "stratified.csv")
        random_df = pd.read_csv(env_eval_dir / "random_heavy.csv")
    except FileNotFoundError:
        print(f"Skipping {env_name}: CSVs not found.")
        return None

    strat_df = parse_lists(strat_df)
    random_df = parse_lists(random_df)
    
    # --- Step 3: Summary Statistics ---
    stats = {}
    for name, df in [("Stratified", strat_df), ("Random-Heavy", random_df)]:
        spikes = df["spikes_per_inference"]
        stats[name] = {
            "mean": spikes.mean(),
            "std": spikes.std(),
            "median": spikes.median(),
            "p25": spikes.quantile(0.25),
            "p75": spikes.quantile(0.75),
            "ci_lower": spikes.mean() - 1.96 * spikes.std() / np.sqrt(len(spikes)),
            "ci_upper": spikes.mean() + 1.96 * spikes.std() / np.sqrt(len(spikes))
        }
    
    # Paired Difference (assuming matched runs by seed/id)
    diff = random_df["spikes_per_inference"] - strat_df["spikes_per_inference"]
    stats["Difference"] = {
        "mean": diff.mean(),
        "ci_lower": diff.mean() - 1.96 * diff.std() / np.sqrt(len(diff)),
        "ci_upper": diff.mean() + 1.96 * diff.std() / np.sqrt(len(diff))
    }
    
    # --- Step 4: Distributions ---
    plt.figure(figsize=(15, 5))
    
    # Histogram
    plt.subplot(1, 3, 1)
    sns.histplot(strat_df["spikes_per_inference"], color="blue", label="Stratified", kde=True, alpha=0.5)
    sns.histplot(random_df["spikes_per_inference"], color="red", label="Random-Heavy", kde=True, alpha=0.5)
    plt.legend()
    plt.title(f"{env_name}: Spike Distribution")
    
    # Boxplot
    plt.subplot(1, 3, 2)
    data_combined = pd.DataFrame({
        "Spikes": pd.concat([strat_df["spikes_per_inference"], random_df["spikes_per_inference"]]),
        "Dataset": ["Stratified"] * len(strat_df) + ["Random-Heavy"] * len(random_df)
    })
    sns.boxplot(data=data_combined, x="Dataset", y="Spikes", palette=["blue", "red"])
    plt.title("Boxplot Comparison")
    
    # CDF
    plt.subplot(1, 3, 3)
    sns.ecdfplot(strat_df["spikes_per_inference"], color="blue", label="Stratified")
    sns.ecdfplot(random_df["spikes_per_inference"], color="red", label="Random-Heavy")
    plt.legend()
    plt.title("CDF Comparison")
    
    plt.tight_layout()
    plt.savefig(env_fig_dir / f"{env_name}_distributions.png")
    plt.close()
    
    # --- Step 5: Diagnostics ---
    # Layer-wise (Mean over runs)
    strat_layers = np.mean(strat_df["spikes_per_layer"].tolist(), axis=0)
    random_layers = np.mean(random_df["spikes_per_layer"].tolist(), axis=0)
    
    plt.figure(figsize=(6, 4))
    x = np.arange(len(strat_layers))
    width = 0.35
    plt.bar(x - width/2, strat_layers, width, label='Stratified', color='blue', alpha=0.7)
    plt.bar(x + width/2, random_layers, width, label='Random-Heavy', color='red', alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("Avg Spikes")
    plt.title(f"{env_name}: Layer-wise Spikes")
    plt.legend()
    plt.savefig(env_fig_dir / f"{env_name}_layers.png")
    plt.close()
    
    # Head-wise
    # Timestep-wise
    strat_time = np.mean(strat_df["spikes_per_timestep"].tolist(), axis=0)
    random_time = np.mean(random_df["spikes_per_timestep"].tolist(), axis=0)
    
    plt.figure(figsize=(6, 4))
    plt.plot(strat_time, label='Stratified', color='blue', marker='o')
    plt.plot(random_time, label='Random-Heavy', color='red', marker='x')
    plt.xlabel("Timestep")
    plt.ylabel("Avg Spikes")
    plt.title(f"{env_name}: Temporal Spike Profile")
    plt.legend()
    plt.savefig(env_fig_dir / f"{env_name}_temporal.png")
    plt.close()

    # --- Step 6: RTG Distribution Confirmation (Re-plotting from frozen) ---
    # Load Frozen Data
    strat_npz = np.load(FROZEN_DIR / env_name / "stratified_dataset.npz")
    random_npz = np.load(FROZEN_DIR / env_name / "random_heavy_dataset.npz")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(strat_npz['returns_to_go'].flatten(), color="blue", label="Stratified", bins=50)
    sns.histplot(random_npz['returns_to_go'].flatten(), color="red", label="Random-Heavy", bins=50, alpha=0.5)
    plt.legend()
    plt.title(f"{env_name}: RTG Distribution")
    
    # Variance
    srtg_var = np.var(strat_npz['returns_to_go'])
    rrtg_var = np.var(random_npz['returns_to_go'])
    
    plt.subplot(1, 2, 2)
    plt.bar(["Stratified", "Random-Heavy"], [srtg_var, rrtg_var], color=["blue", "red"])
    plt.title("RTG Variance")
    plt.tight_layout()
    plt.savefig(env_fig_dir / f"{env_name}_rtg_stats.png")
    plt.close()
    
    stats["RTG_Variance"] = {
        "Stratified": srtg_var,
        "Random-Heavy": rrtg_var
    }
    
    # --- Step 7: Correlation Check ---
    # We want to check correlation between RTG and Spike Counts
    # To see if higher spiking is needed for higher return (or conversely, if low return = collapsed spikes)
    # The hypothesis is "no pathological correlation where lower spikes = collapsed performance"
    
    correlation_results = {}
    for name, df in [("Stratified", strat_df), ("Random-Heavy", random_df)]:
        # Need avg_rtg from df
        if "avg_rtg" not in df.columns:
            print(f"Warning: avg_rtg not in {env_name} {name} logs. Skipping correlation.")
            continue
            
        pearson = df["spikes_per_inference"].corr(df["avg_rtg"], method='pearson')
        spearman = df["spikes_per_inference"].corr(df["avg_rtg"], method='spearman')
        
        correlation_results[name] = {
            "pearson": pearson,
            "spearman": spearman
        }
        
        print(f"  {name} Correlation (Spikes vs RTG): Pearson={pearson:.2f}, Spearman={spearman:.2f}")

    stats["Correlations"] = correlation_results
    
    return stats

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_data = []
    
    for env in ENVS:
        stats = analyze_env(env)
        if stats:
            summary_data.append({
                "Environment": env,
                "Strat_Mean": stats["Stratified"]["mean"],
                "Random_Mean": stats["Random-Heavy"]["mean"],
                "Diff_Mean": stats["Difference"]["mean"],
                "Diff_CI_Low": stats["Difference"]["ci_lower"],
                "Diff_CI_High": stats["Difference"]["ci_upper"],
                "RTG_Var_Strat": stats["RTG_Variance"]["Stratified"],
                "RTG_Var_Random": stats["RTG_Variance"]["Random-Heavy"]
            })
            
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(EVAL_DIR.parent / "spike_summary_table.csv", index=False)
        print("Summary table saved.")

if __name__ == "__main__":
    main()
