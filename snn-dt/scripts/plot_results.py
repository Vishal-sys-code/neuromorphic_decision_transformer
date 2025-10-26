import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_learning_curves(results_dir, out_dir):
    all_metrics = []
    for env_dir in results_dir.iterdir():
        if not env_dir.is_dir() or env_dir.name == "figures":
            continue
        for model_dir in env_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.iterdir():
                metrics_path = run_dir / "metrics.csv"
                if metrics_path.exists():
                    df = pd.read_csv(metrics_path)
                    df["env"] = env_dir.name
                    df["model"] = model_dir.name
                    df["seed"] = run_dir.name
                    all_metrics.append(df)
    
    if not all_metrics:
        print("No metrics files found. Skipping plotting.")
        return

    full_df = pd.concat(all_metrics)
    
    envs = full_df["env"].unique()
    for env in envs:
        plt.figure(figsize=(10, 6))
        env_df = full_df[full_df["env"] == env]
        sns.lineplot(data=env_df, x="epoch", y="return_mean", hue="model")
        plt.title(f"Learning Curves for {env}")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Return")
        plt.grid(True)
        plt.savefig(out_dir / f"learning_curve_{env}.png")
        plt.close()


def create_summary_table(results_dir, out_dir):
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        print("Summary file not found. Skipping summary table.")
        return
        
    df = pd.read_csv(summary_path)
    
    # Create a pivot table
    pivot = df.pivot_table(index="model", columns="env", values="return_mean", aggfunc=["mean", "std"])
    
    # Save to csv
    pivot.to_csv(out_dir / "summary_table.csv")
    
    # Save as latex
    with open(out_dir / "summary_table.tex", "w") as f:
        f.write(pivot.to_latex())


def plot_energy_performance(results_dir, out_dir):
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        print("Summary file not found. Skipping energy plot.")
        return
        
    df = pd.read_csv(summary_path)
    df["energy"] = df["spikes"] * 5e-3 # pJ per spike
    
    for env in df["env"].unique():
        plt.figure(figsize=(10, 6))
        env_df = df[df["env"] == env]
        sns.barplot(data=env_df, x="model", y="return_mean", hue="model")
        plt.title(f"Energy vs Performance for {env}")
        plt.xlabel("Model")
        plt.ylabel("Mean Return")
        
        ax2 = plt.twinx()
        sns.lineplot(data=env_df, x="model", y="energy", ax=ax2, color="red", marker="o")
        ax2.set_ylabel("Energy (pJ)")
        
        plt.savefig(out_dir / f"energy_performance_{env}.png")
        plt.close()


def plot_spike_histogram(results_dir, out_dir):
    all_metrics = []
    for env_dir in results_dir.iterdir():
        if not env_dir.is_dir() or env_dir.name == "figures":
            continue
        for model_dir in env_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name not in ["snn_dt", "dsformer"]:
                continue
            for run_dir in model_dir.iterdir():
                metrics_path = run_dir / "metrics.csv"
                if metrics_path.exists():
                    df = pd.read_csv(metrics_path)
                    df["env"] = env_dir.name
                    df["model"] = model_dir.name
                    all_metrics.append(df)
    
    if not all_metrics:
        print("No spike data found. Skipping spike histogram.")
        return
        
    full_df = pd.concat(all_metrics)
    
    for env in full_df["env"].unique():
        plt.figure(figsize=(10, 6))
        env_df = full_df[full_df["env"] == env]
        sns.histplot(data=env_df, x="spikes", hue="model", multiple="stack")
        plt.title(f"Spike Counts for {env}")
        plt.xlabel("Spikes per Inference")
        plt.savefig(out_dir / f"spike_histogram_{env}.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results", help="Path to the results directory.")
    parser.add_argument("--out", type=str, default="results/figures", help="Directory to save the plots.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Results directory not found at {args.results}. Skipping plotting.")
        return

    plot_learning_curves(results_dir, out_dir)
    create_summary_table(results_dir, out_dir)
    plot_energy_performance(results_dir, out_dir)
    plot_spike_histogram(results_dir, out_dir)
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()