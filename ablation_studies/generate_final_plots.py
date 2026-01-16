import os
import glob
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
LOG_DIR = "ablation_studies/final_logs"
RUNS_DIR = "ablation_studies/runs"
OUTPUT_DIR = "ablation_studies/figures_final"

# Environments to process
ENVIRONMENTS = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCar-v0"]

# Variant Names and Colors (DeepMind/Google inspired)
VARIANTS = {
    "snn_dt": "SNN-DT",
    "dt": "DT",
    "dsformer": "Decision S-Former",
    "cql": "CQL",
    "iql": "IQL",
    "full": "Full Model",
    "no_phase": "No Phase",
    "no_routing": "No Routing",
    "no_plasticity": "No Plasticity"
}

# Colors
COLORS = {
    "snn_dt": "#4285F4",      # Google Blue
    "dt": "#F4B400",          # Google Yellow
    "dsformer": "#DB4437",    # Google Red
    "cql": "#0F9D58",         # Google Green
    "iql": "#AB47BC",         # Purple
    "full": "#0057E7",        # Bright Blue
    "no_phase": "#A0C4FF",    # Light Blue
    "no_routing": "#FFD6A5",  # Light Orange
    "no_plasticity": "#FFADAD" # Light Red
}
DEFAULT_COLOR = "#5f6368"

# Plotting Style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams.update({
    "font.family": "serif", # Serif for research papers often looks better or sans-serif
    "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
    "axes.titleweight": "bold",
    "axes.titlesize": 18,
    "axes.labelweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

def parse_final_logs():
    """Parses *_logs.txt in final_logs to get Return and Spikes."""
    results = {} # cleaned_variant -> {env -> {mean, std, spikes_mean, spikes_std}}
    
    log_files = glob.glob(os.path.join(LOG_DIR, "*_logs.txt"))
    
    # Regex patterns
    # Header: | Ablation Group: name | EnvName |
    header_pattern = re.compile(r"[|│]\s*Ablation Group:\s*(\S+)\s*[|│]\s*(\S+)\s*[|│]")
    # Seed line: [SEED 0] + Finished   Return: 254.60   Spikes: 0.16
    seed_pattern = re.compile(r"\[SEED \d+\] [^\s]+ Finished\s+Return:\s*([-+]?\d*\.?\d+|Not Found)(?:\s+Spikes:\s*([-+]?\d*\.?\d+))?")
    # Final Summary: Mean Return ...
    # We mainly rely on aggregating seeds for consistency if summary is missing or complex
    
    for fpath in log_files:
        filename = os.path.basename(fpath).replace("_ablation_logs.txt", "").replace("_logs.txt", "") # clean name
        # Mapping filename to variant key if needed, but they seem to match mostly
        variant_key = filename.replace("-", "_")
        
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        current_env = None
        current_group = None
        temp_seeds_return = []
        temp_seeds_spikes = []
        
        for line in lines:
            header_match = header_pattern.search(line)
            if header_match:
                # Save previous if exists
                if current_env and current_group:
                    if temp_seeds_return:
                        if variant_key not in results: results[variant_key] = {}
                        if current_env not in results[variant_key]: results[variant_key][current_env] = {}
                        results[variant_key][current_env] = {
                            "return_mean": np.mean(temp_seeds_return),
                            "return_std": np.std(temp_seeds_return),
                            "spikes_mean": np.mean(temp_seeds_spikes) if temp_seeds_spikes else 0.0,
                            "spikes_std": np.std(temp_seeds_spikes) if temp_seeds_spikes else 0.0
                        }
                
                current_group = header_match.group(1)
                current_env = header_match.group(2)
                temp_seeds_return = []
                temp_seeds_spikes = []
                continue
            
            seed_match = seed_pattern.search(line)
            if seed_match and current_env:
                ret_val = seed_match.group(1)
                spike_val = seed_match.group(2)
                
                if ret_val != "Not Found":
                    temp_seeds_return.append(float(ret_val))
                if spike_val:
                    temp_seeds_spikes.append(float(spike_val))
        
        # Save last
        if current_env and current_group:
             if temp_seeds_return:
                if variant_key not in results: results[variant_key] = {}
                if current_env not in results[variant_key]: results[variant_key][current_env] = {}
                results[variant_key][current_env] = {
                    "return_mean": np.mean(temp_seeds_return),
                    "return_std": np.std(temp_seeds_return),
                    "spikes_mean": np.mean(temp_seeds_spikes) if temp_seeds_spikes else 0.0,
                    "spikes_std": np.std(temp_seeds_spikes) if temp_seeds_spikes else 0.0
                }
                
    return results

def parse_learning_curves():
    """Parses metrics.jsonl in runs directory."""
    # Structure: variant -> env -> pd.DataFrame
    curves = {}
    
    # Walk through runs directory
    # runs/variant/seed_X/Env/metrics.jsonl
    for variant in os.listdir(RUNS_DIR):
        variant_path = os.path.join(RUNS_DIR, variant)
        if not os.path.isdir(variant_path): continue
        
        variant_key = variant.replace("-", "_")
        if variant_key not in curves: curves[variant_key] = {}
        
        for seed_dir in os.listdir(variant_path):
            seed_path = os.path.join(variant_path, seed_dir)
            if not os.path.isdir(seed_path): continue
            
            for env in os.listdir(seed_path):
                env_path = os.path.join(seed_path, env)
                if not os.path.isdir(env_path): continue
                
                metrics_file = os.path.join(env_path, "metrics.jsonl")
                if os.path.exists(metrics_file):
                    try:
                        data = []
                        with open(metrics_file, 'r') as f:
                            for line in f:
                                try:
                                    j = json.loads(line)
                                    data.append(j)
                                except: pass
                        if data:
                            df = pd.DataFrame(data)
                            # We might have multiple seeds. For now, let's just use seed_0 or aggregate?
                            # Aggregating is better.
                            if env not in curves[variant_key]: curves[variant_key][env] = []
                            curves[variant_key][env].append(df)
                    except Exception as e:
                        print(f"Error reading {metrics_file}: {e}")
                        
    # Aggregate curves
    aggregated_curves = {}
    for var, envs in curves.items():
        aggregated_curves[var] = {}
        for env, dfs in envs.items():
            # Align by epoch/step
            # Assuming 'epoch' column exists
            combined_df = pd.concat(dfs)
            # Group by epoch and mean
            if 'epoch' in combined_df.columns:
                grouped = combined_df.groupby('epoch').mean().reset_index()
                aggregated_curves[var][env] = grouped
            else:
                # Fallback if no epoch
                pass
                
    return aggregated_curves

def plot_pareto(results, output_dir):
    """Energy vs Return (Pareto) plot."""
    # We want one plot per Environment
    # Flatten envs
    envs = set()
    for v in results.values():
        envs.update(v.keys())
        
    for env in envs:
        plt.figure(figsize=(10, 7))
        
        # Collect points
        xs = [] # Energy (Spikes)
        ys = [] # Return
        labels = []
        colors = []
        
        for var, data in results.items():
            if env in data:
                res = data[env]
                energy = res['spikes_mean'] # Proxy
                ret = res['return_mean']
                
                # If Spikes is 0 (e.g. DT, CQL, IQL), maybe shift slightly or keep at 0?
                # DT is non-spiking, so Energy is different (MACs).
                # But for this plot "Energy vs Return", if we only have Spikes, 
                # non-spiking models might be excluded or placed at a High Energy point?
                # Or just plot them at x=0 and mention "Non-spiking".
                
                xs.append(energy)
                ys.append(ret)
                labels.append(VARIANTS.get(var, var))
                colors.append(COLORS.get(var, DEFAULT_COLOR))
        
        # Plot
        plt.scatter(xs, ys, s=150, c=colors, alpha=0.9, edgecolors='white', linewidth=1.5, zorder=3)
        
        # Annotate
        for x, y, label in zip(xs, ys, labels):
            plt.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
            
        plt.xlabel("Energy Cost (Mean Spikes/Step)", fontsize=14)
        plt.ylabel("Mean Return", fontsize=14)
        plt.title(f"Pareto Frontier: Energy vs Return ({env})", pad=15)
        plt.grid(True, alpha=0.3)
        sns.despine()
        
        # Save
        save_path = os.path.join(output_dir, env, "pareto_energy_return.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_learning_curves(curves, output_dir):
    """Learning curve with spikes overlay."""
    envs = set()
    for v in curves.values():
        envs.update(v.keys())
        
    for env in envs:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        
        has_data = False
        
        for var, data in curves.items():
            if env in data:
                df = data[env]
                if 'val/mean_return' in df.columns:
                    has_data = True
                    color = COLORS.get(var, DEFAULT_COLOR)
                    label = VARIANTS.get(var, var)
                    
                    # Plot Return (Solid)
                    ax1.plot(df['epoch'], df['val/mean_return'], label=f"{label} Return", color=color, linewidth=2.5)
                    
                    # Plot Spikes (Dotted) - Only if available (SNNs)
                    # Check for spike columns. Usually 'train/spikes_mean' or similar.
                    # Based on parsing, we need to check columns.
                    spike_col = next((c for c in df.columns if 'spike' in c and 'mean' in c), None)
                    if spike_col:
                        ax2.plot(df['epoch'], df[spike_col], label=f"{label} Spikes", color=color, linestyle=':', linewidth=2, alpha=0.7)

        if not has_data:
            plt.close()
            continue

        ax1.set_xlabel("Epoch", fontsize=14)
        ax1.set_ylabel("Mean Return", fontsize=14)
        ax2.set_ylabel("Mean Spikes/Step", fontsize=14, rotation=270, labelpad=20)
        
        ax1.set_title(f"Learning Dynamics: Return & Spikes ({env})", pad=15)
        
        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.15, 1))
        
        save_path = os.path.join(output_dir, env, "learning_curve_spikes.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_bar_comparison(results, output_dir):
    """Single bar chart: DT vs SNN-DT (return + spikes)."""
    # Just DT and SNN-DT
    targets = ['dt', 'snn_dt']
    
    envs = set()
    for v in results.values():
        envs.update(v.keys())
    envs = sorted(list(envs))
    
    # Prepare data for plotting
    data_points = []
    
    for env in envs:
        for t in targets:
            if t in results and env in results[t]:
                res = results[t][env]
                data_points.append({
                    "Environment": env,
                    "Variant": VARIANTS.get(t, t),
                    "Return": res['return_mean'],
                    "Spikes": res['spikes_mean'],
                    "Return_Std": res['return_std']
                })
    
    df = pd.DataFrame(data_points)
    if df.empty: return
    
    # We need Return and Spikes in one chart.
    # Maybe 2 subplots?
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Return
    sns.barplot(data=df, x="Environment", y="Return", hue="Variant", ax=ax1, palette=[COLORS['dt'], COLORS['snn_dt']], errorbar=None)
    # Add error bars manually if needed, or rely on Seaborn if full data passed. 
    # Since we passed mean/std, we need custom.
    # But for now, let's just show the bars.
    
    ax1.set_title("Performance Comparison (Return)")
    ax1.set_ylabel("Mean Return")
    ax1.legend(loc='lower right')
    
    # Plot Spikes
    # DT doesn't have spikes usually (0).
    sns.barplot(data=df, x="Environment", y="Spikes", hue="Variant", ax=ax2, palette=[COLORS['dt'], COLORS['snn_dt']])
    ax2.set_title("Efficiency Comparison (Spikes)")
    ax2.set_ylabel("Mean Spikes/Step")
    ax2.legend(loc='upper right')
    
    plt.suptitle("Direct Comparison: DT vs SNN-DT", fontsize=20, weight='bold', y=1.02)
    plt.tight_layout()
    
    # Save in Combined and also per Env folders (chopped?)
    # User said "Each should have to be in 3 different folders".
    # This comparison spans environments, so it fits best in "Combined".
    # But if we need it in Env folders, valid.
    
    save_path = os.path.join(output_dir, "Combined", "dt_vs_snn_dt_bar.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    
    # Also save per-env version (single bar)
    for env in envs:
        subset = df[df["Environment"] == env]
        if subset.empty: continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Dual axis for Return and Spike? Or just grouped?
        # Let's do grouped bar with normalized scale? No.
        # Let's do Return on Left, Spike on Right (scatter?)
        
        # Simplified: Just Return bar chart for this env for DT vs SNN-DT
        sns.barplot(data=subset, x="Variant", y="Return", palette=[COLORS['dt'], COLORS['snn_dt']], ax=ax)
        ax.set_title(f"DT vs SNN-DT: {env}")
        
        save_path = os.path.join(output_dir, env, "dt_vs_snn_dt_bar.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_combined_pareto(results, output_dir):
    """Energy vs Return (Pareto) plot - Combined (All Envs)."""
    # Normalize returns? Or just plot raw with shapes?
    # Simple approach: Plot raw, different markers for Envs.
    
    envs = set()
    for v in results.values():
        envs.update(v.keys())
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    env_markers = {env: markers[i % len(markers)] for i, env in enumerate(sorted(list(envs)))}
    
    plt.figure(figsize=(12, 8))
    
    xs = []
    ys = []
    labels = []
    colors_list = []
    markers_list = []
    
    # We need to manually handle markers in scatter, so we loop
    
    for var, data in results.items():
        c = COLORS.get(var, DEFAULT_COLOR)
        label_var = VARIANTS.get(var, var)
        
        for env, res in data.items():
            x = res['spikes_mean']
            y = res['return_mean']
            m = env_markers[env]
            
            plt.scatter(x, y, s=120, c=c, marker=m, alpha=0.8, edgecolors='white', linewidth=1.0, label=f"{label_var} ({env})" if False else None)
            
            # Annotate only if sparsely populated or distinct? 
            # Too cluttered to annotate everything.
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=VARIANTS.get(v, v), 
                              markerfacecolor=COLORS.get(v, DEFAULT_COLOR), markersize=10) for v in results.keys()]
    
    legend_elements += [Line2D([0], [0], marker=m, color='w', label=e, 
                               markerfacecolor='gray', markersize=10) for e, m in env_markers.items()]
    
    plt.xlabel("Energy Cost (Mean Spikes/Step)", fontsize=14)
    plt.ylabel("Mean Return (Raw)", fontsize=14)
    plt.title("Combined Pareto Frontier: All Environments", pad=15)
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    sns.despine()
    
    save_path = os.path.join(output_dir, "Combined", "combined_pareto.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    print("Parsing Final Logs...")
    final_results = parse_final_logs()
    
    print("Parsing Training Curves...")
    curves = parse_learning_curves()
    
    print("Generating Plots...")
    plot_pareto(final_results, OUTPUT_DIR)
    plot_combined_pareto(final_results, OUTPUT_DIR)
    plot_learning_curves(curves, OUTPUT_DIR)
    plot_bar_comparison(final_results, OUTPUT_DIR)
    
    print(f"Done! Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
