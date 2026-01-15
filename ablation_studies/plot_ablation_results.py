
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# --- Configuration ---
LOG_DIR = "ablation_studies/final_logs"
Performance_OUTPUT_DIR = "ablation_studies/figures/aesthetic_plots"
Research_OUTPUT_DIR = "ablation_studies/figures/research_plots"
Path(Performance_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(Research_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Aesthetic Style (DeepMind/NeurIPS inspired)
# Using seaborn whitegrid with custom font sizes and colors
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "#333333",
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

# Custom Palette (DeepMind-ish colors)
DIM_COLORS = {
    "snn_dt": "#0057E7",      # Bright Blue
    "dsformer": "#D23F31",    # Google Red
    "dt": "#F4B400",          # Google Yellow
    "cql": "#0F9D58",         # Google Green
    "iql": "#AB47BC",         # Purple
    "full": "#4285F4",        # Blue
    "no_phase": "#A0C4FF",    # Light Blue
    "no_routing": "#FFD6A5",  # Light Orange
    "no_plasticity": "#FFADAD", # Light Red
}
DEFAULT_COLOR = "#5f6368"

def parse_log_file(filepath):
    """
    Parses a single log file to extract results for each environment.
    Returns a dict: {env_name: {'mean': float, 'std': float, 'seeds': [float, ...], 'spikes_mean': float, 'spikes_std': float}}
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    results = {}
    
    # Regex for header: | Ablation Group: name | EnvName |
    # Supports ASCII '|' and Unicode '│'
    header_pattern = re.compile(r"[|│]\s*Ablation Group:\s*(\S+)\s*[|│]\s*(\S+)\s*[|│]")
    
    # Regex for seeds
    # Supports '+' and '✓' (and technically any other char there)
    # Also captures optional "Spikes: 0.16" part
    seed_pattern = re.compile(r"\[SEED \d+\] [^\s]+ Finished\s+Return: ([-+]?\d*\.?\d+|Not Found)(?:\s+Spikes: ([-+]?\d*\.?\d+))?")
    
    # Regex for final result
    final_pattern = re.compile(r"Mean Return:\s*([-+]?\d*\.?\d+)\s*[+±]/-?\s*(\d*\.?\d+)")

    lines = content.splitlines()
    current_env = None
    current_group = None
    
    temp_seeds_return = []
    temp_seeds_spikes = []
    
    for line in lines:
        header_match = header_pattern.search(line)
        if header_match:
            current_group = header_match.group(1)
            current_env = header_match.group(2)
            temp_seeds_return = []
            temp_seeds_spikes = []
            continue
            
        seed_match = seed_pattern.search(line)
        if seed_match and current_env:
            val_str = seed_match.group(1)
            
            # Parsing Spikes if present
            spike_str = seed_match.group(2)
            if spike_str:
                temp_seeds_spikes.append(float(spike_str))
            
            if val_str == "Not Found":
                continue 
            else:
                temp_seeds_return.append(float(val_str))
            continue
            
        final_match = final_pattern.search(line)
        if final_match and current_env:
            mean_val = float(final_match.group(1))
            std_val = float(final_match.group(2))
            
            # Calculate spike stats from seeds
            spikes_mean = np.mean(temp_seeds_spikes) if temp_seeds_spikes else 0.0
            spikes_std = np.std(temp_seeds_spikes) if temp_seeds_spikes else 0.0
            
            if current_env not in results:
                results[current_env] = {}
            
            results[current_env] = {
                'mean': mean_val,
                'std': std_val,
                'seeds': temp_seeds_return,
                'spikes_mean': spikes_mean,
                'spikes_std': spikes_std,
                'group': current_group
            }
            current_env = None # Reset
            
    return results

def plot_single_file(filename, data):
    """Generates a bar plot for a single file."""
    
    envs = list(data.keys())
    means = [data[e]['mean'] for e in envs]
    stds = [data[e]['std'] for e in envs]
    group_name = list(data.values())[0]['group']
    
    plt.figure(figsize=(8, 5))
    
    color = DIM_COLORS.get(group_name, DEFAULT_COLOR)
    
    bars = plt.bar(envs, means, yerr=stds, capsize=5, color=color, alpha=0.9, width=0.6, edgecolor='black', linewidth=1.0)
    
    plt.axhline(0, color='black', linewidth=0.8) 
    plt.ylabel("Mean Return")
    plt.title(f"Performance: {group_name.upper()}")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        xy = (bar.get_x() + bar.get_width() / 2, height)
        offset = 5 if height >= 0 else -15
        plt.annotate(f'{height:.1f}', xy=xy, xytext=(0, offset),
                     textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(Performance_OUTPUT_DIR, f"{group_name}_performance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def plot_combined_summary(all_data):
    """Plots a grouped bar chart for RETURNS."""
    # Reorganize data: Env -> Variant -> (Mean, Std)
    env_variant_map = {}
    variants = set()
    
    for fname, fdata in all_data.items():
        variant_name = fname.replace("_ablation_logs.txt", "")
        variants.add(variant_name)
        
        for env, metrics in fdata.items():
            if env not in env_variant_map:
                env_variant_map[env] = {}
            env_variant_map[env][variant_name] = metrics

    envs = sorted(list(env_variant_map.keys()))
    sorted_variants = sorted(list(variants))
    
    # Create grouped bar plot
    n_envs = len(envs)
    n_vars = len(sorted_variants)
    
    x = np.arange(n_envs)
    width = 0.8 / n_vars
    
    plt.figure(figsize=(14, 7))
    
    for i, var in enumerate(sorted_variants):
        means = []
        stds = []
        for env in envs:
            if var in env_variant_map[env]:
                means.append(env_variant_map[env][var]['mean'])
                stds.append(env_variant_map[env][var]['std'])
            else:
                means.append(0)
                stds.append(0)
        
        c_key = var.replace("-", "_")
        color = DIM_COLORS.get(c_key, DEFAULT_COLOR)
        
        offset = (i - n_vars/2) * width + width/2
        plt.bar(x + offset, means, width, yerr=stds, label=var.upper(), 
                color=color, capsize=3, edgecolor='white', linewidth=0.5)

    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("Mean Return")
    plt.title("Ablation Study: Performance Comp.")
    plt.xticks(x, envs)
    plt.legend(title="Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_path = os.path.join(Performance_OUTPUT_DIR, "combined_ablation_summary.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def plot_spikes_comparison(all_data):
    """Plots a grouped bar chart for SPIKES (Research Plot)."""
    # Simply reusing logic but for 'spikes_mean'
    env_variant_map = {}
    variants = set()
    
    for fname, fdata in all_data.items():
        variant_name = fname.replace("_ablation_logs.txt", "")
        variants.add(variant_name)
        for env, metrics in fdata.items():
            if env not in env_variant_map:
                env_variant_map[env] = {}
            env_variant_map[env][variant_name] = metrics

    envs = sorted(list(env_variant_map.keys()))
    sorted_variants = sorted(list(variants))
    
    x = np.arange(n_envs := len(envs))
    width = 0.8 / (n_vars := len(sorted_variants))
    
    plt.figure(figsize=(14, 7))
    
    for i, var in enumerate(sorted_variants):
        means = []
        stds = []
        for env in envs:
            if var in env_variant_map[env]:
                means.append(env_variant_map[env][var]['spikes_mean'])
                stds.append(env_variant_map[env][var]['spikes_std'])
            else:
                means.append(0)
                stds.append(0)
        
        c_key = var.replace("-", "_")
        color = DIM_COLORS.get(c_key, DEFAULT_COLOR)
        
        offset = (i - n_vars/2) * width + width/2
        plt.bar(x + offset, means, width, yerr=stds, label=var.upper(), 
                color=color, capsize=3, edgecolor='white', linewidth=0.5)

    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("Mean Spikes per Step")
    plt.title("Research Plot: Spike Sparsity (Efficiency)")
    plt.xticks(x, envs)
    plt.legend(title="Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add annotation explaining that 0 means conventional ANN
    plt.figtext(0.99, 0.01, "* 0.0 indicates non-spiking model (Standard ANN)", horizontalalignment='right', fontsize=10, style='italic')

    plt.tight_layout()
    save_path = os.path.join(Research_OUTPUT_DIR, "combined_spike_efficiency.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def main():
    log_files = glob.glob(os.path.join(LOG_DIR, "*_logs.txt"))
    all_data = {}
    
    if not log_files:
        print("No log files found!")
        return

    print(f"Found {len(log_files)} log files.")

    for fpath in log_files:
        fname = os.path.basename(fpath)
        data = parse_log_file(fpath)
        if data:
            all_data[fname] = data
            plot_single_file(fname.replace(".txt", ""), data)
        else:
            print(f"Warning: No data parsed from {fname}")

    # Generate combined plot
    if all_data:
        plot_combined_summary(all_data)
        plot_spikes_comparison(all_data)

if __name__ == "__main__":
    main()
