
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# --- Configuration ---
LOG_DIR = "ablation_studies/final_logs"
NEWBIE_OUTPUT_DIR = "ablation_studies/figures/plots_for_newbies"
Path(NEWBIE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Aesthetic Style - "Apple Health" / "Newbie Friendly"
# Clean, soft, approachable.
sns.set_theme(style="white", context="talk") # "talk" context makes everything bigger and clearer
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"], # System fonts usually look good
    "axes.titleweight": "bold",
    "axes.titlesize": 18,
    "axes.labelweight": "normal",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 14,
    "figure.dpi": 200, # Good resolution for screen
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False, # Minimalist
    "axes.spines.bottom": True,
    "grid.color": "#f0f0f0",
    "grid.linestyle": "-",
    "grid.linewidth": 1.0,
    "text.color": "#333333",
    "axes.labelcolor": "#555555",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
})

# Friendly Palette (Soft Pastels)
FRIENDLY_COLORS = {
    "snn_dt": "#4C9EEB",      # Friendly Blue
    "dsformer": "#E74C3C",    # Soft Red
    "dt": "#F1C40F",          # Soft Yellow
    "cql": "#2ECC71",         # Soft Green
    "iql": "#9B59B6",         # Soft Purple
    "full": "#3498DB",        # Blue
    "standard_ann": "#95A5A6", # Grey
}
DEFAULT_COLOR = "#95A5A6"

def parse_log_file(filepath):
    """
    Parses a single log file. Same logic as original script.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

    results = {}
    
    # Regex patterns (same as original)
    header_pattern = re.compile(r"[|│]\s*Ablation Group:\s*(\S+)\s*[|│]\s*(\S+)\s*[|│]")
    seed_pattern = re.compile(r"\[SEED \d+\] [^\s]+ Finished\s+Return: ([-+]?\d*\.?\d+|Not Found)(?:\s+Spikes: ([-+]?\d*\.?\d+))?")
    final_pattern = re.compile(r"Mean Return:\s*([-+]?\d*\.?\d+)\s*[+±]/-?\s*(\d*\.?\d+)")

    lines = content.splitlines()
    current_env = None
    current_group = None
    
    temp_seeds_spikes = []
    
    for line in lines:
        header_match = header_pattern.search(line)
        if header_match:
            current_group = header_match.group(1)
            current_env = header_match.group(2)
            temp_seeds_spikes = []
            continue
            
        seed_match = seed_pattern.search(line)
        if seed_match and current_env:
            spike_str = seed_match.group(2)
            if spike_str:
                temp_seeds_spikes.append(float(spike_str))
            continue
            
        final_match = final_pattern.search(line)
        if final_match and current_env:
            mean_val = float(final_match.group(1))
            std_val = float(final_match.group(2))
            
            spikes_mean = np.mean(temp_seeds_spikes) if temp_seeds_spikes else 0.0
            spikes_std = np.std(temp_seeds_spikes) if temp_seeds_spikes else 0.0
            
            if current_env not in results:
                results[current_env] = {}
            
            results[current_env] = {
                'mean': mean_val,
                'std': std_val,
                'spikes_mean': spikes_mean,
                'spikes_std': spikes_std,
                'group': current_group
            }
            current_env = None 
            
    return results

def setup_friendly_figure(title, subtitle=None):
    """Creates a figure with a friendly title and subtitle."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.suptitle(title, fontsize=20, fontweight='bold', x=0.125, ha='left')
    if subtitle:
        plt.title(subtitle, fontsize=12, loc='left', color='#666666', pad=15)
    return fig, ax

def add_explanatory_text(ax, text, xy, color="#444444"):
    """Adds a small explanatory note to the plot."""
    ax.annotate(text, xy=xy, xycoords='axes fraction', 
                xytext=(0, 0), textcoords='offset points',
                ha='left', va='top', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd", alpha=0.9),
                color=color)

def plot_single_friendly(filename, data):
    """
    Plots a friendly bar chart for a single variant.
    """
    if not data:
        return
        
    group_name = list(data.values())[0]['group']
    envs = list(data.keys())
    means = [data[e]['mean'] for e in envs]
    
    # Setup Figure
    fig, ax = setup_friendly_figure(
        title=f"Performance: {group_name.upper()}",
        subtitle=f"How well did the '{group_name}' AI play across different games?"
    )
    
    x = np.arange(len(envs))
    
    c_key = group_name.replace("-", "_")
    color = FRIENDLY_COLORS.get(c_key, DEFAULT_COLOR)
    
    bars = ax.bar(x, means, width=0.6, color=color, alpha=0.9, edgecolor='white', linewidth=1.5)
    
    ax.set_ylabel("Game Score")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", " ").title() for e in envs])
    
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)
    
    # Annotations
    max_ylim = ax.get_ylim()[1]
    ax.text(-0.05, max_ylim, "Better \u2191", ha='center', va='bottom', color='#2ecc71', fontsize=12, fontweight='bold')
    
    # Add value labels for better readability
    for bar in bars:
        height = bar.get_height()
        xy = (bar.get_x() + bar.get_width() / 2, height)
        ax.annotate(f'{height:.1f}', xy=xy, xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#555555')

    plt.tight_layout()
    # No extensive side text needed for single plots, keep it simple.
    
    save_path = os.path.join(NEWBIE_OUTPUT_DIR, f"single_{filename}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def plot_performance_friendly(all_data):
    """
    Plots 'Performance' (Returns) in a way that answers:
    'Which AI played the game better?'
    """
    # 1. Aggregate data: Variant -> Mean Return (averaged across envs for simplicity or grouped)
    # Let's do a grouped bar chart like before, but simpler.
    
    env_variant_map = {}
    variants = set()
    
    for fname, fdata in all_data.items():
        variant_name = fname.replace("_ablation_logs.txt", "")
        variants.add(variant_name)
        for env, metrics in fdata.items():
            if env not in env_variant_map: env_variant_map[env] = {}
            env_variant_map[env][variant_name] = metrics

    envs = sorted(list(env_variant_map.keys()))
    sorted_variants = sorted(list(variants))
    
    # Setup Figure
    fig, ax = setup_friendly_figure(
        title="AI Game Score Comparison",
        subtitle="Higher score means the AI played the game better."
    )
    
    # Plotting
    x = np.arange(len(envs))
    width = 0.8 / len(sorted_variants)
    
    for i, var in enumerate(sorted_variants):
        means = []
        for env in envs:
            means.append(env_variant_map[env].get(var, {'mean': 0})['mean'])
            
        c_key = var.replace("-", "_")
        color = FRIENDLY_COLORS.get(c_key, DEFAULT_COLOR)
        
        offset = (i - len(sorted_variants)/2) * width + width/2
        bars = ax.bar(x + offset, means, width, label=var.upper(), 
                color=color, edgecolor='white', linewidth=1, alpha=0.85, capsize=0)
        
        # Add values on top of bars if they are the 'winning' storage
        # (Skipping for clarity unless it's very clear)

    ax.set_ylabel("Game Score (Points)")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", " ").title() for e in envs])
    
    # Clean up axes
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)
    
    # Annotations
    max_ylim = ax.get_ylim()[1]
    ax.text(-0.05, max_ylim, "Better \u2191", ha='center', va='bottom', color='#2ecc71', fontsize=12, fontweight='bold')
    
    # Legend - Descriptive
    leg = ax.legend(title="AI Model Type", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    # Story / Why?
    why_text = (
        "WHY THIS MATTERS:\n"
        "We want to know if our new 'Brain-like' AI (SNN-DT)\n"
        "can play as well as the standard AI (DT).\n"
        "If the blue bars are close to the others, it's a success!"
    )
    plt.figtext(1.02, 0.4, why_text, fontsize=10, ha='left', va='top', color='#555555', style='italic', backgroundcolor='#f9f9f9', wrap=True)

    plt.tight_layout()
    # Adjust layout to fit text
    plt.subplots_adjust(right=0.75) 
    
    save_path = os.path.join(NEWBIE_OUTPUT_DIR, "combined_game_score_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def plot_efficiency_friendly(all_data):
    """
    Plots 'Efficiency' (Spikes) in a way that answers:
    'Which AI uses less brain power?'
    """
    env_variant_map = {}
    variants = set()
    
    for fname, fdata in all_data.items():
        variant_name = fname.replace("_ablation_logs.txt", "")
        variants.add(variant_name)
        for env, metrics in fdata.items():
            if env not in env_variant_map: env_variant_map[env] = {}
            env_variant_map[env][variant_name] = metrics

    envs = sorted(list(env_variant_map.keys()))
    sorted_variants = sorted(list(variants))
    
    # Setup Figure
    fig, ax = setup_friendly_figure(
        title="Energy Efficiency (Brain Activity)",
        subtitle="Lower bars are better. Like getting a car with better gas mileage."
    )
    
    x = np.arange(len(envs))
    width = 0.8 / len(sorted_variants)
    
    for i, var in enumerate(sorted_variants):
        means = []
        for env in envs:
            means.append(env_variant_map[env].get(var, {'spikes_mean': 0})['spikes_mean'])
            
        c_key = var.replace("-", "_")
        color = FRIENDLY_COLORS.get(c_key, DEFAULT_COLOR)
        
        offset = (i - len(sorted_variants)/2) * width + width/2
        ax.bar(x + offset, means, width, label=var.upper(), 
               color=color, edgecolor='white', linewidth=1, alpha=0.85)

    ax.set_ylabel("Spikes per Step (Brain Activity)")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", " ").title() for e in envs])
    
    # Clean up axes
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)
    
    # Annotations
    max_ylim = ax.get_ylim()[1]
    ax.text(-0.05, 0, "Better (More Efficient) \u2193", ha='center', va='bottom', color='#2ecc71', fontsize=12, fontweight='bold', rotation=90)
    
    # Add a "Standard AI" line at 0 (invisible really, but conceptual)
    # Actually, let's just annotate that Standard AI is 0.
    
    # Legend
    ax.legend(title="AI Model Type", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    # Story / Why?
    why_text = (
        "WHY THIS MATTERS:\n"
        "Standard AI uses a lot of continuous energy.\n"
        "Our 'Spiking' AI (SNN-DT) only uses energy when it 'spikes'.\n"
        "Fewer spikes = Much less power consumption.\n"
        "A standard AI would be at 0 here because it doesn't spike,\n"
        "but it consumes max power constantly (unlike these)."
    )
    plt.figtext(1.02, 0.4, why_text, fontsize=10, ha='left', va='top', color='#555555', style='italic', backgroundcolor='#f9f9f9')

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    save_path = os.path.join(NEWBIE_OUTPUT_DIR, "combined_energy_efficiency.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def main():
    print("Generating Newbie-Friendly Plots...")
    log_files = glob.glob(os.path.join(LOG_DIR, "*_logs.txt"))
    all_data = {}
    
    if not log_files:
        print("No log files found!")
        return

    for fpath in log_files:
        fname = os.path.basename(fpath)
        data = parse_log_file(fpath)
        if data:
            all_data[fname] = data
            # PLOT SINGLE FILE ("1 of 10")
            plot_single_friendly(fname.replace(".txt", ""), data)
            
    if all_data:
        # PLOT COMBINED ("1")
        plot_performance_friendly(all_data)
        plot_efficiency_friendly(all_data)
        print("All newbie plots generated successfully!")

if __name__ == "__main__":
    main()
