"""
Plot comparison results between SNN-DT and DSF-DT
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def plot_comparison_results():
    """Plot comparison results from CSV file"""
    # Read results
    if not os.path.exists("comparison_results.csv"):
        print("comparison_results.csv not found. Run comparison first.")
        return
    
    if not os.path.exists("training_losses.csv"):
        print("training_losses.csv not found. Run comparison first.")
        return
    
    results_df = pd.read_csv("comparison_results.csv")
    losses_df = pd.read_csv("training_losses.csv")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SNN-DT vs DSF-DT Comparison on CartPole-v1', fontsize=16)
    
    # 1. Return Comparison
    ax1 = axes[0, 0]
    models = results_df['model']
    returns = results_df['avg_return']
    std_returns = results_df['std_return']
    
    bars = ax1.bar(models, returns, yerr=std_returns, capsize=5, alpha=0.7, color=['blue', 'orange'])
    ax1.set_ylabel('Average Return')
    ax1.set_title('Average Return Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, ret, std in zip(bars, returns, std_returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{ret:.1f}±{std:.1f}', ha='center', va='bottom')
    
    # 2. Latency Comparison
    ax2 = axes[0, 1]
    latencies = results_df['avg_latency_ms']
    bars = ax2.bar(models, latencies, alpha=0.7, color=['blue', 'orange'])
    ax2.set_ylabel('Average Latency (ms)')
    ax2.set_title('Inference Latency Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, lat in zip(bars, latencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{lat:.2f} ms', ha='center', va='bottom')
    
    # 3. Spikes Comparison
    ax3 = axes[1, 0]
    spikes = results_df['avg_spikes_per_episode']
    bars = ax3.bar(models, spikes, alpha=0.7, color=['blue', 'orange'])
    ax3.set_ylabel('Average Spikes per Episode')
    ax3.set_title('Spiking Activity Comparison')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, spike in zip(bars, spikes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{spike:,.0f}', ha='center', va='bottom')
    
    # 4. Training Loss Comparison
    ax4 = axes[1, 1]
    epochs = losses_df['epoch']
    snn_losses = losses_df['snn_loss']
    dsf_losses = losses_df['dsf_loss']
    
    ax4.plot(epochs, snn_losses, label='SNN-DT', marker='o', linewidth=2)
    ax4.plot(epochs, dsf_losses, label='DSF-DT', marker='s', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Training Loss')
    ax4.set_title('Training Loss Comparison')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plots saved to comparison_results.png")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for _, row in results_df.iterrows():
        print(f"\n{row['model']}:")
        print(f"  Return: {row['avg_return']:.2f} ± {row['std_return']:.2f}")
        print(f"  Latency: {row['avg_latency_ms']:.2f} ms")
        print(f"  Spikes: {row['avg_spikes_per_episode']:,.0f}")
        print(f"  Parameters: {row['total_params']:,}")

def main():
    plot_comparison_results()

if __name__ == "__main__":
    main()