import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('energy_latency_results.csv')

modes = ['baseline', 'pos_only', 'router_only', 'full']
labels = ['Baseline', 'Pos-Only', 'Route-Only', 'Full']
x = np.arange(len(modes))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot spikes per inference on primary y-axis
bar1 = ax1.bar(x - width/2, df['spikes_per_inference'], width, label='# Spikes per Inference', color='royalblue')
ax1.set_ylabel('# Spikes per Inference', color='royalblue', fontsize=13)
ax1.tick_params(axis='y', labelcolor='royalblue')

# Plot CPU latency on secondary y-axis
ax2 = ax1.twinx()
bar2 = ax2.bar(x + width/2, df['cpu_latency_ms'], width, label='CPU Latency (ms)', color='tomato')
ax2.set_ylabel('CPU Latency (ms)', color='tomato', fontsize=13)
ax2.tick_params(axis='y', labelcolor='tomato')

ax1.set_xlabel('Ablation Mode', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_title('Energy Proxy (# Spikes) and CPU Latency per Inference\n(Lower is better)', fontsize=15)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
fig.tight_layout()
fig.legend([bar1, bar2], ['# Spikes per Inference', 'CPU Latency (ms)'], loc='upper right')
plt.savefig('energy_latency_vs_mode.png', dpi=200)
plt.show()
