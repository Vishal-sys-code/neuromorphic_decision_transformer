import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('novel_phases/phase3/energy_latency_results.csv')

# Filter for baseline and full models
df_filtered = df[df['ablation_mode'].isin(['baseline', 'full'])]

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))

# Bar chart
bars = ax.bar(df_filtered['ablation_mode'], df_filtered['spikes_per_inference'], color=['skyblue', 'salmon'])

# Add labels and title
ax.set_ylabel('# Spikes per Inference')
ax.set_title('Spike Counts: Baseline vs. Full Model')
ax.set_xticks(np.arange(len(df_filtered['ablation_mode'])))
ax.set_xticklabels(['Baseline', 'Full'])

# Add text labels on bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom') # va: vertical alignment

# Save the figure
plt.savefig('spike_histogram.png', dpi=300)
plt.show()