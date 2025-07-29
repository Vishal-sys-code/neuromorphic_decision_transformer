import pandas as pd
import matplotlib.pyplot as plt

# Load validation loss log
df = pd.read_csv('validation_losses.csv')

# Define color mapping for each mode
color_map = {
    'baseline': 'blue',
    'pos_only': 'orange',
    'router_only': 'green',
    'full': 'red'
}

plt.figure(figsize=(8, 5))
for mode, color in color_map.items():
    mode_df = df[df['ablation_mode'] == mode]
    plt.plot(mode_df['epoch'], mode_df['validation_loss'], label=mode.replace('_', ' ').title(), color=color, linewidth=2)

plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Validation Loss', fontsize=13)
plt.title('Validation Loss vs. Epoch for Ablation Modes', fontsize=15)
plt.legend(title='Ablation Mode')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('validation_loss_vs_epoch.png', dpi=200)
plt.show()
