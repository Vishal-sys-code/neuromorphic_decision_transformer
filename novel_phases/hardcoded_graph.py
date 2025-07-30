import matplotlib.pyplot as plt
import numpy as np

# Environments
environments = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"]

# Average returns (replace these with your real values)
baseline =     [452.3, -120.2, -87.1, -155.3]
pos_only =     [474.1, -111.5, -72.0, -140.0]
route_only =   [479.2, -109.8, -68.3, -135.4]
full =         [492.3, -102.4, -59.7, -130.5]

# Standard deviations (optional – can be replaced with real std values)
baseline_std = [11.7, 9.4, 3.2, 5.1]
pos_std =      [7.9, 7.2, 3.6, 4.7]
route_std =    [6.2, 6.9, 3.9, 4.4]
full_std =     [6.8, 5.5, 2.7, 4.2]

# Plotting setup
x = np.arange(len(environments))  # Label locations
width = 0.2  # Bar width

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
rects1 = ax.bar(x - 1.5*width, baseline, width, yerr=baseline_std, label='Baseline', capsize=5)
rects2 = ax.bar(x - 0.5*width, pos_only, width, yerr=pos_std, label='Pos-Only', capsize=5)
rects3 = ax.bar(x + 0.5*width, route_only, width, yerr=route_std, label='Route-Only', capsize=5)
rects4 = ax.bar(x + 1.5*width, full, width, yerr=full_std, label='Full', capsize=5, color='darkgreen')

# Axis config
ax.set_ylabel('Average Return')
ax.set_title('RL Performance across Environments')
ax.set_xticks(x)
ax.set_xticklabels(environments)
ax.legend(loc='lower right')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# Auto-label function (optional)
def autolabel(rects, xpos='center'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Annotate all bars
for group in [rects1, rects2, rects3, rects4]:
    autolabel(group)

# Save or Show
plt.tight_layout()
plt.savefig("rl_performance_plot.png", dpi=300)
plt.show()