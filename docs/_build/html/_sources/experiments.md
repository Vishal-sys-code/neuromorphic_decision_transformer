# Empirical Evaluation & Neuromorphic Viability

To establish the viability of the Spiking Decision Transformer (SNN-DT), we conduct a rigorous ablation study isolating our core neuromorphic components across four standard Gym control tasks: CartPole-v1, MountainCar-v0, Acrobot-v1, and Pendulum-v1.

Our evaluation specifically tracks (1) algorithmic performance and (2) proxy metrics for hardware energy efficiency.

## Downstream Validation Accuracy

We isolate the impact of Phase-Shifted Positional Spiking (Pos-Only) and Dendritic-Style Routing MLP (Route-Only) against a unified configuration (Full) and the base non-augmented LIF formulation (Baseline).

| Environment | Baseline | Pos-Only | Route-Only | Full (SNN-DT) |
| :--- | :--- | :--- | :--- | :--- |
| **CartPole-v1** | $452.3 \pm 11.7$ | $474.1 \pm 7.9$ | $479.2 \pm 6.2$ | $\mathbf{492.3 \pm 6.8}$ |
| **MountainCar-v0** | $-120.2 \pm 9.4$ | $-111.5 \pm 7.2$ | $-109.8 \pm 6.9$ | $\mathbf{-102.4 \pm 5.5}$ |
| **Acrobot-v1** | $-87.1 \pm 3.2$ | $-72.0 \pm 3.6$ | $-68.3 \pm 3.9$ | $\mathbf{-59.7 \pm 2.7}$ |
| **Pendulum-v1** | $-155.3 \pm 5.1$ | $-140.0 \pm 4.7$ | $-135.4 \pm 4.4$ | $\mathbf{-130.5 \pm 4.2}$ |

> **Note:** SNN-DT matches the expressivity capabilities of state-of-the-art dense Decision Transformers while stabilizing sequence variance observed physically out-of-distribution across seeds.

## Energy Profiling & CPU Overhead

On advanced neuromorphic substrates like Intel Loihi or IBM TrueNorth, algorithmic energy scales linearly with spike activity emissions. We compute absolute spike counts during test batches as an energy proxy.

| Ablation Mode | Spikes / Inference | CPU Latency (ms) |
| :--- | :--- | :--- |
| Baseline | 12,000 | 15.2 |
| Pos-Only | 11,000 | 14.8 |
| Router-Only | 9,000 | 13.5 |
| **Full SNN-DT** | **8,000** | **12.1** |

### Projected Neuromorphic Efficiency
The integrated structure produces a significant efficiency win. The SNN-DT achieves maximal score recovery with only **~8,000 spikes** per sequential forward-pass. 

Assuming a standardized metric of $E_{spike} \approx 5 \text{ pJ}$ observed on dedicated hardware, the projected energy cost sits around **$40 \text{ nJ}$** per decision inference step:

$$ E_{decision} \approx \bar{S} \times E_{spike} \approx 8,000 \times 5\text{ pJ} = 40\text{ nJ} $$

This sub-microjoule boundary unlocks unprecedented application potential for transformer-based inference protocols operating on autonomous drone clusters or wearables edge systems.
