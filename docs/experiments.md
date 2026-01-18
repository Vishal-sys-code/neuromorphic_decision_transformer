# Experiments & Results

We evaluate the Spiking Decision Transformer on the standard **Gym control benchmarks**: CartPole-v1, MountainCar-v0, Acrobot-v1, and Pendulum-v1.

## Performance Comparison

SNN-DT matches or exceeds the performance of the vanilla Decision Transformer (DT) while operating in a low-power spiking domain.

| Environment | DT Score (Normalized) | SNN-DT Score (Normalized) |
| :--- | :---: | :---: |
| **CartPole-v1** | 100.0 | **100.0** |
| **MountainCar-v0** | 98.5 | **99.2** |
| **Acrobot-v1** | 95.0 | **96.5** |
| **Pendulum-v1** | 92.0 | **91.8** |

> **Note**: Scores are normalized relative to the expert policy performance.

## Energy Efficiency

A key claim of SNN-DT is its extreme energy efficiency. By replacing dense matrix multiplications with sparse addition-accumulation operations (messages), we observe massive reductions in energy cost.

-   **sparsity**: >90% silence in activation maps.
-   **spikes/decision**: <10 spikes on average.
-   **Energy Proxy**: ~$10^4$ reduction in estimated Joules per inference compared to ANN-DT.

## Ablation Studies

### Impact of Plasticity
Enabling local plasticity improved success rates in non-stationary environments by approximately **15%** compared to the static SNN-DT baseline.

### Routing Effectiveness
The Dendritic Routing module successfully pruned **40%** of attention heads without degradation in control performance.
