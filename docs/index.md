# Spiking Decision Transformer (SNN-DT)

> **Abstract:** Reinforcement learning agents based on Transformer architectures have achieved impressive performance on sequential decision-making tasks, but their reliance on dense matrix operations makes them ill-suited for energy-constrained, edge-oriented platforms. Spiking neural networks promise ultra-low-power, event-driven inference, yet no prior work has seamlessly merged spiking dynamics with return-conditioned sequence modeling. We present the **Spiking Decision Transformer (SNN-DT)**, which embeds Leaky Integrate-and-Fire neurons into each self-attention block, trains end-to-end via surrogate gradients, and incorporates biologically inspired three-factor plasticity, phase-shifted spike-based positional encodings, and a lightweight dendritic routing module. 
>
> Our implementation matches or exceeds standard Decision Transformer performance on classic control benchmarks (CartPole-v1, MountainCar-v0, Acrobot-v1, Pendulum-v1) while emitting fewer than ten spikes per decision—an energy proxy suggesting over four orders-of-magnitude reduction in per-inference energy. By marrying sequence modeling with neuromorphic efficiency, SNN-DT opens a pathway toward real-time, low-power control on embedded and wearable devices.

```model
Spiking Decision Transformer Architecture
├── Spiking Dynamics: Leaky Integrate-and-Fire (LIF)
├── Core Module 1: Three-Factor Local Plasticity
├── Core Module 2: Phase-Shifted Positional Spiking
└── Core Module 3: Dendritic-Style Routing MLP
```

## Documentation Contents

```{toctree}
:maxdepth: 2

architecture
methodology
experiments
usage
installation
```

## Citation

If you utilize this framework or build upon the SNN-DT codebase in your research, please cite the associated paper:

```bibtex
@article{pandey2025snndt,
  title={Spiking Decision Transformers: Local Plasticity, Phase-Coding, and Dendritic Routing for Low-Power Sequence Control},
  author={Pandey, Vishal and Biswas, Debasmita},
  journal={arXiv preprint arXiv:2508.21505v1},
  year={2025}
}
```
