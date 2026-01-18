# Spiking Decision Transformer (SNN-DT)

> **Abstract**: Reinforcement learning agents based on Transformer architectures have achieved impressive performance on sequential decision-making tasks, but their reliance on dense matrix operations makes them ill-suited for energy-constrained, edge-oriented platforms. We present the **Spiking Decision Transformer (SNN-DT)**, which embeds Leaky Integrate-and-Fire neurons into each self-attention block, trains end-to-end via surrogate gradients, and incorporates biologically inspired three-factor plasticity, phase-shifted spike-based positional encodings, and a lightweight dendritic routing module. Our implementation matches or exceeds standard Decision Transformer performance on classic control benchmarks (CartPole-v1, MountainCar-v0, Acrobot-v1, Pendulum-v1) while emitting fewer than ten spikes per decision—an energy proxy suggesting over four orders-of-magnitude reduction in per-inference energy.

```model
Spiking Decision Transformer
├── Core: Transformer Architecture
├── Spiking Dynamics: Leaky Integrate-and-Fire (LIF)
├── Plasticity: STDP-inspired 3-factor rule
└── Efficiency: Dendritic Routing & Event-Driven
```

## Documentation Contents

```{toctree}
:maxdepth: 2

methodology
experiments
architecture
usage
installation
```

## Citation

If you use this code in your research, please cite the associated paper:

```bibtex
@article{pandey2025snn,
  title={Spiking Decision Transformers: Local Plasticity, Phase-Coding, and Dendritic Routing for Low-Power Sequence Control},
  author={Pandey, Vishal and Biswas, Debasmita},
  journal={arXiv preprint arXiv:2508.21505},
  year={2025}
}
```
