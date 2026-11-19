<div align="center">

# 🧠 Neuromorphic Spiking Decision Transformer (SNN-DT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.21505-b31b1b.svg)](https://arxiv.org/abs/2508.21505)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://neuromorphic-decision-transformer.vercel.app/)

**Precision-Preserving Spiking Control, Phase-Coding, and Dendritic Routing for Low-Power Sequence Decision Making**

</div>

---

## 🌟 Overview & Research Direction

This repository contains the PyTorch research framework for **Precision-Aware Spiking Decision Transformers (SNN-DT)** for offline reinforcement learning and continuous control.

### Chronological Context & Prior Art
Following the introduction of spiking transformers to offline decision making by **DecisionSpikeFormer** (*Huang et al., CVPR 2025* — introducing Temporal & Positional Spiking Self-Attention and Progressive Threshold-dependent Batch Normalization), this project investigates why spike-based policies encounter performance gaps in high-precision continuous control environments (e.g. D4RL Hopper, HalfCheetah, Walker2d, Ant).

### The Precision-Aware SNN-DT Hypothesis
Continuous control requires predicting real-valued action vectors (e.g., $a_t \in [-1, 1]^8$ in Ant). When pure event-driven representations discard continuous sub-threshold membrane information, action distributions can suffer from variance collapse and poor return-to-go (RTG) conditioning. 

To address this, SNN-DT develops:
1. **Precision-Preserving Membrane Residuals:** Retains normalized subthreshold residuals ($r_t = u_t - \theta s_t$) to construct hybrid control representations ($z_t = W_s s_t + W_r r_t$) that recover continuous action precision while preserving sparse event communication.
2. **Phase-Coded Positional Encoding:** Rhythmic, spike phase-shifted encodings that encode temporal recency in biological spike timing.
3. **Dendritic-Style Routing:** Context-dependent gating coefficients dynamically modulating attention head contributions.
4. **Decision-Conditioned Adaptive Computation:** Early-stopping temporal simulation steps when action representations converge ($\Delta_k < \epsilon$).
5. **Rigorous CMOS 45nm Energy Accounting:** Transparent separation between dense projection MACs ($\approx 339$ µJ/step in CMOS 45nm) and neuromorphic core synaptic spike events ($\approx 19$ nJ/step), replacing misleading single-operation energy comparisons.

<div align="center">
  <img src="model_architecture.png" alt="SNN-DT Architecture" width="850">
</div>

---

## ✨ Core Neuromorphic Innovations

- 🕒 **Phase-Coded Positional Encoding:** Replaces generic float embeddings with rhythmic, spike phase-shifted encodings.
- 🌳 **Dendritic-Style Routing MLP:** Context-dependent gating coefficients dynamically prune attention heads without uniform averaging.
- 🧬 **Three-Factor Local Plasticity:** Elegantly implements STDP-like localized credit assignment rules, circumventing catastrophic unrolling.

<div align="center">
  <img src="visualizations/routing_coefficients_heatmap.png" alt="Routing Heatmap" width="600">
  <p><em>Dynamic Dendritic Gating Coefficients</em></p>
</div>

---

## 🚀 Quickstart & Installation

**System Requirements:** Linux/Windows, Python 3.8+, CUDA-enabled GPU (Recommended).

```bash
# Clone the repository
git clone https://github.com/Vishal-sys-code/neuromorphic_decision_transformer.git
cd neuromorphic_decision_transformer

# Install core dependencies natively
pip install -r requirements.txt
```

> [!TIP]
> For deploying the documentation locally via Sphinx, execute `make html` inside the `/docs` directory!

---

## 💻 Experimental Workflows

### 1. Training the Architecture
Run the SNN-DT training pipeline, handling automated data orchestration and surrogate gradient optimization.

```bash
python snn-dt/scripts/train.py --model snn_dt --env "Pendulum-v1" --save-dir "results/snn_dt_pendulum"
```
Or execute the entire benchmarking suite across **CartPole-v1**, **MountainCar-v0**, and **Acrobot-v1**:
```bash
./run_all_experiments.sh
```

### 2. Neuromorphic Evaluation & Profiling
Evaluate checkpoint inference trajectories and monitor strict event-driven metrics (absolute spike outputs).
```bash
python eval_snn_dt.py \
    --env "Pendulum-v1" \
    --checkpoint_path "results/snn_dt_pendulum/best_model.pt" \
    --target_return -200
```
> [!NOTE]
> The runtime reports normalized return against the expert policy baseline alongside the average hardware energy proxies (Spikes / timestep).

---

## 📖 Citation

If you build upon this architecture or framework, please consider citing our underlying research:

```bibtex
@article{pandey2025spiking,
  title={Spiking Decision Transformers: Local Plasticity, Phase-Coding, and Dendritic Routing for Low-Power Sequence Control},
  author={Pandey, Vishal and Biswas, Debasmita},
  journal={arXiv preprint arXiv:2508.21505},
  year={2025}
}
```

<div align="center">
  <i>Maintained with ❤️ by the SNN-DT Research Team. Licensed under <a href="LICENSE">MIT</a>.</i>
</div>
