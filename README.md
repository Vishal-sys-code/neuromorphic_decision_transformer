# SpikingMindRL: Energy‑Efficient Spiking Decision Transformers for Sequential Decision‑Making

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python >=3.8](https://img.shields.io/badge/python-%3E%3D3.8-yellow.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D1.10-red.svg)](https://pytorch.org/)  
[![Norse](https://img.shields.io/badge/norse-v0.5.0-blue.svg)](https://norse.github.io/)

**SpikingMindRL** is a first‑of‑its‑kind open‑source framework implementing **Spiking Neural Networks** (SNNs) in the style of **Decision Transformers** for offline and online reinforcement learning. By marrying the sparse, event‑driven dynamics of LIF neurons with the sequence modeling power of transformer architectures, SpikingMindRL demonstrates dramatic energy savings—on the order of **4–5 orders of magnitude**—while retaining competitive control performance.  

> “Bridging the gap between neuromorphic efficiency and modern sequential decision‑making.”  
> Vishal Pandey, Debasmita Biswas

---

## 🚀 Highlights

- **Spiking Self‑Attention**  
  Replace each multi‑head attention block with a spike‑driven, surrogate‑gradient‑trained module in Norse.  
- **Offline & Online Training**  
  – **Offline DT**: expert + random demonstrations via Decision Transformer loss  
  – **Online PG (REINFORCE)**: warm‑start from offline weights  
- **Multi‑Env Support**  
  CartPole‑v1, MountainCar‑v0, Acrobot‑v1, Pendulum‑v1, plus easy extension.  
- **Energy & Latency Benchmarks**  
  – **Avg spikes/forward:** ~10 spikes  
  – **CPU latency:** ~80 ms vs. ~10 ms (ANN)  
  – **Ablations:** time‑window & context‑length trade‑offs  
- **Extensible**  
  Fully modular PyTorch + Norse codebase with utilities for trajectory buffers, metrics, plotting, and demo collection.

---

## 📦 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/SpikingMindRL.git
   cd SpikingMindRL

2. **Create a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate         # Linux / macOS
    venv\Scripts\activate            # Windows
    ```
3. **Install requirements**
    ```bash
    pip install -r requirements.txt
    pip install norse torch torchvision gym[all] stable‑baselines3[extra]
    ```
4. **(Optional) Install Gym Box2D**
    ```bash
    pip install gym[box2d]
    ```

---

## 🛠️ Project Structure
```bash
.
├── LICENSE
├── README.md
├── requirements.txt
├── demos/                     # collected expert trajectories
│   └── expert_<env>.pkl
├── checkpoints/               # saved model weights
├── results/                   # CSVs & figures from experiments
├── scripts/
│   ├── collect_all_experts.py
│   ├── collect_sb3_expert.py
│   └── collect_cartpole_expert.py
└── src/
    ├── config.py              # hyperparameters & paths
    ├── setup_paths.py         # PYTHONPATH hacks
    ├── utils/                 # buffers, logging, helpers
    ├── models/
    │   ├── snn_lif.py         # LIF neuron with spike counting
    │   ├── spiking_layers.py  # spiking self‑attention
    │   └── snn_dt_patch.py    # SNN‑Decision Transformer wrapper
    ├── train_offline_dt.py    # offline DT training script
    ├── train_snn_dt.py        # online PG fine‑tuning
    ├── evaluate_and_plot.py   # multi‑env evaluation & plotting
    ├── energy_profile.py      # spikes & latency benchmarking
    └── ablation_studies.py    # hyperparameter ablations
```
---

## 📚 Quickstart

1. **Collect Expert Demonstrations**
    ```bash
    python scripts/collect_all_experts.py
    ```
2. **Train Offline Decision Transformer**
    ```bash
    python -m src.train_offline_dt --env CartPole-v1
    ```
3. **Evaluate & Plot**
    ```bash
    python -m src.evaluate_and_plot
    ```
4. **Benchmark SNN vs. ANN**
    ```bash
    python -m src.benchmark_snn_vs_ann
    ```
5. **Run Ablation Studies**
    ```bash
    python -m src.ablation_studies
    ```
---

## 📈 Key Results

| Environment    | Offline Return | Avg Spikes/Forward | Latency (ms) | ANN Latency (ms) |
|----------------|----------------|--------------------|--------------|------------------|
| CartPole-v1    | 200 ± 0        | 9.6                | 87.80        | 9.61             |
| MountainCar-v0 | -135 ± 10      | 26.0               | 184.79       | 12.30            |
| Acrobot-v1     | -80 ± 5        | 23.9               | 95.60        | 11.10            |
| Pendulum-v1    | -200 ± 15      | 27.2               | 485.03       | 14.50            |

---

# 📝 Citation

```bash
@article{pandey2025spikingmindrl,
  title   = {Spiking Neural Networks for Sequential Decision‑Making Inspired by Transformer‑RL Frameworks},
  author  = {},
  journal = {},
  year    = {2025}
}
```