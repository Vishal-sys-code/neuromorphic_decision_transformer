# Target Environment

We design SNN-DT targeting research-layer reproducibility integrated natively via standard ML toolings (PyTorch). 

## Dependencies

```bash
# Core Machine Learning Frameworks
pip install torch torchvision torchaudio

# Neuromorphic / Spiking Simulators
pip install norse

# Environmental Support 
pip install gym mujoco-py
```

## Recommended Setup (Conda Platform)

```bash
conda create -n snn-dt python=3.10
conda activate snn-dt
pip install -r requirements.txt
```