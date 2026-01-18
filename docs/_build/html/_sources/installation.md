# Installation

The code is built on PyTorch and requires a CUDA-enabled GPU for efficient training, although CPU execution is supported.

## Prerequisites

-   OS: Linux (Ubuntu 20.04+) or Windows 10/11 (WSL2 recommended)
-   Python: 3.8+
-   CUDA: 11.8+ (for GPU acceleration)

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Vishal-sys-code/neuromorphic_decision_transformer.git
    cd neuromorphic_decision_transformer
    ```

2.  **Create an environment**:
    ```bash
    conda create -n snn-dt python=3.9
    conda activate snn-dt
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## D4RL Datasets

For Mujoco environments, you will need to install the D4RL library and download the datasets.

```bash
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

> **Note**: For `CartPole`, `MountainCar`, etc., the scripts will automatically generate expert trajectories if datasets are not found.