# Methodology

The **Spiking Decision Transformer (SNN-DT)** bridges the gap between the high-level planning capabilities of Transformers and the energy efficiency of Spiking Neural Networks (SNNs).

## Architecture Overview

The SNN-DT architecture replaces the standard Multi-Head Attention (MHA) mechanism with a **Spiking Self-Attention (SSA)** block. Key components include:

1.  **Leaky Integrate-and-Fire (LIF) Neurons**:
    We employ non-leaky and leaky integrate-and-fire dynamics. The membrane potential $u_t$ evolves as:
    
    $$
    u_t = \beta u_{t-1} + W x_t
    $$
    
    $$
    s_t = \Theta(u_t - V_{th})
    $$
    
    where $\beta$ is the decay factor, $V_{th}$ is the threshold, and $s_t$ is the spike output.

2.  **Phase-Coding**:
    To preserve temporal information without high-precision floating point values, we utilize phase-shifted spike-based positional encodings.

3.  **Dendritic Routing**:
    A lightweight routing module that dynamically sparsifies the attention matrix, ensuring that only relevant tokens trigger synaptic operations.

## Local Plasticity

We incorporate biological three-factor plasticity rules to enable online adaptation:

$$
\Delta w_{ij} = \eta \cdot \text{Pre}_j \cdot \text{Post}_i \cdot \text{Modulator}
$$

This allows the network to adapt to distribution shifts in the environment without full backpropagation through time (BPTT).

## Training

The model is trained end-to-end using **Surrogate Gradients**. Since the spiking function $\Theta(\cdot)$ is non-differentiable, we approximate its derivative during the backward pass using a fast sigmoid function:

$$
\frac{\partial s}{\partial u} \approx \frac{1}{(1 + k|u|)^2}
$$

This enables us to optimize the decision control objective:

$$
\mathcal{L} = \sum_{t} (a_t - \hat{a}_t)^2
$$

where $\hat{a}_t$ is the predicted action and $a_t$ is the ground truth action from the expert demonstration.
