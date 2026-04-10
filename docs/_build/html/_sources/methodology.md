# Training Methodology & Dynamics

The convergence of biologically-plausible Spiking Neural Networks (SNNs) and offline reinforcement learning presents unique optimization challenges. The SNN-DT achieves end-to-end learnability via Surrogate Gradients combined with structured dataset preparation.

## 1. Offline Dataset Curation

Following the foundation of the Decision Transformer framework, we define a trajectory sequence parameterized by step returns, states, and action constraints:
$\tau = \{(s_t, a_t, r_t)\}^T_{t=1}$.

The temporal context is unrolled into autoregressive sequence intervals padded as required. Specifically, a return-to-go scalar $G_t = \sum_{k=t}^{T} r_k$ fundamentally binds the sequence optimization trajectory, explicitly directing the modeled distribution towards highest-return outcomes by enforcing sequence prediction across $(G_1, s_1, a_1, \dots, G_N, s_N, a_N)$.

## 2. Leaky Integrate-and-Fire (LIF) Discretization

The SNN-DT relies physically on the simulation of membrane charges. In discrete time frames dictated by integration parameter $\Delta t$, a forward-Euler step gives the membrane potential:

$$ V[t+1] = V[t] + \frac{\Delta t}{\tau_m} (V_{\text{rest}} - V[t]) + \Delta t C_m I[t] $$

When $V[t+1] \geq V_{\text{th}}$, the neuron immediately broadcasts an absolute binary signal $S_{t+1}=1$ and the internal potential is hard-reset towards $V_{\text{rest}}$.

## 3. Surrogate Gradient Learning

Due to the intrinsic, mathematically discontinuous constraint of the Heaviside step $S = \{V \geq V_{\text{th}} \}$, standard computational graphs normally suffer vanishing gradients immediately. 

SNN-DT utilizes a **Fast-Sigmoid Surrogate Gradient** to bypass the non-differentiable threshold. During the backward pass, we replace the Heaviside step's derivative over the potential offset $u = V - V_{\text{th}}$:

$$ \tilde{\sigma}'(u) = \frac{1}{(1 + |k \cdot u|)^2} $$

With $k=10$, this enables a defined gradient flow backward through the dense attention channels all the way into the phase-shifted positional sine representations, supporting full end-to-end regression to target distribution values. 

