# Phase 2 Ablation Study: Summary of Findings

## 1. Introduction

This report summarizes the key findings from the Phase 2 ablation study of the SNN-DT model. The study aimed to investigate the individual contributions of three key components: the phase encoder, the dendritic router, and the local plasticity mechanism.

## 2. Experimental Setup

The experiments were conducted on three environments: CartPole-v1, Acrobot-v1, and Pendulum-v1. Four model variants were tested:

*   **full**: The complete SNN-DT model with all components enabled.
*   **no_phase**: The model with the phase encoder disabled.
*   **no_routing**: The model with the dendritic router disabled.
*   **no_plasticity**: The model with the local plasticity mechanism disabled.

Each experiment was run with three different random seeds (1001, 1002, 1003) to ensure the robustness of the results.

## 3. Results

### 3.1. Learning Curves

*(Insert and discuss the learning curve plots here.)*

### 3.2. Final Performance

*(Insert and discuss the summary table of final performance metrics here.)*

### 3.3. Energy vs. Performance

*(Insert and discuss the Pareto plots here.)*

## 4. Discussion

*(Provide a detailed analysis of the results, interpreting the findings and discussing their implications.)*

## 5. Conclusion

*(Summarize the main conclusions of the study and suggest directions for future work.)*