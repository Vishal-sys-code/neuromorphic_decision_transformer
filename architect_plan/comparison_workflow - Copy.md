# Comparison Workflow

## Overview
This diagram shows the workflow for comparing SNN-DT and DSF-DT on CartPole-v1.

```mermaid
flowchart TD
    A[Start] --> B[Environment Setup]
    B --> C[Data Collection]
    C --> D[DSF-DT Training]
    C --> E[SNN-DT Training]
    D --> F[Evaluation]
    E --> F
    F --> G[Results Analysis]
    G --> H[Comparison Report]
    H --> I[End]

    subgraph Data Collection
        C1[Collect 5000 steps<br/>random policy data]
        C2[Save to shared dataset]
        C1 --> C2
    end

    subgraph DSF-DT Training
        D1[Load shared dataset]
        D2[Train for 10 epochs]
        D3[Save checkpoints]
        D1 --> D2 --> D3
    end

    subgraph SNN-DT Training
        E1[Load shared dataset]
        E2[Train for 10 epochs]
        E3[Save checkpoints]
        E1 --> E2 --> E3
    end

    subgraph Evaluation
        F1[Load best checkpoints]
        F2[Evaluate on CartPole-v1<br/>10 episodes]
        F3[Collect metrics:<br/>- Returns<br/>- Latency<br/>- Spikes<br/>- Parameters]
        F1 --> F2 --> F3
    end

    subgraph Results Analysis
        G1[Compare performance]
        G2[Compare efficiency]
        G3[Compare spiking behavior]
        G1 --> G2 --> G3
    end

    subgraph Comparison Report
        H1[Generate comparison table]
        H2[Create visualizations]
        H3[Write analysis]
        H1 --> H2 --> H3
    end