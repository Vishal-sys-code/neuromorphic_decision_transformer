"""
Neuromorphic Decision Transformer models for continuous MuJoCo control.
"""

from offline_rl.models.snn_dt_mujoco import (
    ContinuousSnnDecisionTransformer,
    DendriticRouter,
    ENV_DIM_MAP,
    PhaseSpikeEncoder,
    SNNDTMuJoCo,
    SnnDtBackbone,
    SpikingTransformerBlock,
    SurrogateSpike,
    ThreeFactorPlasticity,
    VectorizedLIF,
    create_ablation_model,
    get_env_dims,
    surrogate_spike,
)

__all__ = [
    "SNNDTMuJoCo",
    "ContinuousSnnDecisionTransformer",
    "create_ablation_model",
    "get_env_dims",
    "ENV_DIM_MAP",
    "SnnDtBackbone",
    "VectorizedLIF",
    "PhaseSpikeEncoder",
    "DendriticRouter",
    "ThreeFactorPlasticity",
    "SpikingTransformerBlock",
    "SurrogateSpike",
    "surrogate_spike",
]
