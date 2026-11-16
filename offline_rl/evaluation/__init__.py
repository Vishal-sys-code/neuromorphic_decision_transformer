"""
Evaluation modules for MuJoCo environments.
Phase 6: MuJoCo Evaluation & Gate 6: Policy produces valid trajectories.
"""

from offline_rl.evaluation.evaluate_mujoco import (
    D4RL_REF_SCORES,
    EpisodeResult,
    TrajectoryRollout,
    compute_d4rl_normalized_score,
    evaluate_episode,
    evaluate_mujoco,
    evaluate_seed,
    load_trained_model,
    tune_target_returns,
    validate_gate6_policy,
)

__all__ = [
    "D4RL_REF_SCORES",
    "EpisodeResult",
    "TrajectoryRollout",
    "compute_d4rl_normalized_score",
    "evaluate_episode",
    "evaluate_mujoco",
    "evaluate_seed",
    "load_trained_model",
    "tune_target_returns",
    "validate_gate6_policy",
]
