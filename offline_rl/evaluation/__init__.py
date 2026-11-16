"""
Evaluation modules for MuJoCo environments.
Phase 6: MuJoCo Evaluation & Gate 6: Policy produces valid trajectories.
Phase 7: Compare with Decision Transformer and Decision SpikeFormer & Gate 7: Research comparison.
Phase 7.1: Ablation study.
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
from offline_rl.evaluation.research_comparison import (
    OFFICIAL_BENCHMARK_BASELINES,
    BenchmarkComparisonEntry,
    ResearchComparisonReport,
    build_efficiency_comparison_table,
    build_final_comparison_table,
    evaluate_snndt_benchmark,
    get_official_baselines,
    run_gate7_research_comparison,
)
from offline_rl.evaluation.run_ablations import (
    AblationMetrics,
    AblationStudyReport,
    load_ablation_dataset,
    run_full_ablation_study,
    run_single_ablation_experiment,
)

from offline_rl.evaluation.efficiency_metrics import (
    ComprehensiveEfficiencyReport,
    EnergyProxyEstimate,
    HARDWARE_ENERGY_DISCLAIMER,
    LatencyProfile,
    ModelParameterBreakdown,
    TheoreticalOpsProfile,
    compute_energy_proxy,
    compute_theoretical_ops,
    count_model_parameters,
    measure_inference_latency,
    profile_snndt_environment,
)
from offline_rl.evaluation.phase8_scaling import (
    PHASE8_SEQUENCE,
    Phase8EnvironmentResult,
    build_phase8_summary_table,
    run_full_phase8_scaling,
    run_phase8_single_environment,
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
    # Gate 7: Research Comparison
    "OFFICIAL_BENCHMARK_BASELINES",
    "BenchmarkComparisonEntry",
    "ResearchComparisonReport",
    "get_official_baselines",
    "evaluate_snndt_benchmark",
    "build_final_comparison_table",
    "build_efficiency_comparison_table",
    "run_gate7_research_comparison",
    # Phase 7.1: Ablations
    "AblationMetrics",
    "AblationStudyReport",
    "load_ablation_dataset",
    "run_single_ablation_experiment",
    "run_full_ablation_study",
    # Phase 8: Efficiency Metrics
    "ComprehensiveEfficiencyReport",
    "EnergyProxyEstimate",
    "HARDWARE_ENERGY_DISCLAIMER",
    "LatencyProfile",
    "ModelParameterBreakdown",
    "TheoreticalOpsProfile",
    "compute_energy_proxy",
    "compute_theoretical_ops",
    "count_model_parameters",
    "measure_inference_latency",
    "profile_snndt_environment",
    # Phase 8: Scaling Orchestrator
    "PHASE8_SEQUENCE",
    "Phase8EnvironmentResult",
    "run_phase8_single_environment",
    "run_full_phase8_scaling",
    "build_phase8_summary_table",
]

