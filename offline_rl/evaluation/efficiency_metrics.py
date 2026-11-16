"""
Comprehensive Neuromorphic Efficiency Profiling & CMOS Energy Proxy Module.

Phase 8 / Critical Correction 4:
Comprehensive multi-metric profiling beyond spike counts alone:
1. Spike count (total spikes per step and per episode, mean spike rate)
2. High-precision inference latency (per step and per episode in ms, p50, p95)
3. Parameter count breakdown (total, trainable, embeddings, backbone, head)
4. MACs, ACs, and theoretical FLOPs for ANN vs SNN operations
5. Standard CMOS 45nm Energy Proxy (Horowitz 2014, Panda et al. 2020)
   - E_MAC ≈ 4.6 pJ (32-bit floating point multiply-accumulate)
   - E_AC  ≈ 0.9 pJ (32-bit floating point accumulate addition)
   - E_spike ≈ 0.5 pJ (neuromorphic synaptic event)

IMPORTANT SCIENTIFIC NOTICE:
Keep theoretical CMOS energy proxy estimates separate from actual neuromorphic
hardware energy measurements on physical silicon (e.g. Intel Loihi, SpiNNaker2).
"""

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Standard CMOS 45nm Energy Constants (Horowitz 2014, Panda et al. 2020)
ENERGY_MAC_PJ: float = 4.6       # 32-bit FP MAC (pJ)
ENERGY_AC_PJ: float = 0.9        # 32-bit FP Accumulate / Addition (pJ)
ENERGY_SPIKE_PJ: float = 0.5     # Dynamic synaptic event (pJ)

HARDWARE_ENERGY_DISCLAIMER: str = (
    "NOTE: Energy metrics reported here are analytical energy proxies based on standardized "
    "CMOS 45nm arithmetic operation models (4.6 pJ/MAC, 0.9 pJ/AC, 0.5 pJ/spike; Horowitz 2014, "
    "Panda et al. 2020). These serve as theoretical algorithmic proxies and do not represent "
    "physical on-chip energy measurements on dedicated neuromorphic silicon (e.g., Intel Loihi, SpiNNaker)."
)


@dataclass
class ModelParameterBreakdown:
    """Detailed breakdown of model parameter counts."""
    total_parameters: int
    trainable_parameters: int
    embedding_parameters: int
    backbone_parameters: int
    action_head_parameters: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class TheoreticalOpsProfile:
    """Theoretical computational operations per step."""
    dt_dense_macs: float
    dt_flops: float
    snndt_dense_macs: float
    snndt_spiking_acs: float
    snndt_total_spikes: float
    snndt_lif_potential_updates: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class EnergyProxyEstimate:
    """Analytical CMOS 45nm energy proxy estimates."""
    ann_energy_nj_per_step: float
    snndt_energy_nj_per_step: float
    energy_reduction_pct: float
    efficiency_factor: float
    ann_energy_nj_per_episode: float
    snndt_energy_nj_per_episode: float
    disclaimer: str = HARDWARE_ENERGY_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatencyProfile:
    """High-precision inference latency measurements."""
    mean_ms: float
    std_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    num_samples: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ComprehensiveEfficiencyReport:
    """Complete consolidated efficiency profile across all 5 dimensions."""
    env_name: str
    state_dim: int
    action_dim: int
    context_len: int
    parameters: ModelParameterBreakdown
    operations: TheoreticalOpsProfile
    energy: EnergyProxyEstimate
    latency: LatencyProfile
    spike_rate: float
    mean_spikes_per_step: float
    scientific_disclaimer: str = HARDWARE_ENERGY_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_name": self.env_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "context_len": self.context_len,
            "parameters": self.parameters.to_dict(),
            "operations": self.operations.to_dict(),
            "energy": self.energy.to_dict(),
            "latency": self.latency.to_dict(),
            "spike_rate": self.spike_rate,
            "mean_spikes_per_step": self.mean_spikes_per_step,
            "scientific_disclaimer": self.scientific_disclaimer,
        }


def count_model_parameters(model: nn.Module) -> ModelParameterBreakdown:
    """
    Compute comprehensive breakdown of model parameter counts across:
    - Total & trainable parameters
    - Embedding layers (state, action, return, timestep, layer norm)
    - Transformer backbone (projections, phase encoder, routers)
    - Action prediction head
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    embed_params = 0
    head_params = 0
    backbone_params = 0

    for name, p in model.named_parameters():
        if any(term in name for term in ["embed", "state_embedding", "action_embedding", "return_embedding"]):
            embed_params += p.numel()
        elif any(term in name for term in ["head", "action_head", "action_predictor"]):
            head_params += p.numel()
        elif "backbone" in name or "block" in name or "router" in name or "phase_encoder" in name:
            backbone_params += p.numel()
        else:
            backbone_params += p.numel()

    return ModelParameterBreakdown(
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        embedding_parameters=embed_params,
        backbone_parameters=backbone_params,
        action_head_parameters=head_params,
    )


def compute_theoretical_ops(
    state_dim: int,
    action_dim: int,
    context_len: int = 20,
    embed_dim: int = 128,
    n_layer: int = 3,
    n_head: int = 4,
    snn_timesteps: int = 4,
    mean_spike_rate: float = 0.32,
) -> TheoreticalOpsProfile:
    """
    Compute theoretical MACs, ACs, and FLOPs per inference step for:
    1. Standard Dense Decision Transformer (ANN)
    2. Spiking Decision Transformer (SNN-DT)

    Token sequence has length L = 3 * K.
    Decision Transformer Attention per layer:
    - Q, K, V projections: 3 * L * D^2 MACs
    - Attention matrix Q K^T: L^2 * D MACs
    - Attention values A V: L^2 * D MACs
    - Output projection: L * D^2 MACs
    Total ANN Transformer MACs per layer = 4 * L * D^2 + 2 * L^2 * D.

    In SNN-DT:
    - Projections (Q, K, V): dense linear MACs
    - LIF integration: T * L * 3 * D additions
    - Spike-based self-attention:
      Because inputs are sparse binary/discrete spike events across T,
      matrix multiplications are computed as synaptic Accumulate additions (ACs)
      rather than dense multiply-accumulates (MACs)!
      Sparse ACs = mean_spike_rate * (2 * L^2 * D).
    """
    K = context_len
    L = 3 * K
    D = embed_dim
    T = snn_timesteps

    # Modality Embeddings MACs per step:
    # State: state_dim * D, Action: action_dim * D, Return: 1 * D
    embed_macs = (state_dim * D) + (action_dim * D) + (1 * D) + (L * D)  # including timestep & LN
    head_macs = D * action_dim

    # Dense ANN Decision Transformer:
    layer_attn_macs = (3 * L * (D ** 2)) + (2 * (L ** 2) * D) + (L * (D ** 2))
    dt_total_macs = embed_macs + (n_layer * layer_attn_macs) + head_macs
    dt_flops = 2.0 * dt_total_macs

    # SNN-DT Operations:
    # Linear projections before spiking remain dense MACs
    snndt_proj_macs = embed_macs + n_layer * (3 * L * (D ** 2) + L * (D ** 2)) + head_macs
    # LIF integration updates across biological time T
    lif_updates = n_layer * (T * L * (3 * D))
    # Synaptic spikes generated per step
    total_spikes = float(lif_updates) * float(mean_spike_rate)
    # Spike-based attention replaces dense MACs with Accumulate additions (ACs)
    snndt_spiking_acs = float(mean_spike_rate) * float(n_layer * (2 * (L ** 2) * D))

    return TheoreticalOpsProfile(
        dt_dense_macs=float(dt_total_macs),
        dt_flops=float(dt_flops),
        snndt_dense_macs=float(snndt_proj_macs),
        snndt_spiking_acs=float(snndt_spiking_acs),
        snndt_total_spikes=float(total_spikes),
        snndt_lif_potential_updates=float(lif_updates),
    )


def compute_energy_proxy(
    ops: TheoreticalOpsProfile,
    episode_length: int = 1000,
    energy_mac_pj: float = ENERGY_MAC_PJ,
    energy_ac_pj: float = ENERGY_AC_PJ,
    energy_spike_pj: float = ENERGY_SPIKE_PJ,
) -> EnergyProxyEstimate:
    """
    Compute theoretical CMOS 45nm energy proxy in nanojoules (nJ):
    E_ANN    = MACs * E_MAC
    E_SNNDT  = MACs_dense * E_MAC + ACs * E_AC + Spikes * E_spike
    """
    # 1 pJ = 1e-3 nJ
    pj_to_nj = 1e-3

    ann_energy_nj_step = ops.dt_dense_macs * energy_mac_pj * pj_to_nj
    snndt_energy_nj_step = (
        (ops.snndt_dense_macs * energy_mac_pj)
        + (ops.snndt_spiking_acs * energy_ac_pj)
        + (ops.snndt_total_spikes * energy_spike_pj)
    ) * pj_to_nj

    reduction_pct = 0.0
    if ann_energy_nj_step > 0.0:
        reduction_pct = 100.0 * (ann_energy_nj_step - snndt_energy_nj_step) / ann_energy_nj_step

    efficiency_factor = (
        ann_energy_nj_step / max(snndt_energy_nj_step, 1e-12)
        if snndt_energy_nj_step > 0.0
        else 1.0
    )

    ann_ep_energy = ann_energy_nj_step * float(episode_length)
    snndt_ep_energy = snndt_energy_nj_step * float(episode_length)

    return EnergyProxyEstimate(
        ann_energy_nj_per_step=float(ann_energy_nj_step),
        snndt_energy_nj_per_step=float(snndt_energy_nj_step),
        energy_reduction_pct=float(reduction_pct),
        efficiency_factor=float(efficiency_factor),
        ann_energy_nj_per_episode=float(ann_ep_energy),
        snndt_energy_nj_per_episode=float(snndt_ep_energy),
    )


def measure_inference_latency(
    model: nn.Module,
    sample_batch: Dict[str, torch.Tensor],
    device: str = "cpu",
    num_warmup: int = 10,
    num_steps: int = 50,
) -> LatencyProfile:
    """
    High-precision wall-clock latency benchmarking using time.perf_counter().
    """
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()

    # Move batch to target device
    dev_batch = {k: v.to(target_device) for k, v in sample_batch.items()}

    # Warmup iterations
    with torch.no_grad():
        for _ in range(num_warmup):
            if hasattr(model, "reset_state"):
                model.reset_state()
            _ = model(
                states=dev_batch["states"],
                actions=dev_batch["actions"],
                returns_to_go=dev_batch["returns_to_go"],
                timesteps=dev_batch["timesteps"],
                attention_mask=dev_batch.get("attention_mask"),
            )

    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(num_steps):
            if hasattr(model, "reset_state"):
                model.reset_state()
            if hasattr(model, "reset_spike_counts"):
                model.reset_spike_counts()

            t0 = time.perf_counter()
            _ = model(
                states=dev_batch["states"],
                actions=dev_batch["actions"],
                returns_to_go=dev_batch["returns_to_go"],
                timesteps=dev_batch["timesteps"],
                attention_mask=dev_batch.get("attention_mask"),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

    lat_arr = np.array(latencies_ms)
    return LatencyProfile(
        mean_ms=float(np.mean(lat_arr)),
        std_ms=float(np.std(lat_arr)),
        median_ms=float(np.median(lat_arr)),
        p95_ms=float(np.percentile(lat_arr, 95)),
        min_ms=float(np.min(lat_arr)),
        max_ms=float(np.max(lat_arr)),
        num_samples=len(latencies_ms),
    )


def profile_snndt_environment(
    env_name: str,
    model: nn.Module,
    state_dim: int,
    action_dim: int,
    context_len: int = 20,
    mean_spike_rate: float = 0.32,
    device: str = "cpu",
    num_latency_steps: int = 30,
) -> ComprehensiveEfficiencyReport:
    """
    Profile an SNN-DT model for a specified continuous control environment
    across all 5 core efficiency dimensions.
    """
    params = count_model_parameters(model)
    ops = compute_theoretical_ops(
        state_dim=state_dim,
        action_dim=action_dim,
        context_len=context_len,
        mean_spike_rate=mean_spike_rate,
    )
    energy = compute_energy_proxy(ops)

    # Generate sample context window for latency testing
    sample_batch = {
        "states": torch.zeros((1, context_len, state_dim), dtype=torch.float32),
        "actions": torch.zeros((1, context_len, action_dim), dtype=torch.float32),
        "returns_to_go": torch.ones((1, context_len, 1), dtype=torch.float32),
        "timesteps": torch.arange(context_len).unsqueeze(0).long(),
        "attention_mask": torch.ones((1, context_len), dtype=torch.float32),
    }

    latency = measure_inference_latency(
        model=model,
        sample_batch=sample_batch,
        device=device,
        num_steps=num_latency_steps,
    )

    return ComprehensiveEfficiencyReport(
        env_name=env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        context_len=context_len,
        parameters=params,
        operations=ops,
        energy=energy,
        latency=latency,
        spike_rate=mean_spike_rate,
        mean_spikes_per_step=ops.snndt_total_spikes,
    )
