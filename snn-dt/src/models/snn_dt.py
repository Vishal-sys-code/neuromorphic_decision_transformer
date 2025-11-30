# src/models/snn_dt.py
"""
SNN-DT: Spiking Decision Transformer (vectorized, biological time-aware).

Key features:
- Biological time axis T (configurable)
- Vectorized LIF neuron (fast, batched)
- Simple surrogate gradient (custom autograd)
- PhaseSpikeEncoder -> modulates currents across time
- Per-head DendriticRouter with learned gating alpha_h
- Three-factor plasticity skeleton (eligibility traces + weight update)
- Diagnostics: spike counts, spike rates, attn diagnostics
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Surrogate spike function
# -------------------------
class SurrogateSpike(torch.autograd.Function):
    """
    Forward: hard threshold -> spikes (0/1).
    Backward: surrogate gradient (fast sigmoid derivative).
    """
    @staticmethod
    def forward(ctx, membrane_potential, v_th, alpha):
        ctx.save_for_backward(membrane_potential, v_th)
        ctx.alpha = float(alpha)
        out = (membrane_potential >= v_th).to(membrane_potential.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        v, v_th = ctx.saved_tensors
        alpha = ctx.alpha
        # surrogate gradient: derivative of sigmoid(alpha*(v - v_th))
        sigma = torch.sigmoid(alpha * (v - v_th))
        grad_approx = alpha * sigma * (1.0 - sigma)
        return grad_output * grad_approx, None, None


surrogate_spike = SurrogateSpike.apply


# -------------------------
# Vectorized LIF module
# -------------------------
class VectorizedLIF(nn.Module):
    """
    Vectorized LIF that processes an input current tensor across biological time T
    in a fully vectorized manner.

    Currents shape: (T, B, L, N)  where:
      - T = biological timesteps
      - B = batch
      - L = sequence length (tokens)
      - N = neuron dimension (d_model or head_dim)

    Returns:
      spikes_time: (T, B, L, N)  (float 0/1)
      final_state: membrane potential state for continuation (B, L, N)
    """

    def __init__(self, tau: float = 20.0, v_th: float = 0.5, alpha: float = 25.0, dt: float = 1.0):
        super().__init__()
        self.tau = float(tau)
        self.v_th = float(v_th)
        self.alpha = float(alpha)  # surrogate hardness
        self.dt = float(dt)
        # alpha_decay used in Euler update: v = rho * v + current
        # rho = exp(-dt / tau)
        self.register_buffer("rho", torch.tensor(math.exp(-self.dt / max(self.tau, 1e-6))))

    def forward(self, currents: torch.Tensor, v0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # currents: (T, B, L, N)
        T, B, L, N = currents.shape
        device = currents.device
        if v0 is None:
            v = torch.zeros((B, L, N), dtype=currents.dtype, device=device)
        else:
            v = v0

        spikes_time = torch.zeros_like(currents, dtype=currents.dtype)

        # Vectorized loop across T implemented in PyTorch operations
        # We iterate over time axis but the operations are efficient; T is usually small (8-32)
        # If T is large, we can unroll with scan-like custom kernels.
        for t in range(T):
            I_t = currents[t]  # (B, L, N)
            # membrane update: v = rho * v + I_t
            v = self.rho * v + I_t
            # spike generation (surrogate)
            s_t = surrogate_spike(v, torch.tensor(self.v_th, device=device, dtype=v.dtype), self.alpha)
            # reset (soft reset)
            v = v * (1.0 - s_t)
            spikes_time[t] = s_t
        return spikes_time, v


# -------------------------
# PhaseSpikeEncoder
# -------------------------
class PhaseSpikeEncoder(nn.Module):
    """
    Produces a modulation tensor across biological time that modulates the per-token
    projections into time-varying currents.

    Given timesteps (B, L) and d_model, it creates modulation factors shape (T, B, L, 1)
    which are later multiplied with projections.

    Implementation: learnable omegas and phis per feature-dimension block, then
    produce sinusoids across T. We compress the modulation to a scalar per neuron
    (so modulation shape ends with 1) for efficiency, but can be extended to full D.
    """

    def __init__(self, d_model: int, T: int = 8):
        super().__init__()
        self.d_model = d_model
        self.T = int(T)
        # We'll learn a set of frequencies and phases per d_model // 4 groups for capacity
        # but output final scalar modulation per neuron via a small MLP.
        hidden = max(16, d_model // 8)
        self.freqs = nn.Parameter(torch.randn(hidden))  # frequencies
        self.phases = nn.Parameter(torch.randn(hidden))
        # small projection from token embedding -> modulation weights for the hidden sinusoid bank
        self.proj_in = nn.Linear(d_model, hidden)
        self.proj_out = nn.Linear(hidden, 1)  # produce scalar modulation per neuron

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        """
        token_emb: (B, L, d_model)  -- usually the projected state/action/return embedding.
        returns: modulation factors (T, B, L, 1) with values ~ [0.0, 2.0] (we'll offset to >0)
        """
        B, L, D = token_emb.shape
        device = token_emb.device

        # produce base coefficients per token
        h = torch.tanh(self.proj_in(token_emb))  # (B, L, hidden)
        # compute sinusoid bank across T: (T, hidden)
        t = torch.arange(self.T, device=device).float().unsqueeze(1)  # (T, 1)
        freqs = self.freqs.unsqueeze(0)  # (1, hidden)
        phases = self.phases.unsqueeze(0)  # (1, hidden)
        sin_bank = torch.sin(t * freqs + phases)  # (T, hidden)

        # combine: modulation_raw (T, B, L, hidden) = sin_bank (T, hidden) * h (B,L,hidden)
        # we compute outer product efficiently:
        # reshape h -> (1, B, L, hidden)
        h_expand = h.unsqueeze(0)  # (1,B,L,hidden)
        sin_bank_expand = sin_bank.unsqueeze(1).unsqueeze(1)  # (T,1,1,hidden)
        mod_hidden = sin_bank_expand * h_expand  # (T,B,L,hidden)

        # project to scalar modulation and make positive
        mod_scalar = self.proj_out(mod_hidden)  # (T, B, L, 1)
        # shift and scale so modulation is mostly positive and centered ~1
        mod_scalar = 1.0 + 0.5 * torch.tanh(mod_scalar)
        return mod_scalar  # (T, B, L, 1)


# -------------------------
# Dendritic Router (per-head)
# -------------------------
class DendriticRouter(nn.Module):
    """
    Per-head dendritic routing module.
    Accepts y_heads: (B, L, H, head_dim) and returns:
      - y_routed: (B, L, head_dim)  (weighted sum across heads)
      - alpha: (B, L, H) gating coefficients per head (softmax over heads)
    """

    def __init__(self, head_dim: int, n_heads: int, hidden: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        # small network: compute per-head logit from head features; produce softmax over heads
        self.mlp = nn.Sequential(
            nn.Linear(head_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, y_heads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # y_heads: (B, L, H, D_h)
        B, L, H, D_h = y_heads.shape
        assert H == self.n_heads and D_h == self.head_dim
        # compute logits per head
        # flatten (B*L*H, D_h)
        x = y_heads.reshape(-1, D_h)
        logits = self.mlp(x).reshape(B, L, H)  # (B, L, H)
        alpha = torch.softmax(logits, dim=-1)  # (B, L, H)
        # weighted sum: sum_h alpha_h * y_h -> broadcast alpha to head_dim
        alpha_exp = alpha.unsqueeze(-1)  # (B, L, H, 1)
        y_weighted = (alpha_exp * y_heads).sum(dim=2)  # (B, L, D_h)
        return y_weighted, alpha  # routed output, gating coeffs


# -------------------------
# Three-factor plasticity (skeleton)
# -------------------------
class ThreeFactorPlasticity:
    """
    Minimal three-factor plasticity class:
      - maintains eligibility trace E (same shape as a weight matrix)
      - update rule: E <- lambda_decay * E + (pre^T @ post) / (batch*time)
      - apply rule: dW <- eta * reward * E
    This is not a full biologically-plausible implementation but is sufficient
    for ablation / experiments and matches the requested interface.
    """

    def __init__(self, weight_shape: Tuple[int, int], eta: float = 1e-3, lambda_decay: float = 0.99, device: Optional[torch.device] = None):
        self.device = device
        self.eta = float(eta)
        self.lambda_decay = float(lambda_decay)
        self.E = torch.zeros(weight_shape, device=device)

    def update_trace(self, pre: torch.Tensor, post: torch.Tensor):
        """
        pre: (B*T, in_dim)
        post: (B*T, out_dim)
        update E += (pre^T @ post) / (B*T)
        """
        with torch.no_grad():
            BT = max(1, pre.shape[0])
            # (in_dim, BT) @ (BT, out_dim) => (in_dim, out_dim)
            corr = pre.transpose(0, 1) @ post  # (in_dim, out_dim)
            corr = corr / float(BT)
            self.E = self.lambda_decay * self.E + corr

    def apply(self, weight_param: nn.Parameter, reward: float):
        """
        Apply plasticity: weight += eta * reward * E^T (match dims)
        If weight shape is (out_dim, in_dim), E is (in_dim, out_dim) so transpose.
        """
        with torch.no_grad():
            if weight_param.shape == (self.E.shape[1], self.E.shape[0]):
                # W is (out, in), E is (in, out)
                delta = self.eta * float(reward) * self.E.transpose(0, 1)
                weight_param.add_(delta)
            else:
                # Try broadcasting fallback
                delta = self.eta * float(reward) * self.E.transpose(0, 1)
                if delta.shape == weight_param.shape:
                    weight_param.add_(delta)
                else:
                    # shapes mismatch -> no-op (user must ensure shapes align)
                    pass


# -------------------------
# Spiking Transformer Block
# -------------------------
class SpikingTransformerBlock(nn.Module):
    """
    Full per-layer block:
    - Linear projections q/k/v (d_model -> d_model)
    - Phase modulation + vectorized LIF per projection across T
    - Reshape spikes to heads and compute attention using spike-rate (sum across T)
    - Per-head DendriticRouter
    - Output projection back to d_model
    - Spike counting accumulated per-block
    """

    def __init__(self, cfg, d_model: int, n_heads: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        current_scale = float(getattr(self.cfg.snn, "current_scale", 1.0))
        # add a small learnable gain so we can ramp input current easily
        self.register_parameter("input_gain", torch.nn.Parameter(torch.tensor(float(current_scale), dtype=torch.float32)))

        import torch.nn.init as init

        for lin in (self.q_proj, self.k_proj, self.v_proj, getattr(self, 'out_proj', None)):
            if lin is None: continue
            init.xavier_uniform_(lin.weight, gain=1.0)
            if lin.bias is not None:
                lin.bias.data.zero_()

        # Vectorized LIF for full d_model
        lif_tau = float(getattr(cfg.snn, "lif_tau", 20.0))
        v_th = float(getattr(cfg.snn, "v_th", 0.5))
        surrogate_k = float(getattr(cfg.snn, "surrogate_k", 25.0))
        self.lif = VectorizedLIF(tau=lif_tau, v_th=v_th, alpha=surrogate_k)

        # Phase encoder (we will create one globally in SnnDt and pass modulation into block)
        # Dendritic router works on head-dim
        self.router = DendriticRouter(head_dim=self.head_dim, n_heads=n_heads)
        self.out_proj = nn.Linear(self.head_dim, d_model)

        # Plasticity parameters
        self.use_plasticity = bool(getattr(cfg.snn, "use_plasticity", False))
        self.plasticity_rule = None
        if self.use_plasticity:
            # eligibility trace shape for v_proj weight: (out, in) = (d_model, d_model)
            self.plasticity_rule = ThreeFactorPlasticity(weight_shape=(d_model, d_model),
                                                         eta=getattr(cfg.snn, "eta_local", 1e-4),
                                                         lambda_decay=getattr(cfg.snn, "lambda_decay", 0.99),
                                                         device=None)

        # diagnostics
        self.spike_count = 0.0
        self.last_alpha = None
        self.last_attn_scores = None
        self.diagnostics = {}

    def forward(self, x: torch.Tensor, phase_mod: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, L, d_model)
        phase_mod: (T, B, L, 1)  -- modulation produced by PhaseSpikeEncoder (or others)
        returns:
          out: (B, L, d_model)
          alpha: (B, L, n_heads)  gating from dendritic router
        """
        B, L, D = x.shape
        T = phase_mod.shape[0]
        device = x.device

        # Project once (token-level), then create time-varying currents:
        q = self.q_proj(x)  # (B, L, D)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        self.diagnostics['q_proj_min'] = q.min().item()
        self.diagnostics['q_proj_max'] = q.max().item()
        self.diagnostics['q_proj_mean'] = q.mean().item()

        # currents shape: (T, B, L, D)
        # scale current by cfg.snn.current_scale if present
        q_curr = q.unsqueeze(0) * phase_mod * self.input_gain
        k_curr = k.unsqueeze(0) * phase_mod * self.input_gain
        v_curr = v.unsqueeze(0) * phase_mod * self.input_gain

        # run LIF (vectorized across time)
        spikes_q_t, q_state = self.lif(q_curr)  # (T, B, L, D)
        spikes_k_t, _ = self.lif(k_curr)
        spikes_v_t, _ = self.lif(v_curr)

        if isinstance(q_state, torch.Tensor):
            self.diagnostics['v_mem_q_min'] = q_state.min().item()
            self.diagnostics['v_mem_q_max'] = q_state.max().item()

        # accumulate spike counts for diagnostics
        s_sum = spikes_q_t.sum() + spikes_k_t.sum() + spikes_v_t.sum()
        self.spike_count += float(s_sum.detach())

        # spike-rate per token: sum over T => (B, L, D)
        spikes_q = spikes_q_t.sum(dim=0)
        spikes_k = spikes_k_t.sum(dim=0)
        spikes_v = spikes_v_t.sum(dim=0)

        # diagnostics (simple)
        self.diagnostics['q_min'] = q.min().item()
        self.diagnostics['q_max'] = q.max().item()
        self.diagnostics['spike_rate_q'] = spikes_q_t.detach().mean().item()
        self.diagnostics['spike_rate_v'] = spikes_v_t.detach().mean().item()

        # reshape to heads: (B, L, H, head_dim)
        q_h = spikes_q.view(B, L, self.n_heads, self.head_dim)
        k_h = spikes_k.view(B, L, self.n_heads, self.head_dim)
        v_h = spikes_v.view(B, L, self.n_heads, self.head_dim)

        # attention scores: (B, H, L, L)
        # We compute per-head attention
        # q_h: (B, L, H, Dh) -> permute to (B, H, L, Dh)
        q_perm = q_h.permute(0, 2, 1, 3)
        k_perm = k_h.permute(0, 2, 1, 3)
        v_perm = v_h.permute(0, 2, 1, 3)

        # compute attn scores via matmul:
        # (B, H, L, Dh) @ (B, H, Dh, L) -> (B, H, L, L)
        k_trans = k_perm.transpose(-1, -2)  # (B,H, Dh, L)
        attn_scores = torch.matmul(q_perm, k_trans) / math.sqrt(self.head_dim)  # (B,H,L,L)

        self.last_attn_scores = attn_scores.detach()
        # convert to (B, L, H, L) by permuting so we can mask per token if needed -> router expects (B,L,H,D)
        # We will compute attention weights now (softmax over last dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B,H,L,L)

        # compute weighted sum of values: (B,H,L,L) @ (B,H,L,Dh) => (B,H,L,Dh)
        y_heads = torch.matmul(attn_weights, v_perm)  # (B,H,L,Dh)
        # permute to (B,L,H,Dh) for router
        y_heads = y_heads.permute(0, 2, 1, 3)

        # Dendritic routing (per-head gating)
        y_routed, alpha = self.router(y_heads)  # (B, L, Dh), (B,L,H)
        self.last_alpha = alpha.detach()

        # project back to d_model
        out = self.out_proj(y_routed)  # (B, L, D)

        # plasticity: update eligibility using flattened pre/post if configured
        if self.use_plasticity and self.plasticity_rule is not None:
            # pre = input to v_proj (x) flattened by (B*L, D)
            pre = x.reshape(-1, D)
            post = spikes_v.reshape(-1, D)
            self.plasticity_rule.update_trace(pre, post)

        return out, alpha


# -------------------------
# Full SnnDt model
# -------------------------
class SnnDt(nn.Module):
    """
    SnnDt model that stacks SpikingTransformerBlocks with:
     - PhaseSpikeEncoder to generate modulation across biological time T
     - LayerNorm + residuals around blocks (as requested, norm AFTER block to avoid suppressing spiking)
     - Spike counting and diagnostics collection
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(cfg.model.d_model)
        self.n_heads = int(cfg.model.n_heads)
        self.n_layers = int(cfg.model.n_layers)
        self.seq_len = int(cfg.model.seq_len)
        # snn params
        self.T = int(getattr(cfg.snn, "biological_time", 8))
        self.v_th = float(getattr(cfg.snn, "v_th", 0.5))
        self.current_scale = float(getattr(cfg.snn, "current_scale", 1.0))

        # embeddings
        self.embed_state = nn.Linear(cfg.dataset.state_dim, self.d_model)
        if cfg.dataset.is_discrete:
            self.embed_action = nn.Embedding(cfg.dataset.act_dim, self.d_model)
        else:
            self.embed_action = nn.Linear(cfg.dataset.act_dim, self.d_model)
        self.embed_return = nn.Linear(1, self.d_model)

        # Phase encoder (one global instance to produce modulation for all projections)
        self.phase_encoder = PhaseSpikeEncoder(d_model=self.d_model, T=self.T)

        # Blocks
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(cfg=cfg, d_model=self.d_model, n_heads=self.n_heads)
            for _ in range(self.n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layers)])
        self.action_predictor = nn.Linear(self.d_model, cfg.dataset.act_dim)

        # plasticity config (apply after eval checks)
        self.use_plasticity = bool(getattr(cfg.snn, "use_plasticity", False))

        # statefulness / diagnostics
        self.total_spike_count = 0.0
        self.total_spike_opportunities = 0.0
        self.last_diagnostics = {}

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch contains:
          - states: (B, L, state_dim)
          - actions: (B, L, act_dim)
          - returns_to_go: (B, L, 1)
          - timesteps: (B, L) integer timesteps (positions)
        """
        states = batch["states"]
        actions = batch["actions"]
        returns = batch["returns_to_go"]
        timesteps = batch["timesteps"]

        B, L, _ = states.shape
        device = states.device

        # Pad actions if they are shorter than states
        if actions.shape[1] < states.shape[1]:
            padding_needed = states.shape[1] - actions.shape[1]
            if actions.dim() == 3:
                actions = torch.nn.functional.pad(
                    actions, (0, 0, 0, padding_needed), "constant", 0
                )
            else: # (B, L)
                actions = torch.nn.functional.pad(
                    actions, (0, padding_needed), "constant", 0
                )

        # basic embeddings
        s_emb = self.embed_state(states)  # (B,L,D)
        if self.cfg.dataset.is_discrete:
            # Squeeze and convert to long for embedding lookup
            a_emb = self.embed_action(actions.squeeze(-1).long())
        else:
            a_emb = self.embed_action(actions)
        r_emb = self.embed_return(returns)

        # phase modulation factors T x B x L x 1
        # the phase encoder uses token embedding to compute per-token modulation,
        # we feed combined rate embedding (sum of s/a/r) to provide variety
        rate_emb = (s_emb + a_emb + r_emb)  # (B,L,D)
        phase_mod = self.phase_encoder(rate_emb)  # (T,B,L,1)

        # The phase modulation needs to match the new sequence length of 3*L
        # We repeat the modulation for each of the (R, S, A) tokens.
        phase_mod = phase_mod.repeat(1, 1, 3, 1)

        # combine tokens into the typical transformer stacking (R, S, A)
        stacked = torch.stack((r_emb, s_emb, a_emb), dim=1)  # (B, 3, L, D)
        stacked = stacked.permute(0, 2, 1, 3).contiguous()    # (B, L, 3, D)
        stacked = stacked.view(B, 3 * L, self.d_model)        # (B, 3L, D)
        x = stacked

        # reset block states and diagnostics
        for b in self.blocks:
            if hasattr(b, "lif"):
                # zero state handled internally if none provided; clear spike_count for each forward pass
                b.spike_count = 0.0
                b.last_alpha = None
                b.last_attn_scores = None
                b.diagnostics = {}

        self.total_spike_count = 0.0
        self.total_spike_opportunities = 0.0

        # run each block
        for i, block in enumerate(self.blocks):
            out, alpha = block(x, phase_mod)
            # residual + norm (norm after block as requested)
            x = x + out
            x = self.layer_norms[i](x)

            # diagnostics aggregation
            self.total_spike_count += block.spike_count
            # total opportunities for normalization: B * seq * d_model * 3 (Q,K,V) * 1 (per block)
            # We accumulate across blocks for global normalization
            self.total_spike_opportunities += float(B * x.shape[1] * (3 * self.d_model))
            self.last_diagnostics[f"block_{i}_spike_rate_q"] = block.diagnostics.get("spike_rate_q", 0.0)
            self.last_diagnostics[f"block_{i}_spike_rate_v"] = block.diagnostics.get("spike_rate_v", 0.0)
            self.last_diagnostics[f"block_{i}_q_min"] = block.diagnostics.get("q_min", 0.0)
            self.last_diagnostics[f"block_{i}_alpha_mean"] = alpha.mean().item() if alpha is not None else 0.0

        # final action prediction using tokens corresponding to actions:
        action_preds = self.action_predictor(x[:, 1::3])  # select action positions (works with 3*L layout)
        # finalize diagnostics
        self.last_diagnostics["spikes_total_raw"] = float(self.total_spike_count)
        self.last_diagnostics["spikes_norm"] = self.count_spikes()
        # keep last attn max
        self.last_diagnostics["max_attn"] = max(
            (block.last_attn_scores.max().item() if block.last_attn_scores is not None else 0.0)
            for block in self.blocks
        )
        return action_preds

    def count_spikes(self) -> float:
        if self.total_spike_opportunities <= 0.0:
            return float(self.total_spike_count)
        return float(self.total_spike_count / max(1.0, self.total_spike_opportunities))

    def reset_spike_counts(self):
        for b in self.blocks:
            b.spike_count = 0.0
        self.total_spike_count = 0.0
        self.total_spike_opportunities = 0.0
        self.last_diagnostics = {}

    def apply_plasticity(self, reward: float):
        # apply plasticity updates stored in each block
        if not self.use_plasticity:
            return
        for block in self.blocks:
            if block.use_plasticity and block.plasticity_rule is not None:
                # apply to v_proj weight (example)
                block.plasticity_rule.apply(block.v_proj.weight, reward)