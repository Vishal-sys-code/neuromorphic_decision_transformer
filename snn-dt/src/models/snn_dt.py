import torch
import torch.nn as nn
from norse.torch.module.leaky_integrator import LICell
from norse.torch.module.lif import LIF, LIFCell, LIFParameters

from src.models.base import BasePolicy


class SpikingTransformerBlock(nn.Module):
    def __init__(self, cfg, d_model, n_heads, lif_tau, surrogate_k):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Key, Query, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Spiking neurons
        p = LIFParameters(
            tau_mem_inv=torch.tensor(1.0 / lif_tau),
            v_th=torch.tensor(0.3),
            method="super",
            alpha=surrogate_k,
        )

        self.q_lif = LIFCell(p=p)
        self.k_lif = LIFCell(p=p)
        self.v_li = LICell()
        
        self.use_plasticity = False # Will be set by SnnDt
        self.eligibility_trace = None

        # Dendritic routing MLP
        self.routing_mlp = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, d_model),
            nn.Sigmoid(),
        )

        self.spike_count = 0

    def forward(self, x, state_q, state_k, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # --- FIX 1 & 3: Temporal Unfolding and Input Normalization ---
        x_time = x.transpose(0, 1)  # (seq, B, d_model)
        
        spikes_q_list, spikes_k_list = [], []
        state_q, state_k = None, None
        
        # Linear projections outside the loop
        q_proj = torch.tanh(self.q_proj(x_time))
        k_proj = torch.tanh(self.k_proj(x_time))
        v = self.v_proj(x)

        if not hasattr(self, "spike_count"):
            self.spike_count = 0.0

        for t in range(seq_len):
            # Use pre-projected inputs
            sp_q, state_q = self.q_lif(q_proj[t], state_q)
            sp_k, state_k = self.k_lif(k_proj[t], state_k)
            
            spikes_q_list.append(sp_q)
            spikes_k_list.append(sp_k)
            
            # --- FIX 4: Correct Spike Count Accumulation ---
            self.spike_count += float(sp_q.detach().sum() + sp_k.detach().sum())

        spikes_q = torch.stack(spikes_q_list, dim=0).transpose(0, 1) # (B, seq, d_model)
        spikes_k = torch.stack(spikes_k_list, dim=0).transpose(0, 1) # (B, seq, d_model)

        # Attention
        q_reshaped = spikes_q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k_reshaped = spikes_k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v_reshaped = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        attn_scores = torch.einsum("bnhd,bmhd->bhnm", q_reshaped, k_reshaped) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        attn_output = torch.einsum("bhnm,bmhd->bnhd", attn_weights, v_reshaped).reshape(batch_size, seq_len, self.d_model)
        
        # Dendritic routing
        routing_gate = self.routing_mlp(attn_output)
        out = attn_output * routing_gate
        
        # Three-factor plasticity
        if self.training and self.use_plasticity:
            self.update_eligibility_trace(spikes_q, v)
            
        return out

    def update_eligibility_trace(self, presynaptic_spikes, postsynaptic_potential):
        # Simplified eligibility trace update
        self.eligibility_trace += torch.einsum("bshd,bshd->hd", presynaptic_spikes, postsynaptic_potential)

    def apply_plasticity(self, reward):
        if self.use_plasticity:
            # Apply reward-modulated weight update
            self.v_proj.weight.data += self.cfg.snn.eta_local * reward * self.eligibility_trace
            self.eligibility_trace.zero_()


class PhaseEncoder(nn.Module):
    def __init__(self, d_model, max_timesteps):
        super().__init__()
        self.d_model = d_model
        self.max_timesteps = max_timesteps
        
        self.omegas = nn.Parameter(torch.randn(d_model // 2))
        self.phis = nn.Parameter(torch.randn(d_model // 2))

    def forward(self, timesteps):
        t = timesteps.float().unsqueeze(-1)
        omegas = self.omegas.view(1, 1, -1)
        phis = self.phis.view(1, 1, -1)
        
        cos_vals = torch.cos(t * omegas + phis)
        sin_vals = torch.sin(t * omegas + phis)
        
        return torch.cat([cos_vals, sin_vals], dim=-1)


class SnnDt(BasePolicy, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.model.d_model
        
        self.phase_encoder = PhaseEncoder(self.hidden_size, cfg.dataset.max_timesteps)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(cfg.dataset.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(cfg.dataset.act_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                cfg=self.cfg,
                d_model=self.hidden_size,
                n_heads=cfg.model.n_heads,
                lif_tau=cfg.snn.lif_tau,
                surrogate_k=cfg.snn.surrogate_k,
            )
            for _ in range(cfg.model.n_layers)
        ])

        self.action_predictor = nn.Linear(self.hidden_size, cfg.dataset.act_dim)

        self.use_plasticity = getattr(cfg.snn, "use_plasticity", False)
        if self.use_plasticity:
            for block in self.blocks:
                block.use_plasticity = True
                block.eligibility_trace = torch.zeros_like(block.v_proj.weight)

    def forward(self, batch):
        batch_size, seq_len = batch["states"].shape[:2]
        
        state_embeddings = self.embed_state(batch["states"])  # (B, seq_len, hidden)

        actions = batch["actions"]

        # Pad actions if they are shorter than states
        if actions.shape[1] < state_embeddings.shape[1]:
            padding_needed = state_embeddings.shape[1] - actions.shape[1]
            if actions.dim() == 3:
                actions = torch.nn.functional.pad(
                    actions, (0, 0, 0, padding_needed), "constant", 0
                )
            else:
                actions = torch.nn.functional.pad(
                    actions, (0, padding_needed), "constant", 0
                )
        
        # Handle discrete actions by one-hot encoding
        if self.cfg.dataset.is_discrete:
            action_input = torch.nn.functional.one_hot(
                actions.squeeze(-1).to(torch.int64), num_classes=self.cfg.dataset.act_dim
            ).float()
        else:
            action_input = actions
        
        action_embeddings = self.embed_action(action_input)
        return_embeddings = self.embed_return(batch["returns_to_go"])  # (B, seq_len, hidden)
        time_embeddings = self.phase_encoder(batch["timesteps"])  # (B, seq_len, hidden)

        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        return_embeddings += time_embeddings
        
        stacked_inputs = (
            torch.stack((return_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.hidden_size)
        )
        x = self.embed_ln(stacked_inputs)
        
        # Spiking transformer blocks
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        for i, block in enumerate(self.blocks):
            x = block(x, None, None, attn_mask=attn_mask)
        
        action_preds = self.action_predictor(x[:, 1::3])
        return action_preds

    @torch.no_grad()
    def predict_action(self, states, actions, returns_to_go, timesteps):
        device = next(self.parameters()).device
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        returns_to_go = torch.from_numpy(returns_to_go).float().to(device)
        timesteps = torch.from_numpy(timesteps).long().to(device)

        batch = {
            "states": states,
            "actions": actions,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "mask": torch.ones_like(states[..., 0]),
        }
        
        action_preds = self.forward(batch)
        return action_preds[0, -1].cpu().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def count_spikes(self):
        total_spikes = sum(block.spike_count for block in self.blocks)
        return total_spikes / len(self.blocks) if len(self.blocks) > 0 else 0.0

    def reset_spike_counts(self):
        for block in self.blocks:
            block.spike_count = 0.0

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)