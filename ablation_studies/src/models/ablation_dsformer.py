import torch
import torch.nn as nn
import math

# --- Base Class (for compatibility) ---
class BasePolicy:
    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))

# --- Component Modules ---

class PhaseSpikeEncoder(nn.Module):
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.omega = nn.Parameter(torch.randn(n_heads))
        self.phi = nn.Parameter(torch.randn(n_heads))

    def forward(self, timesteps):
        t = timesteps.float().unsqueeze(-1)
        w = self.omega.view(1, 1, -1)
        p = self.phi.view(1, 1, -1)
        wave = torch.sin(t * w + p)
        spikes = (wave > 0).float()
        repeats = self.d_model // self.n_heads
        spikes = spikes.repeat_interleave(repeats, dim=-1)
        if spikes.shape[-1] != self.d_model:
            padding = self.d_model - spikes.shape[-1]
            spikes = torch.cat([spikes, spikes[:, :, :padding]], dim=-1)
        return spikes

class DendriticRouter(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_heads))

    def forward(self, x_heads): # x_heads is (B, L, H, D_h)
        batch, seq_len, _, _ = x_heads.shape
        x_concat = x_heads.view(batch, seq_len, -1)
        gates = self.mlp(x_concat)
        alpha = torch.softmax(gates, dim=-1).unsqueeze(-1)
        y_weighted = x_heads * alpha
        return y_weighted

class ThreeFactorPlasticity:
    def __init__(self, eta_local):
        self.eta_local = eta_local

    def update_weights(self, layer, eligibility_trace, reward):
        if eligibility_trace is None or not torch.is_tensor(eligibility_trace): return
        with torch.no_grad():
            delta_w = self.eta_local * reward * eligibility_trace
            layer.weight.data += delta_w

class SpikingAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.hidden_dim_d
        self.n_heads = cfg.num_heads_H
        self.head_dim = self.d_model // self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.lif = lambda x: (torch.rand_like(x) < torch.sigmoid(x - cfg.surrogate_slope_k)).float()

        if self.cfg.routing.enabled:
            self.router = DendriticRouter(self.d_model, self.n_heads)
        
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(nn.Linear(self.d_model, 4 * self.d_model), nn.GELU(), nn.Linear(4 * self.d_model, self.d_model))
        self.register_buffer('spike_count', torch.tensor(0.0))

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        res = x
        x = self.ln1(x)
        
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        spikes_q, spikes_k = self.lif(q), self.lif(k)

        q_r = spikes_q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k_r = spikes_k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v_r = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q_r @ k_r.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask[:, :, :L, :L] == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_heads = (attn_weights @ v_r).transpose(1, 2).contiguous()

        if self.cfg.routing.enabled:
            attn_heads = self.router(attn_heads)
        else:
            attn_heads = attn_heads.mean(dim=2).view(B, L, 1, self.head_dim).expand_as(attn_heads)
        
        attn_concat = attn_heads.reshape(B, L, D)
        attn_output = self.out_proj(attn_concat)
        x = res + attn_output

        res = x
        x = self.ln2(x)
        x = res + self.ffn(x)

        self.spike_count += (spikes_q.sum() + spikes_k.sum()).detach()
        return x, None

class AblationDsFormer(BasePolicy, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.hidden_dim_d
        
        self.use_phase_encoder = cfg.phase_encoder.enabled
        self.embed_timestep = PhaseSpikeEncoder(d_model=self.d_model, n_heads=cfg.num_heads_H) if self.use_phase_encoder else nn.Embedding(cfg.dataset.max_timesteps + 1, self.d_model)
        self.embed_return = nn.Linear(1, self.d_model)
        self.embed_state = nn.Linear(cfg.dataset.state_dim, self.d_model)
        self.embed_action = nn.Linear(cfg.dataset.act_dim, self.d_model)
        self.embed_ln = nn.LayerNorm(self.d_model)

        self.blocks = nn.ModuleList([SpikingAttentionBlock(cfg) for _ in range(cfg.num_layers_L)])
        self.action_predictor = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, cfg.dataset.act_dim))

        self.use_plasticity = cfg.local_plasticity.enabled
        if self.use_plasticity:
            self.plasticity_rule = ThreeFactorPlasticity(cfg.local_lr_eta_local)
            self.register_buffer('eligibility_trace', torch.zeros_like(self.action_predictor[1].weight))

    def forward(self, batch):
        B, L = batch["states"].shape[:2]
        
        state_embed = self.embed_state(batch["states"])
        action_embed = self.embed_action(batch["actions"])
        rtg_embed = self.embed_return(batch["returns_to_go"].float())
        time_embed = self.embed_timestep(batch["timesteps"].squeeze(-1))

        state_embed, action_embed, rtg_embed = state_embed + time_embed, action_embed + time_embed, rtg_embed + time_embed

        x = torch.stack([rtg_embed, state_embed, action_embed], dim=1).permute(0, 2, 1, 3).reshape(B, 3 * L, self.d_model)
        x = self.embed_ln(x)

        causal_mask = torch.ones((1, 1, 3 * L, 3 * L), device=x.device).tril()
        
        for block in self.blocks:
            x, _ = block(x, attn_mask=causal_mask)

        action_features = x[:, 1::3, :]
        action_preds = self.action_predictor(action_features)

        if self.use_plasticity and self.training:
            with torch.no_grad():
                pre = action_features.mean([0, 1])
                post = action_preds.mean([0, 1])
                self.eligibility_trace = 0.99 * self.eligibility_trace + torch.einsum("o,i->oi", post, pre)

        return action_preds, None

    def apply_plasticity(self, reward):
        if self.use_plasticity:
            self.plasticity_rule.update_weights(self.action_predictor[1], self.eligibility_trace, reward)
            if torch.is_tensor(self.eligibility_trace): self.eligibility_trace.zero_()

    def count_spikes(self):
        return sum(block.spike_count.item() for block in self.blocks)
    
    def reset_spike_counts(self):
        for block in self.blocks:
            block.spike_count.zero_()