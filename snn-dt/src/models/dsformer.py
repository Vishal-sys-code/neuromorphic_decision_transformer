import torch
import torch.nn as nn
from norse.torch.module.lif import LIFCell

from src.models.base import BasePolicy


class SpikingAttention(nn.Module):
    def __init__(self, d_model, n_heads, lif_tau, surrogate_k):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.q_lif = LIFCell()
        self.k_lif = LIFCell()

        self.spike_count = 0

    def forward(self, x, state_q, state_k, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        spikes_q_seq = []
        spikes_k_seq = []
        for t in range(seq_len):
            spikes_q, state_q = self.q_lif(q[:, t], state_q)
            spikes_k, state_k = self.k_lif(k[:, t], state_k)
            spikes_q_seq.append(spikes_q)
            spikes_k_seq.append(spikes_k)
        spikes_q = torch.stack(spikes_q_seq, dim=1)
        spikes_k = torch.stack(spikes_k_seq, dim=1)

        q_reshaped = spikes_q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k_reshaped = spikes_k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v_reshaped = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = (q_reshaped @ k_reshaped.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        attn_output = (attn_weights @ v_reshaped).permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        self.spike_count = spikes_q.sum() + spikes_k.sum()
        return attn_output, state_q, state_k


class DsFormer(BasePolicy, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.model.d_model

        self.embed_timestep = nn.Embedding(cfg.dataset.max_timesteps, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(cfg.dataset.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(cfg.dataset.act_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.blocks = nn.ModuleList([
            SpikingAttention(
                d_model=self.hidden_size,
                n_heads=cfg.model.n_heads,
                lif_tau=cfg.snn.lif_tau,
                surrogate_k=cfg.snn.surrogate_k,
            )
            for _ in range(cfg.model.n_layers)
        ])

        self.action_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, cfg.dataset.act_dim),
            nn.Tanh() if cfg.model.action_tanh else nn.Identity(),
        )

    def forward(self, batch):
        batch_size, seq_len = batch["states"].shape[:2]

        state_embeddings = self.embed_state(batch["states"])
        action_embeddings = self.embed_action(batch["actions"])
        return_embeddings = self.embed_return(batch["returns_to_go"])
        time_embeddings = self.embed_timestep(batch["timesteps"])

        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        return_embeddings += time_embeddings

        stacked_inputs = (
            torch.stack((return_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.hidden_size)
        )
        x = self.embed_ln(stacked_inputs)

        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        q_states = [None] * len(self.blocks)
        k_states = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            x, q_states[i], k_states[i] = block(x, q_states[i], k_states[i], attn_mask=attn_mask)

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
        return sum(b.spike_count for b in self.blocks)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)