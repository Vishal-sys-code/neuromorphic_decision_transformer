import torch
import torch.nn as nn
from norse.torch.module.lif import LIF, LIFCell, LIFParameters

from src.models.base import BasePolicy
from src.modules.fake_lif import FakeLIF as FakeLIFModule

class SpikingAttention(nn.Module):
    def __init__(self, d_model, n_heads, lif_tau, surrogate_k, use_fake_lif=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        if use_fake_lif:
            self.q_lif = FakeLIFModule()
            self.k_lif = FakeLIFModule()
        else:
            p = LIFParameters(tau_syn_inv=lif_tau, tau_mem_inv=lif_tau, v_th=torch.as_tensor(0.8))
            self.q_lif = LIF(p=p)
            self.k_lif = LIF(p=p)

        self.register_buffer('spike_count', torch.tensor(0.0))

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Pass None for initial state to let LIF handle it
        spikes_q, _ = self.q_lif(q, None)
        spikes_k, _ = self.k_lif(k, None)

        q_reshaped = spikes_q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k_reshaped = spikes_k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v_reshaped = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = (q_reshaped @ k_reshaped.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        attn_output = (attn_weights @ v_reshaped).permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        self.spike_count += spikes_q.sum() + spikes_k.sum()
        return attn_output


class DsFormer(BasePolicy, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.model.d_model

        self.embed_timestep = nn.Embedding(cfg.dataset.max_timesteps + 1, self.hidden_size)
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
                use_fake_lif=self.cfg.model.get("use_fake_lif", False),
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
        
        # Handle continuous vs discrete actions
        actions = batch["actions"]
        if self.cfg.dataset.is_discrete:
             action_input = torch.nn.functional.one_hot(
                actions.squeeze(-1).to(torch.int64), num_classes=self.cfg.dataset.act_dim
            ).float()
        else:
            action_input = actions

        action_embeddings = self.embed_action(action_input)

        # Pad action embeddings to match state/return sequence length
        if action_embeddings.shape[1] < seq_len:
            padding_size = seq_len - action_embeddings.shape[1]
            padding = torch.zeros(
                action_embeddings.shape[0],
                padding_size,
                action_embeddings.shape[2],
                device=action_embeddings.device,
            )
            action_embeddings = torch.cat([action_embeddings, padding], dim=1)

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
        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask=attn_mask)

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
        avg_spikes = total_spikes / len(self.blocks) if len(self.blocks) > 0 else 0.0
        return avg_spikes.item() if isinstance(avg_spikes, torch.Tensor) else avg_spikes

    def reset_spike_counts(self):
        for block in self.blocks:
            block.spike_count.zero_()

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)