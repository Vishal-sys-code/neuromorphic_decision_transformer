import numpy as np
import torch
import torch.nn as nn

from src.models.base import BasePolicy


class DecisionTransformer(BasePolicy, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.state_dim = cfg.dataset.state_dim
        self.act_dim = cfg.dataset.act_dim
        self.hidden_size = cfg.model.d_model

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=cfg.model.n_heads,
                dim_feedforward=4 * self.hidden_size,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            ),
            num_layers=cfg.model.n_layers,
        )

        self.embed_timestep = nn.Embedding(cfg.dataset.max_timesteps, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(self.act_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.action_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.act_dim),
            nn.Tanh() if cfg.model.action_tanh else nn.Identity(),
        )

    def forward(self, batch):
        states, actions, returns_to_go, timesteps, mask = (
            batch["states"],
            batch["actions"],
            batch["returns_to_go"],
            batch["timesteps"],
            batch["mask"],
        )

        state_embeddings = self.embed_state(states)

        # Handle discrete actions by one-hot encoding
        if self.act_dim > 1 and actions.shape[-1] == 1:
            action_input = torch.nn.functional.one_hot(
                actions.squeeze(-1).to(torch.int64), num_classes=self.act_dim
            ).float()
        else:
            action_input = actions
        action_embeddings = self.embed_action(action_input)

        return_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Time embeddings are added to state, action, and return embeddings
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        return_embeddings += time_embeddings

        # This is for GPT-2 like stile of stacking embeddings
        stacked_inputs = (
            torch.stack((return_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(states.shape[0], 3 * states.shape[1], self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Causal mask to ensure predictions for a timestep are made using only previous timesteps
        attn_mask = nn.Transformer.generate_square_subsequent_mask(stacked_inputs.shape[1], device=states.device)

        transformer_outputs = self.transformer(stacked_inputs, mask=attn_mask)

        # Predict actions from state embeddings
        action_preds = self.action_predictor(transformer_outputs[:, 1::3]) # Predict action from state embedding
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
        return 0  # Not a spiking model

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)