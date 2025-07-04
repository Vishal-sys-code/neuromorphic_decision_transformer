"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""
import os, sys
# Path setup for external modules, assuming a specific project structure.
# This might need adjustment based on how the project is run.
# Determine the project root directory to allow imports from 'novel_phases'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import math
import torch
import torch.nn as nn
from external.decision_transformer.gym.decision_transformer.models.decision_transformer import DecisionTransformer
from external.decision_transformer.gym.decision_transformer.models.trajectory_gpt2 import TransformerBlock as HuggingFaceTransformerBlock # Renamed to avoid conflict
from .spiking_layers import SpikingSelfAttention, LIFCell, LIFParameters

# Import Phase 3 modules
from ...novel_phases.phase3.positional_spike_encoder import PositionalSpikeEncoder
from ...novel_phases.phase3.dendritic_routing import DendriticRouter

class SpikingTransformerBlock(nn.Module):
    def __init__(self, 
                 block: HuggingFaceTransformerBlock, 
                 time_window: int, 
                 use_phase3_features: bool = True):
        super().__init__()
        self.ln1 = block.ln_1
        self.ln2 = block.ln_2
        
        num_heads = block.attn.n_head
        embed_dim = block.ln_1.normalized_shape[0] # This is typically n_embd

        self.pos_encoder = None
        self.dendritic_router = None

        if use_phase3_features:
            self.pos_encoder = PositionalSpikeEncoder(num_heads=num_heads, time_window=time_window)
            self.dendritic_router = DendriticRouter(num_heads=num_heads)

        self.snn_attn = SpikingSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            time_window=time_window,
            positional_encoder=self.pos_encoder,
            dendritic_router=self.dendritic_router
        )
        self.ff = block.mlp

    def forward(self, hidden_states, *args, **kwargs):
        a = self.snn_attn(self.ln1(hidden_states))
        x = hidden_states + a              
        f = self.ff(self.ln2(x))           
        hidden_states = x + f              
        return hidden_states

class SNNDecisionTransformer(DecisionTransformer):
    def __init__(self, 
                 *args, 
                 time_window: int = 10, 
                 lif_threshold: float = 0.1, 
                 action_lr: float = 1e-3, 
                 use_phase3_features: bool = True, # Added flag for Phase 3 features
                 **kwargs):
        super().__init__(*args, **kwargs) 

        self.transformer.h = nn.ModuleList([
            SpikingTransformerBlock(block, 
                                    time_window, 
                                    use_phase3_features=use_phase3_features) # Pass flag
            for block in self.transformer.h
        ])

        self.action_lin = nn.Linear(self.hidden_size, self.act_dim)
        action_lif_params = LIFParameters(threshold=lif_threshold, decay_constant=0.9, reset_potential=0.0)
        self.action_lif = LIFCell(input_size=self.act_dim, hidden_size=self.act_dim, p=action_lif_params)
        self.action_lif_learning_rate = action_lr
        self.batch_counter = 0

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions) 
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        transformer_outputs_dict = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x_transformer_all_modalities = transformer_outputs_dict['last_hidden_state']
        x_reshaped = x_transformer_all_modalities.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        transformer_hidden_states_for_actions = x_reshaped[:,1]

        x_seq_logits = self.action_lin(transformer_hidden_states_for_actions)

        action_lif_v = torch.zeros(batch_size, self.act_dim, device=x_seq_logits.device)
        list_of_spikes_t = []
        for t in range(seq_length):
            current_x_t = x_seq_logits[:, t, :]
            spikes_t, action_lif_v = self.action_lif(current_x_t, action_lif_v)
            list_of_spikes_t.append(spikes_t)
        all_spikes_seq = torch.stack(list_of_spikes_t, dim=1)

        if self.training and hasattr(self, 'batch_counter') and self.batch_counter < 5:
            total_spikes = all_spikes_seq.sum().item()
            if total_spikes == 0:
                print(f"WARNING: Batch {self.batch_counter}: No spikes from action_lif! Threshold={self.action_lif.p.threshold:.4f}, Max logit: {x_seq_logits.max().item():.4f}, Min logit: {x_seq_logits.min().item():.4f}")
            else:
                avg_spikes_per_neuron_step = total_spikes / (batch_size * seq_length * self.act_dim)
                print(f"Batch {self.batch_counter}: Total spikes from action_lif: {total_spikes}. Avg spikes/neuron/step: {avg_spikes_per_neuron_step:.4f}. Threshold={self.action_lif.p.threshold:.4f}")
        
        if self.training:
            transformer_output_detached = transformer_hidden_states_for_actions.detach()
            all_spikes_seq_detached = all_spikes_seq.detach()
            returns_to_go_detached = returns_to_go.detach()
            
            delta_W = torch.einsum('bs,bsa,bsd->ad', 
                                   returns_to_go_detached.squeeze(-1), 
                                   all_spikes_seq_detached, 
                                   transformer_output_detached)
            delta_W /= batch_size 

            if hasattr(self, 'batch_counter') and self.batch_counter < 5:
                delta_w_norm = torch.norm(delta_W).item()
                if delta_w_norm == 0:
                    print(f"WARNING: Batch {self.batch_counter}: delta_W is all zeros before applying learning rate. Spikes occurred: {all_spikes_seq.sum().item() > 0}. RTG sum: {returns_to_go_detached.sum().item():.4f}")
                else:
                    print(f"Batch {self.batch_counter}: delta_W norm: {delta_w_norm:.4g}. Max abs weight: {self.action_lin.weight.abs().max().item():.4g}, LR: {self.action_lif_learning_rate:.1e}")
            
            with torch.no_grad():
                self.action_lin.weight.data += self.action_lif_learning_rate * delta_W
        
        if self.training and hasattr(self, 'batch_counter') and self.batch_counter < 5:
             self.batch_counter +=1

        return_preds = self.predict_return(x_reshaped[:,2])
        state_preds = self.predict_state(x_reshaped[:,2])
        action_preds = x_seq_logits 

        return state_preds, action_preds, return_preds