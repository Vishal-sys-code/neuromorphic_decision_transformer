"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""
import os, sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ext = os.path.join(root, "external")
if ext not in sys.path:
    sys.path.insert(0, ext)
import math
import torch
import torch.nn as nn
from external.decision_transformer.gym.decision_transformer.models.decision_transformer import DecisionTransformer
from external.decision_transformer.gym.decision_transformer.models.trajectory_gpt2 import TransformerBlock as TransformerBlock
from src.models.spiking_layers import SpikingSelfAttention, LIFCell, LIFParameters

class SpikingTransformerBlock(nn.Module):
    def __init__(self, block: TransformerBlock, time_window: int):
        super().__init__()
        # reuse the original layer norms
        self.ln1 = block.ln_1
        self.ln2 = block.ln_2
        # spiking self‑attention in place of block.attn
        self.snn_attn = SpikingSelfAttention(
            embed_dim=block.ln_1.normalized_shape[0],
            num_heads=block.attn.n_head,
            time_window=time_window
        )
        # keep the original feed‑forward (FFN) as is
        self.ff = block.mlp

    def forward(self, hidden_states, *args, **kwargs):
        # Self‑Attention pass
        a = self.snn_attn(self.ln1(hidden_states))
        x = hidden_states + a               # residual
        # Feed‑forward pass
        f = self.ff(self.ln2(x))           # [B, S, E]
        hidden_states = x + f              # residual
        # Return only the hidden_states tensor to match expected output
        return hidden_states

class SNNDecisionTransformer(DecisionTransformer):
    # Removed enable_action_head_plasticity from __init__
    def __init__(self, *args, time_window: int = 10, lif_threshold: float = 0.1, action_lr: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs) # This will initialize self.hidden_size, self.act_dim, self.predict_action etc.

        # Replace each vanilla TransformerBlock with our Spiking version (as before)
        self.transformer.h = nn.ModuleList([
            SpikingTransformerBlock(block, time_window)
            for block in self.transformer.h
        ])

        # New action head with LIF neurons
        # self.hidden_size and self.act_dim are available from super().__init__
        self.action_lin = nn.Linear(self.hidden_size, self.act_dim)
        
        # Configure LIF parameters for the action head
        # Making threshold configurable via __init__
        action_lif_params = LIFParameters(threshold=lif_threshold, decay_constant=0.9, reset_potential=0.0)
        self.action_lif = LIFCell(input_size=self.act_dim, hidden_size=self.act_dim, p=action_lif_params)

        # Learning rate for the local rule on action_lin weights
        self.action_lif_learning_rate = action_lr

        # Counter for initial batch checks (spike activity, eligibility trace)
        self.batch_counter = 0
        
        # The original self.predict_action is no longer used for action predictions.
        # However, the base class's forward method might try to use it if we call super().forward().
        # We are writing our own forward method, so we control what's called.
        # For other predictions (state, return), we'll use the original layers from the base class.
        # No need to delete self.predict_action, just don't use it for actions.

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # Embed each modality (code from base DecisionTransformer.forward)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions) # Used for input, not target for this model
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
        # x is the last hidden state from the transformer
        x_transformer_all_modalities = transformer_outputs_dict['last_hidden_state']
        
        # Reshape x to separate modalities (R, S, A)
        # x_reshaped has shape (batch_size, 3, seq_length, hidden_size)
        x_reshaped = x_transformer_all_modalities.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # Output of transformer corresponding to states (s_t) is used to predict actions a_t
        # This is (batch_size, seq_length, hidden_size)
        transformer_hidden_states_for_actions = x_reshaped[:,1] # s_t tokens

        # --- New Action Head Logic ---
        # 1. Linear layer
        # transformer_hidden_states_for_actions is (batch, seq_len, d_model)
        # x_seq is (batch, seq_len, action_dim) - these are the "currents" or "logits"
        x_seq_logits = self.action_lin(transformer_hidden_states_for_actions)

        # 2. LIF layer processing sequence
        # Initialize LIF state (membrane potentials)
        action_lif_v = torch.zeros(batch_size, self.act_dim, device=x_seq_logits.device)
        
        list_of_spikes_t = []
        for t in range(seq_length):
            current_x_t = x_seq_logits[:, t, :]  # Input current for this timestep (batch, action_dim)
            spikes_t, action_lif_v = self.action_lif(current_x_t, action_lif_v)
            list_of_spikes_t.append(spikes_t)
        
        all_spikes_seq = torch.stack(list_of_spikes_t, dim=1) # (batch, seq_len, action_dim)

        # Spike Check (for the first few batches during training)
        if self.training and hasattr(self, 'batch_counter') and self.batch_counter < 5:
            if all_spikes_seq.sum().item() == 0:
                print(f"WARNING: Batch {self.batch_counter}: No spikes from action_lif! Threshold={self.action_lif.p.threshold}")
            else:
                print(f"Batch {self.batch_counter}: Total spikes from action_lif: {all_spikes_seq.sum().item()}. Avg spikes/sample/timestep: {all_spikes_seq.sum().item()/(batch_size*seq_length*self.act_dim):.4f}")
        
        # Eligibility Trace and Weight Update (only during training)
        if self.training:
            # Detach pre-synaptic and post-synaptic activities for local rule calculation
            transformer_output_detached = transformer_hidden_states_for_actions.detach()
            all_spikes_seq_detached = all_spikes_seq.detach()
            returns_to_go_detached = returns_to_go.detach() # (batch_size, seq_len, 1)

            # Calculate delta_W = sum_{b,s} ( G[b,s] * outer(spikes[b,s,:], hidden[b,s,:]) ) / batch_size
            # G[b,s] is returns_to_go_detached[b,s,0]
            # spikes[b,s,:] is all_spikes_seq_detached[b,s,:] (shape: action_dim)
            # hidden[b,s,:] is transformer_output_detached[b,s,:] (shape: hidden_size/d_model)
            # Outer product: (action_dim, 1) * (1, hidden_size) -> (action_dim, hidden_size)
            
            # More efficient computation of delta_W using einsum
            # rtg: (B, S, 1), spikes: (B, S, A), hidden: (B, S, D)
            # We want delta_W of shape (A, D)
            # einsum: rtg_bs * spikes_bsa * hidden_bsd -> ad
            delta_W = torch.einsum('bs,bsa,bsd->ad', 
                                   returns_to_go_detached.squeeze(-1), 
                                   all_spikes_seq_detached, 
                                   transformer_output_detached)
            delta_W /= batch_size # Average over batch

            if hasattr(self, 'batch_counter') and self.batch_counter < 5: # Check for a few batches
                if torch.all(delta_W == 0):
                    print(f"WARNING: Batch {self.batch_counter}: delta_W is all zeros before applying learning rate. Non-zero spikes: {all_spikes_seq.sum().item() > 0}")
                else:
                    print(f"Batch {self.batch_counter}: delta_W norm: {torch.norm(delta_W).item()}")
            
            with torch.no_grad():
                self.action_lin.weight.data += self.action_lif_learning_rate * delta_W
        
        if self.training and hasattr(self, 'batch_counter') and self.batch_counter < 5 : # only increment if training and within check window
             self.batch_counter +=1


        # Predictions for return and state (as in base class, using original prediction heads)
        # x_reshaped[:,2] corresponds to the transformer output for the last element in (R_t, s_t, a_t) sequence,
        # which is a_t. The base model uses this to predict R_{t+1} and s_{t+1}.
        return_preds = self.predict_return(x_reshaped[:,2])
        state_preds = self.predict_state(x_reshaped[:,2])
        
        # The `action_preds` for the environment should be the logits before spiking.
        action_preds = x_seq_logits 

        return state_preds, action_preds, return_preds