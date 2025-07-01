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
from src.models.spiking_layers import SpikingSelfAttention

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
    def __init__(self, *args, time_window: int = 10, enable_action_head_plasticity: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace each vanilla TransformerBlock with our Spiking version
        self.transformer.h = nn.ModuleList([
            SpikingTransformerBlock(block, time_window)
            for block in self.transformer.h
        ])

        self.enable_action_head_plasticity = enable_action_head_plasticity
        if self.enable_action_head_plasticity:
            if not isinstance(self.predict_action, nn.Sequential) or not isinstance(self.predict_action[0], nn.Linear):
                raise TypeError("SNNDT: self.predict_action is not nn.Sequential with Linear as first element. Plasticity rule cannot be applied as expected.")

            self.action_linear_layer_ref = self.predict_action[0] # Direct reference to the nn.Linear layer

            self.captured_pre_syn_for_action = None
            self.captured_post_syn_for_action_logits = None # Will store logits output by predict_action

            self._register_hooks()

            # It's useful to have a device property if not already in base class
            if not hasattr(self, 'device'):
                # Attempt to infer device from parameters, assuming model has parameters
                try:
                    self.device = next(self.parameters()).device
                except StopIteration:
                    # Default to CPU if no parameters (e.g., model not fully built)
                    self.device = torch.device('cpu')


    def _hook_capture_pre_syn_for_action(self, module, input_args):
        # input_args is a tuple, input[0] is the actual tensor
        if input_args[0] is not None:
            self.captured_pre_syn_for_action = input_args[0].detach()

    def _hook_capture_post_syn_for_action_logits(self, module, input_args, output):
        # output is the direct output of self.predict_action (logits)
        if output is not None:
            self.captured_post_syn_for_action_logits = output.detach()

    def _register_hooks(self):
        if not hasattr(self, 'action_linear_layer_ref'):
            print("SNNDT Warning: action_linear_layer_ref not found. Hooks for plasticity not registered.")
            return

        # Hook to capture the input to the nn.Linear layer inside self.predict_action
        self.handle_pre_hook = self.action_linear_layer_ref.register_forward_hook(self._hook_capture_pre_syn_for_action)

        # Hook to capture the output of the entire self.predict_action nn.Sequential module
        self.handle_post_hook = self.predict_action.register_forward_hook(self._hook_capture_post_syn_for_action_logits)

    def get_captured_action_head_io(self):
        """
        Returns the captured input to the action head's linear layer and
        the captured output logits from the action head.
        Called by the training loop after a forward pass.
        """
        return self.captured_pre_syn_for_action, self.captured_post_syn_for_action_logits

    def clear_captured_action_head_io(self):
        """
        Clears the stored captured I/O data.
        Called by the training loop before a new forward pass.
        """
        self.captured_pre_syn_for_action = None
        self.captured_post_syn_for_action_logits = None

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        # If plasticity is enabled, ensure io is cleared before forward pass
        if self.enable_action_head_plasticity:
            self.clear_captured_action_head_io()

        # The hooks will capture the necessary data during the super().forward() call
        return super().forward(states, actions, rewards, returns_to_go, timesteps, attention_mask)

    # It's good practice to remove hooks when the model is no longer needed or before saving
    def remove_hooks(self):
        if hasattr(self, 'handle_pre_hook'):
            self.handle_pre_hook.remove()
        if hasattr(self, 'handle_post_hook'):
            self.handle_post_hook.remove()
        print("SNNDT: Removed plasticity hooks.")

    def __del__(self):
        # Ensure hooks are removed when the object is deleted
        if self.enable_action_head_plasticity:
            self.remove_hooks()
