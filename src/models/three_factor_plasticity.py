import torch
import torch.nn as nn

class ThreeFactorPlasticity(nn.Module):
    def __init__(self, eta, lambda_decay):
        super().__init__()
        self.eta = eta
        self.lambda_decay = lambda_decay
        
    def forward(self, eligibility_trace, pre_spikes, post_activity):
        """
        Updates the eligibility trace during forward pass (per timestep).
        E = lambda * E + (pre_spikes * post_activity)
        
        Args:
            eligibility_trace: (batch, out_features, in_features) or compatible shape
            pre_spikes: (batch, in_features)
            post_activity: (batch, out_features) - can be spikes or membrane potential
            
        Returns:
            Updated eligibility_trace
        """
        # Outer product per batch sample: (B, out, 1) @ (B, 1, in) -> (B, out, in)
        # Assuming post_activity is (B, out) and pre_spikes is (B, in)
        
        # Detach to stop gradients flowing through plasticity back to inputs during standard backprop
        pre = pre_spikes.detach()
        post = post_activity.detach()
        
        batch_update = torch.einsum("bo,bi->boi", post, pre)
        
        # Average over batch to update the shared trace?
        # Usually eligibility trace is per-synapse. If it's a global trace for the weight matrix:
        # We accumulate the batch average.
        
        update = batch_update.mean(dim=0)
        
        # Update trace
        return self.lambda_decay * eligibility_trace + update
        
    def update_weights(self, layer, eligibility_trace, reward):
        """
        Updates the weights after reward is known.
        Delta W = eta * reward * E
        W += Delta W
        """
        if eligibility_trace is None:
            return

        with torch.no_grad():
            delta_w = self.eta * reward * eligibility_trace
            layer.weight.data += delta_w